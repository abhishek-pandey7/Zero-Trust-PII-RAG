"""
pipeline.py – Zero-Trust PII Redacting RAG Pipeline.

This module provides a complete RAG pipeline with:
- PII detection and masking before any external API calls
- Thread-safe vault with SQLite persistence
- Error handling with retry logic
- Audit logging for compliance
- Configurable parameters via config.yaml
"""
import hashlib
import logging
import os
import shutil

import requests as _requests
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .audit_logger import AuditLogger
from .config import Config
from .exceptions import APIError, RedactionError, ValidationError, ZeroTrustRAGError
from .pii_redactor import PIIRedactor
from .utils import (
    retry_on_api_error,
    validate_api_key,
    validate_input,
)
from .vault import PIIVault


class _HFApiEmbeddings(Embeddings):
    """Ultra-lightweight HuggingFace Inference API embeddings — no torch, no local models.
    Uses huggingface_hub InferenceClient which handles routing and warm-up automatically.
    """
    def __init__(self, api_key: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=api_key)
        except ImportError:
            self._client = None
        self.api_key = api_key
        self.model = model
        # Keep raw requests fallback URLs
        self._url_pipeline = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        self._url_models   = f"https://api-inference.huggingface.co/models/{model}"

    def _mean_pool(self, token_vecs: list) -> list:
        """Mean-pool token embeddings to a single sentence vector."""
        n = len(token_vecs)
        dim = len(token_vecs[0])
        return [sum(token_vecs[t][d] for t in range(n)) / n for d in range(dim)]

    def _embed_via_client(self, texts: list) -> list:
        """Use huggingface_hub InferenceClient (preferred path)."""
        embeddings = []
        for text in texts:
            vec = self._client.feature_extraction(text=text, model=self.model)
            # vec may be numpy array or nested list
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            # If 2D (seq_len, dim), mean-pool
            if vec and isinstance(vec[0], list):
                vec = self._mean_pool(vec)
            embeddings.append(vec)
        return embeddings

    def _embed_via_requests(self, texts: list) -> list:
        """Fallback: raw HTTP to the pipeline endpoint."""
        for url in [self._url_pipeline, self._url_models]:
            try:
                resp = _requests.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=90
                )
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                result = resp.json()
                embeddings = []
                for item in result:
                    if item and isinstance(item[0], list):
                        embeddings.append(self._mean_pool(item))
                    else:
                        embeddings.append(item)
                return embeddings
            except Exception:
                continue
        raise RuntimeError(
            f"HuggingFace embedding API unavailable for model '{self.model}'. "
            "Check that HUGGINGFACEHUB_API_TOKEN is valid and the model is hosted on the free tier."
        )

    def _embed(self, texts: list) -> list:
        if self._client is not None:
            try:
                return self._embed_via_client(texts)
            except Exception:
                pass  # fall through to raw requests
        return self._embed_via_requests(texts)

    def embed_documents(self, texts: list) -> list:
        return self._embed(texts)

    def embed_query(self, text: str) -> list:
        return self._embed([text])[0]


logger = logging.getLogger(__name__)
config = Config()


class ZeroTrustRAG:
    """
    Zero-Trust PII Redacting RAG Pipeline.
    
    Security Model:
    1. All PII is masked BEFORE sending to any external API
    2. External LLMs only see masked placeholders (e.g., <<PERSON_01>>)
    3. Unmasking happens only on secure local server
    4. All operations are audited for compliance
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        hf_api_key: str | None = None,
        persist_directory: str | None = None
    ):
        self._init_security()
        self._init_components(groq_api_key, hf_api_key, persist_directory)
        self._init_splitter()
        self._init_vector_db()

    def _init_security(self) -> None:
        """Initialize security components (vault and audit logger)."""
        self.vault = PIIVault()
        self.audit = AuditLogger()
        logger.info("Security components initialized")

    def _init_components(
        self,
        groq_api_key: str | None,
        hf_api_key: str | None,
        persist_directory: str | None
    ) -> None:
        """Initialize LLM and embedding components."""
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        hf_api_key = hf_api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        validate_api_key(groq_api_key, "Groq")
        validate_api_key(hf_api_key, "HuggingFace")

        self.embeddings = _HFApiEmbeddings(
            api_key=hf_api_key,
            model=config.get("embeddings.model", "sentence-transformers/all-MiniLM-L6-v2")
        )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=config.get("llm.model", "llama-3.3-70b-versatile"),
            temperature=config.get("llm.temperature", 0)
        )

        self.redactor = PIIRedactor(self.vault, self.audit)
        
        self.audit.log_operation("initialization", "success", {
            "llm_model": config.get("llm.model"),
            "embeddings_model": config.get("embeddings.model")
        })

    def _init_splitter(self) -> None:
        """Initialize text splitter with configured parameters."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("splitter.chunk_size", 500),
            chunk_overlap=config.get("splitter.chunk_overlap", 100),
            separators=config.get("splitter.separators", ["\n\n", "\n", " ", ""]),
            length_function=len
        )

    def _init_vector_db(self) -> None:
        """Initialize vector database with persistence."""
        self._persist_directory = config.get("chroma.persist_directory", "./chroma_store")
        self.vector_db = Chroma(
            persist_directory=self._persist_directory,
            embedding_function=self.embeddings
        )

    @staticmethod
    def _stable_id(text: str) -> str:
        """Generate a deterministic UUID-like ID from text content to prevent duplicate entries."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Format as UUID: 8-4-4-4-12
        return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

    @retry_on_api_error()
    def _add_texts_to_vector_db(self, texts: list[str]) -> None:
        if self.vector_db is None:
            self.vector_db = Chroma(
                persist_directory=self._persist_directory,
                embedding_function=self.embeddings
            )
        # Use content-derived stable IDs so re-ingesting the same text
        # does not produce duplicate ChromaDB entries with new UUIDs.
        ids = [self._stable_id(t) for t in texts]
        self.vector_db.add_texts(texts, ids=ids)

    def ingest(self, text_content: str) -> dict:
        """
        Ingest text content into the RAG pipeline.
        
        SECURITY: PII is masked BEFORE embedding and storage.
        """
        try:
            validate_input(text_content)
        except ValidationError as e:
            self.audit.log_operation("ingest", "error", {"error": str(e)})
            raise

        try:
            masked_text, redaction_log = self.redactor.redact(text_content, "ingest")
            
            validate_input(masked_text)
            
            chunks = self.splitter.split_text(masked_text)
            
            self._add_texts_to_vector_db(chunks)
            
            self.audit.log_operation("ingest", "success", {
                "chunks_created": len(chunks),
                "pii_redacted": len(redaction_log)
            })
            
            logger.info(f"Ingested {len(chunks)} chunks, redacted {len(redaction_log)} PII entities")
            
            return {
                "chunks_created": len(chunks),
                "pii_redacted": len(redaction_log),
                "redaction_details": redaction_log
            }
            
        except Exception as e:
            self.audit.log_operation("ingest", "error", {"error": str(e)})
            raise RedactionError(f"Ingestion failed: {e}") from e

    @retry_on_api_error()
    def ask(self, question: str) -> str:
        """
        Ask a question using the RAG pipeline.
        
        SECURITY: Question is masked BEFORE retrieval and LLM inference.
        """
        try:
            validate_input(question)
        except ValidationError as e:
            self.audit.log_operation("ask", "error", {"error": str(e)})
            raise

        try:
            masked_q, redaction_log_q = self.redactor.redact(question, "query")
            
            top_k = config.get("retrieval.top_k", 3)
            docs = self.vector_db.similarity_search(masked_q, k=top_k)
            context = "\n".join([d.page_content for d in docs])
            
            if not context:
                self.audit.log_operation("ask", "error", {"error": "No relevant context found"})
                return "I don't have enough context to answer that question."

            prompt = ChatPromptTemplate.from_template(
                """You are a secure data assistant. 
                The context below uses placeholders like <<PERSON_01>> to protect privacy.
                Answer the question accurately using these placeholders.

                Context: {context}
                Question: {question}

                Answer:"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            
            try:
                masked_answer = chain.invoke({
                    "context": context,
                    "question": masked_q
                })
            except Exception as e:
                raise APIError(f"LLM inference failed: {e}", provider="Groq") from e

            final_answer = self.redactor.unmask(masked_answer)
            
            self.audit.log_operation("ask", "success", {
                "pii_in_question": len(redaction_log_q),
                "context_chunks": len(docs)
            })
            
            logger.info(f"Query processed: {len(redaction_log_q)} PII masked, {len(docs)} chunks retrieved")
            
            return final_answer
            
        except ZeroTrustRAGError:
            raise
        except Exception as e:
            self.audit.log_operation("ask", "error", {"error": str(e)})
            raise APIError(f"Query failed: {e}", provider="Groq") from e

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "vault_stats": self.vault.get_stats(),
            "persist_directory": self._persist_directory
        }

    def clear_vector_db(self) -> None:
        """Remove all documents from the Chroma collection without dropping the store."""
        if self.vector_db is None:
            self.vector_db = Chroma(
                persist_directory=self._persist_directory,
                embedding_function=self.embeddings
            )
        try:
            all_ids = self.vector_db.get()["ids"]
            if all_ids:
                self.vector_db.delete(ids=all_ids)
                logger.info(f"Deleted {len(all_ids)} documents from vector DB")
            else:
                logger.info("Vector DB already empty")
        except Exception as e:
            logger.warning(f"Could not clear vector DB documents: {e}")

    def reset_vector_db(self) -> None:
        """Wipe the persisted ChromaDB store from disk entirely.

        Use this before evaluation runs to guarantee a clean slate and
        prevent stale document IDs from previous sessions from appearing
        in retrieval results.
        """
        self.vector_db = None
        store_path = self._persist_directory
        if os.path.exists(store_path):
            shutil.rmtree(store_path)
            logger.info(f"Wiped persisted vector store at: {store_path}")
        # Re-initialise a fresh empty store
        self._init_vector_db()
        logger.info("Fresh empty vector DB initialised")