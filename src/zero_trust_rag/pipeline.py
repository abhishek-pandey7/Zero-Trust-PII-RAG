"""
pipeline.py – Zero-Trust PII Redacting RAG Pipeline.

This module provides a complete RAG pipeline with:
- PII detection and masking before any external API calls
- Thread-safe vault with SQLite persistence
- Error handling with retry logic
- Audit logging for compliance
- Configurable parameters via config.yaml
"""
import os
import logging
from typing import Optional
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .vault import PIIVault
from .pii_redactor import PIIRedactor
from .audit_logger import AuditLogger
from .config import Config
from .utils import (
    validate_api_key,
    validate_input,
    retry_on_api_error,
)
from .exceptions import (
    APIError,
    ValidationError,
    RedactionError,
    ZeroTrustRAGError
)

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
        groq_api_key: Optional[str] = None,
        hf_api_key: Optional[str] = None,
        persist_directory: Optional[str] = None
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
        groq_api_key: Optional[str],
        hf_api_key: Optional[str],
        persist_directory: Optional[str]
    ) -> None:
        """Initialize LLM and embedding components."""
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        hf_api_key = hf_api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        validate_api_key(groq_api_key, "Groq")
        validate_api_key(hf_api_key, "HuggingFace")

        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=config.get("embeddings.model", "sentence-transformers/all-MiniLM-L6-v2"),
            huggingfacehub_api_token=hf_api_key
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

    @retry_on_api_error()
    def _add_texts_to_vector_db(self, texts: list[str]) -> None:
        if self.vector_db is None:
            self.vector_db = Chroma(
                persist_directory=self._persist_directory,
                embedding_function=self.embeddings
            )
        self.vector_db.add_texts(texts)

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
                raise APIError(f"LLM inference failed: {e}", provider="Groq")

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
        """Clear the vector database."""
        if self.vector_db:
            self.vector_db = None
            logger.info("Vector database cleared")