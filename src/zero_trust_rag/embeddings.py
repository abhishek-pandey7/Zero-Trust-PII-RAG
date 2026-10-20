"""
embeddings.py – Sentence embeddings computed inside the trust boundary.

all-MiniLM-L6-v2 runs locally through ONNX Runtime, so chunks are embedded on this machine and
never leave it. The model is ~80 MB, downloaded once (via ChromaDB's downloader) and cached.

The forward pass is our own rather than ChromaDB's, for memory. ChromaDB pads every text to 256
tokens, runs batches of 32 and leaves ONNX Runtime's memory arena on, so embedding 100 chunks
grew the process by ~385 MB and got it OOM-killed on a 512 MB host. Padding to the longest text
in the batch gives identical vectors (padding is masked out of attention and pooling) at a
fraction of the memory.
"""
import logging
import os
from pathlib import Path
from typing import Protocol

import numpy as np

# The Rust tokenizer otherwise starts a thread pool with its own buffers; batches here are small
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

# Inside the project by default: on hosts like Render only the project directory survives from the
# build to the running service, so a model cached under ~/.cache at build time would be lost
MODEL_DIR = Path(os.getenv("ZT_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
MAX_TOKENS = 256  # what sentence-transformers uses for this model


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _download_model() -> Path:
    """Fetch the ONNX model and tokenizer if they aren't cached yet; return their folder."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _Pinned(ONNXMiniLM_L6_V2):
        DOWNLOAD_PATH = MODEL_DIR / ONNXMiniLM_L6_V2.MODEL_NAME

    downloader = _Pinned(preferred_providers=["CPUExecutionProvider"])
    downloader._download_model_if_not_exists()
    return Path(_Pinned.DOWNLOAD_PATH) / _Pinned.EXTRACTED_FOLDER_NAME


class LocalEmbedder:
    """all-MiniLM-L6-v2 via ONNX Runtime: 384-dimension vectors, CPU only, no network after download."""

    dimensions = 384

    def __init__(self, batch_size: int = 8, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = _download_model()

        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False  # return activation memory after each batch
        options.enable_mem_pattern = False  # don't pre-plan buffers per input shape; shapes vary
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # to the longest in the batch
        self._batch_size = batch_size

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        hidden = self._session.run(None, {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        })[0]
        # Mean of the token vectors, ignoring padding, then unit length (cosine-ready)
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self._batch_size]).tolist())
        return vectors

    def warm_up(self) -> None:
        """Run one batch now so the first request doesn't pay for session start-up."""
        self.embed(["warm up"])
        logger.info("Local embedding model loaded from %s", MODEL_DIR)
