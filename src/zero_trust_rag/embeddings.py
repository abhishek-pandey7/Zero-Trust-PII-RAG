"""
embeddings.py – Sentence embeddings computed inside the trust boundary.

all-MiniLM-L6-v2 runs locally through ONNX Runtime (the build ChromaDB ships), so chunks are
embedded on this machine and never leave it. The model is ~80 MB, downloaded once and cached.
"""
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """all-MiniLM-L6-v2 via ONNX Runtime: 384-dimension vectors, CPU only, no network after download."""

    dimensions = 384

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._model = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model(texts)]

    def warm_up(self) -> None:
        """Load (and, on first run, download) the model now rather than on the first request."""
        self.embed(["warm up"])
        logger.info("Local embedding model loaded")
