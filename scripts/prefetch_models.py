"""
prefetch_models.py – Download the local embedding model at build time.

Run as part of the deploy build so the first request after a deploy doesn't wait on an 80 MB
download. The spaCy model is installed by requirements.txt.

    python scripts/prefetch_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.zero_trust_rag.embeddings import MODEL_DIR, LocalEmbedder  # noqa: E402

if __name__ == "__main__":
    LocalEmbedder().warm_up()
    print(f"Embedding model ready in {MODEL_DIR}")
