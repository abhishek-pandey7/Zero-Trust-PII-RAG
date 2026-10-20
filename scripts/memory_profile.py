"""
memory_profile.py – Resident memory after each stage of loading and using the pipeline.

Run inside the deploy image to see where the 512 MB budget goes:

    docker run --rm -m 512m zt python scripts/memory_profile.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rss_mb() -> tuple[float, float]:
    """Current and peak resident memory in MB (Linux /proc; psutil elsewhere)."""
    status = Path("/proc/self/status")
    if status.exists():
        fields = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
        return int(fields["VmRSS"].split()[0]) / 1024, int(fields["VmHWM"].split()[0]) / 1024
    import psutil
    info = psutil.Process().memory_info()
    return info.rss / 1048576, getattr(info, "peak_wset", info.rss) / 1048576


def stage(label: str, started: float) -> None:
    now, peak = rss_mb()
    print(f"{label:<34} {now:6.0f} MB   peak {peak:6.0f} MB   {time.time() - started:5.1f}s", flush=True)


def main() -> None:
    t = time.time()
    stage("interpreter", t)
    from src.zero_trust_rag.audit_logger import AuditLogger
    from src.zero_trust_rag.pii_redactor import PIIRedactor
    from src.zero_trust_rag.vault import PIIVault
    stage("imports", t)

    work = Path(tempfile.mkdtemp())
    redactor = PIIRedactor(PIIVault(str(work / "vault.db")), AuditLogger(str(work / "audit.log")))
    stage("redactor (spaCy + Presidio)", t)

    from src.zero_trust_rag.embeddings import LocalEmbedder
    embedder = LocalEmbedder()
    embedder.warm_up()
    stage("embedder (ONNX MiniLM)", t)

    import random

    from scripts.generate_records import make_record
    rng = random.Random(7)
    text = "\n\n---\n\n".join(make_record(rng, i) for i in range(1, 101))
    masked, log = redactor.redact(text)
    stage(f"redact 100 records ({len(log)} values)", t)

    from src.zero_trust_rag.splitter import TextSplitter
    chunks = TextSplitter(500, 100).split_text(masked)
    embedder.embed(chunks)
    stage(f"embed {len(chunks)} chunks", t)

    from src.zero_trust_rag.utils import release_free_memory
    release_free_memory()
    stage("after malloc_trim", t)


if __name__ == "__main__":
    main()
