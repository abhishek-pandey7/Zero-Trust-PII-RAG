"""
utils.py – Utility functions for Zero-Trust RAG.
"""
import os

from .exceptions import ValidationError

MAX_INPUT_CHARS = 1_000_000


def max_input_chars() -> int:
    """The input size limit. A small host sets ZT_MAX_INPUT_CHARS: redacting and embedding cost
    memory in proportion to the text, and 512 MB holds roughly 30,000 characters at a time."""
    return int(os.getenv("ZT_MAX_INPUT_CHARS", MAX_INPUT_CHARS))


def validate_input(text: str, max_length: int | None = None) -> None:
    limit = max_length or max_input_chars()
    if not isinstance(text, str):
        raise ValidationError("Input must be a string")
    if not text.strip():
        raise ValidationError("Input text cannot be empty")
    if len(text) > limit:
        raise ValidationError(
            f"Input is {len(text):,} characters; this deployment accepts up to {limit:,} at a time. "
            "Split the document and ingest it in parts."
        )


def release_free_memory() -> None:
    """Hand memory freed by a large request back to the OS (glibc only; a no-op elsewhere).

    glibc keeps freed heap pages for reuse, so after one big ingest the process stays at its peak
    size. On a 512 MB host that leftover is the headroom the next request needs.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
