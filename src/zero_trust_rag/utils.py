"""
utils.py – Utility functions for Zero-Trust RAG.
"""
from .exceptions import ValidationError

MAX_INPUT_CHARS = 1_000_000


def validate_input(text: str, max_length: int = MAX_INPUT_CHARS) -> None:
    if not isinstance(text, str):
        raise ValidationError("Input must be a string")
    if not text.strip():
        raise ValidationError("Input text cannot be empty")
    if len(text) > max_length:
        raise ValidationError(f"Input text exceeds maximum length of {max_length} chars")
