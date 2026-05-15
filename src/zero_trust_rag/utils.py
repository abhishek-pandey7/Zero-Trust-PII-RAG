"""
utils.py – Utility functions for Zero-Trust RAG.
"""
import time
import logging
from functools import wraps
from typing import Callable, TypeVar, Any

from .config import Config
from .exceptions import APIError, ValidationError

config = Config()
logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_on_api_error(max_retries: int = None, retry_delay: float = None) -> Callable:
    max_retries = max_retries or config.get("security.max_retries", 3)
    retry_delay = retry_delay or config.get("security.retry_delay", 1.0)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except APIError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"API error on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"API error after {max_retries + 1} attempts: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator


def validate_api_key(api_key: str | None, provider: str) -> None:
    if not api_key:
        raise ValidationError(f"Missing {provider} API key. Set {provider}_API_KEY in .env")
    
    if len(api_key) < 10:
        raise ValidationError(f"Invalid {provider} API key. Key seems too short.")


def validate_input(text: str, max_length: int = 1000000) -> None:
    if not text:
        raise ValidationError("Input text cannot be empty")
    
    if not isinstance(text, str):
        raise ValidationError("Input must be a string")
    
    if len(text) > max_length:
        raise ValidationError(f"Input text exceeds maximum length of {max_length} chars")


def format_chunk_metadata(chunk_index: int, total_chunks: int) -> dict[str, Any]:
    return {
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunk_id": f"chunk_{chunk_index:04d}_{total_chunks:04d}"
    }