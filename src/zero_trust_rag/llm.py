"""
llm.py – The one call that leaves the trust boundary: masked context and question to Groq.
"""
import logging
from typing import Protocol

import groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .exceptions import APIError

logger = logging.getLogger(__name__)

# Worth retrying: the request may succeed a moment later. Auth and bad-request errors won't.
TRANSIENT_ERRORS = (
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.RateLimitError,
    groq.InternalServerError,
)


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class GroqLLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0,
                 timeout: float = 30.0, max_attempts: int = 3, reasoning_effort: str | None = None):
        if not api_key:
            raise APIError("Missing Groq API key. Set GROQ_API_KEY.", provider="Groq")
        # The SDK has its own retries; disable them so tenacity is the only retry layer
        self._client = groq.Groq(api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        # gpt-oss models think before answering; "low" keeps latency near a non-reasoning model
        self._extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        self._call = retry(
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            stop=stop_after_attempt(max_attempts),
            # Exponential backoff with full jitter, so retries from many requests don't line up
            wait=wait_random_exponential(multiplier=0.5, max=8),
            reraise=True,
        )(self._create)

    def _create(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self._extra,
        )
        return response.choices[0].message.content or ""

    def complete(self, system: str, user: str) -> str:
        try:
            return self._call(system, user)
        except groq.APIStatusError as e:
            raise APIError(f"Groq returned {e.status_code}", provider="Groq", status_code=e.status_code) from e
        except groq.GroqError as e:
            raise APIError(f"Groq request failed: {type(e).__name__}", provider="Groq") from e
