"""Test doubles for the two pipeline dependencies that are slow or external."""
import hashlib
import math
import re

TOKEN = re.compile(r"<<[A-Z_]+_\d{2,}_[0-9a-f]{4}>>")


class HashingEmbedder:
    """Bag-of-words vectors from hashed words: deterministic, instant, and good enough to rank."""

    dimensions = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class RecordingLLM:
    """Records every prompt it is sent. Answers with the first token in the context, or a canned reply."""

    model = "recording-fake"

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.reply is not None:
            return self.reply
        tokens = TOKEN.findall(user.split("Question:")[0])
        return f"The answer is {tokens[0]}." if tokens else "I don't know."

    @property
    def sent_text(self) -> str:
        return "\n".join(system + "\n" + user for system, user in self.calls)
