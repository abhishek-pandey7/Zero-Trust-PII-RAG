"""
splitter.py – Recursive character text splitter that never cuts a vault token in half.

Splits on the coarsest separator that keeps pieces under chunk_size (paragraphs, then lines, then
words), packs pieces into chunks with some overlap, and only hard-cuts text that has no separator
left. A hard cut never lands inside a <<TOKEN>>: half a token can't be unmasked and would leak the
fact that a token existed without its meaning.
"""
from collections.abc import Sequence

DEFAULT_SEPARATORS = ("\n\n", "\n", " ", "")


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100,
                 separators: Sequence[str] = DEFAULT_SEPARATORS):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)

    def split_text(self, text: str) -> list[str]:
        return [c for c in (c.strip() for c in self._split(text, self.separators)) if c]

    def _split(self, text: str, separators: Sequence[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        index = next((i for i, s in enumerate(separators) if s == "" or s in text), len(separators) - 1)
        separator, finer = separators[index], separators[index + 1:]
        if separator == "":
            return self._hard_cut(text)

        chunks: list[str] = []
        fitting: list[str] = []
        for piece in (p for p in text.split(separator) if p):
            if len(piece) <= self.chunk_size:
                fitting.append(piece)
                continue
            chunks.extend(self._merge(fitting, separator))
            fitting = []
            chunks.extend(self._split(piece, finer) if finer else self._hard_cut(piece))
        chunks.extend(self._merge(fitting, separator))
        return chunks

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        """Pack pieces into chunks of at most chunk_size, repeating up to chunk_overlap chars."""
        chunks: list[str] = []
        window: list[str] = []
        length = 0
        sep_len = len(separator)
        for piece in pieces:
            added = len(piece) + (sep_len if window else 0)
            if window and length + added > self.chunk_size:
                chunks.append(separator.join(window))
                # Slide the window forward until what's left fits as overlap before the next piece
                while window and (length > self.chunk_overlap
                                  or length + sep_len + len(piece) > self.chunk_size):
                    length -= len(window[0]) + (sep_len if len(window) > 1 else 0)
                    window.pop(0)
                added = len(piece) + (sep_len if window else 0)
            window.append(piece)
            length += added
        if window:
            chunks.append(separator.join(window))
        return chunks

    def _hard_cut(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                opened = text.rfind("<<", start, end)
                if opened > start and text.find(">>", opened, end) == -1:
                    end = opened  # back off to just before an unclosed token
            chunks.append(text[start:end])
            start = end
        return chunks
