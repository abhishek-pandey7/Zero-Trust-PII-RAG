import re

import pytest

from src.zero_trust_rag.splitter import TextSplitter

TOKEN = re.compile(r"<<[A-Z_]+_\d{2,}_[0-9a-f]{4}>>")


def test_short_text_is_one_chunk():
    assert TextSplitter(100, 20).split_text("Name: <<PERSON_01_abcd>>") == ["Name: <<PERSON_01_abcd>>"]


def test_chunks_respect_the_size_limit():
    text = "\n\n".join(f"Record {i}: " + "word " * 30 for i in range(20))
    chunks = TextSplitter(200, 40).split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_every_word_lands_in_some_chunk():
    words = [f"w{i}" for i in range(400)]
    chunks = TextSplitter(120, 30).split_text(" ".join(words))
    joined = " ".join(chunks)
    assert all(w in joined.split() for w in words)


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(200))
    first, second = TextSplitter(100, 30).split_text(text)[:2]
    assert first.split()[-1] in second.split()


def test_paragraphs_are_preferred_split_points():
    a, b = "A" * 80, "B" * 80
    assert TextSplitter(100, 10).split_text(f"{a}\n\n{b}") == [a, b]


def test_hard_cut_never_splits_a_token():
    token = "<<PERSON_01_abcd>>"
    text = ("x" * 37 + token) * 20  # no separators at all, tokens straddle the 50-char boundary
    chunks = TextSplitter(50, 10).split_text(text)
    assert all(len(c) <= 50 for c in chunks)
    for chunk in chunks:
        stripped = TOKEN.sub("", chunk)
        assert "<<" not in stripped and ">>" not in stripped, chunk
    assert sum(len(TOKEN.findall(c)) for c in chunks) == 20


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        TextSplitter(100, 100)
