"""Task 3 - chunking strategies for the Practo knowledge base.

Strategy A: fixed-size character window with character overlap.
Strategy B: sentence-based chunking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, KB_DIR


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    strategy: str
    index: int
    text: str

    @property
    def embed_text(self) -> str:
        return f"{self.doc_title}: {self.text}"


_SENT_RE = re.compile(
    r"(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e)\.)"
    r"(?<=[.!?])\s+(?=[A-Z0-9])"
)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    raw = _SENT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def chunk_sentences(doc: Document) -> list[Chunk]:
    sents = split_sentences(doc.text)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}_sent_{i}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            strategy="sentence",
            index=i,
            text=s,
        )
        for i, s in enumerate(sents)
    ]
