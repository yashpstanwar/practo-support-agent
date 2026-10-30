"""Task 3 - two chunking strategies.

1. fixed_size_chunks: sliding character window (size 200, overlap 50) that
   snaps to the nearest word boundary so words are never cut in half.
2. sentence_chunks: one chunk per sentence (regex sentence splitter).

Every chunk carries its parent `doc_id` so retrieval can be scored at the
document level (Task 5).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    strategy: str


def load_kb(extra_dirs: list[Path] | None = None) -> list[Document]:
    docs: list[Document] = []
    dirs = [config.KB_DIR] + list(extra_dirs or [])
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.md")):
            raw = p.read_text(encoding="utf-8").strip().splitlines()
            title = raw[0].lstrip("# ").strip()
            body = " ".join(line.strip() for line in raw[1:] if line.strip())
            docs.append(Document(doc_id=p.stem, title=title, text=body))
    return docs


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def fixed_size_chunks(doc: Document, size: int = config.FIXED_CHUNK_SIZE,
                      overlap: int = config.FIXED_CHUNK_OVERLAP) -> list[Chunk]:
    text = doc.text
    chunks: list[Chunk] = []
    start, i = 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # snap back to a word boundary
            sp = text.rfind(" ", start, end)
            if sp > start + size // 2:
                end = sp
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(f"{doc.doc_id}::fixed::{i}", doc.doc_id, doc.title, piece, "fixed"))
            i += 1
        if end >= len(text):
            break
        nxt = end - overlap
        sp = text.find(" ", nxt, end)  # start the overlap on a word boundary
        start = sp + 1 if sp != -1 else nxt
    return chunks


def sentence_chunks(doc: Document) -> list[Chunk]:
    return [
        Chunk(f"{doc.doc_id}::sent::{i}", doc.doc_id, doc.title, s, "sentence")
        for i, s in enumerate(split_sentences(doc.text))
    ]
