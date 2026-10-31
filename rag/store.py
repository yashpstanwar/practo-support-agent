"""Task 3 - two separate ChromaDB collections + retrieval.

Both collections use cosine space and the same embedding model. Each chunk is
embedded as "<doc title>: <chunk text>" so short sentence chunks keep their
topic context; the stored document text is the plain chunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb

import config
from rag.chunking import Chunk, Document, fixed_size_chunks, load_kb, sentence_chunks
from rag.embeddings import embed, embed_one

STRATEGY_TO_COLLECTION = {"fixed": config.COLLECTION_FIXED, "sentence": config.COLLECTION_SENTENCE}
CHUNKERS = {"fixed": fixed_size_chunks, "sentence": sentence_chunks}


@lru_cache(maxsize=1)
def client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(name: str):
    return client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _embed_text(c: Chunk) -> str:
    return f"{c.title}: {c.text}"


def add_chunks(collection_name: str, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    col = get_collection(collection_name)
    col.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed([_embed_text(c) for c in chunks]),
        metadatas=[{"doc_id": c.doc_id, "title": c.title, "strategy": c.strategy} for c in chunks],
    )
    return len(chunks)


def index_documents(docs: list[Document]) -> dict[str, int]:
    counts = {}
    for strategy, chunker in CHUNKERS.items():
        chunks = [ch for d in docs for ch in chunker(d)]
        counts[strategy] = add_chunks(STRATEGY_TO_COLLECTION[strategy], chunks)
    return counts


def build_index(reset: bool = True) -> dict[str, int]:
    if reset:
        for name in STRATEGY_TO_COLLECTION.values():
            try:
                client().delete_collection(name)
            except Exception:
                pass
    return index_documents(load_kb(extra_dirs=[config.ADDED_DOCS_DIR]))


def ensure_index() -> None:
    for name in STRATEGY_TO_COLLECTION.values():
        if get_collection(name).count() == 0:
            build_index(reset=True)
            return


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float  # cosine similarity


def retrieve(query: str, k: int = config.TOP_K, collection: str = config.AGENT_COLLECTION) -> list[Hit]:
    ensure_index()
    col = get_collection(collection)
    res = col.query(query_embeddings=[embed_one(query)], n_results=min(k, col.count()),
                    include=["documents", "metadatas", "distances"])
    hits = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(Hit(cid, meta["doc_id"], meta["title"], doc, round(1.0 - float(dist), 4)))
    return hits


def delete_document(doc_id: str) -> None:
    for name in STRATEGY_TO_COLLECTION.values():
        get_collection(name).delete(where={"doc_id": doc_id})
