"""Local, free embeddings with SentenceTransformers (all-MiniLM-L6-v2).

Vectors are L2-normalised, so cosine similarity == dot product and
ChromaDB's cosine distance d gives similarity 1 - d.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL_NAME, device="cpu")


def embed(texts: list[str]) -> list[list[float]]:
    vecs = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    return float(np.dot(np.asarray(a), np.asarray(b)))
