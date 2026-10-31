"""Task 3 - chunk the KB both ways, embed, and index into two Chroma collections.

Run:  python -m rag.build_index
"""
from __future__ import annotations

import config
from evaluation.queries import SAMPLE_QUERY
from rag.chunking import fixed_size_chunks, load_kb, sentence_chunks
from rag.store import STRATEGY_TO_COLLECTION, build_index, get_collection, retrieve


def main() -> None:
    docs = load_kb()
    print(f"Loaded {len(docs)} knowledge-base documents from kb/")
    for d in docs:
        n_sent = len(sentence_chunks(d))
        print(f"  {d.doc_id:<32} sentences={n_sent}  fixed_chunks={len(fixed_size_chunks(d))}  chars={len(d.text)}")
    counts = build_index(reset=True)
    print(f"\nEmbedding model: {config.EMBED_MODEL_NAME}")
    print(f"Fixed-size chunking: size={config.FIXED_CHUNK_SIZE} chars, overlap={config.FIXED_CHUNK_OVERLAP} chars")
    for strat, name in STRATEGY_TO_COLLECTION.items():
        print(f"Collection '{name}' ({strat}): {get_collection(name).count()} chunks indexed (added {counts[strat]})")

    print("\nExample fixed-size chunks of 02_cancellation_rescheduling:")
    d2 = next(d for d in docs if d.doc_id.startswith("02"))
    for c in fixed_size_chunks(d2):
        print(f"  [{c.chunk_id}] {c.text}")

    print(f"\nSample query on BOTH collections: '{SAMPLE_QUERY}'")
    for strat, name in STRATEGY_TO_COLLECTION.items():
        print(f"  -- {name}")
        for h in retrieve(SAMPLE_QUERY, k=3, collection=name):
            print(f"     sim={h.score:.3f}  {h.doc_id:<28} | {h.text[:90]}")


if __name__ == "__main__":
    main()
