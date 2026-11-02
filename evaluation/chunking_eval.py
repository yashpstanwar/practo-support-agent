"""Task 5 - document-level Precision@3 / Recall@3 for BOTH collections.

Procedure per query and collection:
  1. retrieve the top-15 chunks (over-fetch),
  2. map every chunk to its parent doc_id and de-duplicate in rank order,
  3. keep the first 3 unique documents  -> "top-3 documents",
  4. P@3 = |relevant ∩ top3| / 3,  R@3 = |relevant ∩ top3| / |relevant|.

Run:  python -m evaluation.chunking_eval
"""
from __future__ import annotations

from evaluation.queries import IN_SCOPE
from rag.store import STRATEGY_TO_COLLECTION, retrieve

K = 3


def top_docs(q: str, collection: str, k: int = K) -> tuple[list[str], list[str]]:
    hits = retrieve(q, k=15, collection=collection)
    raw = [h.doc_id for h in hits]
    dedup: list[str] = []
    for d in raw:
        if d not in dedup:
            dedup.append(d)
    return raw[:6], dedup[:k]


def evaluate(collection: str) -> tuple[float, float]:
    ps, rs = [], []
    for i, item in enumerate(IN_SCOPE, 1):
        raw, top = top_docs(item["q"], collection)
        rel = item["relevant"]
        hit = [d for d in top if d in rel]
        p, r = len(hit) / K, len(hit) / len(rel)
        ps.append(p)
        rs.append(r)
        print(f"  Q{i}: {item['q']}")
        print(f"      first chunk parents: {raw}")
        print(f"      top-3 unique docs:   {top}")
        print(f"      relevant:            {sorted(rel)}")
        print(f"      P@3 = {len(hit)}/{K} = {p:.3f}    R@3 = {len(hit)}/{len(rel)} = {r:.3f}")
    mp, mr = sum(ps) / len(ps), sum(rs) / len(rs)
    print(f"  MEAN P@3 = ({' + '.join(f'{x:.3f}' for x in ps)}) / {len(ps)} = {mp:.3f}")
    print(f"  MEAN R@3 = ({' + '.join(f'{x:.3f}' for x in rs)}) / {len(rs)} = {mr:.3f}")
    return mp, mr


def mean_rank_of_first_relevant(collection: str) -> float:
    ranks = []
    for item in IN_SCOPE:
        hits = retrieve(item["q"], k=15, collection=collection)
        r = next((i + 1 for i, h in enumerate(hits) if h.doc_id in item["relevant"]), 16)
        ranks.append(r)
    return sum(ranks) / len(ranks)


def mean_top1_relevant_sim(collection: str) -> float:
    sims = []
    for item in IN_SCOPE:
        hits = retrieve(item["q"], k=15, collection=collection)
        sims.append(next((h.score for h in hits if h.doc_id in item["relevant"]), 0.0))
    return sum(sims) / len(sims)


def main() -> None:
    results = {}
    for strat, name in STRATEGY_TO_COLLECTION.items():
        print(f"=== {name} ({strat}) ===")
        results[strat] = evaluate(name)
        print(f"  mean rank of first relevant chunk: {mean_rank_of_first_relevant(name):.2f}")
        print(f"  mean similarity of best relevant chunk: {mean_top1_relevant_sim(name):.3f}\n")
    print("SUMMARY")
    for strat, (p, r) in results.items():
        print(f"  {strat:<9} mean P@3={p:.3f}  mean R@3={r:.3f}")


if __name__ == "__main__":
    main()
