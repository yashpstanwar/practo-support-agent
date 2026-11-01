"""Task 4 - empirically calibrate the "I don't know" threshold.

Measures top-1 cosine similarity for in-scope and out-of-scope queries on BOTH
collections and proposes the midpoint of the gap between the two clusters.

Run:  python -m evaluation.calibrate_threshold
"""
from __future__ import annotations

import config
from evaluation.queries import IN_SCOPE, OUT_OF_SCOPE_CALIBRATION
from rag.store import STRATEGY_TO_COLLECTION, retrieve


def main() -> None:
    for strat, name in STRATEGY_TO_COLLECTION.items():
        print(f"=== Collection: {name} ({strat}) ===")
        ins, outs = [], []
        for item in IN_SCOPE:
            s = retrieve(item["q"], k=1, collection=name)[0].score
            ins.append(s)
            print(f"  IN   top1={s:.4f}  {item['q']}")
        for q in OUT_OF_SCOPE_CALIBRATION:
            s = retrieve(q, k=1, collection=name)[0].score
            outs.append(s)
            print(f"  OUT  top1={s:.4f}  {q}")
        lo_in, hi_out = min(ins), max(outs)
        print(f"  in-scope cluster:     min={lo_in:.4f} max={max(ins):.4f}")
        print(f"  out-of-scope cluster: min={min(outs):.4f} max={hi_out:.4f}")
        if lo_in > hi_out:
            print(f"  gap = ({hi_out:.4f}, {lo_in:.4f}) -> midpoint {(lo_in + hi_out) / 2:.4f}")
        else:
            print("  WARNING: clusters overlap on this collection")
        print()
    print(f"Configured IDK_THRESHOLD (config.py) = {config.IDK_THRESHOLD}")
    print("It is chosen inside the gap observed on the agent's collection "
          f"({config.AGENT_COLLECTION}), not taken from a tutorial preset.")


if __name__ == "__main__":
    main()
