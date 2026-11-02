"""Task 4 - grounded generation on 6 in-scope queries + 1 out-of-scope query.

Run:  python -m demos.demo_grounded_generation
"""
from __future__ import annotations

import config
from evaluation.queries import IN_SCOPE, OUT_OF_SCOPE_DEMO
from rag.generate import grounded_answer


def show(q: str) -> dict:
    r = grounded_answer(q)
    print(f"Q: {q}")
    for h in r["hits"]:
        print(f"   retrieved sim={h.score:.3f} {h.chunk_id}")
    print(f"   decision: {r['reason']} -> {'FALLBACK' if r['fallback'] else 'ANSWER'}")
    print(f"A: {r['answer']}\n")
    return r


def main() -> None:
    print(f"MOCK_LLM={config.MOCK_LLM}  collection={config.AGENT_COLLECTION}  "
          f"IDK_THRESHOLD={config.IDK_THRESHOLD}\n")
    print("---- In-scope queries ----")
    for item in IN_SCOPE:
        r = show(item["q"])
        assert not r["fallback"], "in-scope query wrongly fell back"
    print("---- Deliberately out-of-scope query ----")
    r = show(OUT_OF_SCOPE_DEMO)
    assert r["fallback"], "out-of-scope query did not trigger the fallback"
    print("OK: all in-scope queries answered from context; out-of-scope query triggered the fallback.")


if __name__ == "__main__":
    main()
