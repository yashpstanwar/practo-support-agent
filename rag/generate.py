"""Task 4 - grounded generation with an empirically calibrated IDK threshold."""
from __future__ import annotations

import config
from llm import generate_grounded
from rag.store import Hit, retrieve

IDK_ANSWER = "I don't know based on the clinic policy knowledge base."


def grounded_answer(question: str, collection: str = config.AGENT_COLLECTION,
                    k: int = config.TOP_K, threshold: float = config.IDK_THRESHOLD) -> dict:
    hits: list[Hit] = retrieve(question, k=k, collection=collection)
    top1 = hits[0].score if hits else 0.0
    if top1 < threshold:
        return {"answer": IDK_ANSWER, "fallback": True, "top1": top1, "hits": hits,
                "prompt": None, "reason": f"top-1 similarity {top1:.3f} < threshold {threshold:.2f}"}
    answer, prompt = generate_grounded(question, [h.text for h in hits])
    return {"answer": answer, "fallback": False, "top1": top1, "hits": hits, "prompt": prompt,
            "reason": f"top-1 similarity {top1:.3f} >= threshold {threshold:.2f}"}
