"""Task 10 - guardrails.

Input side
  * mask_pii: masks Indian mobile numbers (the fixed-format PII field), e.g.
    "+91 98765 43210", "09876543210", "9876543210", "98765-43210".
    Also masks e-mail addresses as a bonus. Patient name, diagnosis and
    insurance ID have no universal fixed format and are out of scope for a
    keyless regex masker (see README).
  * detect_injection: pattern-based prompt-injection detector.
Output side
  * check_groundedness: refuses when retrieval similarity is below the
    calibrated threshold OR when any answer sentence is not supported by the
    retrieved context (cosine >= SUPPORT_SIM_THRESHOLD or verbatim match).
"""
from __future__ import annotations

import re

import config
from rag.chunking import split_sentences

PHONE_RE = re.compile(r"(?<![\w-])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_MASK = "[PHONE_MASKED]"
EMAIL_MASK = "[EMAIL_MASKED]"

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|your) (instructions|rules|guidelines)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"(reveal|show|print|leak) (me )?(your|the) (system|hidden|developer) (prompt|instructions)",
    r"\bsystem prompt\b",
    r"you are now (in )?(dan|developer mode|an? unrestricted)",
    r"\bjailbreak\b",
    r"act as (an? )?(admin|administrator|root|developer)",
    r"(dump|list|export) (all|every) (patients?|records|appointments)",
    r"override (the )?(safety|guardrails?|policy)",
]
_INJ = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def mask_pii(text: str) -> tuple[str, dict]:
    phones = PHONE_RE.findall(text or "")
    emails = EMAIL_RE.findall(text or "")
    masked = PHONE_RE.sub(PHONE_MASK, text or "")
    masked = EMAIL_RE.sub(EMAIL_MASK, masked)
    return masked, {"phones_masked": len(phones), "emails_masked": len(emails),
                    "pii_masked": bool(phones or emails)}


def detect_injection(text: str) -> tuple[bool, str | None]:
    for rx in _INJ:
        m = rx.search(text or "")
        if m:
            return True, m.group(0)
    return False, None


def check_groundedness(answer: str, contexts: list[str], top1: float,
                       threshold: float = config.IDK_THRESHOLD,
                       support_threshold: float = config.SUPPORT_SIM_THRESHOLD) -> dict:
    """Return {"grounded": bool, "reason": str, "support": [...]}."""
    from rag.embeddings import cosine, embed

    if top1 < threshold:
        return {"grounded": False, "support": [],
                "reason": f"retrieval_below_threshold (top1={top1:.3f} < {threshold:.2f})"}
    sents = split_sentences(answer)
    if not sents or not contexts:
        return {"grounded": False, "support": [], "reason": "empty_answer_or_context"}
    ctx_sents = [s for c in contexts for s in split_sentences(c)]
    ctx_vecs = embed(ctx_sents)
    support = []
    for s, v in zip(sents, embed(sents)):
        verbatim = any(s in c for c in contexts)
        best = max(cosine(v, cv) for cv in ctx_vecs)
        support.append({"sentence": s, "best_context_sim": round(best, 3),
                        "supported": verbatim or best >= support_threshold})
    ok = all(x["supported"] for x in support)
    return {"grounded": ok, "support": support,
            "reason": "all_sentences_supported" if ok else "answer_not_supported_by_context"}
