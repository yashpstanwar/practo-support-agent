"""Task 10 - input & output guardrails for the Practo support agent.
"""
from __future__ import annotations

import re

import config
from rag.chunking import split_sentences

PHONE_RE = re.compile(r"(?<![\w-])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_MASK = "[PHONE_MASKED]"
EMAIL_MASK = "[EMAIL_MASKED]"


def mask_pii(text: str) -> tuple[str, dict]:
    phones = PHONE_RE.findall(text or "")
    emails = EMAIL_RE.findall(text or "")
    masked = PHONE_RE.sub(PHONE_MASK, text or "")
    masked = EMAIL_RE.sub(EMAIL_MASK, masked)
    return masked, {"phones_masked": len(phones), "emails_masked": len(emails),
                    "pii_masked": bool(phones or emails)}
