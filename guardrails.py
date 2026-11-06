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
