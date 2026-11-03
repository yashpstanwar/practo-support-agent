"""Task 6 - appointment lookup tool + designed escalation score.
"""
from __future__ import annotations

import re
from typing import Optional

from dataset import APPOINTMENTS, summarize

APPOINTMENT_BY_ID = {r["record_id"]: dict(r) for r in APPOINTMENTS}
_STATS = summarize(APPOINTMENTS)
MEDIAN_DAYS = 15.0

W_FOLLOW_UP = 0.6
W_RECENCY = 0.4
RECENCY_MAX_DAYS = 30.0
ESCALATION_THRESHOLD = 0.80

RECORD_ID_RE = re.compile(r"\bAPT-(\d{4})\b", re.IGNORECASE)


class TransientBackendError(RuntimeError):
    pass


def normalize_record_id(query: str) -> Optional[str]:
    m = RECORD_ID_RE.search(query or "")
    return f"APT-{m.group(1)}" if m else None


def check_appointment_status(record_id: str) -> dict:
    rid = (record_id or "").strip().upper()
    r = APPOINTMENT_BY_ID.get(rid)
    if not r:
        return {"found": False, "record_id": rid, "error": f"No appointment found with ID {rid}."}
    return {
        "found": True,
        "record_id": r["record_id"],
        "category": r["category"],
        "status": r["status"],
        "consultation_fee_inr": r["consultation_fee_inr"],
        "follow_up_required": r["follow_up_required"],
        "days_since_created": r["days_since_created"],
    }
