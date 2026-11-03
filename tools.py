"""Task 6 - check_appointment_status with a designed escalation score.

escalation_score = 0.6 * follow_up_required + 0.4 * (days_since_created / 30)

* follow_up_required (0/1) carries 60% of the weight: a patient the doctor
  flagged for follow-up is the main clinical reason to involve a human.
* recency/age signal: days_since_created / 30 (normalised to [0, 1] by the
  dataset's 30-day maximum) - the longer a flagged appointment has been
  open, the more likely the follow-up window (kb/10) is being missed.
* Weights sum to 1, so the score is always in [0, 1].

ESCALATION_THRESHOLD = 0.80 = 0.6 + 0.4 * (15 / 30), where 15 days is the
50th percentile (median) of days_since_created in the generated data
(SEED=42). So a record escalates only when follow-up is required AND it is
older than the median appointment. A record without follow-up can score at
most 0.40 and never escalates on age alone. In this dataset 4/60 records
(6.7%) cross the threshold - a workload the support team can absorb.
"""
from __future__ import annotations

import re
import statistics

from dataset import APPOINTMENTS, APPOINTMENTS_BY_ID

W_FOLLOW_UP = 0.6
W_AGE = 0.4
MAX_DAYS = 30
MEDIAN_DAYS = statistics.median(r["days_since_created"] for r in APPOINTMENTS)  # 15.0 for SEED=42
ESCALATION_THRESHOLD = round(W_FOLLOW_UP + W_AGE * MEDIAN_DAYS / MAX_DAYS, 2)    # 0.80

RECORD_ID_RE = re.compile(r"\bAPT[-\s]?(\d{4})\b", re.IGNORECASE)


class TransientBackendError(RuntimeError):
    """Raised by the simulated flaky records backend (Task 16)."""


def escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    age = min(max(days_since_created, 0), MAX_DAYS) / MAX_DAYS
    return round(W_FOLLOW_UP * float(follow_up_required) + W_AGE * age, 4)


def normalize_record_id(text: str) -> str | None:
    m = RECORD_ID_RE.search(text or "")
    return f"APT-{m.group(1)}" if m else None


def check_appointment_status(record_id: str) -> dict:
    """Look up one Practo appointment by its record ID (e.g. "APT-1007").

    Returns the appointment's status, consultation fee in INR and an
    escalation_score in [0, 1] (0.6 * follow_up_required + 0.4 * days/30),
    plus `escalate`, True when the score is >= ESCALATION_THRESHOLD (0.80).
    Returns {"found": False, ...} for unknown IDs.
    """
    rid = normalize_record_id(record_id) or (record_id or "").strip().upper()
    rec = APPOINTMENTS_BY_ID.get(rid)
    if rec is None:
        return {"found": False, "record_id": rid, "error": f"No appointment found with ID {rid}."}
    score = escalation_score(rec["follow_up_required"], rec["days_since_created"])
    return {
        "found": True,
        "record_id": rid,
        "category": rec["category"],
        "status": rec["status"],
        "consultation_fee_inr": rec["consultation_fee_inr"],
        "follow_up_required": rec["follow_up_required"],
        "days_since_created": rec["days_since_created"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "escalate": score >= ESCALATION_THRESHOLD,
    }


if __name__ == "__main__":
    import numpy as np

    days = [r["days_since_created"] for r in APPOINTMENTS]
    print("Formula: escalation_score = 0.6 * follow_up_required + 0.4 * (days_since_created / 30)")
    print("days_since_created percentiles in generated data:",
          {p: float(np.percentile(days, p)) for p in (25, 50, 75, 80, 90)})
    print(f"Threshold = 0.6 + 0.4 * (P50={MEDIAN_DAYS} / 30) = {ESCALATION_THRESHOLD}")
    scores = sorted((escalation_score(r["follow_up_required"], r["days_since_created"]), r["record_id"])
                    for r in APPOINTMENTS)
    esc = [rid for s, rid in scores if s >= ESCALATION_THRESHOLD]
    print(f"Records at/above threshold: {len(esc)}/{len(APPOINTMENTS)} -> {sorted(esc)}")
    print("Score distribution percentiles:",
          {p: round(float(np.percentile([s for s, _ in scores], p)), 3) for p in (50, 75, 90, 95)})
    for rid in ["APT-1002", "APT-1007", "APT-1022", "APT-9999"]:
        print(rid, "->", check_appointment_status(rid))
