"""Task 1 - seeded, deterministic appointment dataset for the Practo track.

Run:  python dataset.py

Design choices (also stated in README.md):
  * SEED = 42, N_RECORDS = 60
  * Category weights: General Medicine 0.30, Cardiology 0.18, Dermatology 0.18,
    Pediatrics 0.17, Orthopedics 0.17
  * Status weights: Scheduled 0.34, Completed 0.30, Rescheduled 0.14,
    Cancelled 0.12, No-Show 0.10
  * follow_up_required ~ Bernoulli(0.20)
  * consultation_fee_inr: drawn uniformly inside a per-specialty band and
    rounded to the nearest INR 50; overall range INR 300-1500.
    Reasoning: in-clinic consultations booked through Indian aggregators
    typically cost a few hundred rupees for a GP and up to roughly INR 1,500 for
    a senior cardiologist, so specialty bands inside 300-1500 are realistic.
  * days_since_created: uniform integer 0-30.

The generator never edits individual records. If a structural check fails,
`validate()` raises and the fix is to change SEED or the weights.
All data is fabricated; there is no patient name, diagnosis, phone or
insurance ID in this dataset.
"""
from __future__ import annotations

import random
from collections import Counter

SEED = 42
N_RECORDS = 60

CATEGORY_WEIGHTS = {
    "General Medicine": 0.30,
    "Cardiology": 0.18,
    "Dermatology": 0.18,
    "Pediatrics": 0.17,
    "Orthopedics": 0.17,
}
STATUS_WEIGHTS = {
    "Scheduled": 0.34,
    "Completed": 0.30,
    "Rescheduled": 0.14,
    "Cancelled": 0.12,
    "No-Show": 0.10,
}
FOLLOW_UP_PROBABILITY = 0.20

# Per-specialty fee bands (INR) - match kb/03_consultation_fees.md
FEE_BANDS = {
    "General Medicine": (300, 600),
    "Pediatrics": (400, 800),
    "Dermatology": (500, 1000),
    "Orthopedics": (600, 1200),
    "Cardiology": (800, 1500),
}
FEE_MIN, FEE_MAX = 300, 1500

REQUIRED_CATEGORIES = list(CATEGORY_WEIGHTS)
REQUIRED_STATUSES = list(STATUS_WEIGHTS)


def generate(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    cats = rng.choices(list(CATEGORY_WEIGHTS), weights=list(CATEGORY_WEIGHTS.values()), k=n)
    stats = rng.choices(list(STATUS_WEIGHTS), weights=list(STATUS_WEIGHTS.values()), k=n)
    records = []
    for i in range(n):
        lo, hi = FEE_BANDS[cats[i]]
        fee = int(round(rng.uniform(lo, hi) / 50.0) * 50)
        records.append(
            {
                "record_id": f"APT-{1001 + i}",
                "category": cats[i],
                "status": stats[i],
                "consultation_fee_inr": fee,
                "days_since_created": rng.randint(0, 30),
                "follow_up_required": rng.random() < FOLLOW_UP_PROBABILITY,
            }
        )
    return records


def summarize(records: list[dict]) -> dict:
    cat = Counter(r["category"] for r in records)
    st = Counter(r["status"] for r in records)
    fu = sum(r["follow_up_required"] for r in records)
    return {
        "n": len(records),
        "category_counts": {c: cat.get(c, 0) for c in REQUIRED_CATEGORIES},
        "status_counts": {s: st.get(s, 0) for s in REQUIRED_STATUSES},
        "follow_up_true": fu,
        "follow_up_pct": round(100.0 * fu / len(records), 2),
        "fee_min": min(r["consultation_fee_inr"] for r in records),
        "fee_max": max(r["consultation_fee_inr"] for r in records),
    }


def validate(records: list[dict]) -> dict:
    s = summarize(records)
    assert s["n"] >= 40, "need >= 40 records"
    for c, k in s["category_counts"].items():
        assert k >= 3, f"category {c} has only {k} records (< 3)"
    for st, k in s["status_counts"].items():
        assert k >= 1, f"status {st} missing"
    assert 10.0 <= s["follow_up_pct"] <= 30.0, f"follow-up % {s['follow_up_pct']} outside 10-30"
    for r in records:
        assert FEE_MIN <= r["consultation_fee_inr"] <= FEE_MAX
        assert 0 <= r["days_since_created"] <= 30
        assert isinstance(r["follow_up_required"], bool)
    return s


APPOINTMENTS: list[dict] = generate()
APPOINTMENTS_BY_ID: dict[str, dict] = {r["record_id"]: r for r in APPOINTMENTS}


if __name__ == "__main__":
    s = validate(APPOINTMENTS)
    print(f"Generated {s['n']} appointments  (SEED={SEED})")
    print("\nCount per category (every given category must be >= 3):")
    for c, k in s["category_counts"].items():
        print(f"  {c:<18} {k:>3}")
    print("\nCount per status (every given status must be >= 1):")
    for st, k in s["status_counts"].items():
        print(f"  {st:<18} {k:>3}")
    print(
        f"\nfollow_up_required=True: {s['follow_up_true']}/{s['n']} = {s['follow_up_pct']}%"
        "  (required band 10%-30%)"
    )
    print(f"consultation_fee_inr observed range: INR {s['fee_min']} - {s['fee_max']}")
    print("\nFirst 5 records:")
    for r in APPOINTMENTS[:5]:
        print("  ", r)
    print("\nALL STRUCTURAL CHECKS PASSED")
