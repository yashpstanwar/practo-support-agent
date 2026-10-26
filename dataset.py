"""Task 1 - seeded, deterministic appointment dataset for the Practo track.

Run:  python dataset.py
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
