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
