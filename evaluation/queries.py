"""Shared query sets with hand-labelled gold documents."""

# Tasks 4 + 5: the same in-scope queries are used for grounded generation and
# for Precision@3 / Recall@3. `relevant` = gold parent documents.
IN_SCOPE = [
    {"q": "How late can I cancel my appointment without losing any money?",
     "relevant": {"02_cancellation_rescheduling"}},
    {"q": "How much does a cardiology consultation cost?",
     "relevant": {"03_consultation_fees"}},
    {"q": "How long will my thyroid test report take?",
     "relevant": {"06_lab_turnaround"}},
    {"q": "Can my first orthopedic consultation be done over a video call?",
     "relevant": {"07_telemedicine"}},
    {"q": "Is a follow-up visit free, and can a doctor visit me at home after my surgery?",
     "relevant": {"10_follow_up_discount", "12_home_visits"}},
    {"q": "What documents do I need to upload to claim insurance for my consultation?",
     "relevant": {"04_insurance_claims"}},
]

OUT_OF_SCOPE_DEMO = "Who won the cricket world cup in 2011?"

# Task 4 calibration: deliberately out-of-scope queries.
OUT_OF_SCOPE_CALIBRATION = [
    "Who won the cricket world cup in 2011?",
    "Recommend a good gaming laptop under 80000 rupees.",
    "What is the capital city of Australia?",
    "Write me a poem about the monsoon.",
]

SAMPLE_QUERY = "What is the fee for a dermatologist?"
