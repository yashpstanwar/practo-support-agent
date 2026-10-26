"""Central configuration for the Practo Domain Support Agent.

Everything defaults to MOCK_LLM mode: zero API keys, zero network calls for
generation or judging. Only the one-time download of the local
SentenceTransformers model needs the internet (it is then cached).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- LLM mode -------------------------------------------------------------
# MOCK_LLM=1 (default) -> deterministic extractive generator + rubric judge.
# MOCK_LLM=0 + LLM_PROVIDER=openai + OPENAI_API_KEY -> optional real LLM.
MOCK_LLM: bool = os.getenv("MOCK_LLM", "1") != "0"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ---- Embeddings / vector store -------------------------------------------
EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR: Path = ROOT / "chroma_db"
KB_DIR: Path = ROOT / "kb"
ADDED_DOCS_DIR: Path = ROOT / "data" / "added_docs"

COLLECTION_FIXED = "practo_kb_fixed"        # fixed-size-with-overlap chunks
COLLECTION_SENTENCE = "practo_kb_sentence"  # sentence-based chunks

FIXED_CHUNK_SIZE = 200      # characters
FIXED_CHUNK_OVERLAP = 50    # characters

# Collection the agent uses (chosen in Task 5 - see README / transcripts/05).
AGENT_COLLECTION: str = os.getenv("AGENT_COLLECTION", COLLECTION_SENTENCE)
TOP_K = 3

# ---- "I don't know" threshold (Task 4) -----------------------------------
# Calibrated empirically: see evaluation/calibrate_threshold.py and
# transcripts/03_task4_threshold_calibration.txt. Measured gap on the
# sentence collection: out-of-scope max 0.3065, in-scope min 0.5548 -> 0.43.
IDK_THRESHOLD: float = float(os.getenv("IDK_THRESHOLD", "0.50"))

# Output-side groundedness: an answer sentence counts as supported when its
# cosine similarity to some retrieved chunk is at least this value.
SUPPORT_SIM_THRESHOLD: float = 0.80

# ---- Memory / logs / checkpoints -----------------------------------------
MEMORY_DIR: Path = ROOT / "memory_store"
LOG_FILE: Path = ROOT / "logs" / "requests.jsonl"
CHECKPOINT_DB: Path = ROOT / "checkpoints.sqlite"

# ---- Resilience (Task 16) -------------------------------------------------
NODE_TIMEOUT_SECONDS = 5.0      # per-node timeout on the RAG retrieval node
GLOBAL_TIMEOUT_SECONDS = 8.0    # wall-clock cap for one whole graph run
RETRY_MAX_ATTEMPTS = 4
RETRY_INITIAL_INTERVAL = 0.5    # seconds
RETRY_BACKOFF_FACTOR = 2.0
RETRY_MAX_INTERVAL = 4.0        # seconds
RETRY_JITTER = True

# ---- MCP -----------------------------------------------------------------
MCP_HOST = "127.0.0.1"
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"
