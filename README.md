# Practo Domain Support Agent (LangGraph) — Final Capstone

**Track completed: Practo (Healthcare).**

A production-minded patient-support agent for Practo's patient-experience team. It
answers clinic-policy questions from a knowledge base (RAG over ChromaDB), looks up a
specific appointment's status from a validated synthetic dataset, remembers the
conversation, is guarded against PII leaks, prompt injection and ungrounded answers,
survives interruptions (SQLite checkpointing) and transient failures (retries,
per-node and global timeouts), exposes its lookup tool over MCP, and is deployed
behind FastAPI with JSON-Lines logging.

**Everything runs under `MOCK_LLM=1` (the default) with zero API keys and zero paid
services.** Embeddings use the free local SentenceTransformers model
`all-MiniLM-L6-v2` (downloaded once from Hugging Face, then cached); the vector store
is local ChromaDB; the MCP server is local `fastmcp`. All graded transcripts in
[`transcripts/`](transcripts) were produced with `MOCK_LLM=1`.

---

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_all.py            # runs every task demo and rewrites transcripts/*.txt
# (first run downloads all-MiniLM-L6-v2 from Hugging Face; to use a local copy:
#  set EMBED_MODEL_NAME=<path to the model folder>)
uvicorn api:app --port 8080  # optional: live API (POST /ask, POST /add-document, GET /health)
```

| Task | Command | Transcript |
|---|---|---|
| 1 Dataset | `python dataset.py` | `transcripts/01_task1_dataset.txt` |
| 2–3 KB, chunking, two collections | `python -m rag.build_index` | `transcripts/02_task3_chunk_and_index.txt` |
| 4 Threshold calibration | `python -m evaluation.calibrate_threshold` | `transcripts/03_task4_threshold_calibration.txt` |
| 4 Grounded generation | `python -m demos.demo_grounded_generation` | `transcripts/04_task4_grounded_generation.txt` |
| 5 P@3 / R@3 both collections | `python -m evaluation.chunking_eval` | `transcripts/05_task5_chunking_precision_recall.txt` |
| 6 Escalation tool | `python tools.py` | `transcripts/06_task6_escalation_tool.txt` |
| 7–8 Routing + memory | `python -m demos.demo_routing_memory` | `transcripts/07_task7_8_routing_and_memory.txt` |
| 9–10 Schema + guardrails | `python -m demos.demo_guardrails` | `transcripts/08_task9_10_schema_and_guardrails.txt` |
| 11–12 FastAPI + JSONL logs | `python -m demos.demo_api` | `transcripts/09_task11_12_fastapi_and_logging.txt`, `logs/requests.jsonl` |
| 13 RAG triad (15 queries) | `python -m evaluation.rag_triad` | `transcripts/10_task13_rag_triad_eval.txt`, `evaluation/rag_triad_results.json` |
| 14 MCP server + separate client | `python -m demos.demo_mcp` (or `python mcp_server.py` + `python mcp_client.py APT-1002 APT-1007`) | `transcripts/11_task14_mcp_client_server.txt` |
| 15 SQLite checkpoint resume | `python -m demos.demo_checkpoint` | `transcripts/12_task15_sqlite_checkpointing.txt` |
| 16 Retries + timeouts | `python -m demos.demo_resilience` | `transcripts/13_task16_retries_and_timeouts.txt` |

## Repository layout

```
dataset.py                 Task 1  seeded generator -> APPOINTMENTS (60 records)
kb/*.md                    Task 2  12 knowledge-base documents (one per required topic)
rag/chunking.py            Task 3  fixed-size-with-overlap + sentence chunkers
rag/store.py               Task 3  two ChromaDB collections, embed + retrieve
rag/embeddings.py                  SentenceTransformers all-MiniLM-L6-v2 (normalised)
rag/generate.py, llm.py    Task 4  grounded generation, MOCK_LLM extractive generator
evaluation/                Tasks 4,5,13  calibration, P@3/R@3, RAG triad judge
tools.py                   Task 6  check_appointment_status + escalation_score
agent/graph.py             Task 7  LangGraph graph (10 nodes, conditional routing)
agent/memory.py            Task 8  JSON-file conversation memory (memory_store/)
agent/schema.py            Task 9  JSON Schema for every response + validator
guardrails.py              Task 10 PII masking, injection detection, groundedness
api.py                     Tasks 11-12 FastAPI app + JSON-Lines logging middleware
mcp_server.py, mcp_client.py  Task 14 fastmcp server (/mcp) and separate client
demos/demo_checkpoint.py   Task 15 SQLite checkpoint interrupt + resume
demos/demo_resilience.py   Task 16 retry policy, node timeout, global timeout
config.py                  all thresholds, timeouts, retry settings, MOCK_LLM flag
run_all.py                 runs everything and writes transcripts/
```

---

## Part 1 — Dataset design & RAG core

### Task 1 — dataset design choices (reproducible)

| Setting | Value |
|---|---|
| Seed | `SEED = 42` (Python `random.Random`) |
| Records | `N_RECORDS = 60` (`APT-1001` … `APT-1060`) |
| Category weights | General Medicine 0.30, Cardiology 0.18, Dermatology 0.18, Pediatrics 0.17, Orthopedics 0.17 |
| Status weights | Scheduled 0.34, Completed 0.30, Rescheduled 0.14, Cancelled 0.12, No-Show 0.10 |
| `follow_up_required` | Bernoulli(p = 0.20) |
| `days_since_created` | uniform integer 0–30 |
| `consultation_fee_inr` | uniform within a per-specialty band, rounded to ₹50: GM 300–600, Pediatrics 400–800, Dermatology 500–1000, Orthopedics 600–1200, Cardiology 800–1500 (overall range ₹300–₹1500) |

**Fee reasoning:** in-clinic consultations booked through Indian aggregators typically
cost a few hundred rupees for a general physician and up to roughly ₹1,500 for a senior
cardiologist, so specialty bands inside ₹300–₹1,500 are realistic (and the same bands
are written into `kb/03_consultation_fees.md`, so the KB and data agree).

**Result of the first draw (seed 42 — no regeneration was needed, nothing hand-edited):**

| Category | Count | | Status | Count |
|---|---|---|---|---|
| General Medicine | 24 | | Scheduled | 21 |
| Cardiology | 7 | | Completed | 18 |
| Dermatology | 13 | | Rescheduled | 6 |
| Pediatrics | 10 | | Cancelled | 7 |
| Orthopedics | 6 | | No-Show | 8 |

`follow_up_required=True`: **16 / 60 = 26.67 %** (inside the 10–30 % band).
`validate()` asserts every threshold and raises instead of patching records.
All data is fabricated; the dataset contains no names, diagnoses, phone numbers or insurance IDs.

### Task 2 — knowledge base

12 documents in `kb/`, 4 sentences each, written for this brief (the policies are
illustrative, not Practo's real terms):
appointment booking · cancellation/rescheduling window · consultation fees by specialty ·
insurance claims · prescription refills · lab turnaround times · telemedicine
eligibility · emergency protocol · patient-data privacy · follow-up discount ·
second opinion · home-visit eligibility.

### Task 3 — two chunking strategies, two collections

* **Fixed-size with overlap** → collection `practo_kb_fixed`: 200-character window,
  50-character overlap, snapped to word boundaries (37 chunks).
* **Sentence-based** → collection `practo_kb_sentence`: one chunk per sentence (48 chunks).
* Both embedded with `all-MiniLM-L6-v2` (L2-normalised; each chunk embedded as
  `"<doc title>: <chunk text>"` so short sentences keep their topic), stored in
  separate ChromaDB collections with cosine distance. A sample query against both
  collections is in transcript 02.

### Task 4 — grounded generation & the calibrated "I don't know" threshold

`MOCK_LLM` generation is **extractive**: it can only return sentences that exist in the
top-k retrieved chunks, ranked by similarity to the question — it cannot invent text.
If the top-1 cosine similarity is below `IDK_THRESHOLD`, it answers
*"I don't know based on the clinic policy knowledge base."*

**Calibration (measured, `transcripts/03_task4_threshold_calibration.txt`)** — top-1
cosine similarity of 6 in-scope and 4 deliberately out-of-scope queries:

| Query | Type | `practo_kb_fixed` | `practo_kb_sentence` |
|---|---|---|---|
| How late can I cancel my appointment without losing any money? | in | 0.6276 | 0.6339 |
| How much does a cardiology consultation cost? | in | 0.8387 | 0.8370 |
| How long will my thyroid test report take? | in | 0.6142 | 0.7221 |
| Can my first orthopedic consultation be done over a video call? | in | 0.6803 | 0.7169 |
| Is a follow-up visit free, and can a doctor visit me at home after my surgery? | in | 0.6430 | 0.7049 |
| What documents do I need to upload to claim insurance for my consultation? | in | 0.5969 | 0.5548 |
| Who won the cricket world cup in 2011? | out | 0.1197 | 0.0891 |
| Recommend a good gaming laptop under 80000 rupees. | out | 0.3324 | 0.3065 |
| What is the capital city of Australia? | out | 0.0517 | 0.0461 |
| Write me a poem about the monsoon. | out | 0.0914 | 0.0740 |
| **Cluster range** | in | 0.597 – 0.839 | **0.555 – 0.837** |
| **Cluster range** | out | 0.052 – 0.332 | **0.046 – 0.307** |

On the agent's collection (`practo_kb_sentence`) the out-of-scope cluster tops out at
**0.3065** and the in-scope cluster starts at **0.5548**, so I set
**`IDK_THRESHOLD = 0.43`** — the midpoint of that gap (0.4306), which also sits inside
the fixed-size collection's gap (0.332–0.597). A tutorial default of 0.6 or 0.7 would
have wrongly refused the insurance query (0.555).

**Demonstration (`transcripts/04_task4_grounded_generation.txt`)**: all 6 in-scope
queries are answered with sentences taken verbatim from the retrieved chunks (top-1
0.555–0.837); the out-of-scope *"Who won the cricket world cup in 2011?"* scores
0.089 and returns the "I don't know" fallback.
