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

### Task 5 — Precision@3 / Recall@3 (document level)

Per query and collection: retrieve the top-15 chunks, map each chunk to its parent
document, de-duplicate in rank order, keep the first 3 unique documents, then
P@3 = |relevant ∩ top-3| / 3 and R@3 = |relevant ∩ top-3| / |relevant|. Because most
queries have exactly one relevant document, the maximum attainable P@3 is 0.333 for
those queries (0.667 for the two-document query). Full per-query arithmetic for both
collections: `transcripts/05_task5_chunking_precision_recall.txt`.

| Query (relevant docs) | Fixed P@3 | Fixed R@3 | Sentence P@3 | Sentence R@3 |
|---|---|---|---|---|
| Q1 cancel without losing money (02) | 1/3 = 0.333 | 1/1 = 1.000 | 1/3 = 0.333 | 1/1 = 1.000 |
| Q2 cardiology fee (03) | 1/3 = 0.333 | 1/1 = 1.000 | 1/3 = 0.333 | 1/1 = 1.000 |
| Q3 thyroid report time (06) | 1/3 = 0.333 | 1/1 = 1.000 | 1/3 = 0.333 | 1/1 = 1.000 |
| Q4 first ortho visit by video (07) | 1/3 = 0.333 | 1/1 = 1.000 | 1/3 = 0.333 | 1/1 = 1.000 |
| Q5 free follow-up + home visit (10, 12) | 2/3 = 0.667 | 2/2 = 1.000 | 2/3 = 0.667 | 2/2 = 1.000 |
| Q6 insurance claim documents (04) | 1/3 = 0.333 | 1/1 = 1.000 | 1/3 = 0.333 | 1/1 = 1.000 |
| **Mean** | **2.333/6 = 0.389** | **1.000** | **2.333/6 = 0.389** | **1.000** |
| Mean similarity of best relevant chunk | 0.667 | | **0.695** | |

**Recommendation:** both strategies tie on the graded metrics — mean P@3 = 0.389 (the
ceiling for this query set) and mean R@3 = 1.000, with the relevant document ranked
first for every query — so I deploy the **sentence-based collection**. It gives the
higher mean similarity for the best relevant chunk (0.695 vs 0.667, and a higher top-1
on 5 of 6 queries), and its chunks are whole sentences that the extractive generator
can quote cleanly, whereas fixed-size chunks cut sentences mid-way
(e.g. *"…A patient who does"*).

---

## Part 2 — LangGraph agent with tools, memory & guardrails

### Task 6 — `check_appointment_status` and the escalation score

```
escalation_score = 0.6 * follow_up_required + 0.4 * (days_since_created / 30)
escalate if escalation_score >= 0.80
```

* `follow_up_required` carries 60 % of the weight — a doctor-flagged follow-up is the
  main clinical reason for a human to step in.
* The age signal `days_since_created / 30` is normalised by the dataset's 30-day
  maximum; the longer a flagged appointment stays open, the more likely the free /
  discounted follow-up window (`kb/10`) is being missed.
* **Threshold 0.80 = 0.6 + 0.4 × (15 / 30)**, where **15 days is the 50th percentile
  (median) of `days_since_created` in my generated data** (P25 = 7.75, P50 = 15,
  P75 = 23, P80 = 25.2). So a record escalates only when follow-up is required **and**
  it is older than the median appointment; a record without follow-up scores at most
  0.40 and never escalates on age alone. In this dataset **4 / 60 records (6.7 %)**
  escalate (`APT-1002`, `APT-1025`, `APT-1044`, `APT-1047`) — a manageable queue.

### Task 7 — the graph

```
START → load_memory → input_guard → classify_intent ──(conditional edge)──┬─ rag_tool ─────┐
                                                                         ├─ status_tool ──┤
                                                                         ├─ blocked ──────┤
                                                                         └─ clarify ──────┤
                     END ← save_memory ← format_response ← output_guard ←─────────────────┘
```

10 nodes. `classify_intent` routes on intent: a message containing an appointment ID
(`APT-XXXX`), or a status/fee question that refers back to the appointment in memory
("what is the fee for **it**?"), goes to `status_tool`; injection attempts go to
`blocked`; status questions without any ID go to `clarify`; everything else goes to
`rag_tool`. Transcript 07 shows both tool routes firing.

### Task 8 — persisted memory

`memory_store/<thread_id>.json` stores every turn (masked user text, answer, route) and
`last_record_id`. Transcript 07 has **two separate transcripts**: a 4-turn conversation
where turns 2–3 never mention the ID but are answered about `APT-1022` from memory, and
a fresh thread where the same question finds no state, starts at turn 1, and is asked
for an appointment ID.

### Task 9 — structured output

`agent/schema.py` defines a JSON Schema (draft 2020-12, `additionalProperties: false`)
for every response: `trace_id, thread_id, turn, route, answer, sources[], appointment,
guardrails{pii_masked, injection_detected, grounded, reason}, mock_llm`. The
`format_response` node validates every response with `jsonschema` and raises on any
violation; the demos re-validate and a negative test shows a malformed response being rejected.

### Task 10 — guardrails (each demonstrated firing in transcript 08)

* **PII masking (input)** — Indian mobile numbers, the fixed-format PII field
  (`+91 98765 43210`, `09812345678`, `98765-43210`, …) → `[PHONE_MASKED]`; e-mails are
  masked too. The masked text replaces the raw question inside the graph, so memory,
  logs and the model only ever see the masked value. Patient name, diagnosis/condition
  and insurance ID are free text with no universal format, so — as the brief states —
  they are out of scope for a keyless regex masker; only fabricated examples are used.
* **Prompt-injection detection (input)** — pattern detector (ignore/disregard previous
  instructions, reveal system prompt, developer mode, dump all records, …); a hit routes
  to the `blocked` node and no tool runs.
* **Groundedness (output)** — `output_guard` refuses when (a) top-1 retrieval similarity
  is below the calibrated threshold, or (b) any answer sentence is not supported by the
  retrieved context (verbatim, or cosine ≥ 0.80 to a context sentence). Demonstrated with
  an out-of-scope medical question and with a deliberately hallucinated draft answer.

---

## Part 3 — Evaluation, observability & deployment

* **Task 11** — `api.py`: `POST /ask` (`AskRequest` → `AskResponse`), `POST /add-document`
  (`AddDocumentRequest` → `AddDocumentResponse`, chunks the new doc into **both**
  collections), `GET /health`. Pydantic models throughout; empty questions return 422.
* **Task 12** — an HTTP middleware writes exactly one JSON line per request to
  `logs/requests.jsonl`: `ts, trace_id, method, path, status, latency_ms,
  request_text (masked), pii_masked, route`. The request body is passed through the
  same `mask_pii` as the agent, so the phone number in the demo is stored only as
  `[PHONE_MASKED]` (the demo asserts the raw digits are absent from the log file).
* **Task 13** — 15 test queries: one per required KB topic (12), two out-of-scope
  (sports, medication dosing) and one edge case (`"refund?"`). The LLM-as-judge prompt is
  in `evaluation/rag_triad.py`; under `MOCK_LLM` the same rubric is scored
  deterministically (definitions in the file header).

**Results (`transcripts/10_task13_rag_triad_eval.txt`, `evaluation/rag_triad_results.json`):**

| # | Topic | Context relevance | Groundedness | Answer relevance |
|---|---|---|---|---|
| Q01 | appointment-booking policy | 1.000 | 1.000 | 0.658 |
| Q02 | cancellation/rescheduling window | 1.000 | 1.000 | 0.596 |
| Q03 | consultation-fee structure | 1.000 | 1.000 | 0.708 |
| Q04 | insurance-claim process | 1.000 | 1.000 | 0.588 |
| Q05 | prescription-refill policy | 1.000 | 1.000 | 0.717 |
| Q06 | lab-test turnaround times | 1.000 | 1.000 | 0.661 |
| Q07 | telemedicine eligibility | 1.000 | 1.000 | 0.672 |
| Q08 | emergency-visit protocol | 0.333 | 1.000 | 0.502 |
| Q09 | patient-data privacy policy | 1.000 | 1.000 | 0.812 |
| Q10 | follow-up-visit discount policy | 1.000 | 1.000 | 0.800 |
| Q11 | second-opinion process | 1.000 | 1.000 | 0.724 |
| Q12 | home-visit eligibility | 1.000 | 1.000 | 0.707 |
| Q13 | OUT-OF-SCOPE: "Who won the IPL in 2023?" | 0.000 | 1.000 | 1.000 |
| Q14 | OUT-OF-SCOPE: paracetamol dose | 0.000 | 1.000 | 1.000 |
| Q15 | EDGE: "refund?" | 0.000 | 1.000 | 0.000 |
| | **Average (15 queries)** | **0.756** | **1.000** | **0.676** |

Reading the numbers: groundedness is 1.000 by construction — the extractive mock
generator can only emit retrieved sentences, and the output guard would block anything
else. Both out-of-scope queries are correctly refused. The two weak spots are honest
findings: Q08 (stroke) retrieves only one strongly relevant chunk because the emergency
policy is a single sentence among unrelated ones, and the one-word edge case `"refund?"`
scores 0.4265 — just under the 0.43 threshold — so the agent abstains rather than
guessing which refund policy is meant (a real deployment would ask a clarifying question
here). Q07 shows the extractive generator's limit: it answers with related telemedicine
sentences rather than the exact "not suitable for chest pain" sentence.

---
