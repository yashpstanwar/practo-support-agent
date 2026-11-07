"""Tasks 11 + 12 - FastAPI deployment with structured JSON-Lines logging.

Endpoints
  POST /ask            {"question", "thread_id"} -> AgentResponse
  POST /add-document   {"doc_id", "title", "text"} -> indexes into BOTH collections
  GET  /health

Every request (any path) produces exactly one JSON line in logs/requests.jsonl
with trace_id, method, path, status, latency_ms and the request text AFTER the
same PII masking the agent applies - raw phone numbers never reach disk.

Run:  uvicorn api:app --port 8080
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import config
from agent.graph import GlobalTimeoutError, arun_agent
from guardrails import mask_pii
from rag.chunking import Document
from rag.store import index_documents

app = FastAPI(title="Practo Domain Support Agent", version="1.0.0")


# ----------------------------------------------------------------- models
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field("default", min_length=1, max_length=100)


class Source(BaseModel):
    doc_id: str
    chunk_id: str
    score: float


class Appointment(BaseModel):
    record_id: str
    found: bool
    status: Optional[str] = None
    consultation_fee_inr: Optional[int] = None
    escalation_score: Optional[float] = None
    escalate: Optional[bool] = None


class Guardrails(BaseModel):
    pii_masked: bool
    injection_detected: bool
    grounded: Optional[bool]
    reason: Optional[str]


class AskResponse(BaseModel):
    trace_id: str
    thread_id: str
    turn: int
    route: Literal["rag", "appointment_status", "blocked", "clarify"]
    answer: str
    sources: list[Source]
    appointment: Optional[Appointment]
    guardrails: Guardrails
    mock_llm: bool


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., pattern=r"^[a-z0-9_]{3,60}$")
    title: str = Field(..., min_length=3, max_length=120)
    text: str = Field(..., min_length=20, max_length=5000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_indexed: dict[str, int]


# ----------------------------------------------------------------- logging
def write_log(entry: dict) -> None:
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@app.middleware("http")
async def jsonl_logger(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
    request.state.trace_id = trace_id
    request.state.route = None
    body = (await request.body()).decode("utf-8", errors="replace")
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-trace-id"] = trace_id
        return response
    finally:
        masked_body, pii = mask_pii(body)
        write_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "request_text": masked_body,
            "pii_masked": pii["pii_masked"],
            "route": request.state.route,
            "mock_llm": config.MOCK_LLM,
        })


# ----------------------------------------------------------------- endpoints
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "mock_llm": config.MOCK_LLM, "collection": config.AGENT_COLLECTION}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    try:
        resp = await arun_agent(req.question, req.thread_id, trace_id=request.state.trace_id)
    except GlobalTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    request.state.route = resp["route"]
    return AskResponse(**resp)


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(req: AddDocumentRequest, request: Request) -> AddDocumentResponse:
    config.ADDED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    text = re.sub(r"\s+", " ", req.text).strip()
    (config.ADDED_DOCS_DIR / f"{req.doc_id}.md").write_text(f"# {req.title}\n\n{text}\n", encoding="utf-8")
    counts = index_documents([Document(doc_id=req.doc_id, title=req.title, text=text)])
    request.state.route = "add_document"
    return AddDocumentResponse(trace_id=request.state.trace_id, doc_id=req.doc_id, chunks_indexed=counts)
