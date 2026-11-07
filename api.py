"""Tasks 11 + 12 - FastAPI deployment with structured JSON-Lines logging.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import config
from agent.graph import GlobalTimeoutError, arun_agent
from rag.chunking import Document
from rag.store import index_documents

app = FastAPI(title="Practo Domain Support Agent", version="1.0.0")


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": config.MOCK_LLM, "collection": config.AGENT_COLLECTION}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    trace_id = getattr(request.state, "trace_id", None) or uuid.uuid4().hex
    try:
        res = await arun_agent(body.question, body.thread_id, trace_id=trace_id)
    except GlobalTimeoutError:
        raise HTTPException(status_code=504, detail="Agent run exceeded global timeout")
    request.state.route = res.get("route")
    return AskResponse(**res)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document_endpoint(body: AddDocumentRequest, request: Request) -> AddDocumentResponse:
    trace_id = getattr(request.state, "trace_id", None) or uuid.uuid4().hex
    doc = Document(doc_id=body.doc_id, title=body.title, text=body.text)
    config.ADDED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.ADDED_DOCS_DIR / f"{body.doc_id}.md"
    out.write_text(f"# {body.title}\n\n{body.text}\n", encoding="utf-8")
    counts = index_documents([doc], persist_directory=config.CHROMA_DIR)
    return AddDocumentResponse(trace_id=trace_id, doc_id=body.doc_id, chunks_indexed=counts)
