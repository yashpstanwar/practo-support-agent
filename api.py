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
