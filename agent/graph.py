"""Task 7 - LangGraph support agent with format_response node.
"""
from __future__ import annotations

import operator
import os
import re
import uuid
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import config
from agent import memory as mem
from agent.schema import validate_response
from rag.generate import IDK_ANSWER, grounded_answer
from tools import check_appointment_status, normalize_record_id

VERBOSE = os.getenv("AGENT_VERBOSE", "1") == "1"

STATUS_WORDS = re.compile(r"\b(status|fee|cost|charge|escalat\w*|follow[- ]?up|confirmed|scheduled|cancelled)\b", re.I)
REFERS_BACK = re.compile(r"\b(its|for it|is it|does it|that one|this one|that appointment|this appointment|"
                         r"same appointment|my appointment|that booking|my booking)\b", re.I)
EXPLICIT_STATUS = re.compile(r"\b(status of|check (my|the) (appointment|booking)|appointment status|booking status)\b", re.I)
CLARIFY_ANSWER = "Please share your appointment ID (format APT-XXXX) so I can check its status."


class AgentState(TypedDict, total=False):
    thread_id: str
    trace_id: str
    question: str
    turn: int
    history: list
    last_record_id: Optional[str]
    pii: dict
    injection: bool
    injection_match: Optional[str]
    route: str
    record_id: Optional[str]
    hits: list
    top1: float
    draft_answer: str
    appointment: Optional[dict]
    grounded: Optional[bool]
    guard_reason: Optional[str]
    answer: str
    response: dict
    node_log: Annotated[list, operator.add]


def _log(node: str, msg: str = "") -> None:
    if VERBOSE:
        print(f"    [node:{node}] {msg}".rstrip(), flush=True)


async def load_memory(state: AgentState) -> dict:
    tid = state.get("thread_id") or "default"
    rec = mem.load(tid)
    trace = state.get("trace_id") or uuid.uuid4().hex
    _log("load_memory", f"thread_id={tid} existing_turns={len(rec['turns'])} last_id={rec['last_record_id']}")
    return {
        "thread_id": tid,
        "trace_id": trace,
        "turn": len(rec["turns"]) + 1,
        "history": rec["turns"],
        "last_record_id": rec["last_record_id"],
        "node_log": ["load_memory"],
    }


async def save_memory(state: AgentState) -> dict:
    tid = state["thread_id"]
    ans = state.get("answer", "")
    route = state.get("route", "")
    q = state.get("question", "")
    rec_id = state.get("record_id") or state.get("last_record_id")
    turns = list(state.get("history", [])) + [{"turn": state.get("turn", 1), "user": q, "assistant": ans, "route": route}]
    mem.save(tid, {"thread_id": tid, "turns": turns, "last_record_id": rec_id})
    _log("save_memory", f"saved turn={state.get('turn')} last_record_id={rec_id}")
    return {"node_log": ["save_memory"]}


async def classify_intent(state: AgentState) -> dict:
    q = state.get("question", "")
    last_id = state.get("last_record_id")
    rid = normalize_record_id(q)
    if not rid and last_id and (REFERS_BACK.search(q) or STATUS_WORDS.search(q)):
        rid = last_id
    if rid:
        route = "appointment_status"
    elif EXPLICIT_STATUS.search(q) and not rid:
        route = "clarify"
    else:
        route = "rag"
    _log("classify_intent", f"route={route} record_id={rid}")
    return {"route": route, "record_id": rid, "node_log": ["classify_intent"]}


def route_intent(state: AgentState) -> str:
    r = state.get("route", "rag")
    return r if r in {"rag", "appointment_status", "blocked", "clarify"} else "rag"


async def rag_tool(state: AgentState, config_: RunnableConfig | None = None) -> dict:
    res = grounded_answer(state.get("question", ""))
    _log("rag_tool", f"top1={res['top1']:.3f} hits={len(res['hits'])} draft={res['answer'][:60]}...")
    return {
        "hits": res["hits"],
        "top1": res["top1"],
        "draft_answer": res["answer"],
        "node_log": ["rag_tool"],
    }


async def status_tool(state: AgentState, config_: RunnableConfig | None = None) -> dict:
    rid = state.get("record_id") or ""
    info = check_appointment_status(rid)
    if not info.get("found"):
        ans = f"I couldn't find an appointment with ID {rid}. Please check the ID."
    else:
        parts = [
            f"Appointment {info['record_id']} ({info['category']}) is currently {info['status']}.",
            f"Consultation fee: INR {info['consultation_fee_inr']}.",
            f"Created {info['days_since_created']} day(s) ago; follow-up required: "
            f"{'yes' if info['follow_up_required'] else 'no'}."
        ]
        if info["escalate"]:
            parts.append(
                f"Attention: This appointment has an escalation score of {info['escalation_score']} "
                f"(>= {info['escalation_threshold']}). A patient-experience specialist has been notified."
            )
        ans = " ".join(parts)
    _log("status_tool", f"record={rid} escalate={info.get('escalate')} ans={ans[:60]}...")
    return {
        "appointment": info,
        "draft_answer": ans,
        "node_log": ["status_tool"],
    }


async def clarify(state: AgentState) -> dict:
    _log("clarify", "asking for appointment ID")
    return {"draft_answer": CLARIFY_ANSWER, "node_log": ["clarify"]}


async def format_response(state: AgentState) -> dict:
    hits = state.get("hits") or []
    sources = [
        {"doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "score": round(float(h["score"]), 3)}
        for h in hits
    ]
    ans = state.get("answer") or state.get("draft_answer") or ""
    app = state.get("appointment")
    appointment_out = None
    if app:
        appointment_out = {
            "record_id": app["record_id"],
            "found": app["found"],
            "status": app.get("status"),
            "consultation_fee_inr": app.get("consultation_fee_inr"),
            "escalation_score": app.get("escalation_score"),
            "escalate": app.get("escalate"),
        }
    pii = state.get("pii") or {}
    resp = {
        "trace_id": state["trace_id"],
        "thread_id": state["thread_id"],
        "turn": state.get("turn", 1),
        "route": state.get("route", "rag"),
        "answer": ans,
        "sources": sources,
        "appointment": appointment_out,
        "guardrails": {
            "pii_masked": bool(pii.get("pii_masked", False)),
            "injection_detected": bool(state.get("injection", False)),
            "grounded": state.get("grounded"),
            "reason": state.get("guard_reason"),
        },
        "mock_llm": config.MOCK_LLM,
    }
    validate_response(resp)
    _log("format_response", f"valid response, answer len={len(ans)}")
    return {"answer": ans, "response": resp, "node_log": ["format_response"]}
