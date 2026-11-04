"""Task 7 - LangGraph support agent with intent classifier.
"""
from __future__ import annotations

import operator
import os
import re
import uuid
from typing import Annotated, Any, Optional, TypedDict

import config
from agent import memory as mem
from tools import normalize_record_id

VERBOSE = os.getenv("AGENT_VERBOSE", "1") == "1"

STATUS_WORDS = re.compile(r"\b(status|fee|cost|charge|escalat\w*|follow[- ]?up|confirmed|scheduled|cancelled)\b", re.I)
REFERS_BACK = re.compile(r"\b(its|for it|is it|does it|that one|this one|that appointment|this appointment|"
                         r"same appointment|my appointment|that booking|my booking)\b", re.I)
EXPLICIT_STATUS = re.compile(r"\b(status of|check (my|the) (appointment|booking)|appointment status|booking status)\b", re.I)


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
