"""Task 7 - LangGraph support agent with memory nodes.
"""
from __future__ import annotations

import operator
import os
import uuid
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.runnables import RunnableConfig

import config
from agent import memory as mem

VERBOSE = os.getenv("AGENT_VERBOSE", "1") == "1"


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
