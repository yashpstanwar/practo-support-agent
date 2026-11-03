"""Task 7 - LangGraph support agent state definition.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

import config


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
