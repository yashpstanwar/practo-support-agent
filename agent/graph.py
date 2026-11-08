"""Task 7 (+8, 9, 10) - the LangGraph support agent.

Nodes (10):
  load_memory -> input_guard -> classify_intent --(conditional edge)--> one of
      rag_tool | status_tool | blocked | clarify
  -> output_guard -> format_response -> save_memory -> END
"""
from __future__ import annotations

import asyncio
import operator
import os
import re
import time
import uuid
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

import config
from agent import memory as mem
from agent.schema import validate_response
from guardrails import check_groundedness, detect_injection, mask_pii
from rag.generate import IDK_ANSWER, grounded_answer
from tools import TransientBackendError, check_appointment_status, normalize_record_id

VERBOSE = os.getenv("AGENT_VERBOSE", "1") == "1"

REFUSAL_UNGROUNDED = ("I can't answer that reliably from the clinic policy knowledge base, "
                      "so I won't guess. Please contact Practo support for help.")
BLOCKED_ANSWER = ("I can't help with that request. I can answer clinic-policy questions or "
                  "check an appointment by its ID (for example APT-1007).")
CLARIFY_ANSWER = "Please share your appointment ID (format APT-XXXX) so I can check its status."

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


_FLAKY_CALLS: dict[str, int] = {}


def _log(node: str, msg: str = "") -> None:
    if VERBOSE:
        print(f"    [node:{node}] {msg}".rstrip(), flush=True)


def _cfg(config_: RunnableConfig | None) -> dict:
    return (config_ or {}).get("configurable", {}) or {}


async def _maybe_delay(node: str, config_: RunnableConfig | None) -> None:
    d = _cfg(config_).get("simulate_delay", {}).get(node)
    if d:
        _log(node, f"simulated slow call: sleeping {d}s")
        await asyncio.sleep(d)


async def load_memory(state: AgentState, config: RunnableConfig) -> dict:
    await _maybe_delay("load_memory", config)
    m = mem.load(state["thread_id"])
    _log("load_memory", f"thread={state['thread_id']} prior_turns={len(m['turns'])} "
                        f"last_record_id={m['last_record_id']}")
    return {"history": m["turns"], "last_record_id": m["last_record_id"],
            "turn": len(m["turns"]) + 1, "node_log": ["load_memory"]}


async def input_guard(state: AgentState, config: RunnableConfig) -> dict:
    await _maybe_delay("input_guard", config)
    masked, pii = mask_pii(state["question"])
    inj, match = detect_injection(masked)
    _log("input_guard", f"pii_masked={pii['pii_masked']} injection={inj}"
                        + (f" match='{match}'" if match else ""))
    return {"question": masked, "pii": pii, "injection": inj, "injection_match": match,
            "node_log": ["input_guard"]}


async def classify_intent(state: AgentState, config: RunnableConfig) -> dict:
    await _maybe_delay("classify_intent", config)
    q = state["question"]
    rid = normalize_record_id(q)
    if state.get("injection"):
        route = "blocked"
    elif rid:
        route = "appointment_status"
    elif state.get("last_record_id") and STATUS_WORDS.search(q) and REFERS_BACK.search(q):
        route, rid = "appointment_status", state["last_record_id"]
    elif EXPLICIT_STATUS.search(q) or (STATUS_WORDS.search(q) and REFERS_BACK.search(q)):
        route = "clarify"
    else:
        route = "rag"
    _log("classify_intent", f"route={route}" + (f" record_id={rid}" if rid else ""))
    return {"route": route, "record_id": rid, "node_log": ["classify_intent"]}


def route_by_intent(state: AgentState) -> str:
    return {"rag": "rag_tool", "appointment_status": "status_tool",
            "blocked": "blocked", "clarify": "clarify"}[state["route"]]


async def rag_tool(state: AgentState, config: RunnableConfig) -> dict:
    await _maybe_delay("rag_tool", config)
    r = await asyncio.to_thread(grounded_answer, state["question"])
    draft = _cfg(config).get("force_draft_answer") or r["answer"]
    hits = [{"doc_id": h.doc_id, "chunk_id": h.chunk_id, "score": h.score, "text": h.text} for h in r["hits"]]
    _log("rag_tool", f"top1={r['top1']:.3f} fallback={r['fallback']} docs={[h['doc_id'] for h in hits]}")
    return {"hits": hits, "top1": r["top1"], "draft_answer": draft, "appointment": None,
            "node_log": ["rag_tool"]}


async def status_tool(state: AgentState, config: RunnableConfig) -> dict:
    await _maybe_delay("status_tool", config)
    rec = check_appointment_status(state["record_id"])
    if not rec["found"]:
        draft = f"I couldn't find an appointment with ID {rec['record_id']}. Please check the ID and try again."
    else:
        q = state["question"].lower()
        fee = f"The consultation fee is INR {rec['consultation_fee_inr']}."
        status = f"Appointment {rec['record_id']} ({rec['category']}) is currently {rec['status']}."
        fu = "A follow-up visit is required." if rec["follow_up_required"] else "No follow-up visit is required."
        esc = (f"Escalation score {rec['escalation_score']:.2f} (threshold {rec['escalation_threshold']:.2f}): "
               + ("escalated to the patient-experience team." if rec["escalate"] else "no escalation needed."))
        parts = [fee, status, fu, esc] if re.search(r"\b(fee|cost|charge)\b", q) else [status, fee, fu, esc]
        draft = " ".join(parts)
    _log("status_tool", f"success, found={rec['found']}")
    return {"appointment": rec, "draft_answer": draft, "hits": [], "top1": 1.0,
            "node_log": ["status_tool"]}


async def blocked(state: AgentState, config: RunnableConfig) -> dict:
    _log("blocked", f"prompt injection blocked: '{state.get('injection_match')}'")
    return {"draft_answer": BLOCKED_ANSWER, "hits": [], "appointment": None, "node_log": ["blocked"]}


async def clarify(state: AgentState, config: RunnableConfig) -> dict:
    _log("clarify", "no appointment ID in message or memory")
    return {"draft_answer": CLARIFY_ANSWER, "hits": [], "appointment": None, "node_log": ["clarify"]}


async def output_guard(state: AgentState, config: RunnableConfig) -> dict:
    await _maybe_delay("output_guard", config)
    route = state["route"]
    if route == "rag":
        g = await asyncio.to_thread(check_groundedness, state["draft_answer"],
                                    [h["text"] for h in state["hits"]], state["top1"])
        if g["grounded"]:
            answer = state["draft_answer"]
        elif g["reason"].startswith("retrieval_below_threshold"):
            answer = IDK_ANSWER
        else:
            answer = REFUSAL_UNGROUNDED
        _log("output_guard", f"grounded={g['grounded']} reason={g['reason']}")
        return {"answer": answer, "grounded": g["grounded"], "guard_reason": g["reason"],
                "node_log": ["output_guard"]}
    if route == "appointment_status":
        _log("output_guard", "answer built from tool result -> grounded")
        return {"answer": state["draft_answer"], "grounded": True, "guard_reason": "tool_result",
                "node_log": ["output_guard"]}
    reason = f"prompt_injection: {state.get('injection_match')}" if route == "blocked" else "missing_record_id"
    _log("output_guard", f"pass-through ({reason})")
    return {"answer": state["draft_answer"], "grounded": None, "guard_reason": reason,
            "node_log": ["output_guard"]}


async def format_response(state: AgentState, config: RunnableConfig) -> dict:
    appt = state.get("appointment")
    if appt is not None:
        appt = {k: appt[k] for k in ("record_id", "found", "status", "consultation_fee_inr",
                                     "escalation_score", "escalate") if k in appt}
    resp = {
        "trace_id": state["trace_id"],
        "thread_id": state["thread_id"],
        "turn": state["turn"],
        "route": state["route"],
        "answer": state["answer"],
        "sources": [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "score": float(h["score"])}
                    for h in state.get("hits", [])],
        "appointment": appt,
        "guardrails": {"pii_masked": bool(state["pii"]["pii_masked"]),
                       "injection_detected": bool(state.get("injection")),
                       "grounded": state.get("grounded"),
                       "reason": state.get("guard_reason")},
        "mock_llm": bool(config.MOCK_LLM),
    }
    validate_response(resp)
    _log("format_response", "response validated against RESPONSE_SCHEMA")
    return {"response": resp, "node_log": ["format_response"]}


async def save_memory(state: AgentState, config: RunnableConfig) -> dict:
    turns = list(state.get("history", [])) + [
        {"turn": state["turn"], "user": state["question"], "assistant": state["answer"],
         "route": state["route"]}]
    last = state.get("last_record_id")
    appt = state.get("appointment")
    if appt and appt.get("found"):
        last = appt["record_id"]
    path = mem.save(state["thread_id"], {"thread_id": state["thread_id"], "turns": turns,
                                         "last_record_id": last})
    _log("save_memory", f"saved {len(turns)} turn(s), last_record_id={last} -> {path.name}")
    return {"last_record_id": last, "node_log": ["save_memory"]}


STATUS_RETRY = RetryPolicy(
    max_attempts=config.RETRY_MAX_ATTEMPTS,
    initial_interval=config.RETRY_INITIAL_INTERVAL,
    backoff_factor=config.RETRY_BACKOFF_FACTOR,
    max_interval=config.RETRY_MAX_INTERVAL,
    jitter=config.RETRY_JITTER,
    retry_on=TransientBackendError,
)

def build_graph(checkpointer: Any = None, interrupt_before: list[str] | None = None):
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory)
    g.add_node("input_guard", input_guard)
    g.add_node("classify_intent", classify_intent)
    g.add_node("rag_tool", rag_tool)
    g.add_node("status_tool", status_tool, retry_policy=STATUS_RETRY)
    g.add_node("blocked", blocked)
    g.add_node("clarify", clarify)
    g.add_node("output_guard", output_guard)
    g.add_node("format_response", format_response)
    g.add_node("save_memory", save_memory)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "input_guard")
    g.add_edge("input_guard", "classify_intent")
    g.add_conditional_edges("classify_intent", route_by_intent,
                            ["rag_tool", "status_tool", "blocked", "clarify"])
    for n in ("rag_tool", "status_tool", "blocked", "clarify"):
        g.add_edge(n, "output_guard")
    g.add_edge("output_guard", "format_response")
    g.add_edge("format_response", "save_memory")
    g.add_edge("save_memory", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


_DEFAULT_GRAPH = None


def default_graph():
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        _DEFAULT_GRAPH = build_graph()
    return _DEFAULT_GRAPH


async def arun_agent(question: str, thread_id: str = "default", *, trace_id: str | None = None,
                     simulate: dict | None = None, graph=None) -> dict:
    trace_id = trace_id or uuid.uuid4().hex
    graph = graph or default_graph()
    inputs = {"thread_id": thread_id, "trace_id": trace_id, "question": question, "node_log": []}
    cfg = {"configurable": {"thread_id": f"{thread_id}:{trace_id}", **(simulate or {})}}
    state = await graph.ainvoke(inputs, cfg)
    return state["response"]


def run_agent(question: str, thread_id: str = "default", **kw) -> dict:
    return asyncio.run(arun_agent(question, thread_id, **kw))
