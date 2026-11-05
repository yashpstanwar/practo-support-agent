"""Tasks 7 + 8 + 9 - routing to both tools, persisted multi-turn memory, a fresh
conversation with memory absent, and schema validation of every response.

Run:  python -m demos.demo_routing_memory
"""
from __future__ import annotations

import json

import config
from agent import memory as mem
from agent.graph import run_agent
from agent.schema import validate_response


def ask(q: str, thread: str) -> dict:
    print(f"\nUSER [{thread}]: {q}")
    r = run_agent(q, thread)
    validate_response(r)
    print(f"AGENT route={r['route']}  turn={r['turn']}  (schema: VALID)")
    print(f"  answer: {r['answer']}")
    if r["appointment"]:
        print(f"  appointment: {json.dumps(r['appointment'])}")
    if r["sources"]:
        print(f"  sources: {[s['chunk_id'] for s in r['sources']]}")
    return r


def main() -> None:
    print(f"MOCK_LLM={config.MOCK_LLM}")
    print("=" * 78)
    print("PART A - conditional edge routes to BOTH tools")
    print("=" * 78)
    mem.reset("routing-demo")
    a = ask("How much does a cardiology consultation cost?", "routing-demo")
    b = ask("Can you check the status of appointment APT-1007?", "routing-demo")
    assert a["route"] == "rag" and b["route"] == "appointment_status"
    print("\nOK: query 1 -> rag_tool, query 2 -> status_tool")

    print("\n" + "=" * 78)
    print("PART B - TRANSCRIPT 1: multi-turn conversation, state carried via memory_store/patient-42.json")
    print("=" * 78)
    mem.reset("patient-42")
    ask("Hi, please check my appointment APT-1022.", "patient-42")
    r2 = ask("What is the fee for it?", "patient-42")
    r3 = ask("Does it need a follow-up, and is it escalated?", "patient-42")
    r4 = ask("What is the discount on a follow-up visit?", "patient-42")
    assert r2["appointment"]["record_id"] == "APT-1022" and r3["appointment"]["record_id"] == "APT-1022"
    assert r4["route"] == "rag" and r4["turn"] == 4
    print("\nPersisted memory file memory_store/patient-42.json:")
    print(json.dumps(mem.load("patient-42"), indent=2))
    print("OK: turns 2 and 3 never mention an ID - APT-1022 was carried over from turn 1 via persisted memory")

    print("\n" + "=" * 78)
    print("PART C - TRANSCRIPT 2: fresh conversation (new thread) - memory correctly absent")
    print("=" * 78)
    mem.reset("patient-77")
    f1 = ask("What is the fee for it?", "patient-77")
    f2 = ask("What's my appointment status?", "patient-77")
    assert f1["turn"] == 1 and f1["appointment"] is None and f1["route"] != "appointment_status"
    assert f2["route"] == "clarify"
    print("\nPersisted memory file memory_store/patient-77.json:")
    print(json.dumps(mem.load("patient-77"), indent=2))
    print("OK: new thread starts at turn 1 with last_record_id=None; the agent does NOT know APT-1022 "
          "and asks for an ID instead.")


if __name__ == "__main__":
    main()
