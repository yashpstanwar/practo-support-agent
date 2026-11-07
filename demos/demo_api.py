"""Tasks 11 + 12 - exercise the FastAPI endpoints and show the JSON-Lines log.

Uses FastAPI's TestClient (real ASGI app, no network). To run the live server
instead:  uvicorn api:app --port 8080

Run:  python -m demos.demo_api
"""
from __future__ import annotations

import json
import shutil

from fastapi.testclient import TestClient

import config
from agent import memory as mem
from api import app
from rag.store import delete_document


def main() -> None:
    if config.LOG_FILE.exists():
        config.LOG_FILE.unlink()
    shutil.rmtree(config.ADDED_DOCS_DIR, ignore_errors=True)
    mem.reset("api-user-1")
    c = TestClient(app)

    calls = [
        ("GET", "/health", None),
        ("POST", "/ask", {"question": "How long do culture test reports take?", "thread_id": "api-user-1"}),
        ("POST", "/ask", {"question": "Call me on +91 98765 43210 about appointment APT-1044 status",
                          "thread_id": "api-user-1"}),
        ("POST", "/add-document", {"doc_id": "13_parking_policy", "title": "Clinic Parking Policy",
                                   "text": "Free parking for patients is available for two hours at every Practo "
                                           "partner clinic. Patients must show their appointment SMS at the parking "
                                           "desk to get the free parking token."}),
        ("POST", "/ask", {"question": "Is parking free for patients at the clinic?", "thread_id": "api-user-1"}),
        ("POST", "/ask", {"question": "", "thread_id": "api-user-1"}),  # 422 validation error
    ]
    for method, path, body in calls:
        r = c.request(method, path, json=body)
        print(f"{method} {path}  body={json.dumps(body) if body else '-'}")
        print(f"  -> HTTP {r.status_code}  x-trace-id={r.headers.get('x-trace-id')}")
        print("  " + json.dumps(r.json(), indent=2)[:900].replace("\n", "\n  ") + "\n")

    print("=" * 78)
    print(f"Structured log: {config.LOG_FILE.relative_to(config.ROOT)}  (one JSON line per request)")
    print("=" * 78)
    lines = config.LOG_FILE.read_text(encoding="utf-8").splitlines()
    for line in lines:
        print(line)
    raw_log = "\n".join(lines)
    assert len(lines) == len(calls), "expected exactly one log line per request"
    assert all("trace_id" in json.loads(x) and "latency_ms" in json.loads(x) for x in lines)
    assert "98765" not in raw_log and "43210" not in raw_log, "raw phone number leaked into log!"
    print(f"\nOK: {len(lines)} requests -> {len(lines)} JSONL entries, each with trace_id + latency_ms; "
          "the phone number appears only as [PHONE_MASKED].")
    shutil.rmtree(config.ADDED_DOCS_DIR, ignore_errors=True)
    delete_document("13_parking_policy")  # keep the graded KB at exactly the 12 authored docs


if __name__ == "__main__":
    main()
