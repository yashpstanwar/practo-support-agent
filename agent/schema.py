"""Task 9 - JSON Schema every agent response must satisfy (validated in code)."""
from __future__ import annotations

from jsonschema import Draft202012Validator

RESPONSE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PractoAgentResponse",
    "type": "object",
    "additionalProperties": False,
    "required": ["trace_id", "thread_id", "turn", "route", "answer", "sources",
                 "appointment", "guardrails", "mock_llm"],
    "properties": {
        "trace_id": {"type": "string", "minLength": 8},
        "thread_id": {"type": "string", "minLength": 1},
        "turn": {"type": "integer", "minimum": 1},
        "route": {"enum": ["rag", "appointment_status", "blocked", "clarify"]},
        "answer": {"type": "string", "minLength": 1},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["doc_id", "chunk_id", "score"],
                "properties": {
                    "doc_id": {"type": "string"},
                    "chunk_id": {"type": "string"},
                    "score": {"type": "number", "minimum": -1, "maximum": 1},
                },
            },
        },
        "appointment": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "required": ["record_id", "found"],
                    "properties": {
                        "record_id": {"type": "string", "pattern": "^APT-\\d{4}$"},
                        "found": {"type": "boolean"},
                        "status": {"enum": ["Scheduled", "Completed", "Cancelled", "No-Show", "Rescheduled"]},
                        "consultation_fee_inr": {"type": "integer", "minimum": 0},
                        "escalation_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "escalate": {"type": "boolean"},
                    },
                },
            ]
        },
        "guardrails": {
            "type": "object",
            "additionalProperties": False,
            "required": ["pii_masked", "injection_detected", "grounded", "reason"],
            "properties": {
                "pii_masked": {"type": "boolean"},
                "injection_detected": {"type": "boolean"},
                "grounded": {"type": ["boolean", "null"]},
                "reason": {"type": ["string", "null"]},
            },
        },
        "mock_llm": {"type": "boolean"},
    },
}

_VALIDATOR = Draft202012Validator(RESPONSE_SCHEMA)


class SchemaValidationError(ValueError):
    pass


def validate_response(resp: dict) -> dict:
    errors = sorted(_VALIDATOR.iter_errors(resp), key=lambda e: e.path)
    if errors:
        raise SchemaValidationError("; ".join(f"{list(e.path)}: {e.message}" for e in errors))
    return resp
