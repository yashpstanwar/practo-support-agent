"""Task 8 - conversation memory persisted to one JSON file per thread.

memory_store/<thread_id>.json = {
  "thread_id": ..., "turns": [{"user": masked text, "assistant": ..., "route": ...}],
  "last_record_id": "APT-1022" | null
}
Only masked user text is persisted, so phone numbers never reach disk.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import config

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _path(thread_id: str) -> Path:
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return config.MEMORY_DIR / f"{_SAFE.sub('_', thread_id)}.json"


def load(thread_id: str) -> dict:
    p = _path(thread_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"thread_id": thread_id, "turns": [], "last_record_id": None}


def save(thread_id: str, data: dict) -> Path:
    p = _path(thread_id)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def reset(thread_id: str) -> None:
    p = _path(thread_id)
    if p.exists():
        p.unlink()
