"""Task 16 - retries with exponential backoff, per-node timeout, global timeout.

Configuration (config.py):
  RetryPolicy on `status_tool`: max_attempts=4, initial_interval=0.5s,
    backoff_factor=2.0, max_interval=4.0s, jitter=True,
    retry_on=TransientBackendError  -> waits ~0.5s, ~1s, ~2s (+ jitter)
  Per-node timeout on `rag_tool`: 5.0 s  (LangGraph `timeout=` on add_node)
  Global timeout on the whole graph run: 8.0 s (asyncio.wait_for in arun_agent)

Run:  python -m demos.demo_resilience
"""
from __future__ import annotations

import asyncio
import time

import config
from agent import memory as mem
from agent.graph import GlobalTimeoutError, arun_agent
from tools import TransientBackendError


async def main() -> None:
    print(f"Retry: max_attempts={config.RETRY_MAX_ATTEMPTS} initial_interval={config.RETRY_INITIAL_INTERVAL}s "
          f"backoff_factor={config.RETRY_BACKOFF_FACTOR} max_interval={config.RETRY_MAX_INTERVAL}s "
          f"jitter={config.RETRY_JITTER}")
    print(f"Per-node timeout (rag_tool) = {config.NODE_TIMEOUT_SECONDS}s; "
          f"global graph timeout = {config.GLOBAL_TIMEOUT_SECONDS}s\n")

    print("=" * 78)
    print("(a) RETRY - status backend fails the first 2 calls, then succeeds")
    print("=" * 78)
    mem.reset("resilience")
    t0 = time.perf_counter()
    r = await arun_agent("Check the status of APT-1047", "resilience",
                         simulate={"simulate_transient_failures": 2})
    print(f"  -> succeeded after {time.perf_counter() - t0:.2f}s: {r['answer']}")
    print("  OK: recovered on attempt 3 of 4 (within the configured max_attempts)\n")

    print("  (a2) control: 5 consecutive failures > max_attempts=4 -> clean, bounded failure")
    t0 = time.perf_counter()
    try:
        await arun_agent("Check the status of APT-1047", "resilience",
                         simulate={"simulate_transient_failures": 5}, global_timeout=30)
    except TransientBackendError as e:
        print(f"  -> gave up after {time.perf_counter() - t0:.2f}s with TransientBackendError: {e}\n")

    print("=" * 78)
    print(f"(b) PER-NODE TIMEOUT - rag_tool simulates a 30s vector-store call (limit {config.NODE_TIMEOUT_SECONDS}s)")
    print("=" * 78)
    t0 = time.perf_counter()
    try:
        await arun_agent("What is the fee for a dermatologist?", "resilience",
                         simulate={"simulate_delay": {"rag_tool": 30}}, global_timeout=60)
        print("  -> ERROR: no timeout fired")
    except Exception as e:  # noqa: BLE001 - we want to show the exact exception type
        el = time.perf_counter() - t0
        print(f"  -> {type(e).__module__}.{type(e).__name__} after {el:.2f}s: {e}")
        assert el < config.NODE_TIMEOUT_SECONDS + 3, "timeout did not fire promptly"
        print(f"  OK: clean error at ~{config.NODE_TIMEOUT_SECONDS}s instead of hanging for 30s\n")

    print("=" * 78)
    print(f"(c) GLOBAL TIMEOUT - three nodes each take 3s (each under any per-node limit), "
          f"total 9s > {config.GLOBAL_TIMEOUT_SECONDS}s")
    print("=" * 78)
    t0 = time.perf_counter()
    try:
        await arun_agent("What is the fee for a dermatologist?", "resilience",
                         simulate={"simulate_delay": {"load_memory": 3, "input_guard": 3, "classify_intent": 3}})
        print("  -> ERROR: no global timeout fired")
    except GlobalTimeoutError as e:
        el = time.perf_counter() - t0
        print(f"  -> GlobalTimeoutError after {el:.2f}s: {e}")
        assert el < config.GLOBAL_TIMEOUT_SECONDS + 1.5
        turns = mem.load("resilience")["turns"]
        assert len(turns) == 1
        print(f"  OK: the whole run was cancelled at the global limit; save_memory never ran "
              f"(memory still holds {len(turns)} turn, from step (a)).")


if __name__ == "__main__":
    asyncio.run(main())
