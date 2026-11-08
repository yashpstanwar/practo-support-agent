"""Task 15 - SQLite checkpointing (langgraph-checkpoint-sqlite) keyed by thread ID.

Phase 1 (process A): run the graph on thread "ckpt-demo-1" with
  interrupt_before=["output_guard"] -> load_memory, input_guard,
  classify_intent and rag_tool execute (4 of 7 nodes on this path), then the
  run deliberately STOPS before output_guard / format_response / save_memory.
Phase 2 (process B, a brand-new Python process): reopen checkpoints.sqlite,
  print the state loaded from the checkpoint, and resume the SAME thread ID.
  Only the remaining 3 nodes execute. `node_log` uses an append reducer, so if
  any finished node had re-run it would appear twice - it does not.

Run:  python -m demos.demo_checkpoint
"""
from __future__ import annotations

import asyncio
import subprocess
import sys

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import config
from agent import memory as mem
from agent.graph import build_graph

THREAD = "ckpt-demo-1"
CFG = {"configurable": {"thread_id": THREAD}}
QUESTION = "How long will my thyroid test report take?"


async def phase1() -> None:
    if config.CHECKPOINT_DB.exists():
        config.CHECKPOINT_DB.unlink()
    mem.reset(THREAD)
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = build_graph(checkpointer=saver, interrupt_before=["output_guard"])
        print(f"PHASE 1 - run thread_id={THREAD!r}, deliberately stopping before 'output_guard'")
        await graph.ainvoke({"thread_id": THREAD, "trace_id": "trace-ckpt-0001",
                             "question": QUESTION, "node_log": []}, CFG)
        snap = await graph.aget_state(CFG)
        print(f"\n  STOPPED. next node(s) pending: {snap.next}")
        print(f"  nodes completed so far (node_log): {snap.values['node_log']}")
        print(f"  checkpoint_id: {snap.config['configurable']['checkpoint_id']}")
        print(f"  checkpoints stored in {config.CHECKPOINT_DB.name}: "
              f"{len([c async for c in saver.alist(CFG)])}")


async def phase2() -> None:
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = build_graph(checkpointer=saver, interrupt_before=["output_guard"])
        snap = await graph.aget_state(CFG)
        print(f"PHASE 2 - NEW PROCESS, reopened {config.CHECKPOINT_DB.name}, thread_id={THREAD!r}")
        print("  State LOADED FROM CHECKPOINT (nothing recomputed):")
        print(f"    next              = {snap.next}")
        print(f"    node_log          = {snap.values['node_log']}")
        print(f"    route             = {snap.values['route']}")
        print(f"    top1 similarity   = {snap.values['top1']}")
        print(f"    retrieved chunks  = {[h['chunk_id'] for h in snap.values['hits']]}")
        print(f"    draft_answer      = {snap.values['draft_answer']}")
        before = list(snap.values["node_log"])
        print("\n  Resuming with graph.ainvoke(None, same thread_id) - watch which nodes execute:")
        final = await graph.ainvoke(None, CFG)
        after = final["node_log"]
        newly_run = after[len(before):]
        print(f"\n  final node_log    = {after}")
        print(f"  executed in phase 2 only: {newly_run}")
        print(f"  final answer      = {final['response']['answer']}")
        assert newly_run == ["output_guard", "format_response", "save_memory"], newly_run
        assert all(after.count(n) == 1 for n in after), "a completed node was re-executed!"
        print("\nOK: the 4 nodes finished in phase 1 were loaded from the SQLite checkpoint and NOT re-executed;"
              " only the 3 remaining nodes ran.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"phase1", "phase2"}:
        asyncio.run(phase1() if sys.argv[1] == "phase1" else phase2())
        return
    for phase in ("phase1", "phase2"):
        out = subprocess.run([sys.executable, "-m", "demos.demo_checkpoint", phase], cwd=config.ROOT,
                             capture_output=True, text=True)
        print(out.stdout)
        if out.returncode != 0:
            print(out.stderr)
            raise SystemExit(out.returncode)
        print("-" * 78)


if __name__ == "__main__":
    main()
