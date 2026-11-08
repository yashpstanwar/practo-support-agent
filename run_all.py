"""Run every demo/evaluation in order and save its output to transcripts/.

Run:  python run_all.py          (MOCK_LLM=1 by default - no API keys needed)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import config

STEPS = [
    ("01_task1_dataset.txt", ["dataset.py"]),
    ("02_task3_chunk_and_index.txt", ["-m", "rag.build_index"]),
    ("03_task4_threshold_calibration.txt", ["-m", "evaluation.calibrate_threshold"]),
    ("04_task4_grounded_generation.txt", ["-m", "demos.demo_grounded_generation"]),
    ("05_task5_chunking_precision_recall.txt", ["-m", "evaluation.chunking_eval"]),
    ("06_task6_escalation_tool.txt", ["tools.py"]),
    ("07_task7_8_routing_and_memory.txt", ["-m", "demos.demo_routing_memory"]),
    ("08_task9_10_schema_and_guardrails.txt", ["-m", "demos.demo_guardrails"]),
    ("09_task11_12_fastapi_and_logging.txt", ["-m", "demos.demo_api"]),
    ("10_task13_rag_triad_eval.txt", ["-m", "evaluation.rag_triad"]),
    ("11_task14_mcp_client_server.txt", ["-m", "demos.demo_mcp"]),
    ("12_task15_sqlite_checkpointing.txt", ["-m", "demos.demo_checkpoint"]),
    ("13_task16_retries_and_timeouts.txt", ["-m", "demos.demo_resilience"]),
]


def main() -> None:
    env = {**os.environ, "MOCK_LLM": "1", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
    out_dir = config.ROOT / "transcripts"
    out_dir.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    failed = []
    for name, args in STEPS:
        if only and not any(o in name for o in only):
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, *args], cwd=config.ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8")
        header = f"$ MOCK_LLM=1 python {' '.join(args)}\n\n"
        (out_dir / name).write_text(header + p.stdout + (("\n[stderr]\n" + p.stderr) if p.returncode else ""),
                                    encoding="utf-8")
        status = "OK  " if p.returncode == 0 else "FAIL"
        print(f"[{status}] {name:<45} {time.time() - t0:6.1f}s")
        if p.returncode:
            failed.append(name)
            print(p.stderr[-2000:])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
