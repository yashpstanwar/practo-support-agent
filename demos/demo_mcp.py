"""Task 14 - launch the MCP server as its own process, then run the separate
client script against http://127.0.0.1:8000/mcp.

Run:  python -m demos.demo_mcp
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time

import config


def wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.3)
    raise RuntimeError("MCP server did not start")


def main() -> None:
    server = subprocess.Popen([sys.executable, "mcp_server.py"], cwd=config.ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_port(config.MCP_HOST, config.MCP_PORT)
        print(f"MCP server process started (pid={server.pid}) at {config.MCP_URL}\n")
        out = subprocess.run([sys.executable, "mcp_client.py", "APT-1002", "APT-1007", "APT-9999"],
                             cwd=config.ROOT, capture_output=True, text=True, timeout=60)
        print(out.stdout)
        if out.returncode != 0:
            print(out.stderr)
            raise SystemExit(out.returncode)
        print("OK: separate client process called the MCP tool for 3 record IDs (2 valid, 1 unknown).")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
