"""Task 14 - expose check_appointment_status as an MCP tool with fastmcp.

Run:  python mcp_server.py           # serves streamable HTTP at http://127.0.0.1:8000/mcp
"""
from __future__ import annotations

from fastmcp import FastMCP

import config
from tools import check_appointment_status as _lookup

mcp = FastMCP("practo-appointments")


@mcp.tool
def check_appointment_status(record_id: str) -> dict:
    """Look up a Practo appointment by its record ID and return its status.

    Args:
        record_id: The appointment ID in the format "APT-XXXX" (for example "APT-1007").

    Returns:
        A dict with `found`, `record_id`, `category`, `status`
        (Scheduled / Completed / Cancelled / No-Show / Rescheduled),
        `consultation_fee_inr`, `follow_up_required`, `days_since_created`,
        `escalation_score` in [0, 1] computed as
        0.6 * follow_up_required + 0.4 * (days_since_created / 30),
        `escalation_threshold` (0.80) and `escalate` (score >= threshold).
        Unknown IDs return {"found": false, "error": ...}.
    """
    return _lookup(record_id)


if __name__ == "__main__":
    mcp.run(transport="http", host=config.MCP_HOST, port=config.MCP_PORT)
