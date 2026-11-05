"""Official Model Context Protocol (MCP) Server for Engines Health Monitor.

Exposes domain predictive maintenance tools and telemetry resources via FastMCP:
- Tools: list_fleet, get_engine_status, explain_engine, recommend_maintenance, schedule_hangar, run_backtest_summary
- Resources: fleet://summary

Allows external AI clients (Antigravity, Claude Desktop, Cursor) to directly query
and orchestrate maintenance actions via standard MCP stdio protocol.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.agents.tools import (
    explain_engine as _explain_engine_tool,
    get_engine_details as _get_engine_details_tool,
    get_engines_by_status as _get_engines_by_status_tool,
    get_fleet_statistics as _get_fleet_statistics_tool,
    get_fleet_summary as _get_fleet_summary_tool,
)
from src.decision.backtest import run_fleet_backtest as _run_backtest_tool
from src.decision.policy import default_policy
from src.decision.scheduler import EngineTask, default_scheduler
from src.uncertainty import default_calibrator

mcp = FastMCP("Engines-Health-Monitor")


@mcp.resource("fleet://summary")
def get_fleet_summary_resource() -> str:
    """Read-only fleet telemetry and health KPI overview resource."""
    summary = _get_fleet_summary_tool()
    stats = _get_fleet_statistics_tool()
    return json.dumps({"summary": summary, "distribution_statistics": stats}, indent=2)


@mcp.tool()
def list_fleet(status_filter: str | None = None) -> list[dict[str, Any]]:
    """List engines across the fleet, optionally filtered by status (CRITICAL, WARNING, HEALTHY)."""
    if status_filter:
        res = _get_engines_by_status_tool(status_filter.upper())
        return res.get("engines", [])
    summary = _get_fleet_summary_tool()
    return summary.get("top_at_risk", [])


@mcp.tool()
def get_engine_status(engine_id: int) -> dict[str, Any]:
    """Retrieve full health status, RUL point prediction, and calibrated uncertainty for one engine."""
    details = _get_engine_details_tool(engine_id)
    if "error" in details:
        return details
    rul = details["predicted_rul_cycles"]
    unc = default_calibrator.estimate(rul)
    return {
        **details,
        "rul_p10": unc.rul_p10,
        "rul_p50": unc.rul_p50,
        "rul_p90": unc.rul_p90,
        "prob_fail_10": unc.prob_fail_10,
        "prob_fail_30": unc.prob_fail_30,
        "conformal_confidence": unc.confidence_score,
    }


@mcp.tool()
def explain_engine(engine_id: int, top_n: int = 5) -> dict[str, Any]:
    """Explain prediction root-causes using SHAP feature contributions for a specific engine."""
    return _explain_engine_tool(engine_id, top_n=top_n)


@mcp.tool()
def recommend_maintenance(engine_id: int) -> dict[str, Any]:
    """Generate cost-optimal replacement recommendation and replace-by deadline."""
    details = _get_engine_details_tool(engine_id)
    if "error" in details:
        return details
    rul = details["predicted_rul_cycles"]
    unc = default_calibrator.estimate(rul)
    cycle = details.get("last_observed_cycle", 1)
    dec = default_policy.evaluate_engine(
        predicted_rul=rul,
        rul_p10=unc.rul_p10,
        prob_fail_30=unc.prob_fail_30,
        current_cycle=cycle,
    )
    return dec.__dict__


@mcp.tool()
def schedule_hangar(engine_ids: list[int]) -> dict[str, Any]:
    """Schedule maintenance bays for given engines using OR-Tools CP-SAT."""
    tasks: list[EngineTask] = []
    for eid in engine_ids:
        details = _get_engine_details_tool(eid)
        if "error" in details:
            continue
        rul = details["predicted_rul_cycles"]
        unc = default_calibrator.estimate(rul)
        dec = default_policy.evaluate_engine(rul, unc.rul_p10, unc.prob_fail_30)
        tasks.append(
            EngineTask(
                engine_id=eid,
                duration_days=2,
                deadline_days=max(1, int(dec.deadline_days)),
                urgency_weight=10 if dec.urgency == "IMMEDIATE_GROUNDING" else 5,
                rul_p10=unc.rul_p10,
            )
        )

    schedule_res = default_scheduler.solve(tasks)
    return {
        "solver_status": schedule_res.solver_status,
        "total_tardiness_days": schedule_res.total_weighted_tardiness,
        "makespan_days": schedule_res.makespan_days,
        "schedule": [s.__dict__ for s in schedule_res.schedule],
    }


@mcp.tool()
def run_backtest_summary() -> dict[str, Any]:
    """Execute the 3-policy financial backtest (Reactive vs Fixed vs Predictive)."""
    return _run_backtest_tool()


if __name__ == "__main__":
    mcp.run()
