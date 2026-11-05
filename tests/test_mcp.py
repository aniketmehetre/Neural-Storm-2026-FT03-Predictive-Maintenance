"""Tests for Model Context Protocol (MCP) server integration."""

import json
import pytest

from src.mcp_server import (
    explain_engine,
    get_engine_status,
    get_fleet_summary_resource,
    list_fleet,
    mcp,
    recommend_maintenance,
    run_backtest_summary,
    schedule_hangar,
)


def test_mcp_tools_registration():
    """Verify all 6 core maintenance tools are registered on FastMCP server."""
    # FastMCP stores tools in _tool_manager._tools or similar
    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    expected = [
        "list_fleet",
        "get_engine_status",
        "explain_engine",
        "recommend_maintenance",
        "schedule_hangar",
        "run_backtest_summary",
    ]
    for exp in expected:
        assert exp in tool_names, f"Tool {exp} missing from MCP server"


def test_mcp_resource_summary():
    """Verify fleet://summary resource returns valid JSON."""
    raw = get_fleet_summary_resource()
    data = json.loads(raw)
    assert "summary" in data
    assert "distribution_statistics" in data
    assert data["summary"]["total_engines"] == 100


def test_mcp_tool_execution():
    """Verify tool functions return valid structured outputs."""
    # list_fleet
    criticals = list_fleet(status_filter="CRITICAL")
    assert isinstance(criticals, list)

    # get_engine_status
    st = get_engine_status(engine_id=1)
    assert st["unit_id"] == 1
    assert "rul_p10" in st
    assert "prob_fail_30" in st

    # recommend_maintenance
    rec = recommend_maintenance(engine_id=1)
    assert "action_required" in rec
    assert "recommended_replace_by_cycle" in rec
