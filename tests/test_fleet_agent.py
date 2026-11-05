"""Tests for Agent v2: LangGraph pipeline, Critic, Approval, and Offline execution."""

import pytest
from src.agents.fleet_agent import (
    FleetMonitoringAgent,
    compute_shap_subsystem_ranking,
)
from src.llm.provider import OfflineProvider, get_llm_provider


def test_offline_llm_provider_generation():
    """Verify OfflineProvider generates structured narratives without any keys."""
    provider = OfflineProvider()
    resp = provider.generate("sensor analysis agent overall fleet health")
    assert "FLEET TRIAGE SUMMARY" in resp
    assert len(resp) > 50


def test_shap_subsystem_ranking():
    """Verify SHAP attributions are correctly mapped to physical turbofan subsystems."""
    fake_shap_features = [
        {"feature": "HPC outlet static pressure (Ps30)_max", "shap_value": -6.2},
        {"feature": "HPC outlet total temp (T30)_mean", "shap_value": -4.1},
        {"feature": "LPT outlet temp (T50)_trend", "shap_value": -2.3},
    ]
    ranking = compute_shap_subsystem_ranking(fake_shap_features)
    assert len(ranking) >= 2
    # Verify rankings are sorted by weight descending
    assert ranking[0]["shap_weight"] >= ranking[1]["shap_weight"]
    assert sum(r["sensitivity_pct"] for r in ranking) == pytest.approx(100.0, abs=1.0)


def test_fleet_monitoring_agent_end_to_end_offline():
    """Verify full 7-node LangGraph pipeline executes offline and validates critic."""
    agent = FleetMonitoringAgent()
    result = agent.run(human_approved=True)

    assert "triage_data" in result
    assert "diagnostics" in result
    assert "plan" in result
    assert result["critic_passed"] is True
    assert len(result["critic_notes"]) > 0

    # Human-in-the-loop check
    assert result["human_approved"] is True
    assert len(result["simulated_orders"]) > 0
    for order in result["simulated_orders"]:
        assert order["order_id"].startswith("SIMULATED-PO-")
        assert order["status"] == "DISPATCHED_SIMULATED"

    # Report verification
    report = result["final_report"]
    assert "Executive Fleet Maintenance Briefing" in report
    assert "**Numeric Integrity Check:** PASSED" in report


def test_fleet_monitoring_agent_approval_gate():
    """Verify that rejecting approval halts purchase order dispatch."""
    agent = FleetMonitoringAgent()
    result = agent.run(human_approved=False)

    assert result["human_approved"] is False
    assert len(result["simulated_orders"]) == 0, "No orders should be dispatched without human approval"
