"""Tests for Cost-Optimal Maintenance Policy, Backtest, and Hangar Scheduler."""

import json
from pathlib import Path
import pytest

from src.decision.policy import default_policy, MaintenancePolicy
from src.decision.scheduler import EngineTask, HangarScheduler
from src.decision.backtest import run_fleet_backtest


def test_maintenance_policy_triggers():
    """Verify decision engine triggers on high probability or low p10."""
    policy = MaintenancePolicy()

    # Critical case
    crit_dec = policy.evaluate_engine(
        predicted_rul=18.0,
        rul_p10=6.0,
        prob_fail_30=0.88,
        current_cycle=100,
    )
    assert crit_dec.action_required
    assert crit_dec.urgency in ("IMMEDIATE_GROUNDING", "SCHEDULED_MAINTENANCE")
    assert crit_dec.recommended_replace_by_cycle == 106
    assert crit_dec.deadline_days == pytest.approx(6.0 / 2.5, abs=0.1)

    # Healthy case
    healthy_dec = policy.evaluate_engine(
        predicted_rul=110.0,
        rul_p10=90.0,
        prob_fail_30=0.01,
        current_cycle=50,
    )
    assert not healthy_dec.action_required
    assert healthy_dec.urgency == "MONITOR_IN_SERVICE"


def test_hangar_scheduler_capacity_constraint():
    """Verify OR-Tools scheduler never violates concurrent slot capacity."""
    num_slots = 2
    scheduler = HangarScheduler(num_slots=num_slots)

    # 5 engines competing for 2 slots
    tasks = [
        EngineTask(engine_id=1, duration_days=3, deadline_days=4, urgency_weight=10, rul_p10=10.0),
        EngineTask(engine_id=2, duration_days=2, deadline_days=3, urgency_weight=10, rul_p10=7.5),
        EngineTask(engine_id=3, duration_days=4, deadline_days=8, urgency_weight=5, rul_p10=20.0),
        EngineTask(engine_id=4, duration_days=2, deadline_days=6, urgency_weight=5, rul_p10=15.0),
        EngineTask(engine_id=5, duration_days=3, deadline_days=10, urgency_weight=2, rul_p10=25.0),
    ]

    result = scheduler.solve(tasks)
    assert result.solver_status in ("OPTIMAL", "FEASIBLE")
    assert len(result.schedule) == 5

    # Check that at any day, active jobs <= num_slots
    max_day = result.makespan_days
    for day in range(max_day):
        active = sum(1 for s in result.schedule if s.start_day <= day < s.end_day)
        assert active <= num_slots, f"Day {day} active jobs {active} exceeded capacity {num_slots}"


def test_backtest_economic_superiority():
    """Assert 3-policy backtest proves predictive maintenance saves money and avoids failures."""
    root = Path(__file__).resolve().parents[1]
    report_file = root / "reports" / "business_case.json"
    if not report_file.exists():
        run_fleet_backtest()

    data = json.loads(report_file.read_text())
    p = data["policies"]
    fi = data["financial_impact"]

    assert p["predictive"]["catastrophic_failures"] == 0
    assert p["reactive"]["catastrophic_failures"] == 100
    assert fi["savings_vs_reactive_usd"] > 20_000_000.0  # saves > $20M
    assert fi["savings_vs_reactive_pct"] > 70.0

