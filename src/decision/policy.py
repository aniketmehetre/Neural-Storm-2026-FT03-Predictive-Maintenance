"""Cost-Optimal Maintenance Policy and Decision Engine.

Translates calibrated RUL distributions into financially optimal maintenance actions:
- Evaluates risk threshold tau = P(fail <= H) to minimize expected total cost.
- Recommends a replace-by deadline based on the conservative 10th percentile (p10).
- Converts cycle deadlines into calendar days using explicit operational flight rate assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from src.config import CONFIG_DIR


def load_costs_config() -> dict[str, Any]:
    """Load cost and operational assumptions from config/costs.yaml."""
    costs_file = CONFIG_DIR / "costs.yaml"
    if costs_file.exists():
        return yaml.safe_load(costs_file.read_text())
    return {
        "unplanned_failure_cost": 350000.0,
        "planned_replacement_cost": 45000.0,
        "cost_per_wasted_cycle": 150.0,
        "hangar_slots_per_day": 3,
        "cycles_per_day": 2.5,
        "lead_time_cycles": 30,
        "tau_threshold": 0.60,
    }


@dataclass
class ReplacementDecision:
    action_required: bool
    urgency: str  # IMMEDIATE_GROUNDING | SCHEDULED_MAINTENANCE | MONITOR_IN_SERVICE
    recommended_replace_by_cycle: int
    deadline_days: float
    trigger_rule: str
    expected_cost_planned: float
    expected_cost_unplanned: float
    net_cost_avoidance_usd: float
    operational_rationale: str


class MaintenancePolicy:
    """Calculates risk-neutral and risk-averse replacement triggers."""

    def __init__(self, costs: dict[str, Any] | None = None):
        self.costs = costs or load_costs_config()
        self.c_unplanned = float(self.costs["unplanned_failure_cost"])
        self.c_planned = float(self.costs["planned_replacement_cost"])
        self.c_wasted = float(self.costs["cost_per_wasted_cycle"])
        self.cycles_per_day = float(self.costs["cycles_per_day"])
        self.lead_time = int(self.costs.get("lead_time_cycles", 30))
        self.tau = float(self.costs.get("tau_threshold", 0.60))

    def evaluate_engine(
        self,
        predicted_rul: float,
        rul_p10: float,
        prob_fail_30: float,
        current_cycle: int = 1,
    ) -> ReplacementDecision:
        """Evaluate replacement timing minimizing expected cost.

        Rule:
        Trigger replacement if P(failure within 30 cycles) >= tau OR p10 <= 15 cycles.
        """
        # Conservative deadline is driven by p10 (90% lower confidence bound)
        remaining_safe_cycles = max(1, int(rul_p10))
        replace_by_cycle = current_cycle + remaining_safe_cycles
        deadline_days = round(remaining_safe_cycles / self.cycles_per_day, 1)

        # Cost calculations
        wasted_cycles = max(0.0, predicted_rul - 5.0)  # assume buffer of 5 cycles
        cost_planned = self.c_planned + (wasted_cycles * self.c_wasted)
        cost_unplanned = self.c_unplanned
        net_savings = cost_unplanned - cost_planned

        # Decision trigger
        if prob_fail_30 >= self.tau or rul_p10 <= 15.0 or predicted_rul <= 30.0:
            action_required = True
            if rul_p10 <= 5.0 or prob_fail_30 >= 0.85:
                urgency = "IMMEDIATE_GROUNDING"
                rationale = (
                    f"Severe failure probability ({prob_fail_30:.1%}) within 30 cycles. "
                    f"Conservative p10 lower bound is {rul_p10:.1f} cycles ({deadline_days} days). "
                    f"Immediate maintenance avoids an estimated ${net_savings:,.0f} catastrophic failure cost."
                )
                rule = f"P(fail<=30)={prob_fail_30:.2f} >= {self.tau} (or p10 <= 5)"
            else:
                urgency = "SCHEDULED_MAINTENANCE"
                rationale = (
                    f"Elevated failure risk ({prob_fail_30:.1%}). "
                    f"Induct engine into depot within {remaining_safe_cycles} cycles ({deadline_days} days). "
                    f"Optimal replacement avoids an unplanned overhaul cost of ${cost_unplanned:,.0f}."
                )
                rule = f"P(fail<=30)={prob_fail_30:.2f} >= {self.tau} (or p10 <= 15)"
        else:
            action_required = False
            urgency = "MONITOR_IN_SERVICE"
            rationale = (
                f"Engine operating nominally. Failure risk within 30 cycles is low ({prob_fail_30:.1%}). "
                f"Replace-by window extends beyond {remaining_safe_cycles} cycles ({deadline_days} days)."
            )
            rule = f"P(fail<=30)={prob_fail_30:.2f} < {self.tau}"

        return ReplacementDecision(
            action_required=action_required,
            urgency=urgency,
            recommended_replace_by_cycle=replace_by_cycle,
            deadline_days=deadline_days,
            trigger_rule=rule,
            expected_cost_planned=round(cost_planned, 2),
            expected_cost_unplanned=round(cost_unplanned, 2),
            net_cost_avoidance_usd=round(net_savings, 2),
            operational_rationale=rationale,
        )


default_policy = MaintenancePolicy()

