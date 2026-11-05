"""Data Quality Guard for Real-time IoT Ingestion.

Evaluates sensor telemetry integrity before downstream inference:
- Flags stuck / frozen sensor signals across rolling cycles.
- Detects physical out-of-range extremes (e.g., negative temperatures, vacuum pressures).
- Computes an auditable `trust_score` in [0.0, 1.0].
- Low trust scores trigger dynamic uncertainty widening to protect decisions.
"""

from __future__ import annotations

from typing import Any
import numpy as np

# Physical operational bounds for turbofan sensors (Saxena et al., 2008)
SENSOR_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "sensor_02": (500.0, 800.0),    # T24 LPC outlet temp (°R)
    "sensor_03": (1200.0, 1800.0),  # T30 HPC outlet temp (°R)
    "sensor_04": (1100.0, 1700.0),  # T50 LPT outlet temp (°R)
    "sensor_06": (10.0, 35.0),      # P15 bypass duct pressure (psia)
    "sensor_07": (400.0, 700.0),    # P30 HPC outlet pressure (psia)
    "sensor_08": (2000.0, 2600.0),  # Nf physical fan speed (rpm)
    "sensor_09": (8500.0, 9600.0),  # Nc physical core speed (rpm)
    "sensor_11": (35.0, 60.0),      # Ps30 static pressure (psia)
    "sensor_12": (400.0, 650.0),    # phi fuel flow ratio
    "sensor_13": (2000.0, 2600.0),  # NRf corrected fan speed (rpm)
    "sensor_14": (7500.0, 8800.0),  # NRc corrected core speed (rpm)
    "sensor_15": (7.0, 10.0),       # BPR bypass ratio
    "sensor_17": (350.0, 450.0),    # htBleed enthalpy
    "sensor_20": (30.0, 45.0),      # W31 HPT coolant bleed (pps)
    "sensor_21": (18.0, 30.0),      # W32 LPT coolant bleed (pps)
}


class DataQualityGuard:
    """Evaluates telemetry stream validity and flags anomalies."""

    @staticmethod
    def inspect(
        current_reading: dict[str, Any],
        recent_history: list[dict[str, Any]],
        sensors: list[str],
    ) -> tuple[float, list[str]]:
        """Assess quality of the reading against history and physical limits.

        Returns:
            trust_score in [0.0, 1.0], list of quality warnings.
        """
        issues: list[str] = []
        score = 1.0

        for s in sensors:
            val = current_reading.get(s)

            # 1. Missing or NaN check
            if val is None or (isinstance(val, float) and np.isnan(val)):
                issues.append(f"Sensor {s} missing or NaN")
                score -= 0.15
                continue

            # 2. Physical range check
            bounds = SENSOR_PHYSICAL_BOUNDS.get(s)
            if bounds:
                lo, hi = bounds
                if val < lo or val > hi:
                    issues.append(f"Sensor {s}={val:.2f} out of physical bounds [{lo}, {hi}]")
                    score -= 0.10

            # 3. Stuck sensor check (last 5 cycles zero variation)
            if len(recent_history) >= 5:
                past_vals = [r.get(s) for r in recent_history[-5:] if r.get(s) is not None]
                if len(past_vals) == 5 and all(abs(pv - val) < 1e-6 for pv in past_vals):
                    issues.append(f"Sensor {s} appears stuck across last 5 cycles")
                    score -= 0.08

        trust_score = float(np.clip(score, 0.05, 1.0))
        return round(trust_score, 3), issues
