"""Per-engine in-memory rolling state buffer for high-throughput stream processing.

Guarantees low-latency (<50ms) inference pipeline:
Incoming Cycle -> Data Quality Guard -> Rolling Features -> RUL + Conformal -> Anomaly Detection -> Status
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np

from src.config import DATA_PROC_DIR, MODELS_DIR
from src.features import readings_to_feature_row
from src.uncertainty import default_calibrator
from src.anomaly import default_anomaly_detector
from src.stream.guard import DataQualityGuard

log = logging.getLogger(__name__)


@dataclass
class EngineLiveState:
    unit_id: int
    current_cycle: int
    predicted_rul: float
    rul_p10: float
    rul_p50: float
    rul_p90: float
    prob_fail_10: float
    prob_fail_30: float
    health_index: float
    anomaly_score: float
    onset_detected: bool
    onset_cycle: int | None
    status: str
    trust_score: float
    quality_warnings: list[str] = field(default_factory=list)


class FleetStateManager:
    """Maintains 30-cycle FIFO buffers per active unit and executes live pipeline."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._buffers: dict[int, collections.deque[dict[str, Any]]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.window_size)
        )
        self._latest_states: dict[int, EngineLiveState] = {}

        # Load models and column metadata
        self.model = joblib.load(MODELS_DIR / "xgb_rul.joblib")
        eda_meta = json.loads((DATA_PROC_DIR / "eda_meta.json").read_text())
        feat_meta = json.loads((DATA_PROC_DIR / "feature_meta.json").read_text())
        self.sensors = eda_meta["useful_sensors"]
        self.feat_cols = feat_meta["feature_cols"]

    def ingest_reading(self, reading: dict[str, Any]) -> EngineLiveState:
        """Process a single incoming telemetry reading.

        Executes full guard, feature extraction, regression, uncertainty, and anomaly checks.
        """
        unit_id = int(reading.get("unit_id", 0))
        buf = self._buffers[unit_id]

        # 1. Inspect data quality
        trust_score, warnings = DataQualityGuard.inspect(
            current_reading=reading,
            recent_history=list(buf),
            sensors=self.sensors,
        )

        # 2. Append to buffer
        buf.append(reading)
        cycle = int(reading.get("cycle", len(buf)))

        # 3. Extract causal features
        history_list = list(buf)
        feat_vec = readings_to_feature_row(
            cycle_readings=history_list,
            sensors=self.sensors,
            feature_cols=self.feat_cols,
            window=self.window_size,
        )

        # 4. Predict RUL
        raw_pred = float(self.model.predict(feat_vec.reshape(1, -1))[0])
        pred_rul = float(np.clip(raw_pred, 0.0, 125.0))

        # 5. Uncertainty with dynamic trust widening
        unc = default_calibrator.estimate(pred_rul)
        if trust_score < 0.80:
            penalty = (1.0 - trust_score) * 20.0
            p10 = max(0.0, unc.rul_p10 - penalty)
            p90 = min(125.0, unc.rul_p90 + penalty)
        else:
            p10, p90 = unc.rul_p10, unc.rul_p90

        # 6. Anomaly & Onset tracking
        anom_res = default_anomaly_detector.evaluate_sequence(history_list)

        # 7. Alert status mapping
        if pred_rul <= 30.0 or unc.prob_fail_30 >= 0.70:
            status = "CRITICAL"
        elif pred_rul <= 80.0:
            status = "WARNING"
        else:
            status = "HEALTHY"

        state = EngineLiveState(
            unit_id=unit_id,
            current_cycle=cycle,
            predicted_rul=round(pred_rul, 2),
            rul_p10=round(p10, 2),
            rul_p50=round(unc.rul_p50, 2),
            rul_p90=round(p90, 2),
            prob_fail_10=unc.prob_fail_10,
            prob_fail_30=unc.prob_fail_30,
            health_index=anom_res.health_index,
            anomaly_score=anom_res.anomaly_score,
            onset_detected=anom_res.onset_detected,
            onset_cycle=anom_res.onset_cycle,
            status=status,
            trust_score=trust_score,
            quality_warnings=warnings,
        )

        self._latest_states[unit_id] = state
        return state

    def get_fleet_snapshot(self) -> list[dict[str, Any]]:
        """Return the latest state for all ingested engines."""
        return [state.__dict__ for state in self._latest_states.values()]

    def get_engine_state(self, unit_id: int) -> dict[str, Any] | None:
        """Return the current state for a specific engine."""
        st = self._latest_states.get(unit_id)
        return st.__dict__ if st else None


fleet_state_manager = FleetStateManager()

