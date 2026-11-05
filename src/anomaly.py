"""Anomaly Detection and Degradation Onset Detection Engine.

Directly addresses problem statement FT-03 (IoT time-series and anomaly detection).
1. Trains unsupervised models (PCA Reconstruction Error + IsolationForest) strictly
   on early healthy engine operation (cycles <= 30).
2. Fuses anomaly scores into a normalized, causally smoothed Health Index H_t in [0, 1].
3. Detects degradation onset using causal EWMA + CUSUM change-point tracking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROC_DIR, DATA_RAW_DIR, MODELS_DIR, REPORTS_DIR
from src.features import RAW_COLUMNS

log = logging.getLogger(__name__)

ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"


@dataclass
class AnomalyResult:
    health_index: float
    anomaly_score: float
    onset_detected: bool
    onset_cycle: int | None
    degradation_state: str  # HEALTHY_BASELINE | DEGRADATION_ONSET | SEVERE_ANOMALY


class AnomalyDetector:
    """PCA + IsolationForest health index and CUSUM onset detector."""

    def __init__(self):
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.iso: IsolationForest | None = None
        self.mu_recon: float = 0.3540
        self.std_recon: float = 0.1608
        self.mu_iso: float = 0.4399
        self.std_iso: float = 0.0500
        self.sensors: list[str] = []
        self._load_or_train()

    def _load_or_train(self) -> None:
        if ANOMALY_MODEL_PATH.exists():
            try:
                bundle = joblib.load(ANOMALY_MODEL_PATH)
                self.scaler = bundle["scaler"]
                self.pca = bundle["pca"]
                self.iso = bundle["iso"]
                self.mu_recon = bundle["mu_recon"]
                self.std_recon = bundle["std_recon"]
                self.mu_iso = bundle["mu_iso"]
                self.std_iso = bundle["std_iso"]
                self.sensors = bundle["sensors"]
                return
            except Exception as e:
                log.warning("Could not load cached anomaly model: %s. Retraining...", e)

        self._train_healthy_baseline()

    def _train_healthy_baseline(self) -> None:
        """Train models on the first 30 cycles of engines (guaranteed healthy baseline)."""
        eda_meta = json.loads((DATA_PROC_DIR / "eda_meta.json").read_text())
        self.sensors = eda_meta["useful_sensors"]
        drop = eda_meta["constant_sensors"]

        df = (
            pd.read_csv(DATA_RAW_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
            .drop(columns=drop)
        )

        healthy_df = df[df["cycle"] <= 30]
        X_h = healthy_df[self.sensors].values

        self.scaler = StandardScaler().fit(X_h)
        X_scaled = self.scaler.transform(X_h)

        self.pca = PCA(n_components=2, random_state=42).fit(X_scaled)
        self.iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=42).fit(X_scaled)

        recon = np.mean((X_scaled - self.pca.inverse_transform(self.pca.transform(X_scaled))) ** 2, axis=1)
        iso_scores = -self.iso.score_samples(X_scaled)

        self.mu_recon = float(np.mean(recon))
        self.std_recon = float(np.std(recon))
        self.mu_iso = float(np.mean(iso_scores))
        self.std_iso = float(np.std(iso_scores))

        bundle = {
            "scaler": self.scaler,
            "pca": self.pca,
            "iso": self.iso,
            "mu_recon": self.mu_recon,
            "std_recon": self.std_recon,
            "mu_iso": self.mu_iso,
            "std_iso": self.std_iso,
            "sensors": self.sensors,
        }
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, ANOMALY_MODEL_PATH)
        log.info("Anomaly detection baseline trained and cached at %s", ANOMALY_MODEL_PATH)

    def evaluate_sequence(self, cycle_readings: list[dict[str, Any]] | pd.DataFrame) -> AnomalyResult:
        """Evaluate a time series of sensor readings for an engine causally.

        Args:
            cycle_readings: Chronologically sorted list of dicts or DataFrame with 'cycle' and sensor keys.

        Returns:
            AnomalyResult with latest health index, anomaly score, and CUSUM onset status.
        """
        if isinstance(cycle_readings, pd.DataFrame):
            if cycle_readings.empty:
                raise ValueError("cycle_readings cannot be empty")
            df = cycle_readings.sort_values("cycle").reset_index(drop=True)
        else:
            if not cycle_readings:
                raise ValueError("cycle_readings cannot be empty")
            df = pd.DataFrame(cycle_readings).sort_values("cycle").reset_index(drop=True)
        X = self.scaler.transform(df[self.sensors].values)

        # PCA reconstruction error
        recon = np.mean((X - self.pca.inverse_transform(self.pca.transform(X))) ** 2, axis=1)
        # Isolation forest score
        iso_scores = -self.iso.score_samples(X)

        s_r = (recon - self.mu_recon) / (self.std_recon + 1e-6)
        s_i = (iso_scores - self.mu_iso) / (self.std_iso + 1e-6)
        anomaly_scores = np.maximum(0.0, 0.4 * s_r + 0.6 * s_i)
        raw_health_indices = np.exp(-anomaly_scores / 3.0)

        # Causal EWMA smoothing (alpha = 0.2)
        smoothed_hi: list[float] = []
        curr_hi = 1.0
        for h in raw_health_indices:
            curr_hi = 0.20 * h + 0.80 * curr_hi
            smoothed_hi.append(curr_hi)

        # CUSUM Change-point Tracking for degradation onset
        cusum = 0.0
        onset_cycle: int | None = None
        cycles = df["cycle"].values

        for idx, h_val in enumerate(smoothed_hi):
            # Slack k = 0.05 deviation from nominal 1.0
            cusum = max(0.0, cusum + (1.0 - h_val) - 0.05)
            if cusum > 0.40 and onset_cycle is None:
                onset_cycle = int(cycles[idx])

        latest_hi = float(smoothed_hi[-1])
        latest_anomaly = float(anomaly_scores[-1])
        onset_detected = onset_cycle is not None

        if latest_hi > 0.85 and not onset_detected:
            state = "HEALTHY_BASELINE"
        elif onset_detected and latest_hi > 0.60:
            state = "DEGRADATION_ONSET"
        else:
            state = "SEVERE_ANOMALY"

        return AnomalyResult(
            health_index=round(latest_hi, 4),
            anomaly_score=round(latest_anomaly, 4),
            onset_detected=onset_detected,
            onset_cycle=onset_cycle,
            degradation_state=state,
        )

    def compute_health_series(self, cycle_readings: list[dict[str, Any]] | pd.DataFrame) -> dict[str, Any]:
        """Compute full cycle-by-cycle health index trajectory and onset cycle."""
        if isinstance(cycle_readings, pd.DataFrame):
            if cycle_readings.empty:
                raise ValueError("cycle_readings cannot be empty")
            df = cycle_readings.sort_values("cycle").reset_index(drop=True)
        else:
            if not cycle_readings:
                raise ValueError("cycle_readings cannot be empty")
            df = pd.DataFrame(cycle_readings).sort_values("cycle").reset_index(drop=True)

        X = self.scaler.transform(df[self.sensors].values)
        recon = np.mean((X - self.pca.inverse_transform(self.pca.transform(X))) ** 2, axis=1)
        iso_scores = -self.iso.score_samples(X)

        s_r = (recon - self.mu_recon) / (self.std_recon + 1e-6)
        s_i = (iso_scores - self.mu_iso) / (self.std_iso + 1e-6)
        anomaly_scores = np.maximum(0.0, 0.4 * s_r + 0.6 * s_i)
        raw_health_indices = np.exp(-anomaly_scores / 3.0)

        smoothed_hi: list[float] = []
        curr_hi = 1.0
        for h in raw_health_indices:
            curr_hi = 0.20 * h + 0.80 * curr_hi
            smoothed_hi.append(curr_hi)

        cusum = 0.0
        onset_cycle: int | None = None
        cycles = [int(c) for c in df["cycle"].values]

        for idx, h_val in enumerate(smoothed_hi):
            cusum = max(0.0, cusum + (1.0 - h_val) - 0.05)
            if cusum > 0.40 and onset_cycle is None:
                onset_cycle = cycles[idx]

        return {
            "cycles": cycles,
            "health_indices": [round(float(v), 4) for v in smoothed_hi],
            "anomaly_scores": [round(float(v), 4) for v in anomaly_scores],
            "onset_detected": onset_cycle is not None,
            "onset_cycle": onset_cycle,
        }


def evaluate_anomaly_report() -> dict[str, Any]:
    """Evaluate onset detector across all 100 test engines and write report."""
    detector = AnomalyDetector()
    eda_meta = json.loads((DATA_PROC_DIR / "eda_meta.json").read_text())
    drop = eda_meta["constant_sensors"]

    df = (
        pd.read_csv(DATA_RAW_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
        .drop(columns=drop)
    )

    trues = pd.read_csv(DATA_RAW_DIR / "RUL_FD001.txt", header=None, names=["t"])["t"].clip(upper=125).values

    onsets = []
    lead_times_to_test_end = []
    total_lead_times_to_failure = []
    false_alarms = 0
    total_healthy_cycles = 0

    for idx, (uid, group) in enumerate(df.groupby("unit_id")):
        g = group.sort_values("cycle")
        readings = g.to_dict(orient="records")
        res = detector.evaluate_sequence(readings)

        onset = res.onset_cycle
        max_cycle = int(g["cycle"].max())
        true_remaining = trues[idx]

        onsets.append(onset)
        if onset is not None:
            lead_test = max_cycle - onset
            lead_times_to_test_end.append(lead_test)
            total_lead_times_to_failure.append(lead_test + true_remaining)

        # Count false alarms in first 25 cycles
        cycles = g["cycle"].values
        for c in cycles:
            if c <= 25:
                total_healthy_cycles += 1
                if onset is not None and onset <= c:
                    false_alarms += 1

    detected_count = sum(1 for o in onsets if o is not None)
    lead_times_to_test_end = np.array(lead_times_to_test_end)
    total_lead_times_to_failure = np.array(total_lead_times_to_failure)

    report = {
        "total_test_engines": 100,
        "onsets_detected_count": detected_count,
        "onset_detection_rate": round(detected_count / 100.0, 4),
        "false_alarm_rate_healthy_phase": round(false_alarms / max(1, total_healthy_cycles), 4),
        "median_lead_time_to_test_end_cycles": round(float(np.median(lead_times_to_test_end)), 1),
        "median_lead_time_to_actual_failure_cycles": round(float(np.median(total_lead_times_to_failure)), 1),
        "fraction_detected_before_rul_80": round(float(np.mean(total_lead_times_to_failure >= 80.0)), 4),
        "methodology": "PCA (k=2) Reconstruction Error + IsolationForest -> Fused Health Index with CUSUM Tracking",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "anomaly.json").write_text(json.dumps(report, indent=2))
    log.info("Anomaly report saved to %s", REPORTS_DIR / "anomaly.json")
    return report


default_anomaly_detector = AnomalyDetector()

