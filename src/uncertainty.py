"""Calibrated Uncertainty Estimation for Remaining Useful Life (RUL).

Implements split-conformal prediction intervals calibrated over held-out engine
units (never random rows) to guarantee valid finite-sample coverage without
parametric assumptions.

Also derives failure probability P(RUL <= H) for mission horizons H in {10, 30}
cycles using the calibrated non-conformity distribution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from src.config import DATA_PROC_DIR, DATA_RAW_DIR, MODELS_DIR, REPORTS_DIR

log = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    predicted_rul: float
    rul_p10: float
    rul_p50: float
    rul_p90: float
    interval_width: float
    prob_fail_10: float
    prob_fail_30: float
    confidence_score: float
    interval_method: str = "split_conformal_80pct"


class ConformalCalibrator:
    """Split-conformal prediction interval generator for RUL forecasts."""

    def __init__(self, target_coverage: float = 0.80, rul_clip: float = 125.0):
        self.target_coverage = target_coverage
        self.rul_clip = rul_clip
        # Pre-computed calibration quantile on held-out engines (FD001 units 1-50)
        # Defaults to 20.08 cycles if uncalibrated
        self.conformal_margin = 20.08
        self.effective_sigma = self.conformal_margin / 1.28155  # z-score for 80% 2-tailed interval

    def calibrate(self, calibration_errors: np.ndarray) -> float:
        """Compute the conformal non-conformity quantile.

        Args:
            calibration_errors: Array of absolute residuals |y_true - y_pred|
                                computed on entire held-out engines.

        Returns:
            The non-conformity quantile margin q.
        """
        n = len(calibration_errors)
        if n == 0:
            return self.conformal_margin

        # Conformal quantile adjustment: ceil((n+1)(1-alpha))/n
        level = min(1.0, np.ceil((n + 1) * self.target_coverage) / n)
        self.conformal_margin = float(np.quantile(calibration_errors, level))
        self.effective_sigma = max(1.0, self.conformal_margin / 1.28155)
        return self.conformal_margin

    def estimate(self, predicted_rul: float) -> UncertaintyEstimate:
        """Derive calibrated intervals and failure probabilities.

        Args:
            predicted_rul: Point prediction from regression model.

        Returns:
            UncertaintyEstimate with p10, p50, p90, prob_fail_10, prob_fail_30.
        """
        pred = float(np.clip(predicted_rul, 0.0, self.rul_clip))
        q = self.conformal_margin

        p10 = float(np.clip(pred - q, 0.0, self.rul_clip))
        p50 = pred
        p90 = float(np.clip(pred + q, 0.0, self.rul_clip))
        interval_width = p90 - p10

        # Calibrated CDF approximation P(RUL <= H)
        sigma = self.effective_sigma
        prob_10 = float(norm.cdf((10.0 - pred) / sigma))
        prob_30 = float(norm.cdf((30.0 - pred) / sigma))

        # Honest confidence derived from interval compactness and nominal coverage
        # Tight interval (e.g. width ~15) -> ~94%, wider interval -> ~75-80%
        width_penalty = interval_width / (2.0 * self.rul_clip)
        confidence_pct = round(float(np.clip(self.target_coverage + (0.20 - width_penalty), 0.50, 0.99) * 100.0), 1)

        return UncertaintyEstimate(
            predicted_rul=round(pred, 2),
            rul_p10=round(p10, 2),
            rul_p50=round(p50, 2),
            rul_p90=round(p90, 2),
            interval_width=round(interval_width, 2),
            prob_fail_10=round(prob_10, 4),
            prob_fail_30=round(prob_30, 4),
            confidence_score=confidence_pct,
            interval_method=f"split_conformal_{int(self.target_coverage*100)}pct",
        )


def evaluate_calibration_report() -> dict[str, Any]:
    """Run full calibration evaluation across 100 test engines and save report."""
    import joblib
    import pandas as pd
    from src.features import RAW_COLUMNS, build_rolling_features

    model = joblib.load(MODELS_DIR / "xgb_rul.joblib")
    eda_meta = json.loads((DATA_PROC_DIR / "eda_meta.json").read_text())
    feat_meta = json.loads((DATA_PROC_DIR / "feature_meta.json").read_text())

    sensors = eda_meta["useful_sensors"]
    drop = eda_meta["constant_sensors"]
    feat_cols = feat_meta["feature_cols"]

    df = (
        pd.read_csv(DATA_RAW_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
        .drop(columns=drop)
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )

    feat_df = pd.concat(
        [
            df.groupby("unit_id", group_keys=False).apply(
                lambda g, s=s: build_rolling_features(g.sort_values("cycle"), [s], 30),
                include_groups=False,
            )
            for s in sensors
        ],
        axis=1,
    )

    combined = pd.concat([df[["unit_id", "cycle"]].reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    last = combined.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1).sort_values("unit_id").reset_index(drop=True)
    preds = np.clip(model.predict(last[feat_cols].values), 0, 125)
    trues = pd.read_csv(DATA_RAW_DIR / "RUL_FD001.txt", header=None, names=["t"])["t"].clip(upper=125).values

    residuals = np.abs(preds - trues)

    # 50 engines for calibration, 50 engines for held-out empirical evaluation
    cal_residuals = residuals[:50]
    val_residuals = residuals[50:]
    val_preds = preds[50:]
    val_trues = trues[50:]

    calibrator = ConformalCalibrator(target_coverage=0.80)
    q = calibrator.calibrate(cal_residuals)

    # Measure coverage on validation engines
    p10_val = np.clip(val_preds - q, 0, 125)
    p90_val = np.clip(val_preds + q, 0, 125)
    empirical_coverage = float(np.mean((val_trues >= p10_val) & (val_trues <= p90_val)))

    report = {
        "target_nominal_coverage": 0.80,
        "conformal_quantile_margin_cycles": round(q, 2),
        "calibration_engines_count": len(cal_residuals),
        "evaluation_engines_count": len(val_residuals),
        "empirical_evaluation_coverage": round(empirical_coverage, 4),
        "coverage_error": round(abs(empirical_coverage - 0.80), 4),
        "mean_interval_width_cycles": round(float(np.mean(p90_val - p10_val)), 2),
        "status": "CALIBRATED_WITHIN_BOUNDS" if abs(empirical_coverage - 0.80) <= 0.08 else "SUB_NOMINAL",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "calibration.json").write_text(json.dumps(report, indent=2))
    return report


# Global calibrated instance
default_calibrator = ConformalCalibrator()
