"""Official Multi-Dataset Benchmark and NASA Score Evaluation.

Evaluates RUL regression performance on official CMAPSS test trajectories:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 (Coefficient of Determination)
- NASA Asymmetric PHM Score:
  d = y_pred - y_true
  S = sum(exp(-d/13) - 1 for d < 0) + sum(exp(d/10) - 1 for d >= 0)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROC_DIR, DATA_RAW_DIR, MODELS_DIR, REPORTS_DIR
from src.features import RAW_COLUMNS, build_rolling_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the official NASA asymmetric prognostics scoring function.

    Late predictions (d > 0) are penalized exponentially higher (factor of 10)
    than early predictions (factor of 13) because late maintenance leads to
    in-service failures.

    Args:
        y_true: Ground truth RUL array.
        y_pred: Predicted RUL array.

    Returns:
        Scalar NASA PHM score (lower is better).
    """
    d = y_pred - y_true
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)


def evaluate_dataset(
    dataset_name: str = "FD001",
    rul_clip: int = 125,
    window: int = 30,
) -> dict[str, Any]:
    """Evaluate test set for a given subset (FD001).

    Args:
        dataset_name: Subset identifier (FD001, etc.).
        rul_clip: Maximum RUL clipping value.
        window: Rolling window size.

    Returns:
        Dictionary containing metrics and metadata.
    """
    test_file = DATA_RAW_DIR / f"test_{dataset_name}.txt"
    rul_file = DATA_RAW_DIR / f"RUL_{dataset_name}.txt"

    if not test_file.exists() or not rul_file.exists():
        log.warning("Files for %s not found in %s", dataset_name, DATA_RAW_DIR)
        return {
            "dataset": dataset_name,
            "status": "DATA_UNAVAILABLE",
            "message": f"Raw files test_{dataset_name}.txt not found in CMAPSSData.",
        }

    # Load artifacts
    model = joblib.load(MODELS_DIR / "xgb_rul.joblib")
    eda_meta = json.loads((DATA_PROC_DIR / "eda_meta.json").read_text())
    feat_meta = json.loads((DATA_PROC_DIR / "feature_meta.json").read_text())

    sensors = eda_meta["useful_sensors"]
    drop = eda_meta["constant_sensors"]
    feat_cols = feat_meta["feature_cols"]

    df = (
        pd.read_csv(
            test_file,
            sep=r"\s+",
            header=None,
            names=RAW_COLUMNS,
            engine="python",
        )
        .drop(columns=drop)
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )

    # Compute rolling features per engine
    feat_df = pd.concat(
        [
            df.groupby("unit_id", group_keys=False).apply(
                lambda g, s=s: build_rolling_features(g.sort_values("cycle"), [s], window),
                include_groups=False,
            )
            for s in sensors
        ],
        axis=1,
    )

    combined = pd.concat(
        [df[["unit_id", "cycle"]].reset_index(drop=True), feat_df.reset_index(drop=True)],
        axis=1,
    )

    last_cycle_df = (
        combined.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    X_last = last_cycle_df[feat_cols].values
    raw_preds = model.predict(X_last)
    preds = np.clip(raw_preds, 0, rul_clip).astype(float)

    rul_true = (
        pd.read_csv(rul_file, header=None, names=["t"])["t"]
        .clip(upper=rul_clip)
        .values.astype(float)
    )

    # Compute metrics
    errors = preds - rul_true
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    ss_tot = np.sum((rul_true - np.mean(rul_true)) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    nasa_score = compute_nasa_score(rul_true, preds)

    # Error distribution
    early_preds = int(np.sum(errors < 0))
    late_preds = int(np.sum(errors > 0))
    on_target = int(np.sum(np.abs(errors) <= 2.0))

    return {
        "dataset": dataset_name,
        "status": "VERIFIED_OFFICIAL",
        "sample_count": len(rul_true),
        "metrics": {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "nasa_score": round(nasa_score, 2),
        },
        "error_distribution": {
            "early_predictions_count": early_preds,
            "late_predictions_count": late_preds,
            "within_2_cycles_count": on_target,
            "mean_error_cycles": round(float(np.mean(errors)), 2),
            "median_absolute_error_cycles": round(float(np.median(np.abs(errors))), 2),
        },
        "evaluation_protocol": {
            "tested_rows": "Final observed operational cycle per test engine (100 total)",
            "rul_capping": f"Piecewise linear at {rul_clip} cycles",
            "feature_pipeline": f"Causal rolling statistics ({window}-cycle window)",
        },
    }


def run_benchmark(quick: bool = False) -> dict[str, Any]:
    """Execute official benchmark suite and write reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    datasets = ["FD001"] if quick else ["FD001", "FD002", "FD003", "FD004"]

    for ds in datasets:
        log.info("Benchmarking dataset %s...", ds)
        results[ds] = evaluate_dataset(ds)

    json_path = REPORTS_DIR / "benchmark.json"
    json_path.write_text(json.dumps(results, indent=2))
    log.info("Benchmark JSON written to %s", json_path)

    # Generate Markdown report
    fd001_metrics = results["FD001"]["metrics"]
    fd001_dist = results["FD001"]["error_distribution"]

    md_content = f"""# NASA CMAPSS Benchmark Evaluation Report

**Generated:** {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Evaluation Protocol:** Official test split evaluated at the final observed cycle of each engine against ground-truth remaining life.

## 1. Summary of Benchmark Results

| Dataset | Operating Conditions | Fault Modes | Engines | RMSE (cycles) | MAE (cycles) | R² Score | NASA Asymmetric Score | Status |
|---|---|---|---|---|---|---|---|---|
| **FD001** | Sea Level (1) | HPC Degradation (1) | 100 | **{fd001_metrics['rmse']}** | **{fd001_metrics['mae']}** | **{fd001_metrics['r2']}** | **{fd001_metrics['nasa_score']}** | Verified Official |
| **FD002** | 6 Regimes | HPC Degradation (1) | 259 | — | — | — | — | Raw data not packaged |
| **FD003** | Sea Level (1) | HPC + Fan Degradation (2) | 100 | — | — | — | — | Raw data not packaged |
| **FD004** | 6 Regimes | HPC + Fan Degradation (2) | 248 | — | — | — | — | Raw data not packaged |

## 2. FD001 Detailed Diagnostics

- **Root Mean Squared Error (RMSE):** {fd001_metrics['rmse']} cycles
- **Mean Absolute Error (MAE):** {fd001_metrics['mae']} cycles
- **Median Absolute Error:** {fd001_dist['median_absolute_error_cycles']} cycles
- **NASA Asymmetric Penalty Score:** {fd001_metrics['nasa_score']}
- **Prediction Bias:** {fd001_dist['early_predictions_count']} early predictions vs {fd001_dist['late_predictions_count']} late predictions (Mean bias: {fd001_dist['mean_error_cycles']} cycles).
- **Precision (Within ±2 cycles):** {fd001_dist['within_2_cycles_count']} / 100 engines.

*Note: NASA asymmetric scoring heavily penalizes late estimates ($d > 0$) where maintenance would be scheduled after actual component failure.*
"""
    md_path = REPORTS_DIR / "benchmark.md"
    md_path.write_text(md_content)
    log.info("Benchmark Markdown written to %s", md_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CMAPSS benchmark evaluation")
    parser.add_argument("--quick", action="store_true", help="Run only FD001 verification")
    args = parser.parse_args()
    res = run_benchmark(quick=args.quick)
    print(json.dumps(res, indent=2))
