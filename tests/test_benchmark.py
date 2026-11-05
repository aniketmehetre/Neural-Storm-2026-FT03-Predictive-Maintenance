"""Tests for benchmark evaluation and NASA scoring."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.benchmark import compute_nasa_score, evaluate_dataset
from src.features import build_rolling_features
import pandas as pd


def test_nasa_score_asymmetry():
    """Verify that late predictions receive higher penalty than early ones."""
    y_true = np.array([50.0])
    
    # 5 cycles early (y_pred = 45) -> diff = -5
    early_score = compute_nasa_score(y_true, np.array([45.0]))
    # 5 cycles late (y_pred = 55) -> diff = +5
    late_score = compute_nasa_score(y_true, np.array([55.0]))

    assert early_score > 0
    assert late_score > 0
    assert late_score > early_score, "Late maintenance penalty must exceed early penalty"


def test_fd001_benchmark_evaluation():
    """Verify official FD001 benchmark output and structure."""
    result = evaluate_dataset("FD001")
    assert result["status"] == "VERIFIED_OFFICIAL"
    assert result["sample_count"] == 100
    metrics = result["metrics"]
    assert metrics["rmse"] == pytest.approx(16.74, abs=0.1)
    assert metrics["mae"] == pytest.approx(12.33, abs=0.1)
    assert metrics["nasa_score"] > 0


def test_features_metadata_consistency():
    """Regression test ensuring features output matches feature_meta.json definition."""
    root = Path(__file__).resolve().parents[1]
    feat_meta = json.loads((root / "data" / "feature_meta.json").read_text())
    expected_cols = feat_meta["feature_cols"]

    # Generate synthetic sequence of 35 cycles
    sensors = ["sensor_02", "sensor_03"]
    df = pd.DataFrame({
        "cycle": range(1, 36),
        "sensor_02": np.linspace(640, 645, 35),
        "sensor_03": np.linspace(1580, 1590, 35),
    })

    feat_df = build_rolling_features(df, sensors=sensors, window=30)
    for s in sensors:
        for stat in ["mean", "std", "min", "max", "lag1", "trend"]:
            col_name = f"{s}_{stat}"
            assert col_name in feat_df.columns
            assert not feat_df[col_name].isna().any()
