"""Tests for Anomaly Detection and Degradation Onset (FT-03 requirement)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.anomaly import default_anomaly_detector, evaluate_anomaly_report


def test_anomaly_healthy_baseline():
    """Verify that early nominal cycles yield high health index and no early onset."""
    import pandas as pd
    from src.config import DATA_RAW_DIR
    from src.features import RAW_COLUMNS

    df = pd.read_csv(DATA_RAW_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    # Take first 10 cycles of engine 1 (known healthy)
    eng1_healthy = df[(df["unit_id"] == 1) & (df["cycle"] <= 10)]
    readings = eng1_healthy.to_dict(orient="records")

    res = default_anomaly_detector.evaluate_sequence(readings)

    assert 0.0 <= res.health_index <= 1.0
    assert res.health_index >= 0.70
    assert not res.onset_detected
    assert res.degradation_state == "HEALTHY_BASELINE"


def test_anomaly_report_fractions():
    """Verify that the anomaly report verifies early onset before RUL <= 80."""
    root = Path(__file__).resolve().parents[1]
    report_file = root / "reports" / "anomaly.json"
    if not report_file.exists():
        evaluate_anomaly_report()

    report = json.loads(report_file.read_text())
    assert report["total_test_engines"] == 100
    assert report["onset_detection_rate"] >= 0.85
    assert report["fraction_detected_before_rul_80"] >= 0.85
    assert report["median_lead_time_to_actual_failure_cycles"] > 100.0


def test_api_anomaly_endpoint():
    """Verify POST /anomaly endpoint responses."""
    from tests.test_api import _make_client, _make_reading

    client = _make_client()
    payload = {
        "unit_id": 7,
        "readings": [_make_reading(cycle=c) for c in range(1, 15)],
    }
    resp = client.post("/anomaly", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["unit_id"] == 7
    assert 0.0 <= data["health_index"] <= 1.0
    assert "anomaly_score" in data
    assert "onset_detected" in data
    assert "degradation_state" in data
