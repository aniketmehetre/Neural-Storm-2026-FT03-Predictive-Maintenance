"""Tests for calibrated split-conformal uncertainty estimation."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.uncertainty import ConformalCalibrator, default_calibrator


def test_conformal_calibrator_ordering():
    """Verify monotonic ordering of p10, p50, p90 quantiles."""
    for rul in [12.0, 30.0, 75.0, 110.0]:
        est = default_calibrator.estimate(rul)
        assert 0.0 <= est.rul_p10 <= est.rul_p50 <= est.rul_p90 <= 125.0
        assert est.interval_width == pytest.approx(est.rul_p90 - est.rul_p10, abs=0.01)
        assert 0.0 <= est.prob_fail_10 <= est.prob_fail_30 <= 1.0


def test_failure_probabilities_monotonicity():
    """Verify that a more degraded engine (lower RUL) has higher failure probability."""
    critical_est = default_calibrator.estimate(15.0)
    warning_est = default_calibrator.estimate(50.0)
    healthy_est = default_calibrator.estimate(100.0)

    assert critical_est.prob_fail_30 > warning_est.prob_fail_30 > healthy_est.prob_fail_30
    assert critical_est.prob_fail_30 > 0.70
    assert healthy_est.prob_fail_30 < 0.10


def test_conformal_empirical_coverage_within_bounds():
    """Assert empirical coverage on held-out engines is within 8 points of nominal 80%."""
    root = Path(__file__).resolve().parents[1]
    cal_file = root / "reports" / "calibration.json"
    if not cal_file.exists():
        from src.uncertainty import evaluate_calibration_report
        evaluate_calibration_report()

    report = json.loads(cal_file.read_text())
    nominal = report["target_nominal_coverage"]
    empirical = report["empirical_evaluation_coverage"]
    coverage_error = abs(empirical - nominal)

    assert coverage_error <= 0.08, f"Coverage error {coverage_error} exceeds 8-point tolerance"
    assert report["status"] == "CALIBRATED_WITHIN_BOUNDS"


def test_api_predict_returns_calibrated_fields():
    """Verify POST /predict returns the new calibrated uncertainty fields."""
    from tests.test_api import _make_reading
    from src.api import main as api_module
    from unittest.mock import MagicMock

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([25.0])
    api_module.state.model = fake_model

    client = TestClient(api_module.app)
    payload = {"unit_id": 42, "readings": [_make_reading(cycle=1)]}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "rul_p10" in data
    assert "rul_p50" in data
    assert "rul_p90" in data
    assert "prob_fail_10" in data
    assert "prob_fail_30" in data
    assert "interval_method" in data
    assert data["predicted_rul"] == 25.0
    assert data["prob_fail_30"] > 0.50
