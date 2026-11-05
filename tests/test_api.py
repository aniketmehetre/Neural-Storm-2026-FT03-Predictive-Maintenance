"""Pytest tests for the FastAPI RUL prediction endpoint.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Minimal fake model and metadata so tests don't need the real artefacts ─
ROOT = Path(__file__).resolve().parents[1]

FAKE_FEATURE_COLS = [
    f"sensor_{s:02d}_{stat}"
    for s in [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    for stat in ["mean", "std", "min", "max", "lag1", "trend"]
]

FAKE_MODEL_META = {
    "model_type"  : "XGBRegressor",
    "feature_cols": FAKE_FEATURE_COLS,
    "target_col"  : "rul",
    "rul_clip"    : 125,
    "window"      : 30,
    "xgb_params"  : {},
    "val_metrics" : {"rmse": 15.02, "mae": 10.99, "r2": 0.870},
    "test_metrics": {"rmse": 16.74, "mae": 12.33, "r2": 0.825},
    "mlflow_run_id": "fake-run-id",
}

FAKE_EDA_META = {
    "useful_sensors"  : [f"sensor_{s:02d}" for s in [6,15,8,13,21,20,11,2,12,7,17,3,4,14,9]],
    "constant_sensors": ["sensor_01", "sensor_05", "sensor_10", "sensor_16", "sensor_18", "sensor_19"],
    "rul_clip"        : 125,
}


def _make_reading(cycle: int, value: float = 500.0) -> dict:
    """Build a minimal valid sensor reading dict."""
    sensors = [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    return {"cycle": cycle, **{f"sensor_{s:02d}": value for s in sensors}}


def _make_client(predicted_rul: float = 80.0) -> TestClient:
    """Create a TestClient with a mocked model and file system."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([predicted_rul])

    with (
        patch("src.api.main.joblib.load", return_value=fake_model),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "pathlib.Path.read_text",
            side_effect=lambda: json.dumps(FAKE_MODEL_META),
        ),
        patch(
            "builtins.open",
            side_effect=lambda p, *a, **kw: __import__("io").StringIO(
                json.dumps(FAKE_EDA_META)
            ),
        ),
    ):
        from src.api.main import app
        return TestClient(app)


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture()
def client():
    """TestClient with fully mocked model and artefacts."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([80.0])

    from src.api import main as api_module

    api_module.state.model        = fake_model
    api_module.state.model_meta   = FAKE_MODEL_META
    api_module.state.feature_cols = FAKE_FEATURE_COLS
    api_module.state.sensors      = FAKE_EDA_META["useful_sensors"]
    api_module.state.window       = 30
    api_module.state.rul_clip     = 125

    from src.api.main import app
    return TestClient(app, raise_server_exceptions=True)


# ── Health & root ────────────────────────────────────────────────────────────
class TestHealthAndRoot:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_loaded"] is True

    def test_root_returns_metadata(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "XGBRegressor"
        assert data["rolling_window"] == 30
        assert "val_rmse" in data


# ── Predict endpoint ─────────────────────────────────────────────────────────
class TestPredict:
    def test_single_reading_returns_200(self, client: TestClient) -> None:
        payload = {"unit_id": 1, "readings": [_make_reading(cycle=1)]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        payload = {"unit_id": 42, "readings": [_make_reading(cycle=10)]}
        data = client.post("/predict", json=payload).json()
        assert "predicted_rul" in data
        assert "status" in data
        assert "cycles_provided" in data
        assert "window_used" in data
        assert data["unit_id"] == 42

    def test_thirty_readings(self, client: TestClient) -> None:
        readings = [_make_reading(cycle=i) for i in range(1, 31)]
        payload = {"unit_id": 5, "readings": readings}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["cycles_provided"] == 30
        assert resp.json()["window_used"] == 30

    def test_unit_id_is_optional(self, client: TestClient) -> None:
        payload = {"readings": [_make_reading(cycle=1)]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["unit_id"] is None

    def test_predicted_rul_clipped_at_125(self, client: TestClient) -> None:
        from src.api import main as api_module
        api_module.state.model.predict.return_value = np.array([999.0])
        payload = {"readings": [_make_reading(cycle=1)]}
        data = client.post("/predict", json=payload).json()
        assert data["predicted_rul"] <= 125.0

    def test_predicted_rul_not_negative(self, client: TestClient) -> None:
        from src.api import main as api_module
        api_module.state.model.predict.return_value = np.array([-50.0])
        payload = {"readings": [_make_reading(cycle=1)]}
        data = client.post("/predict", json=payload).json()
        assert data["predicted_rul"] >= 0.0


# ── Alert status thresholds ──────────────────────────────────────────────────
class TestAlertStatus:
    @pytest.mark.parametrize("rul,expected_status", [
        (10.0,  "CRITICAL"),
        (30.0,  "CRITICAL"),
        (31.0,  "WARNING"),
        (80.0,  "WARNING"),
        (81.0,  "HEALTHY"),
        (125.0, "HEALTHY"),
    ])
    def test_status_thresholds(
        self,
        client: TestClient,
        rul: float,
        expected_status: str,
    ) -> None:
        from src.api import main as api_module
        api_module.state.model.predict.return_value = np.array([rul])
        payload = {"readings": [_make_reading(cycle=1)]}
        data = client.post("/predict", json=payload).json()
        assert data["status"] == expected_status


# ── Validation errors ────────────────────────────────────────────────────────
class TestValidation:
    def test_empty_readings_rejected(self, client: TestClient) -> None:
        payload = {"readings": []}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_out_of_order_cycles_rejected(self, client: TestClient) -> None:
        readings = [_make_reading(cycle=10), _make_reading(cycle=5)]
        payload = {"readings": readings}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_sensor_field_rejected(self, client: TestClient) -> None:
        bad_reading = {"cycle": 1, "sensor_02": 500.0}  # all others missing
        payload = {"readings": [bad_reading]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ── Maintenance Strategy & Parts Ordering ───────────────────────────────────
class TestMaintenanceStrategy:
    def test_critical_strategy_generation(self, client: TestClient) -> None:
        payload = {
            "unit_id": 42,
            "predicted_rul": 15.5,
            "dominant_sensor": "sensor_11",
            "operating_cycle": 210,
        }
        resp = client.post("/maintenance-strategy", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["unit_id"] == 42
        assert data["predicted_rul"] == 15.5
        assert data["alert_level"] == "CRITICAL"
        assert "AOG" in data["urgency_rating"] or "LEVEL 1" in data["urgency_rating"]
        assert "High-Pressure Compressor" in data["primary_affected_subsystem"]
        assert "GROUND IMMEDIATELY" in " ".join(data["operational_restrictions"])
        assert data["estimated_downtime_hours"] > 0
        
        # Verify parts order
        order = data["parts_order"]
        assert order["order_id"].startswith("PO-AERO-2026-")
        assert order["order_priority"] == "AOG_EMERGENCY"
        assert len(order["items"]) > 0
        assert order["total_cost_usd"] > 0
        assert any("HPC" in item["sku"] for item in order["items"])

    def test_warning_strategy_generation(self, client: TestClient) -> None:
        payload = {
            "unit_id": 12,
            "predicted_rul": 55.0,
            "dominant_sensor": "sensor_03",
        }
        resp = client.post("/maintenance-strategy", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["alert_level"] == "WARNING"
        assert data["parts_order"]["order_priority"] == "EXPEDITED"
        assert data["parts_order"]["estimated_delivery_hours"] <= 48

    def test_healthy_strategy_generation(self, client: TestClient) -> None:
        payload = {
            "unit_id": 99,
            "predicted_rul": 115.0,
        }
        resp = client.post("/maintenance-strategy", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["alert_level"] == "HEALTHY"
        assert data["parts_order"]["order_priority"] == "ROUTINE"
        assert "nominal" in data["plain_english_strategy"].lower() or "healthy" in data["plain_english_strategy"].lower()

    def test_validation_negative_rul_rejected(self, client: TestClient) -> None:
        payload = {"predicted_rul": -10.0}
        resp = client.post("/maintenance-strategy", json=payload)
        assert resp.status_code == 422

    def test_root_lists_new_endpoint(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "POST /maintenance-strategy" in data.get("endpoints", [])

