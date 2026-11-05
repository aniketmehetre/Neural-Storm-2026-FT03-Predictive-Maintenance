"""Tests for Real-time Streaming, Replay, and Data Quality Guard."""

import json
import pytest
from fastapi.testclient import TestClient

from src.stream.guard import DataQualityGuard
from src.stream.state import fleet_state_manager
from src.api.main import app


def test_data_quality_guard_nominal():
    """Verify nominal readings pass with high trust score and zero warnings."""
    from tests.test_api import _make_reading
    clean_reading = _make_reading(cycle=1, value=640.0)
    sensors = ["sensor_02", "sensor_03", "sensor_04"]
    clean_reading["sensor_02"] = 642.0
    clean_reading["sensor_03"] = 1585.0
    clean_reading["sensor_04"] = 1400.0

    trust_score, warnings = DataQualityGuard.inspect(clean_reading, [], sensors)
    assert trust_score >= 0.90
    assert len(warnings) == 0


def test_data_quality_guard_fault_injection():
    """Verify data guard flags missing (dropout), stuck, and out-of-range (spike) faults."""
    sensors = ["sensor_02", "sensor_03"]

    # 1. Missing / NaN
    bad_reading = {"cycle": 1, "sensor_02": float("nan"), "sensor_03": 1580.0}
    trust_score, warnings = DataQualityGuard.inspect(bad_reading, [], sensors)
    assert trust_score < 0.90
    assert any("missing or NaN" in w for w in warnings)

    # 2. Out of physical bounds
    spike_reading = {"cycle": 2, "sensor_02": 2500.0, "sensor_03": 1580.0}
    trust_score, warnings = DataQualityGuard.inspect(spike_reading, [], sensors)
    assert trust_score < 1.0
    assert any("out of physical bounds" in w for w in warnings)

    # 3. Stuck sensor
    history = [{"cycle": c, "sensor_02": 640.0, "sensor_03": 1580.0} for c in range(1, 7)]
    stuck_reading = {"cycle": 7, "sensor_02": 640.0, "sensor_03": 1580.0}
    trust_score, warnings = DataQualityGuard.inspect(stuck_reading, history, sensors)
    assert any("appears stuck" in w for w in warnings)


def test_ingest_and_fleet_endpoints():
    """Verify IoT /ingest and /fleet snapshot endpoints."""
    from tests.test_api import _make_reading

    client = TestClient(app)
    reading = {"unit_id": 99, **_make_reading(cycle=1)}

    resp = client.post("/ingest", json=reading)
    assert resp.status_code == 200
    state = resp.json()
    assert state["unit_id"] == 99
    assert "predicted_rul" in state
    assert "health_index" in state
    assert "trust_score" in state

    # Test /fleet endpoint
    fleet_resp = client.get("/fleet")
    assert fleet_resp.status_code == 200
    fleet_data = fleet_resp.json()
    assert any(eng["unit_id"] == 99 for eng in fleet_data)


def test_sse_stream_endpoint():
    """Verify GET /stream/{unit_id} yields valid Server-Sent Events."""
    client = TestClient(app)
    with client.stream("GET", "/stream/1?delay=0.01") as response:
        assert response.status_code == 200
        # Read the first few lines of the stream
        lines = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                lines.append(line)
                if len(lines) >= 3:
                    break
        assert len(lines) >= 3
        first_event = json.loads(lines[0].replace("data:", "").strip())
        assert first_event["unit_id"] == 1
        assert "predicted_rul" in first_event
        assert "health_index" in first_event
