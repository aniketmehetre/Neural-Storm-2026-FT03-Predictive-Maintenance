"""FastAPI application for RUL (Remaining Useful Life) prediction.

Endpoints
---------
GET  /           API info and model metadata.
GET  /health     Liveness / readiness probe (used by Cloud Run).
POST /predict    Predict RUL from a sequence of sensor readings.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features import readings_to_feature_row
from src.uncertainty import default_calibrator
from src.agents.strategy_agent import (
    MaintenanceStrategyRequest,
    MaintenanceStrategyResponse,
    SimulatedMaintenanceAgent,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / "models"
DATA_DIR    = ROOT / "data"


# ── RUL alert thresholds ────────────────────────────────────────────────────
ALERT_CRITICAL = 30    # cycles remaining — imminent failure
ALERT_WARNING  = 80    # degradation detected


def _rul_status(rul: float) -> str:
    """Map a predicted RUL value to a human-readable alert level.

    Args:
        rul: Predicted remaining useful life in cycles.

    Returns:
        One of ``"CRITICAL"``, ``"WARNING"``, or ``"HEALTHY"``.
    """
    if rul <= ALERT_CRITICAL:
        return "CRITICAL"
    if rul <= ALERT_WARNING:
        return "WARNING"
    return "HEALTHY"


# ── App state (loaded once at startup) ─────────────────────────────────────
class _AppState:
    model: Any = None
    model_meta: dict = {}
    feature_cols: list[str] = []
    sensors: list[str] = []
    window: int = 30
    rul_clip: int = 125


state = _AppState()
strategy_agent = SimulatedMaintenanceAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts on startup; release on shutdown."""
    model_path = MODELS_DIR / "xgb_rul.joblib"
    meta_path  = MODELS_DIR / "model_meta.json"

    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Run notebook 03 first.")

    state.model      = joblib.load(model_path)
    state.model_meta = json.loads(meta_path.read_text())
    state.feature_cols = state.model_meta["feature_cols"]
    state.window       = state.model_meta["window"]
    state.rul_clip     = state.model_meta["rul_clip"]

    eda_meta       = json.loads((DATA_DIR / "eda_meta.json").read_text())
    state.sensors  = eda_meta["useful_sensors"]

    log.info(
        "Model loaded — val RMSE=%.2f  features=%d  window=%d",
        state.model_meta["val_metrics"]["rmse"],
        len(state.feature_cols),
        state.window,
    )
    yield
    log.info("Shutting down.")


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance — RUL API",
    description=(
        "Predicts Remaining Useful Life (RUL) of industrial engines "
        "from a rolling window of sensor readings. "
        "Built on NASA CMAPSS FD001 dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    """One operational cycle of sensor measurements for a single engine.

    Variable names follow the CMAPSS documentation (Saxena et al., 2008).

    Attributes:
        cycle:     Cycle number (must be >= 1).
        sensor_02: LPC outlet total temperature (T24, °R).
        sensor_03: HPC outlet total temperature (T30, °R).
        sensor_04: LPT outlet total temperature (T50, °R).
        sensor_06: Bypass-duct total pressure (P15, psia).
        sensor_07: HPC outlet total pressure (P30, psia).
        sensor_08: Physical fan speed (Nf, rpm).
        sensor_09: Physical core speed (Nc, rpm).
        sensor_11: HPC outlet static pressure (Ps30, psia).
        sensor_12: Fuel flow / Ps30 ratio (phi).
        sensor_13: Corrected fan speed (NRf, rpm).
        sensor_14: Corrected core speed (NRc, rpm).
        sensor_15: Bypass ratio (BPR).
        sensor_17: Bleed enthalpy (htBleed).
        sensor_20: HPT coolant bleed (W31).
        sensor_21: LPT coolant bleed (W32).
    """

    cycle: int = Field(..., ge=1, description="Operating cycle number")
    sensor_02: float = Field(..., description="LPC outlet total temperature (T24, °R)")
    sensor_03: float = Field(..., description="HPC outlet total temperature (T30, °R)")
    sensor_04: float = Field(..., description="LPT outlet total temperature (T50, °R)")
    sensor_06: float = Field(..., description="Bypass-duct total pressure (P15, psia)")
    sensor_07: float = Field(..., description="HPC outlet total pressure (P30, psia)")
    sensor_08: float = Field(..., description="Physical fan speed (Nf, rpm)")
    sensor_09: float = Field(..., description="Physical core speed (Nc, rpm)")
    sensor_11: float = Field(..., description="HPC outlet static pressure (Ps30, psia)")
    sensor_12: float = Field(..., description="Fuel flow / Ps30 ratio (phi)")
    sensor_13: float = Field(..., description="Corrected fan speed (NRf, rpm)")
    sensor_14: float = Field(..., description="Corrected core speed (NRc, rpm)")
    sensor_15: float = Field(..., description="Bypass ratio (BPR)")
    sensor_17: float = Field(..., description="Bleed enthalpy (htBleed)")
    sensor_20: float = Field(..., description="HPT coolant bleed (W31)")
    sensor_21: float = Field(..., description="LPT coolant bleed (W32)")


class PredictRequest(BaseModel):
    """Prediction request for a single engine.

    Attributes:
        unit_id:  Optional engine identifier (used in response only).
        readings: Ordered list of recent sensor readings, oldest first.
                  Send at least 1 reading; sending up to 30 gives the
                  most accurate rolling-window features.
    """

    unit_id: int | None = Field(None, description="Engine identifier (optional)")
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Recent sensor readings, oldest first (up to 30 ideal)",
    )

    @field_validator("readings")
    @classmethod
    def cycles_must_be_ordered(
        cls, readings: list[SensorReading]
    ) -> list[SensorReading]:
        """Ensure readings are in non-decreasing cycle order."""
        cycles = [r.cycle for r in readings]
        if cycles != sorted(cycles):
            raise ValueError("readings must be ordered by cycle (ascending)")
        return readings


class PredictResponse(BaseModel):
    """Prediction result for one engine.

    Attributes:
        unit_id:          Engine identifier (echoed from request).
        predicted_rul:    Predicted remaining useful life in cycles.
        status:           Alert level based on predicted RUL.
        cycles_provided:  How many sensor readings were sent.
        window_used:      How many were actually used for feature computation.
        model_val_rmse:   Validation RMSE of the loaded model (confidence proxy).
        rul_p10:          10th percentile RUL from split-conformal interval.
        rul_p50:          Median RUL estimate.
        rul_p90:          90th percentile RUL from split-conformal interval.
        prob_fail_10:     Probability of failure within 10 cycles.
        prob_fail_30:     Probability of failure within 30 cycles.
        interval_method:  Uncertainty estimation methodology.
    """

    unit_id: int | None
    predicted_rul: float
    status: str
    cycles_provided: int
    window_used: int
    model_val_rmse: float
    rul_p10: float | None = None
    rul_p50: float | None = None
    rul_p90: float | None = None
    prob_fail_10: float | None = None
    prob_fail_30: float | None = None
    interval_method: str | None = None


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/", summary="API info")
def root() -> dict:
    """Return API metadata and loaded model info."""
    return {
        "name"          : "Predictive Maintenance — RUL API",
        "version"       : "1.1.0",
        "model_type"    : state.model_meta.get("model_type", "unknown"),
        "val_rmse"      : state.model_meta.get("val_metrics", {}).get("rmse"),
        "test_rmse"     : state.model_meta.get("test_metrics", {}).get("rmse"),
        "features"      : len(state.feature_cols),
        "rolling_window": state.window,
        "rul_clip"      : state.rul_clip,
        "thresholds"    : {"critical": ALERT_CRITICAL, "warning": ALERT_WARNING},
        "endpoints"     : [
            "GET /health",
            "POST /predict",
            "POST /maintenance-strategy",
        ],
    }


@app.get("/health", summary="Health check")
def health() -> dict:
    """Liveness and readiness probe for Cloud Run / Docker."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse, summary="Predict RUL")
def predict(request: PredictRequest) -> PredictResponse:
    """Predict Remaining Useful Life for one engine.

    Accepts the last N cycles of sensor readings (oldest first).
    Computes the same 90 rolling-window features used during training,
    runs inference, and returns the predicted RUL plus an alert level.

    Args:
        request: Engine ID (optional) + list of recent sensor readings.

    Returns:
        Predicted RUL, alert status, and model metadata.

    Raises:
        HTTPException 422: If any sensor value is missing or readings
                           are not in ascending cycle order.
        HTTPException 500: If feature computation or inference fails.
    """
    readings_dicts = [r.model_dump() for r in request.readings]

    try:
        feature_vector = readings_to_feature_row(
            cycle_readings=readings_dicts,
            sensors=state.sensors,
            feature_cols=state.feature_cols,
            window=state.window,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        raw_pred = float(state.model.predict(feature_vector.reshape(1, -1))[0])
    except Exception as exc:
        log.exception("Model inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    predicted_rul = round(float(np.clip(raw_pred, 0, state.rul_clip)), 2)
    cycles_provided = len(request.readings)
    window_used = min(cycles_provided, state.window)

    # Derive calibrated conformal intervals & failure probabilities
    unc = default_calibrator.estimate(predicted_rul)

    log.info(
        "unit_id=%s  cycles=%d  rul=%.1f [p10=%.1f, p90=%.1f] P(fail<=30)=%.2f status=%s",
        request.unit_id,
        cycles_provided,
        predicted_rul,
        unc.rul_p10,
        unc.rul_p90,
        unc.prob_fail_30,
        _rul_status(predicted_rul),
    )

    return PredictResponse(
        unit_id         = request.unit_id,
        predicted_rul   = predicted_rul,
        status          = _rul_status(predicted_rul),
        cycles_provided = cycles_provided,
        window_used     = window_used,
        model_val_rmse  = state.model_meta["val_metrics"]["rmse"],
        rul_p10         = unc.rul_p10,
        rul_p50         = unc.rul_p50,
        rul_p90         = unc.rul_p90,
        prob_fail_10    = unc.prob_fail_10,
        prob_fail_30    = unc.prob_fail_30,
        interval_method = unc.interval_method,
    )


@app.post(
    "/maintenance-strategy",
    response_model=MaintenanceStrategyResponse,
    summary="Generate Actionable Maintenance Strategy & Parts Order",
)
def generate_maintenance_strategy(
    request: MaintenanceStrategyRequest,
) -> MaintenanceStrategyResponse:
    """Evaluate predicted RUL and generate an automated maintenance strategy and parts order.

    Takes the Remaining Useful Life (RUL) prediction and uses an autonomous
    simulated maintenance engineering AI agent to generate:
    - A plain-English operational maintenance strategy narrative.
    - Specific maintenance action windows and operational flight restrictions.
    - An automated, simulated parts-order procurement request with aerospace SKUs,
      estimated costs, priority tier, and logistics lead times.
    - Required technician qualifications and estimated maintenance hangar downtime.

    Args:
        request: Target engine ID, predicted RUL in cycles, optional dominant degrading
                 sensor (e.g. 'sensor_11'), and optional cumulative operating cycles.

    Returns:
        Structured maintenance response including full procurement order.
    """
    try:
        response = strategy_agent.generate_strategy(
            predicted_rul=request.predicted_rul,
            unit_id=request.unit_id,
            dominant_sensor=request.dominant_sensor,
            operating_cycle=request.operating_cycle,
        )
        log.info(
            "Strategy generated: unit_id=%s  rul=%.1f  alert=%s  subsystem=%s  order=%s",
            request.unit_id,
            request.predicted_rul,
            response.alert_level,
            response.primary_affected_subsystem,
            response.parts_order.order_id,
        )
        return response
    except Exception as exc:
        log.exception("Maintenance strategy generation failed")
        raise HTTPException(
            status_code=500, detail="Failed to generate maintenance strategy"
        ) from exc

