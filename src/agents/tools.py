"""Shared data-access tools for all AI agents.

All public functions return plain Python dicts/lists so they are JSON-serializable
and can be used directly as OpenAI function call results or LangChain tool outputs.
Results are computed once and cached via ``lru_cache``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap as shap_lib

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
DATA_RAW   = ROOT / "CMAPSSData"
DATA_PROC  = ROOT / "data"
MODELS_DIR = ROOT / "models"

RAW_COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_set_{i}" for i in range(1, 4)]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)

ALERT_CRITICAL = 30
ALERT_WARNING  = 80
RUL_CLIP       = 125
WINDOW         = 30

SENSOR_LABELS: dict[str, str] = {
    "sensor_02": "LPC outlet temp (T24, °R)",
    "sensor_03": "HPC outlet temp (T30, °R)",
    "sensor_04": "LPT outlet temp (T50, °R)",
    "sensor_06": "Bypass-duct pressure (P15, psia)",
    "sensor_07": "HPC outlet pressure (P30, psia)",
    "sensor_08": "Physical fan speed (Nf, rpm)",
    "sensor_09": "Physical core speed (Nc, rpm)",
    "sensor_11": "HPC outlet static pressure (Ps30, psia)",
    "sensor_12": "Fuel flow / Ps30 ratio (phi)",
    "sensor_13": "Corrected fan speed (NRf, rpm)",
    "sensor_14": "Corrected core speed (NRc, rpm)",
    "sensor_15": "Bypass ratio (BPR)",
    "sensor_17": "Bleed enthalpy (htBleed)",
    "sensor_20": "HPT coolant bleed (W31)",
    "sensor_21": "LPT coolant bleed (W32)",
}


# ── Private helpers ────────────────────────────────────────────────────────────
def _rul_status(rul: float) -> str:
    if rul <= ALERT_CRITICAL:
        return "CRITICAL"
    if rul <= ALERT_WARNING:
        return "WARNING"
    return "HEALTHY"


def _feature_label(feat: str) -> str:
    """Map 'sensor_11_mean' → 'HPC outlet static pressure (Ps30, psia)_mean'."""
    for key, name in SENSOR_LABELS.items():
        if feat.startswith(key):
            return f"{name}{feat[len(key):]}"
    return feat


@lru_cache(maxsize=1)
def _load_artifacts():
    """Load model, SHAP explainer, and metadata exactly once."""
    model      = joblib.load(MODELS_DIR / "xgb_rul.joblib")
    model_meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    eda_meta   = json.loads((DATA_PROC  / "eda_meta.json").read_text())
    feat_meta  = json.loads((DATA_PROC  / "feature_meta.json").read_text())
    explainer  = shap_lib.TreeExplainer(model)
    return model, model_meta, eda_meta, feat_meta, explainer


@lru_cache(maxsize=1)
def _compute_fleet() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Compute per-engine last-cycle predictions. Returns (fleet_df, X_last, feat_cols)."""
    from src.features import build_rolling_features  # local import avoids circular refs

    model, _, eda_meta, feat_meta, _ = _load_artifacts()
    sensors   = eda_meta["useful_sensors"]
    drop      = eda_meta["constant_sensors"]
    feat_cols = feat_meta["feature_cols"]

    df = (
        pd.read_csv(
            DATA_RAW / "test_FD001.txt",
            sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python",
        )
        .drop(columns=drop)
        .sort_values(["unit_id", "cycle"])
        .reset_index(drop=True)
    )

    feat_df = pd.concat(
        [
            df.groupby("unit_id", group_keys=False).apply(
                lambda g, s=s: build_rolling_features(g.sort_values("cycle"), [s], WINDOW),
                include_groups=False,
            )
            for s in sensors
        ],
        axis=1,
    )

    result = pd.concat(
        [df[["unit_id", "cycle"]].reset_index(drop=True), feat_df.reset_index(drop=True)],
        axis=1,
    )

    last = (
        result.sort_values(["unit_id", "cycle"])
        .groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    X_last = last[feat_cols].values
    preds  = np.clip(model.predict(X_last), 0, RUL_CLIP)
    rul_true = (
        pd.read_csv(DATA_RAW / "RUL_FD001.txt", header=None, names=["t"])
        ["t"]
        .clip(upper=RUL_CLIP)
        .values
    )

    fleet = last[["unit_id", "cycle"]].copy()
    fleet["predicted_rul"] = preds.round(1)
    fleet["true_rul"]      = rul_true
    fleet["error"]         = (fleet["predicted_rul"] - fleet["true_rul"]).round(1)
    fleet["status"]        = fleet["predicted_rul"].apply(_rul_status)

    return fleet, X_last, feat_cols


# ── Public tool functions ──────────────────────────────────────────────────────

def get_fleet_summary() -> dict:
    """High-level fleet health: counts, top at-risk engines, fleet accuracy."""
    fleet, _, _ = _compute_fleet()

    critical = fleet[fleet["status"] == "CRITICAL"]
    warning  = fleet[fleet["status"] == "WARNING"]
    healthy  = fleet[fleet["status"] == "HEALTHY"]
    rmse = float(np.sqrt(((fleet["predicted_rul"] - fleet["true_rul"]) ** 2).mean()))
    mae  = float((fleet["predicted_rul"] - fleet["true_rul"]).abs().mean())

    return {
        "total_engines":    int(len(fleet)),
        "critical_count":   int(len(critical)),
        "warning_count":    int(len(warning)),
        "healthy_count":    int(len(healthy)),
        "fleet_rmse_cycles": round(rmse, 2),
        "fleet_mae_cycles":  round(mae, 2),
        "top_at_risk": (
            critical.sort_values("predicted_rul")
            .head(5)[["unit_id", "predicted_rul", "status"]]
            .to_dict(orient="records")
        ),
    }


def get_engine_details(engine_id: int) -> dict:
    """Detailed status for one engine: RUL, true RUL, error, status, urgency."""
    fleet, _, _ = _compute_fleet()
    row = fleet[fleet["unit_id"] == engine_id]
    if row.empty:
        return {"error": f"Engine {engine_id} not found. Valid IDs: 1–100."}
    r = row.iloc[0]
    return {
        "unit_id":                  int(r["unit_id"]),
        "predicted_rul_cycles":     float(r["predicted_rul"]),
        "true_rul_cycles":          float(r["true_rul"]),
        "prediction_error_cycles":  float(r["error"]),
        "status":                   str(r["status"]),
        "last_observed_cycle":      int(r["cycle"]),
        "urgency_note": (
            "IMMEDIATE MAINTENANCE REQUIRED — engine approaching end of life"
            if r["status"] == "CRITICAL"
            else "Schedule borescope inspection within 50 cycles"
            if r["status"] == "WARNING"
            else "Operating within normal parameters"
        ),
    }


def get_engines_by_status(status: str) -> dict:
    """All engines with a specific health status (CRITICAL, WARNING, or HEALTHY)."""
    status = status.upper()
    if status not in ("CRITICAL", "WARNING", "HEALTHY"):
        return {"error": "status must be CRITICAL, WARNING, or HEALTHY"}
    fleet, _, _ = _compute_fleet()
    subset = fleet[fleet["status"] == status].sort_values("predicted_rul")
    return {
        "status":  status,
        "count":   int(len(subset)),
        "engines": subset[["unit_id", "predicted_rul", "true_rul", "error"]].to_dict(
            orient="records"
        ),
    }


def get_fleet_statistics() -> dict:
    """Statistical distribution of fleet RUL: mean, median, percentiles, etc."""
    fleet, _, _ = _compute_fleet()
    rul = fleet["predicted_rul"]
    return {
        "mean_rul":                round(float(rul.mean()), 1),
        "median_rul":              round(float(rul.median()), 1),
        "min_rul":                 round(float(rul.min()), 1),
        "max_rul":                 round(float(rul.max()), 1),
        "std_rul":                 round(float(rul.std()), 1),
        "percentile_25":           round(float(rul.quantile(0.25)), 1),
        "percentile_75":           round(float(rul.quantile(0.75)), 1),
        "engines_below_50_cycles": int((rul < 50).sum()),
        "engines_below_20_cycles": int((rul < 20).sum()),
    }


def explain_engine(engine_id: int, top_n: int = 8) -> dict:
    """SHAP-based explanation of why the model predicted a specific RUL."""
    fleet, X_last, feat_cols = _compute_fleet()
    _, _, _, _, explainer    = _load_artifacts()

    row = fleet[fleet["unit_id"] == engine_id]
    if row.empty:
        return {"error": f"Engine {engine_id} not found."}

    idx      = row.index[0]
    sv       = explainer.shap_values(X_last[idx : idx + 1])[0]
    top_idx  = np.argsort(np.abs(sv))[::-1][:top_n]

    features = [
        {
            "feature":    _feature_label(feat_cols[i]),
            "shap_value": round(float(sv[i]), 3),
            "direction":  "increases RUL" if sv[i] > 0 else "decreases RUL",
        }
        for i in top_idx
    ]

    r = row.iloc[0]
    top = features[0]
    return {
        "unit_id":       int(r["unit_id"]),
        "predicted_rul": float(r["predicted_rul"]),
        "status":        str(r["status"]),
        "top_features":  features,
        "interpretation": (
            f"Predicted {r['predicted_rul']:.0f} cycles remaining. "
            f"The dominant factor {'reducing' if top['shap_value'] < 0 else 'extending'} "
            f"the estimated RUL is {top['feature']} "
            f"(SHAP contribution: {top['shap_value']:+.2f} cycles)."
        ),
    }
