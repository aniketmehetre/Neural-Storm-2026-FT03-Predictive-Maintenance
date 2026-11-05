"""Shared feature engineering logic used by both notebooks and the API.

Keeping this in one place guarantees that inference-time features are
computed identically to training-time features — the most common source
of silent model bugs in production.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Column names expected in raw CMAPSS data files.
RAW_COLUMNS: list[str] = (
    ["unit_id", "cycle"]
    + [f"op_set_{i}" for i in range(1, 4)]
    + [f"sensor_{i:02d}" for i in range(1, 22)]
)


def build_rolling_features(
    readings: pd.DataFrame,
    sensors: list[str],
    window: int = 30,
) -> pd.DataFrame:
    """Compute rolling-window features for a sequence of sensor readings.

    Each sensor produces six derived features over the last ``window`` cycles:
    mean, std, min, max, lag-1, and trend (raw − rolling mean).

    The computation is causal: at cycle t only cycles [t-window+1, t] are used,
    so this function is safe to call at inference time with partial histories.

    Args:
        readings: DataFrame with one row per cycle, containing at minimum a
                  ``cycle`` column and all columns listed in ``sensors``.
                  Rows must be sorted ascending by cycle.
        sensors:  List of sensor column names to featurise.
        window:   Rolling window size in cycles (default 30).

    Returns:
        DataFrame with 6 × len(sensors) feature columns and the same index
        as ``readings``.  The original sensor columns are not included.
    """
    if readings.empty:
        raise ValueError("readings DataFrame is empty")

    parts: list[pd.DataFrame] = []
    for sensor in sensors:
        if sensor not in readings.columns:
            raise KeyError(f"Sensor column '{sensor}' not found in readings")
        parts.append(_rolling_one_sensor(readings[sensor], sensor, window))

    return pd.concat(parts, axis=1)


def _rolling_one_sensor(
    series: pd.Series,
    name: str,
    window: int,
) -> pd.DataFrame:
    """Compute the six rolling statistics for a single sensor series.

    Args:
        series: Sensor values ordered by ascending cycle.
        name:   Column name prefix (e.g. ``"sensor_11"``).
        window: Rolling window size.

    Returns:
        DataFrame with columns ``{name}_mean``, ``{name}_std``,
        ``{name}_min``, ``{name}_max``, ``{name}_lag1``, ``{name}_trend``.
    """
    roll = series.rolling(window=window, min_periods=1)
    mean = roll.mean()
    return pd.DataFrame(
        {
            f"{name}_mean" : mean,
            f"{name}_std"  : roll.std().fillna(0.0),
            f"{name}_min"  : roll.min(),
            f"{name}_max"  : roll.max(),
            f"{name}_lag1" : series.shift(1).fillna(series.iloc[0]),
            f"{name}_trend": series - mean,
        },
        index=series.index,
    )


def readings_to_feature_row(
    cycle_readings: list[dict[str, Any]],
    sensors: list[str],
    feature_cols: list[str],
    window: int = 30,
) -> np.ndarray:
    """Convert a list of recent sensor readings into a single feature vector.

    This is the function the API calls for every prediction request.
    It mirrors exactly what notebook 02 does at training time.

    Args:
        cycle_readings: List of dicts, each representing one cycle.
                        Keys must include ``cycle`` and all sensor names.
                        Must be in chronological order (oldest first).
                        At least 1 entry required; up to ``window`` is ideal.
        sensors:        Ordered list of sensor column names (from EDA metadata).
        feature_cols:   Ordered list of feature column names expected by the
                        model (from feature metadata).
        window:         Rolling window size used during training.

    Returns:
        1-D numpy array of shape ``(len(feature_cols),)`` ready for
        ``model.predict()``.

    Raises:
        ValueError: If ``cycle_readings`` is empty or a sensor is missing.
    """
    if not cycle_readings:
        raise ValueError("cycle_readings must contain at least one entry")

    df = pd.DataFrame(cycle_readings).sort_values("cycle").reset_index(drop=True)

    feature_df = build_rolling_features(df, sensors, window)

    last_row = feature_df.iloc[[-1]][feature_cols]
    return last_row.values[0].astype(np.float32)
