"""Telemetry Replay Streamer with Fault Injection.

Simulates real-time IoT gateway telemetry transmission for testing and live demonstrations.
Supports injecting common industrial sensor faults:
- "dropout": Missing / NaN readings.
- "stuck": Sensor value is frozen across cycles.
- "spike": Extreme electrical or sensor noise impulse.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import pandas as pd

from src.config import DATA_RAW_DIR
from src.features import RAW_COLUMNS
from src.stream.state import fleet_state_manager


async def replay_engine_stream(
    unit_id: int,
    delay_sec: float = 0.1,
    fault_type: str | None = None,
    fault_cycle: int = 15,
) -> AsyncGenerator[str, None]:
    """Asynchronously replay test cycles for an engine and stream SSE JSON events.

    Args:
        unit_id: Target engine identifier (1-100).
        delay_sec: Delay between replayed cycles in seconds.
        fault_type: Optional fault injection ("dropout", "stuck", "spike").
        fault_cycle: Cycle at which the fault should be triggered.

    Yields:
        Server-Sent Event JSON string per cycle.
    """
    test_path = DATA_RAW_DIR / "test_FD001.txt"
    df = pd.read_csv(test_path, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    eng_df = df[df["unit_id"] == unit_id].sort_values("cycle").reset_index(drop=True)

    if eng_df.empty:
        yield json.dumps({"error": f"Engine {unit_id} not found"})
        return

    stuck_val: float | None = None

    for _, row in eng_df.iterrows():
        reading = row.to_dict()
        cycle = int(reading["cycle"])

        # Fault Injection Logic
        if fault_type and cycle >= fault_cycle:
            if fault_type == "dropout":
                reading["sensor_02"] = float("nan")
                reading["sensor_11"] = float("nan")
            elif fault_type == "stuck":
                if stuck_val is None:
                    stuck_val = float(reading["sensor_08"])
                reading["sensor_08"] = stuck_val
            elif fault_type == "spike":
                if cycle == fault_cycle:
                    reading["sensor_03"] = 2800.0  # extreme temperature spike

        # Ingest into live state manager
        live_state = fleet_state_manager.ingest_reading(reading)

        # Yield SSE event payload
        event_data = {
            "unit_id": live_state.unit_id,
            "cycle": live_state.current_cycle,
            "predicted_rul": live_state.predicted_rul,
            "rul_p10": live_state.rul_p10,
            "rul_p90": live_state.rul_p90,
            "prob_fail_30": live_state.prob_fail_30,
            "health_index": live_state.health_index,
            "onset_detected": live_state.onset_detected,
            "onset_cycle": live_state.onset_cycle,
            "status": live_state.status,
            "trust_score": live_state.trust_score,
            "quality_warnings": live_state.quality_warnings,
        }

        yield json.dumps(event_data)
        await asyncio.sleep(delay_sec)

