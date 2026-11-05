"""Cryptographically Verified Append-Only Audit Decision Ledger.

Implements tamper-evident SHA-256 hash chaining (blockchain-style ledger) for
aerospace predictive maintenance governance:
- Logs every maintenance recommendation, human approval decision, and purchase order.
- Each record embeds the SHA-256 hash of the immediately preceding record.
- Provides `verify_chain()` to prove the audit trail has not been tampered with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import REPORTS_DIR

log = logging.getLogger(__name__)

LEDGER_FILE = REPORTS_DIR / "decision_ledger.jsonl"
GENESIS_HASH = "0" * 64


def _compute_hash(data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a dictionary (excluding entry_hash)."""
    clean_copy = {k: v for k, v in data.items() if k != "entry_hash"}
    canonical_json = json.dumps(clean_copy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class DecisionLedger:
    """Tamper-evident append-only ledger for maintenance decisions."""

    def __init__(self, ledger_path: Path | None = None):
        self.ledger_path = ledger_path or LEDGER_FILE
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def get_last_entry(self) -> dict[str, Any] | None:
        """Return the most recent ledger entry, or None if empty."""
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return None
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                return None
            return json.loads(lines[-1])

    def append_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        unit_id: int | None = None,
    ) -> dict[str, Any]:
        """Record an immutable event into the hash chain."""
        last_entry = self.get_last_entry()
        if last_entry is None:
            entry_id = 1
            prev_hash = GENESIS_HASH
        else:
            entry_id = int(last_entry["entry_id"]) + 1
            prev_hash = str(last_entry["entry_hash"])

        entry_data: dict[str, Any] = {
            "entry_id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "unit_id": unit_id,
            "payload": payload,
            "prev_hash": prev_hash,
        }

        entry_hash = _compute_hash(entry_data)
        entry_data["entry_hash"] = entry_hash

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_data) + "\n")

        log.info("Ledger entry #%d [%s] recorded (hash: %s)", entry_id, event_type, entry_hash[:8])
        return entry_data

    def verify_chain(self) -> tuple[bool, str]:
        """Validate integrity of the entire ledger chain.

        Returns:
            (True, msg) if valid, (False, error_msg) if any record was tampered with.
        """
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return True, "Ledger is empty (0 records)."

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        expected_prev_hash = GENESIS_HASH
        for idx, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except Exception as e:
                return False, f"JSON parse failure at line {idx}: {e}"

            # 1. Verify previous hash link
            if record.get("prev_hash") != expected_prev_hash:
                return False, (
                    f"TAMPERING DETECTED at record #{record.get('entry_id', idx)}: "
                    f"prev_hash '{record.get('prev_hash')[:12]}' does not match expected '{expected_prev_hash[:12]}'"
                )

            # 2. Verify record hash integrity
            recorded_hash = record.get("entry_hash")
            calculated_hash = _compute_hash(record)
            if recorded_hash != calculated_hash:
                return False, (
                    f"TAMPERING DETECTED at record #{record.get('entry_id', idx)}: "
                    f"hash '{recorded_hash[:12]}' != calculated '{calculated_hash[:12]}'"
                )

            expected_prev_hash = recorded_hash

        return True, f"Chain integrity verified: {len(lines)} cryptographic records intact."


default_ledger = DecisionLedger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify decision ledger cryptographic integrity")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify the hash chain")
    args = parser.parse_args()

    ledger = DecisionLedger()
    valid, msg = ledger.verify_chain()
    print("STATUS:", "PASS" if valid else "FAIL")
    print("DETAIL:", msg)

