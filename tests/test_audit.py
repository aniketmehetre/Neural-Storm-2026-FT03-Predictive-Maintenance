"""Tests for Cryptographic Decision Ledger and Audit Trail Integrity."""

import json
from pathlib import Path
import pytest

from src.audit import DecisionLedger, GENESIS_HASH


def test_ledger_append_and_chain_verification(tmp_path: Path):
    """Verify clean appending and cryptographic chain validation."""
    ledger_file = tmp_path / "test_ledger.jsonl"
    ledger = DecisionLedger(ledger_path=ledger_file)

    # 1. Initially empty
    valid, msg = ledger.verify_chain()
    assert valid

    # 2. Append event 1
    e1 = ledger.append_event("RECOMMENDATION", {"rul": 24.5, "status": "CRITICAL"}, unit_id=42)
    assert e1["entry_id"] == 1
    assert e1["prev_hash"] == GENESIS_HASH
    assert len(e1["entry_hash"]) == 64

    # 3. Append event 2
    e2 = ledger.append_event("APPROVAL", {"approved": True}, unit_id=42)
    assert e2["entry_id"] == 2
    assert e2["prev_hash"] == e1["entry_hash"]

    # 4. Verify chain integrity
    valid, msg = ledger.verify_chain()
    assert valid
    assert "2 cryptographic records intact" in msg


def test_ledger_tampering_detection(tmp_path: Path):
    """Verify that tampering with any record triggers verification failure."""
    ledger_file = tmp_path / "tamper_ledger.jsonl"
    ledger = DecisionLedger(ledger_path=ledger_file)

    ledger.append_event("EVENT_A", {"cost": 45000.0}, unit_id=1)
    ledger.append_event("EVENT_B", {"cost": 35000.0}, unit_id=2)
    ledger.append_event("EVENT_C", {"cost": 15000.0}, unit_id=3)

    valid_before, _ = ledger.verify_chain()
    assert valid_before

    # Tamper with record 2: change cost from 35000.0 to 1.0
    lines = ledger_file.read_text(encoding="utf-8").splitlines()
    rec2 = json.loads(lines[1])
    rec2["payload"]["cost"] = 1.0  # malicious edit!
    lines[1] = json.dumps(rec2)
    ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify tampering is caught
    valid_after, error_msg = ledger.verify_chain()
    assert not valid_after
    assert "TAMPERING DETECTED" in error_msg

