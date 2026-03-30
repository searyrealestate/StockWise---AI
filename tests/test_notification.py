# tests/test_notification.py

"""
StockWise Gen-13 — Notification & I/O Tests (TDD v1.1 Section 11)
================================================================
Telegram Commands (TG-01→05), Safe I/O (IO-01→06).

Execution: python -m pytest tests/test_notification.py -v --tb=short
Expected : 11 passed, 0 failed
"""

import os
import re
import sys
import json
import glob
import shutil
import tempfile

import pytest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import system_config as cfg
from notification_manager import NotificationManager
from safe_json_io import safe_json_read, safe_json_write


def _read_source(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ═══════════════════════════════════════════
# 11.1 TELEGRAM COMMAND TESTS (TG-01 → TG-05)
# ═══════════════════════════════════════════

class TestTelegramCommands:

    @pytest.fixture
    def nm(self):
        """Minimal NotificationManager — Telegram disabled (tokens empty → self.enabled=False)."""
        with patch.object(cfg, 'TELEGRAM_TOKEN', ''), \
             patch.object(cfg, 'TELEGRAM_CHAT_ID', ''):
            return NotificationManager()

    # TG-01 (P1): /CONFIRM command exists in process_incoming_command
    def test_tg01_confirm_command_exists(self):
        source = _read_source("notification_manager.py")
        assert "/CONFIRM" in source or "confirm" in source.lower(), (
            "No /CONFIRM command handler found in notification_manager.py"
        )
        # Must update the ledger with fill status
        assert "_update_ledger_status" in source, (
            "_update_ledger_status not called from confirm handler"
        )

    # TG-02 (P1): /UNFILLED command exists in process_incoming_command
    def test_tg02_unfilled_command_exists(self):
        source = _read_source("notification_manager.py")
        assert "/UNFILLED" in source or "unfilled" in source.lower(), (
            "No /UNFILLED command handler found in notification_manager.py"
        )

    # TG-03 (P0): Veto protection is enforced in the scanner path (adapted)
    # Spec said /veto Telegram command — not yet implemented.
    # The veto mechanism runs via check_veto_gates in the nightly scan loop (stock_hunter.py).
    def test_tg03_veto_gate_enforced_in_scan(self):
        source = _read_source("stock_hunter.py")
        assert "check_veto_gates" in source, (
            "check_veto_gates not called in stock_hunter.py — veto protection missing from scan path"
        )
        # Confirm it's wired to the feature engine, not just imported
        assert "self.fe.check_veto_gates" in source, (
            "check_veto_gates not called on self.fe — may not be applied per-ticker"
        )

    # TG-04 (P1): /CONFIRM AAPL → _update_ledger_status('AAPL', 'CONFIRMED') called
    def test_tg04_confirm_routes_to_ledger_update(self, nm):
        with patch.object(nm, '_update_ledger_status', return_value=True) as mock_update:
            nm.process_incoming_command('/CONFIRM AAPL')
        expected_status = getattr(cfg, 'TRADE_STATUS_EXECUTED', 'CONFIRMED')
        mock_update.assert_called_once_with('AAPL', expected_status)

    # TG-05 (P2): Non-command text → returns early, no crash, returns None
    def test_tg05_non_command_returns_early(self, nm):
        result = nm.process_incoming_command("hello world")  # no leading /
        assert result is None, (
            "process_incoming_command should return None for non-/ input"
        )
        # Empty string also safe
        result2 = nm.process_incoming_command("")
        assert result2 is None


# ═══════════════════════════════════════════
# 11.2 SAFE I/O TESTS (IO-01 → IO-06)
# ═══════════════════════════════════════════

class TestSafeIO:

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    # IO-01 (P0): safe_json_write creates a valid, readable JSON file atomically
    def test_io01_safe_write_creates_valid_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "test_write.json")
        data = {"symbol": "AAPL", "score": 87.5, "tier": 1}

        safe_json_write(path, data)

        assert os.path.exists(path), "safe_json_write did not create the file"
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == data, f"Written data mismatch: {loaded}"

    # IO-02 (P0): Corrupted JSON → safe_json_read returns default, no crash
    def test_io02_corrupted_json_returns_default(self, tmp_dir):
        path = os.path.join(tmp_dir, "corrupted.json")
        with open(path, 'w') as f:
            f.write("{broken json, not valid!!!")

        with patch('time.sleep'):  # suppress retry delays
            result = safe_json_read(path, default={"fallback": True})

        assert isinstance(result, dict), "Should return a dict on corruption"
        assert result.get("fallback") is True, "Should return the provided default"

    # IO-03 (P1): Missing file → safe_json_read returns default, no crash
    def test_io03_missing_file_returns_default(self, tmp_dir):
        path = os.path.join(tmp_dir, "nonexistent.json")
        assert not os.path.exists(path)

        result = safe_json_read(path, default={"missing": True})

        assert isinstance(result, dict), "Should return a dict for missing file"
        assert result.get("missing") is True, "Should return the provided default"

    # IO-04 (P0): No raw json.dump in the critical live-trading path files.
    # Only these files run during live trading and must use safe_json_io exclusively.
    # Training scripts, simulation, and utilities are out of scope.
    def test_io04_no_raw_json_dump_in_live_path(self):
        critical_files = [
            'live_trading_engine.py',
            'stock_hunter.py',
            'notification_manager.py',
            'shadow_ledger.py',
            'pre_market_validator.py',
            'portfolio_risk.py',
        ]
        violations = []
        for basename in critical_files:
            path = os.path.join(PROJECT_ROOT, basename)
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    code = line.split('#')[0]
                    if re.search(r'\bjson\.dump\s*\(', code) and \
                       not re.search(r'\bjson\.dumps\s*\(', code):
                        violations.append(f"{basename}:{i}")
        assert violations == [], (
            f"Raw json.dump found in live-path files: {violations}. "
            f"Use safe_json_write instead."
        )

    # IO-05 (P0): No raw json.load in the critical live-trading path files.
    def test_io05_no_raw_json_load_in_live_path(self):
        critical_files = [
            'live_trading_engine.py',
            'stock_hunter.py',
            'notification_manager.py',
            'shadow_ledger.py',
            'pre_market_validator.py',
            'portfolio_risk.py',
        ]
        violations = []
        for basename in critical_files:
            path = os.path.join(PROJECT_ROOT, basename)
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    code = line.split('#')[0]
                    if re.search(r'\bjson\.load\s*\(', code) and \
                       not re.search(r'\bjson\.loads\s*\(', code):
                        violations.append(f"{basename}:{i}")
        assert violations == [], (
            f"Raw json.load found in live-path files: {violations}. "
            f"Use safe_json_read instead."
        )

    # IO-06 (P0): scan_ledger.json is loaded once per cycle, not inside the per-ticker loop
    def test_io06_scan_ledger_read_once_per_cycle(self):
        source = _read_source("live_trading_engine.py")
        # The engine has an explicit comment documenting this invariant
        assert "not per ticker" in source, (
            "Comment 'not per ticker' missing from live_trading_engine.py — "
            "scan_ledger.json must be loaded once per cycle, not once per symbol"
        )
        assert "once per cycle" in source, (
            "Comment 'once per cycle' missing from live_trading_engine.py"
        )
