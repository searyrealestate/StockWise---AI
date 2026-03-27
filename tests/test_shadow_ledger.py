# tests/test_shadow_ledger.py

"""
StockWise Gen-13 — Shadow Ledger Tests (TDD v1.1 Section 10)
============================================================
SL-01→09: Candle-by-candle evaluation, virtual signal tracking,
per-template stats, safe I/O, offline execution.

Execution: python -m pytest tests/test_shadow_ledger.py -v --tb=short
Expected : 9 passed, 0 failed
"""

import os
import re
import sys
import json
import shutil
import tempfile

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shadow_ledger import ShadowLedger
import system_config as cfg


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_eval_df(rows=300, trend='up'):
    """
    DataFrame suitable for shadow evaluation.
    Must have >= min_candles_for_eval (200) + lookahead (20) = 220 rows minimum.
    Includes OHLCV + indicator columns templates may access.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=rows, freq='B')

    if trend == 'up':
        close = np.linspace(80, 150, rows) + np.random.uniform(-0.5, 0.5, rows)
    elif trend == 'down':
        close = np.linspace(150, 80, rows) + np.random.uniform(-0.5, 0.5, rows)
    else:  # flat/sideways
        close = 100 + np.sin(np.linspace(0, 8 * np.pi, rows)) * 5

    df = pd.DataFrame({
        'open':    close * 0.998,
        'high':    close * 1.015,
        'low':     close * 0.985,
        'close':   close,
        'volume':  np.full(rows, 2_000_000.0),
        'rsi':     np.clip(np.random.uniform(30, 70, rows), 0, 100),
        'adx':     np.random.uniform(15, 40, rows),
        'atr':     np.abs(close * 0.02),
        'sma_50':  pd.Series(close).rolling(50, min_periods=1).mean().values,
        'sma_200': pd.Series(close).rolling(200, min_periods=1).mean().values,
        'er_slow': np.random.uniform(0.3, 0.7, rows),
    }, index=dates)
    df.index.name = 'Date'
    return df


def _make_mock_template(tid, always_signal=False, stop_offset=-3.0, target_offset=5.0):
    """
    Create a mock SetupTemplate.

    always_signal=True  → fires on every bar
    always_signal=False → fires every ~30 bars (for coverage variation)

    stop_offset / target_offset are applied relative to current close:
      stop_loss   = close + stop_offset   (negative → below entry)
      take_profit = close + target_offset (positive → above entry)
    """
    t = MagicMock()
    t.id   = tid
    t.name = tid

    if always_signal:
        t.evaluate_conditions.return_value = (True, [])
    else:
        call_count = {'n': 0}
        def _eval(row):
            call_count['n'] += 1
            return (call_count['n'] % 30 == 0, [])
        t.evaluate_conditions.side_effect = _eval

    def _stop(row):
        return float(row.get('close', 100)) + stop_offset

    def _target(row):
        return float(row.get('close', 100)) + target_offset

    t.calculate_stop_loss.side_effect   = _stop
    t.calculate_take_profit.side_effect = _target
    return t


@pytest.fixture
def tmp_ledger_dir():
    """Temp directory for ledger JSON files — cleaned up after each test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── Shadow Ledger factory ───────────────────────────────────────────────────────

def _make_sl(tmp_dir, templates=None):
    """
    Instantiate ShadowLedger with:
      - mocked TemplateManager (controls which templates fire)
      - patched SHADOW_LEDGER_CONFIG pointing ledger_path to temp dir
    """
    mock_tm = MagicMock()
    tpls = templates or []
    mock_tm.get_enabled.return_value   = tpls
    mock_tm.get_for_state.return_value = tpls

    ledger_path = os.path.join(tmp_dir, "test_shadow_ledger.json")

    # Patch BEFORE __init__ runs so self.config picks up the temp path.
    # After the context exits, sl.config still holds the patched dict (already assigned).
    patched_config = {
        'enabled':                True,
        'ledger_path':            ledger_path,
        'eval_days_back':         1095,
        'max_templates':          5,
        'lookahead_candles':      20,
        'min_candles_for_eval':   200,
        'min_bars_between_signals': 20,
        'run_mode':               'offline',
    }
    with patch.object(cfg, 'SHADOW_LEDGER_CONFIG', patched_config):
        sl = ShadowLedger(template_manager=mock_tm)
    # sl.config / sl.ledger_path already set from patched dict; they persist after patch exits
    return sl


# ═══════════════════════════════════════════════════════
# SL-01 → SL-09
# ═══════════════════════════════════════════════════════

class TestShadowLedger:

    # SL-01 (P0): All signals generated are recorded in the ledger
    def test_sl01_records_every_signal(self, tmp_ledger_dir):
        t1 = _make_mock_template("T1", always_signal=True)
        sl = _make_sl(tmp_ledger_dir, templates=[t1])
        df = _make_eval_df(300, trend='up')

        results = sl.evaluate_history("AAPL", df)

        total = sum(r['signal_count'] for r in results.values())
        assert total > 0, "Should record at least 1 signal across the evaluation window"

    # SL-02 (P0): Uptrend + tight target → target hit → wins recorded
    def test_sl02_tracks_target_hit(self, tmp_ledger_dir):
        # stop_offset=-5.0 keeps stop far below; target_offset=2.0 → hit quickly in uptrend
        t1 = _make_mock_template("T1", always_signal=True, stop_offset=-5.0, target_offset=2.0)
        sl = _make_sl(tmp_ledger_dir, templates=[t1])
        df = _make_eval_df(300, trend='up')

        results = sl.evaluate_history("AAPL", df)
        wins = results.get("T1", {}).get("wins", 0)
        assert wins > 0, "Uptrend with tight target (close+2) should register wins"

    # SL-03 (P0): Downtrend + tight stop → stop hit first → losses recorded
    def test_sl03_tracks_stop_hit(self, tmp_ledger_dir):
        # stop_offset=-1.0 → close-1 stop; downtrend drops low = close*0.985 < close-1 in 1 bar
        # _resolve_outcome checks stop FIRST (conservative), so losses are recorded
        t1 = _make_mock_template("T1", always_signal=True, stop_offset=-1.0, target_offset=10.0)
        sl = _make_sl(tmp_ledger_dir, templates=[t1])
        df = _make_eval_df(300, trend='down')

        results = sl.evaluate_history("AAPL", df)
        losses = results.get("T1", {}).get("losses", 0)
        assert losses > 0, "Downtrend with tight stop (close-1) should register losses"

    # SL-04 (P0): Shadow evaluates ALL signals virtually, independent of live execution
    def test_sl04_independent_of_execution(self, tmp_ledger_dir):
        # Shadow Ledger has no concept of "was this signal actually traded" —
        # it tracks every qualifying signal regardless of live order state.
        t1 = _make_mock_template("T1", always_signal=True)
        sl = _make_sl(tmp_ledger_dir, templates=[t1])
        df = _make_eval_df(300)

        results = sl.evaluate_history("AAPL", df)
        total = sum(r['signal_count'] for r in results.values())
        assert total > 0, "Shadow evaluates every qualifying signal (no execution filter)"

    # SL-05 (P0): Each candle evaluates ALL active templates independently
    def test_sl05_candle_by_candle_eval(self, tmp_ledger_dir):
        t1 = _make_mock_template("T1", always_signal=True)
        t2 = _make_mock_template("T2", always_signal=True)
        sl = _make_sl(tmp_ledger_dir, templates=[t1, t2])
        df = _make_eval_df(300)

        results = sl.evaluate_history("AAPL", df)
        assert "T1" in results, "T1 should appear in results"
        assert "T2" in results, "T2 should appear in results"
        # Both templates were evaluated independently
        assert results["T1"]["signal_count"] > 0, "T1 should have signals"
        assert results["T2"]["signal_count"] > 0, "T2 should have signals"

    # SL-06 (P1): Per-template stats include all required keys
    def test_sl06_per_template_stats_shape(self, tmp_ledger_dir):
        t1 = _make_mock_template("T1", always_signal=True)
        sl = _make_sl(tmp_ledger_dir, templates=[t1])
        df = _make_eval_df(300, trend='up')

        results = sl.evaluate_history("AAPL", df)
        stats = results.get("T1", {})
        required_keys = ['signal_count', 'wins', 'losses', 'win_rate', 'avg_pnl_pct']
        for key in required_keys:
            assert key in stats, f"Template stats missing required key: '{key}'"

    # SL-07 (P0): Shadow Ledger uses safe_json_io — no raw json.dump
    def test_sl07_uses_safe_json_io(self):
        path = os.path.join(PROJECT_ROOT, "shadow_ledger.py")
        if not os.path.exists(path):
            pytest.skip("shadow_ledger.py not found")
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "safe_json_write" in source, "shadow_ledger.py must use safe_json_write"
        assert "safe_json_read"  in source, "shadow_ledger.py must use safe_json_read"
        # No raw json.dump calls (json.dump on its own line, not inside a string literal)
        raw_dumps = re.findall(r"\bjson\.dump\s*\(", source)
        assert len(raw_dumps) == 0, (
            f"Found raw json.dump in shadow_ledger.py — must use safe_json_write instead"
        )

    # SL-08 (P1): Shadow runs offline (weekend batch), not in the nightly scan loop
    def test_sl08_run_mode_is_offline(self):
        config = getattr(cfg, 'SHADOW_LEDGER_CONFIG', {})
        mode = config.get('run_mode')
        assert mode == 'offline', (
            f"SHADOW_LEDGER_CONFIG['run_mode'] should be 'offline', got '{mode}'"
        )

    # SL-09 (P0): Corrupted JSON file → safe fallback to empty dict, no crash
    def test_sl09_corrupted_file_recovery(self, tmp_ledger_dir):
        ledger_path = os.path.join(tmp_ledger_dir, "corrupted_ledger.json")
        with open(ledger_path, 'w') as f:
            f.write("{invalid json content!!!}")

        mock_tm = MagicMock()
        mock_tm.get_enabled.return_value   = []
        mock_tm.get_for_state.return_value = []

        patched_config = {
            'enabled':                True,
            'ledger_path':            ledger_path,
            'min_candles_for_eval':   200,
            'lookahead_candles':      20,
            'min_bars_between_signals': 20,
        }
        # safe_json_read retries 3× on parse failure then returns default — should not raise
        with patch('time.sleep'):  # suppress retry delays so test runs fast
            with patch.object(cfg, 'SHADOW_LEDGER_CONFIG', patched_config):
                sl = ShadowLedger(template_manager=mock_tm)

        assert isinstance(sl.ledger, dict), (
            "Ledger should be a dict after corrupted-file recovery"
        )
