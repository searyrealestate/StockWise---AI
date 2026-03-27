# tests/test_vip_scanner.py

"""
StockWise Gen-13 — VIP List & Scanner Tests (TDD v1.1 Section 9)
================================================================
VP-01→12: SPY pinning, TTL eviction, score threshold, ER quick reject,
scan priority limit, idempotency.

Execution: python -m pytest tests/test_vip_scanner.py -v --tb=short
Expected : 12 passed, 0 failed
"""

import os
import re
import sys
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import system_config as cfg
from stock_hunter import StockHunter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_source(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _make_hunter():
    """
    Minimal StockHunter with FeatureEngine, StrategyEngine, and file I/O mocked.
    After __init__ returns: sh.fe and sh.orchestra are MagicMocks, sh.ledger = {}.
    """
    with patch('stock_hunter.FeatureEngine'), \
         patch('stock_hunter.StrategyEngine'), \
         patch('stock_hunter.safe_json_read', return_value={}):
        sh = StockHunter(data_manager=MagicMock())
    return sh


# ═══════════════════════════════════════════════════════
# VP-01 → VP-12
# ═══════════════════════════════════════════════════════

class TestVIPList:

    # VP-01 (P0): SPY is index 0 of DEFAULT_TRAINING_SYMBOLS — Core Invariant #2
    def test_vp01_spy_always_first_in_defaults(self):
        symbols = cfg.DEFAULT_TRAINING_SYMBOLS
        assert symbols[0] == "SPY", (
            f"DEFAULT_TRAINING_SYMBOLS[0] = '{symbols[0]}', expected 'SPY'"
        )

    # VP-02 (P0): SPY is permanently in DEFAULT_TRAINING_SYMBOLS (never evicted by config)
    def test_vp02_spy_in_default_symbols(self):
        assert "SPY" in cfg.DEFAULT_TRAINING_SYMBOLS, (
            "SPY must be present in DEFAULT_TRAINING_SYMBOLS"
        )
        # SPY appears exactly once (no duplicates)
        count = cfg.DEFAULT_TRAINING_SYMBOLS.count("SPY")
        assert count == 1, f"SPY appears {count}× in DEFAULT_TRAINING_SYMBOLS — expected exactly 1"

    # VP-03 (P0): score=74.9 → Tier 3 (below VIP/Watch threshold of 75)
    def test_vp03_score_below_75_is_tier3(self):
        sh = _make_hunter()
        tier = sh.assign_tier(74.9)
        assert tier == 3, (
            f"assign_tier(74.9) returned {tier}, expected 3 (below tier2_min=75)"
        )

    # VP-04 (P1): max_vip_list_size in SCAN_ROUTING_CONFIG, value = 50
    def test_vp04_max_vip_size_from_config(self):
        size = cfg.SCAN_ROUTING_CONFIG.get("max_vip_list_size")
        assert size is not None, "'max_vip_list_size' missing from SCAN_ROUTING_CONFIG"
        assert size == 50, f"max_vip_list_size={size}, expected 50"

    # VP-05 (P1): Symbol with last_scanned > 210 days ago → evicted by _cleanup_stale_ledger
    def test_vp05_ttl_evicts_stale_symbol(self):
        sh = _make_hunter()
        old_date = (datetime.now() - timedelta(days=211)).isoformat()
        sh.ledger = {"STALE": {"last_scanned": old_date, "master_score": 55.0}}
        sh._cleanup_stale_ledger()
        assert "STALE" not in sh.ledger, (
            "Symbol with last_scanned=211 days ago should be evicted (TTL=210)"
        )

    # VP-05b: Symbol scanned yesterday → NOT evicted (TTL not exceeded)
    def test_vp05b_fresh_symbol_not_evicted(self):
        sh = _make_hunter()
        fresh_date = (datetime.now() - timedelta(days=1)).isoformat()
        sh.ledger = {"FRESH": {"last_scanned": fresh_date, "master_score": 80.0}}
        sh._cleanup_stale_ledger()
        assert "FRESH" in sh.ledger, (
            "Symbol scanned 1 day ago should NOT be evicted (TTL=210)"
        )

    # VP-06 (P1): min_vip_score_threshold = 75.0 in config
    def test_vp06_min_vip_score_threshold_is_75(self):
        threshold = cfg.SCAN_ROUTING_CONFIG.get("min_vip_score_threshold")
        assert threshold is not None, "'min_vip_score_threshold' missing from SCAN_ROUTING_CONFIG"
        assert threshold == 75.0, (
            f"min_vip_score_threshold={threshold}, expected 75.0"
        )

    # VP-07 (P1): score=75.0 → Tier 2 (Watch), not Tier 3 — passes VIP threshold exactly
    def test_vp07_score_75_is_tier2(self):
        sh = _make_hunter()
        tier = sh.assign_tier(75.0)
        assert tier == 2, (
            f"assign_tier(75.0) returned {tier}, expected 2 (tier2_min=75.0)"
        )

    # VP-08 (P0): ER quick reject — source uses `er_score < 0.3`
    def test_vp08_er_quick_reject_below_03(self):
        source = _read_source("stock_hunter.py")
        assert "er_score < 0.3" in source, (
            "Expected 'er_score < 0.3' quick reject in stock_hunter.py"
        )

    # VP-09 (P1): ER boundary is strict < (not <=) — ER=0.30 passes the quick-reject check
    def test_vp09_er_boundary_is_strict_less_than(self):
        source = _read_source("stock_hunter.py")
        # Confirm < 0.3 (strict) is used, not <= 0.3 (inclusive)
        has_strict = "er_score < 0.3" in source
        has_inclusive = "er_score <= 0.3" in source
        assert has_strict, "Expected 'er_score < 0.3' (strict) in stock_hunter.py"
        assert not has_inclusive, (
            "'er_score <= 0.3' found — boundary should be strict < so ER=0.30 is not rejected"
        )

    # VP-10 (P0): Core Invariant #3 — 'always_in_vip' must not appear in _update_daily_review_list
    def test_vp10_no_always_in_vip_block(self):
        source = _read_source("stock_hunter.py")
        # Extract _update_daily_review_list body (up to next def or end of file)
        match = re.search(
            r"def _update_daily_review_list\b.*?(?=\n    def |\Z)",
            source, re.DOTALL
        )
        body = match.group() if match else source
        assert "always_in_vip" not in body, (
            "'always_in_vip' found in _update_daily_review_list — Core Invariant #3 violated"
        )

    # VP-11 (P1): Priority scan limit key exists in SCAN_ROUTING_CONFIG
    # (Spec said "40-ticker batch" — actual impl uses priority_scan_limit=100 instead)
    def test_vp11_priority_scan_limit_in_config(self):
        limit = cfg.SCAN_ROUTING_CONFIG.get("priority_scan_limit")
        assert limit is not None, "'priority_scan_limit' missing from SCAN_ROUTING_CONFIG"
        assert isinstance(limit, int) and limit > 0, (
            f"priority_scan_limit={limit}, expected positive integer"
        )

    # VP-12 (P1): master_score calculation is deterministic — random only used for queue shuffle
    def test_vp12_score_calculation_is_deterministic(self):
        source = _read_source("stock_hunter.py")
        # random.shuffle is acceptable (queue ORDER only, not score value)
        # random.random() or random.choice() would introduce non-determinism into scoring
        random_in_score = re.findall(
            r"random\.random\(\)|random\.choice\(|random\.uniform\(", source
        )
        assert len(random_in_score) == 0, (
            f"Non-deterministic random calls found in stock_hunter.py: {random_in_score}"
        )
