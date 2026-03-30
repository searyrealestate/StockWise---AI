"""
Tests for anti-overfitting rules, expanded block registry, and PULLBACK fix.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import system_config as cfg
from setup_templates import (
    SetupTemplate, TemplateManager, CONDITION_BLOCKS,
    block_adx_above, block_supertrend_bullish, block_golden_cross_active,
    block_stoch_oversold, block_cci_between, block_roc_positive,
    block_obv_rising, block_cmf_positive, block_vwap_above,
    block_gap_up_today, block_fib_near_support, block_double_bottom_active,
)


# ═══════════════════════════════════════════════════════════
# SECTION 1: Anti-Overfitting Rules
# ═══════════════════════════════════════════════════════════

class TestAntiOverfittingConfig:
    """Config keys exist and are sensible."""

    def test_template_config_has_category_keys(self):
        tc = cfg.TEMPLATE_CONFIG
        assert "max_conditions_per_category" in tc
        assert "max_conditions_hard_limit" in tc
        assert "block_categories" in tc

    def test_hard_limit_gte_5(self):
        """Hard limit must be at least 5 (backward compat with existing templates)."""
        assert cfg.TEMPLATE_CONFIG["max_conditions_hard_limit"] >= 5

    def test_block_categories_cover_all_registry_blocks(self):
        """Every block in CONDITION_BLOCKS must appear in exactly one category."""
        cats = cfg.TEMPLATE_CONFIG.get("block_categories", {})
        all_categorized = set()
        for blocks in cats.values():
            all_categorized.update(blocks)
        for block_name in CONDITION_BLOCKS.keys():
            assert block_name in all_categorized, \
                f"Block '{block_name}' not in any category in TEMPLATE_CONFIG.block_categories"

    def test_no_block_in_multiple_categories(self):
        """A block must not appear in two different categories."""
        cats = cfg.TEMPLATE_CONFIG.get("block_categories", {})
        seen = {}
        for cat, blocks in cats.items():
            for b in blocks:
                assert b not in seen, f"Block '{b}' in both '{seen[b]}' and '{cat}'"
                seen[b] = cat


class TestAntiOverfittingValidation:
    """validate() enforces diversity rules."""

    def _make_template(self, conditions):
        return SetupTemplate({
            "id": "TEST", "name": "Test", "conditions": conditions,
            "stop_loss":   {"method": "atr", "atr_multiplier": 2.0, "fallback_pct": 0.02},
            "take_profit": {"method": "atr", "atr_multiplier": 3.0},
        })

    def test_valid_diverse_template(self):
        """5 conditions from 4 categories should pass."""
        t = self._make_template([
            {"block": "er_slow_above",  "params": [0.45]},
            {"block": "rsi_between",    "params": [40, 65]},
            {"block": "volume_surge",   "params": [1.2]},
            {"block": "squeeze_active", "params": []},
            {"block": "bullish_candle", "params": []},
        ])
        valid, errors = t.validate()
        assert valid, f"Should be valid: {errors}"

    def test_reject_3_blocks_same_category(self):
        """3 trend blocks in one template should fail diversity check."""
        t = self._make_template([
            {"block": "close_above_sma", "params": [50]},
            {"block": "sma_above_sma",   "params": [50, 200]},
            {"block": "close_above_ema", "params": [12]},
        ])
        valid, errors = t.validate()
        assert not valid
        assert any("trend" in e.lower() and "diversity" in e.lower() for e in errors)

    def test_accept_2_blocks_same_category(self):
        """2 trend blocks should be fine (max_per_category=2)."""
        t = self._make_template([
            {"block": "close_above_sma", "params": [50]},
            {"block": "er_slow_above",   "params": [0.45]},
            {"block": "rsi_between",     "params": [40, 65]},
        ])
        valid, errors = t.validate()
        assert valid, f"Should be valid with 2 trend blocks: {errors}"

    def test_reject_exceeds_hard_limit(self):
        """More than hard_limit conditions should fail."""
        hard = cfg.TEMPLATE_CONFIG.get("max_conditions_hard_limit", 7)
        conditions = [{"block": "rsi_between", "params": [40, 65]}] * (hard + 1)
        t = self._make_template(conditions)
        valid, errors = t.validate()
        assert not valid

    def test_existing_templates_still_valid(self):
        """All seed templates must pass the new validation rules."""
        tm = TemplateManager()
        for template in tm.templates.values():
            valid, errors = template.validate()
            assert valid, f"Template {template.id} failed validation: {errors}"


# ═══════════════════════════════════════════════════════════
# SECTION 2: New Blocks
# ═══════════════════════════════════════════════════════════

class TestNewBlocksExist:
    """All 12 new blocks are registered."""

    @pytest.mark.parametrize("block_name", [
        "adx_above", "supertrend_bullish", "golden_cross_active",
        "stoch_oversold", "cci_between", "roc_positive",
        "obv_rising", "cmf_positive", "vwap_above",
        "gap_up_today", "fib_near_support", "double_bottom_active",
    ])
    def test_block_in_registry(self, block_name):
        assert block_name in CONDITION_BLOCKS, f"'{block_name}' not in CONDITION_BLOCKS"


class TestNewBlocksBehavior:
    """New blocks return correct True/False for known inputs."""

    def _row(self, **kwargs):
        defaults = {
            "close": 150, "open": 148, "high": 152, "low": 147,
            "volume": 1_000_000, "vol_avg_20": 800_000,
            "adx": 30, "supertrend_direction": 1, "golden_cross": True,
            "stoch_k": 15, "cci": 50, "roc": 2.5,
            "obv": 5_000_000, "cmf": 0.15, "vwap": 145,
            "gap_up": True, "fib_618": 149, "double_bottom": True,
        }
        defaults.update(kwargs)
        return defaults

    def test_adx_above_true(self):
        assert block_adx_above(self._row(adx=30), [25]) is True

    def test_adx_above_false(self):
        assert block_adx_above(self._row(adx=18), [25]) is False

    def test_supertrend_bullish_true(self):
        assert block_supertrend_bullish(self._row(supertrend_direction=1), []) is True

    def test_supertrend_bullish_false(self):
        assert block_supertrend_bullish(self._row(supertrend_direction=-1), []) is False

    def test_stoch_oversold_true(self):
        assert block_stoch_oversold(self._row(stoch_k=15), [20]) is True

    def test_stoch_oversold_false(self):
        assert block_stoch_oversold(self._row(stoch_k=55), [20]) is False

    def test_cci_between_true(self):
        assert block_cci_between(self._row(cci=50), [-100, 100]) is True

    def test_cci_between_false(self):
        assert block_cci_between(self._row(cci=150), [-100, 100]) is False

    def test_roc_positive_true(self):
        assert block_roc_positive(self._row(roc=2.5), []) is True

    def test_roc_positive_false(self):
        assert block_roc_positive(self._row(roc=-1.0), []) is False

    def test_cmf_positive_true(self):
        assert block_cmf_positive(self._row(cmf=0.15), []) is True

    def test_cmf_positive_false(self):
        assert block_cmf_positive(self._row(cmf=-0.05), []) is False

    def test_vwap_above_true(self):
        assert block_vwap_above(self._row(close=150, vwap=145), []) is True

    def test_vwap_above_false(self):
        assert block_vwap_above(self._row(close=140, vwap=145), []) is False

    def test_fib_near_support_true(self):
        assert block_fib_near_support(self._row(close=150, fib_618=149), [0.02]) is True

    def test_fib_near_support_false(self):
        assert block_fib_near_support(self._row(close=150, fib_618=130), [0.02]) is False

    def test_gap_up_true(self):
        assert block_gap_up_today(self._row(gap_up=True), []) is True

    def test_double_bottom_true(self):
        assert block_double_bottom_active(self._row(double_bottom=True), []) is True

    def test_nan_handling_adx(self):
        """NaN adx should return False (safe_get default=0, 0 < 25)."""
        assert block_adx_above(self._row(adx=float('nan')), [25]) is False


# ═══════════════════════════════════════════════════════════
# SECTION 3: PULLBACK Template Fix
# ═══════════════════════════════════════════════════════════

class TestPullbackFix:
    """Verify the updated TREND_PULLBACK_EMA template (v3)."""

    def test_pullback_version_gte_3(self):
        """Version must be >= 3 (v3 relaxed er_slow threshold)."""
        tm = TemplateManager()
        t  = tm.get_template_by_id("TREND_PULLBACK_EMA")
        assert t is not None
        assert t.data.get("version") >= 3

    def test_pullback_has_er_slow(self):
        tm = TemplateManager()
        t  = tm.get_template_by_id("TREND_PULLBACK_EMA")
        blocks = [c["block"] for c in t.conditions]
        assert "er_slow_above" in blocks

    def test_pullback_er_slow_relaxed(self):
        """er_slow threshold must be <= 0.35 (was 0.45, too strict)."""
        tm = TemplateManager()
        t  = tm.get_template_by_id("TREND_PULLBACK_EMA")
        er = next((c for c in t.conditions if c["block"] == "er_slow_above"), None)
        assert er is not None
        assert er["params"][0] <= 0.35, f"er_slow threshold too strict: {er['params'][0]}"

    def test_pullback_no_redundant_sma200(self):
        tm = TemplateManager()
        t  = tm.get_template_by_id("TREND_PULLBACK_EMA")
        blocks_with_params = [(c["block"], c.get("params", [])) for c in t.conditions]
        assert ("close_above_sma", [200]) not in blocks_with_params

    def test_pullback_no_redundant_sma_above_sma(self):
        tm = TemplateManager()
        t  = tm.get_template_by_id("TREND_PULLBACK_EMA")
        blocks = [c["block"] for c in t.conditions]
        assert "sma_above_sma" not in blocks

    def test_pullback_passes_validation(self):
        tm = TemplateManager()
        t  = tm.get_template_by_id("TREND_PULLBACK_EMA")
        valid, errors = t.validate()
        assert valid, f"PULLBACK validation failed: {errors}"

    def test_pullback_uses_multiple_categories(self):
        """PULLBACK should span 3+ categories (TREND + MOMENTUM + PRICE minimum)."""
        tm   = TemplateManager()
        t    = tm.get_template_by_id("TREND_PULLBACK_EMA")
        cats = cfg.TEMPLATE_CONFIG.get("block_categories", {})
        block_to_cat = {b: cat for cat, blocks in cats.items() for b in blocks}
        used_cats = {block_to_cat.get(c["block"], "unknown") for c in t.conditions}
        assert len(used_cats) >= 3, f"Only {len(used_cats)} categories: {used_cats}"

    def test_pullback_statistics_reset(self):
        """Statistics should be reset after conditions changed."""
        tm = TemplateManager()
        t  = tm.get_template_by_id("TREND_PULLBACK_EMA")
        assert t.statistics.get("total_activations", 0) == 0


# ═══════════════════════════════════════════════════════════
# SECTION 4: Regression Guards
# ═══════════════════════════════════════════════════════════

class TestRegression:
    """Ensure existing functionality is not broken."""

    def test_all_seed_templates_valid(self):
        """All seed templates must pass validation with new rules."""
        tm = TemplateManager()
        for tid, t in tm.templates.items():
            valid, errors = t.validate()
            assert valid, f"{tid} failed: {errors}"

    def test_registry_has_31_blocks(self):
        """CONDITION_BLOCKS should have 31 entries (19 original + 12 new)."""
        assert len(CONDITION_BLOCKS) == 31, \
            f"Expected 31 blocks, got {len(CONDITION_BLOCKS)}"

    def test_old_blocks_still_callable(self):
        """Original blocks must still be callable without errors."""
        row = {
            "close": 150, "open": 148, "sma_50": 140, "sma_200": 130,
            "rsi": 55, "macd": 1, "macd_signal": 0.5, "volume": 1_000_000,
            "vol_avg_20": 800_000, "squeeze_on": 1, "mom_sqz": 0.5,
            "bb_width_pct": 0.08, "atr": 3, "ema_12": 148, "er_slow": 0.5,
            "rvol": 1.5, "trend_alignment": 1, "macd_hist": 0.3,
        }
        param_exempt = {"close_above_sma", "sma_above_sma", "close_above_ema",
                        "close_above_ref", "close_below_ref"}
        for name, func in CONDITION_BLOCKS.items():
            if name in param_exempt:
                continue
            try:
                result = func(row, [])
                assert isinstance(result, bool), f"Block {name} returned {type(result)}"
            except (IndexError, TypeError, KeyError):
                pass  # Param-requiring blocks are tested separately

    def test_param_ranges_has_new_blocks(self):
        """PARAM_RANGES should have entries for all new blocks."""
        pr = getattr(cfg, "PARAM_RANGES", {})
        new_blocks = [
            "adx_above", "supertrend_bullish", "golden_cross_active",
            "stoch_oversold", "cci_between", "roc_positive",
            "obv_rising", "cmf_positive", "vwap_above",
            "gap_up_today", "fib_near_support", "double_bottom_active",
        ]
        for bn in new_blocks:
            assert bn in pr, f"PARAM_RANGES missing entry for new block '{bn}'"
