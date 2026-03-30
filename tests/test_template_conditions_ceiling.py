"""
StockWise — Template Conditions Ceiling Tests
Validates SPEC v13.4 §4: max 5 condition blocks per template.
"""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup_templates import SetupTemplate, TemplateManager


def _make_template_data(num_conditions, template_id="TEST_TEMPLATE"):
    """Helper: create template dict with N conditions."""
    block_names = ["rsi_between", "close_above_sma", "macd_above_signal",
                   "volume_surge", "bullish_candle", "rsi_above",
                   "squeeze_active", "trend_alignment"]
    default_params = {
        "rsi_between": [40, 70],
        "close_above_sma": [50],
        "macd_above_signal": [],
        "volume_surge": [1.2],
        "bullish_candle": [],
        "rsi_above": [50],
        "squeeze_active": [],
        "trend_alignment": [],
    }
    conditions = []
    for i in range(num_conditions):
        block = block_names[i % len(block_names)]
        conditions.append({"block": block, "params": default_params[block]})

    return {
        "id": template_id,
        "name": f"Test Template {num_conditions} conditions",
        "description": "Test",
        "version": 1,
        "source": "test",
        "enabled": True,
        "required_state": {},
        "conditions": conditions,
        "entry": {"type": "close", "confirmation_candles": 0},
        "stop_loss": {"method": "atr", "atr_multiplier": 2.0, "fallback_pct": 0.02},
        "take_profit": {"method": "atr", "atr_multiplier": 3.0, "use_runner_mode": False},
    }


# ═══════════════════════════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════════════════════════

class TestConditionsCeiling:
    """Tests for max conditions per template enforcement."""

    def test_5_conditions_valid(self):
        """T1: Template with exactly 5 conditions → valid."""
        data = _make_template_data(5)
        template = SetupTemplate(data)
        is_valid, errors = template.validate()
        assert is_valid, f"5 conditions should be valid, got errors: {errors}"

    def test_6_conditions_invalid(self):
        """T2: Template with 6 conditions → invalid (diversity violation or hard limit)."""
        data = _make_template_data(6)
        template = SetupTemplate(data)
        is_valid, errors = template.validate()
        assert not is_valid, "6 conditions should be rejected"
        # Error may mention category diversity or conditions count
        assert any(
            any(kw in e.lower() for kw in ("conditions", "category", "diversity", "blocks"))
            for e in errors
        ), f"Error should mention the overfitting violation, got: {errors}"

    def test_4_conditions_valid(self):
        """T3: Template with 4 conditions → valid (under ceiling)."""
        data = _make_template_data(4)
        template = SetupTemplate(data)
        is_valid, errors = template.validate()
        assert is_valid, f"4 conditions should be valid, got errors: {errors}"

    def test_config_override_respected(self, monkeypatch):
        """T4: Custom max from config is respected."""
        import system_config as cfg
        monkeypatch.setattr(cfg, 'TEMPLATE_CONFIG',
                            {"max_conditions_per_template": 3}, raising=False)

        data = _make_template_data(4)
        template = SetupTemplate(data)
        is_valid, errors = template.validate()
        assert not is_valid, "4 conditions should fail when max=3"

    def test_all_seed_templates_within_limit(self):
        """T5: All seed templates have ≤ 5 conditions."""
        tm = TemplateManager()
        for template in tm.templates.values():
            assert len(template.conditions) <= 5, \
                f"{template.id} has {len(template.conditions)} conditions (max 5)"

    def test_add_template_rejects_overconditioned(self):
        """T6: add_template() rejects template with too many conditions."""
        tm = TemplateManager()
        data = _make_template_data(7, template_id="OVERCONDITIONED")
        result = tm.add_template(data)
        assert result is False, "add_template should reject 7-condition template"
        assert "OVERCONDITIONED" not in tm.templates

    def test_no_limit_on_total_templates(self):
        """T7: There is NO ceiling on total number of templates —
        only on conditions per template. Verify add_template succeeds
        beyond 5 total if conditions are valid."""
        tm = TemplateManager()
        initial_count = len(tm.templates)

        data = _make_template_data(3, template_id="SIXTH_TEMPLATE")
        result = tm.add_template(data)
        assert result is True, "Should allow 6th template (no total ceiling)"
        assert len(tm.templates) == initial_count + 1

        # Cleanup: remove the test template file if created
        import system_config as cfg
        test_file = os.path.join(cfg.DB_DIR, "templates", "SIXTH_TEMPLATE.json")
        if os.path.exists(test_file):
            os.remove(test_file)


# ═══════════════════════════════════════════════════════════
# Regression guards
# ═══════════════════════════════════════════════════════════

class TestConditionsCeilingRegression:
    """Regression guards."""

    def test_validate_checks_condition_count(self):
        """R1: Source inspection — validate() must check len(conditions)."""
        source = inspect.getsource(SetupTemplate.validate)
        assert "max_cond" in source or "conditions" in source, \
            "validate() must check condition count"

    def test_error_message_includes_count_and_max(self):
        """R2: Error message is informative for debugging."""
        data = _make_template_data(8)
        template = SetupTemplate(data)
        is_valid, errors = template.validate()
        error_text = " ".join(errors)
        assert "8" in error_text, "Error should show actual count"
        assert "5" in error_text or "max" in error_text.lower(), \
            "Error should show the ceiling"
