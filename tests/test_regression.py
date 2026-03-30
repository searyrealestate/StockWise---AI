# tests/test_regression.py

"""
StockWise Gen-13 — Regression Guards (TDD v1.1 Section 13)
==========================================================
Source-code inspection tests that protect critical invariants and
architectural constraints. These check code STRUCTURE, not runtime behavior.

All P0 — must pass before any deployment.
Zero mocking, zero API calls — pure file inspection.
"""

import os
import re
import pytest

# Project root — tests/ is one level below project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_source(filename):
    """Read a source file from project root. Skips test if file missing."""
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _read_sources(*filenames):
    """Read multiple source files, return concatenated content."""
    combined = ""
    for fn in filenames:
        path = os.path.join(PROJECT_ROOT, fn)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                combined += f.read() + "\n"
    return combined


class TestRegressionGuards:
    """TDD v1.1 §13: 15 invariant-protection regression guards."""

    # ═══════════════════════════════════════════════════════════
    # RG-01: Waterfall routing active, not single-provider
    # ═══════════════════════════════════════════════════════════
    def test_rg01_waterfall_routing_active(self):
        """RG-01 (P0): DDR #2 — Waterfall routing exists; ≥3 provider downloaders present."""
        source = _read_source("data_source_manager.py")

        assert "def get_stock_data" in source, \
            "get_stock_data() function missing from data_source_manager.py"

        # All four provider downloaders must exist
        downloaders = re.findall(r"def _download_from_(\w+)", source)
        assert len(set(downloaders)) >= 3, (
            f"Expected ≥3 _download_from_X methods in waterfall, found: {set(downloaders)}"
        )

    # ═══════════════════════════════════════════════════════════
    # RG-02: SPY always first in DEFAULT_TRAINING_SYMBOLS
    # ═══════════════════════════════════════════════════════════
    def test_rg02_spy_first_in_defaults(self):
        """RG-02 (P0): Core #2 — DEFAULT_TRAINING_SYMBOLS[0] == 'SPY'."""
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        try:
            import importlib
            import system_config as cfg
            importlib.reload(cfg)  # Ensure fresh load
            symbols = cfg.DEFAULT_TRAINING_SYMBOLS
            assert len(symbols) > 0, "DEFAULT_TRAINING_SYMBOLS is empty"
            assert symbols[0] == "SPY", (
                f"DEFAULT_TRAINING_SYMBOLS[0] = '{symbols[0]}', expected 'SPY'"
            )
        finally:
            if PROJECT_ROOT in sys.path:
                sys.path.remove(PROJECT_ROOT)

    # ═══════════════════════════════════════════════════════════
    # RG-03: No always_in_vip in daily review list
    # ═══════════════════════════════════════════════════════════
    def test_rg03_no_always_in_vip(self):
        """RG-03 (P0): Core #3 — 'always_in_vip' must NOT appear in _update_daily_review_list."""
        source = _read_source("stock_hunter.py")

        match = re.search(
            r"def _update_daily_review_list.*?(?=\n    def |\nclass |\Z)",
            source, re.DOTALL
        )
        body = match.group() if match else source  # Fall back to whole file

        assert "always_in_vip" not in body, (
            "'always_in_vip' found in stock_hunter.py — violates Core #3 "
            "(all stocks must earn VIP position dynamically)"
        )

    # ═══════════════════════════════════════════════════════════
    # RG-04: manage_kinetic_stop returns exactly 3 values
    # ═══════════════════════════════════════════════════════════
    def test_rg04_kinetic_stop_returns_3_values(self):
        """RG-04 (P0): Core #4 — manage_kinetic_stop() every multi-value return = 3 values."""
        source = _read_source("live_trading_engine.py")

        match = re.search(
            r"def manage_kinetic_stop.*?(?=\n    def |\nclass |\Z)",
            source, re.DOTALL
        )
        assert match, "manage_kinetic_stop() not found in live_trading_engine.py"

        func_body = match.group()
        returns = re.findall(r"\breturn\s+(.+)", func_body)
        assert len(returns) > 0, "No return statements found in manage_kinetic_stop()"

        for ret in returns:
            ret_val = ret.strip().rstrip(")")
            clean = ret_val.lstrip("(").strip()
            commas = clean.count(",")
            if commas == 0:
                continue  # Simple return (None, single value) — skip
            assert commas == 2, (
                f"manage_kinetic_stop return has {commas + 1} values (expected 3): "
                f"return {ret.strip()}"
            )

    # ═══════════════════════════════════════════════════════════
    # RG-05: No raw json.load/dump in wave-updated money path files
    # ═══════════════════════════════════════════════════════════
    def test_rg05_json_io_via_safe_layer(self):
        """RG-05 (P0): Core #5 — No raw json.load/dump in wave-updated files (safe_json_io required)."""
        # Scope: files explicitly converted to safe_json_io in Waves 1-4.
        # strategy_engine.py excluded — not yet wave-updated.
        wave_updated = [
            "data_source_manager.py",
            "portfolio_manager.py",
            "live_trading_engine.py",
        ]

        violations = []
        for filename in wave_updated:
            path = os.path.join(PROJECT_ROOT, filename)
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for i, line in enumerate(lines, 1):
                code = line.split("#")[0]  # Strip inline comments
                # json.load( or json.dump( but NOT json.loads( or json.dumps(
                if re.search(r"\bjson\.(load|dump)\s*\(", code) and \
                   not re.search(r"\bjson\.(loads|dumps)\s*\(", code):
                    violations.append(f"{filename}:{i}: {line.rstrip()}")

        assert not violations, (
            "Raw json.load/dump found in wave-updated files "
            "(use safe_json_read/safe_json_write):\n" + "\n".join(violations)
        )

    # ═══════════════════════════════════════════════════════════
    # RG-06: API credentials use defensive getattr fallback
    # ═══════════════════════════════════════════════════════════
    def test_rg06_api_creds_defensive_init(self):
        """RG-06 (P0): Core #6 — API keys accessed via getattr with None fallback."""
        source = _read_source("data_source_manager.py")

        for cred in ["ALPACA_KEY", "ALPACA_SECRET", "MASSIVE_API_KEY"]:
            # Acceptable patterns: getattr(cfg, 'KEY', None) or getattr(cfg, "KEY", None)
            pattern = rf"""getattr\s*\(\s*cfg\s*,\s*['"]{cred}['"]\s*,\s*None\s*\)"""
            assert re.search(pattern, source), (
                f"{cred} not accessed via getattr(..., None) in data_source_manager.py. "
                f"All API credentials must have a None fallback for safe initialization."
            )

    # ═══════════════════════════════════════════════════════════
    # RG-07: No programmatic profit-taking
    # ═══════════════════════════════════════════════════════════
    def test_rg07_no_programmatic_profit_taking(self):
        """RG-07 (P0): Core #7 — No programmatic profit-taking exit patterns in execution code."""
        source = _read_source("live_trading_engine.py")

        # Code-only (strip comments per line before checking)
        code_lines = []
        for line in source.split("\n"):
            code_lines.append(line.split("#")[0])
        code_only = "\n".join(code_lines)

        bad_patterns = [
            r"profit_exit",
            r"exit_at_target",
            r"close_at_profit",
        ]
        for pattern in bad_patterns:
            assert not re.search(pattern, code_only, re.IGNORECASE), (
                f"Programmatic profit-taking pattern '{pattern}' found in "
                f"live_trading_engine.py — exits must be stop-driven only"
            )

    # ═══════════════════════════════════════════════════════════
    # RG-08: FeatureEngine not instantiated per-ticker in scan
    # ═══════════════════════════════════════════════════════════
    def test_rg08_feature_engine_not_per_ticker(self):
        """RG-08 (P0): Core #8 — FeatureEngine() instantiated at startup, not per-ticker."""
        source = _read_source("live_trading_engine.py")

        instantiations = re.findall(r"\bFeatureEngine\s*\(\s*\)", source)
        assert len(instantiations) <= 2, (
            f"FeatureEngine() instantiated {len(instantiations)} times in live_trading_engine.py "
            f"(expected ≤2 — one at startup). "
            f"Per-ticker instantiation wastes memory and initialization cost."
        )

    # ═══════════════════════════════════════════════════════════
    # RG-09: scan_ledger referenced, not read per-ticker
    # ═══════════════════════════════════════════════════════════
    def test_rg09_scan_ledger_not_read_per_ticker(self):
        """RG-09 (P0): Core #9 — scan_ledger.json not read in per-ticker scan loop."""
        source = _read_source("live_trading_engine.py")

        # Count safe_json_read calls that reference scan_ledger
        reads = re.findall(
            r"safe_json_read[^)]*scan_ledger|scan_ledger[^\n]*safe_json_read",
            source
        )
        # ≤2 allowed: one load per cycle + one possible fallback path
        assert len(reads) <= 2, (
            f"scan_ledger.json read {len(reads)} times in live_trading_engine.py "
            f"(expected ≤2 per cycle). Check for per-ticker reads inside scan loop."
        )

    # ═══════════════════════════════════════════════════════════
    # RG-10: Alpha threshold is 0.5%, no 1.3% remnant
    # ═══════════════════════════════════════════════════════════
    def test_rg10_alpha_threshold_05pct(self):
        """RG-10 (P0): DDR #3 — Alpha threshold is 0.5% (0.005). Old 1.3% (0.013) gone."""
        se_source = _read_source("strategy_engine.py")
        cfg_source = _read_source("system_config.py")
        combined = se_source + cfg_source

        # Old threshold must NOT be set as min_net_profit_pct value
        assert not re.search(r"min_net_profit_pct[^#\n]*0\.013", combined), (
            "Old alpha threshold 0.013 (1.3%) still present as min_net_profit_pct — "
            "should be 0.005 (0.5%) per DDR #3"
        )

        # New threshold must be present
        assert "0.005" in combined, (
            "Alpha threshold 0.005 (0.5%) not found in strategy_engine.py or system_config.py"
        )

    # ═══════════════════════════════════════════════════════════
    # RG-11: Phase 4 Runner in KINETIC_STOP_CONFIG
    # ═══════════════════════════════════════════════════════════
    def test_rg11_runner_phase4_in_kinetic_config(self):
        """RG-11 (P0): DDR #4 — Phase 4 Runner params exist in KINETIC_STOP_CONFIG."""
        source = _read_source("system_config.py")

        assert "KINETIC_STOP_CONFIG" in source, \
            "KINETIC_STOP_CONFIG not found in system_config.py"

        # Find the KINETIC_STOP_CONFIG block
        match = re.search(
            r"KINETIC_STOP_CONFIG\s*=\s*\{.*?(?=\n\w|\Z)",
            source, re.DOTALL
        )
        block = match.group() if match else source

        has_runner = any(
            term in block.lower()
            for term in ["runner", "phase_4", "phase4", "runner_atr"]
        )
        assert has_runner, (
            "No Phase 4 Runner reference (runner/phase_4/runner_atr) found in "
            "KINETIC_STOP_CONFIG — DDR #4 requires Phase 4 runner mode params"
        )

    # ═══════════════════════════════════════════════════════════
    # RG-12: No MARKET orders in execution code
    # ═══════════════════════════════════════════════════════════
    def test_rg12_no_market_orders(self):
        """RG-12 (P0): SPEC §5 — No MARKET or MKT order_type assignment in execution code."""
        source = _read_source("live_trading_engine.py")

        violations = []
        for i, line in enumerate(source.split("\n"), 1):
            code = line.split("#")[0]  # Strip comments
            if re.search(r"""order_type\s*=\s*['"]MARKET['"]""", code, re.IGNORECASE) or \
               re.search(r"""order_type\s*=\s*['"]MKT['"]""", code, re.IGNORECASE):
                violations.append(f"line {i}: {line.strip()}")

        assert not violations, (
            "MARKET/MKT order_type found in live_trading_engine.py — "
            "only LIMIT orders allowed (SPEC §5):\n" + "\n".join(violations)
        )

    # ═══════════════════════════════════════════════════════════
    # RG-13: Normalization layer wired for all providers
    # ═══════════════════════════════════════════════════════════
    def test_rg13_normalization_layer_wired(self):
        """RG-13 (P0): SPEC §2 — normalize_ohlcv() exists and called once per provider (≥4 calls)."""
        source = _read_source("data_source_manager.py")

        assert "def normalize_ohlcv" in source, \
            "normalize_ohlcv() not defined in data_source_manager.py"

        # Count calls (excludes the def line via exact call pattern)
        calls = re.findall(r"\bnormalize_ohlcv\s*\(", source)
        # def line matches too, subtract 1
        call_count = len(calls) - 1
        assert call_count >= 4, (
            f"normalize_ohlcv() called {call_count} times (expected ≥4, one per provider: "
            f"MASSIVE, ALPACA, IBKR, YFINANCE)"
        )

    # ═══════════════════════════════════════════════════════════
    # RG-14: Template ceiling ≤5 files and config enforces it
    # ═══════════════════════════════════════════════════════════
    def test_rg14_max_5_templates(self):
        """RG-14 (P0): SPEC §4 — ≤5 template JSON files AND MAX_TEMPLATES in config."""
        templates_dir = os.path.join(PROJECT_ROOT, "data", "templates")
        if os.path.exists(templates_dir):
            template_files = [f for f in os.listdir(templates_dir) if f.endswith(".json")]
            assert len(template_files) <= 5, (
                f"Found {len(template_files)} template JSON files — maximum is 5: "
                f"{template_files}"
            )

        source = _read_source("system_config.py")
        assert "MAX_TEMPLATES" in source, \
            "MAX_TEMPLATES not found in system_config.py — ceiling not enforced"

        # Verify value is 5
        match = re.search(r"MAX_TEMPLATES\s*=\s*(\d+)", source)
        if match:
            assert int(match.group(1)) <= 5, (
                f"MAX_TEMPLATES = {match.group(1)}, expected ≤5"
            )

    # ═══════════════════════════════════════════════════════════
    # RG-15: No single DATA_PROVIDER hardcoded
    # ═══════════════════════════════════════════════════════════
    def test_rg15_no_single_provider_hardcoded(self):
        """RG-15 (P0): DDR #2 — No DATA_PROVIDER = 'X' constant. Waterfall flags used instead."""
        dsm_source = _read_source("data_source_manager.py")
        cfg_source = _read_source("system_config.py")
        combined = dsm_source + cfg_source

        # Strip comments before checking
        code_lines = []
        for line in combined.split("\n"):
            code_lines.append(line.split("#")[0])
        code_only = "\n".join(code_lines)

        bad_patterns = [
            r"""DATA_PROVIDER\s*=\s*['"]ALPACA['"]""",
            r"""DATA_PROVIDER\s*=\s*['"]IBKR['"]""",
            r"""DATA_PROVIDER\s*=\s*['"]YFINANCE['"]""",
            r"""DATA_PROVIDER\s*=\s*['"]MASSIVE['"]""",
        ]
        for pattern in bad_patterns:
            matches = re.findall(pattern, code_only)
            assert not matches, (
                f"Single DATA_PROVIDER hardcoded: {matches} — "
                f"DDR #2 requires EN_MASSIVE/EN_ALPACA/EN_IBKR/EN_YFINANCE waterfall flags"
            )
