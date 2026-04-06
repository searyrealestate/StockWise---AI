# shadow_ledger.py

"""
StockWise Gen-13 Shadow Ledger
==============================
SPEC v13.4 §4: Candle-by-candle template evaluation engine.

Walks through historical data bar-by-bar, evaluates ALL templates at each candle,
records virtual signals, and tracks outcomes (target hit / stop hit).

Runs OFFLINE (weekends) per DDR Part C — does not block nightly scan.
Output: Per-template statistics used by template_matcher for confidence scoring.

Architecture:
  run_full_evaluation(symbols)
    → per symbol: fetch data → calculate_features → evaluate_history()
      → per candle: classify state → match templates → evaluate conditions
        → on SIGNAL: check cooldown → record virtual entry → resolve outcome via lookahead

Phase 2 planned: MTFA (Multi-Timeframe Analysis) will add 4H/1H/15m confluence scoring.
Current version: Daily candles only.
"""

import logging
import math
import pandas as pd
from datetime import datetime
import system_config as cfg
from safe_json_io import safe_json_read, safe_json_write
from setup_templates import TemplateManager

logger = logging.getLogger("ShadowLedger")


class ShadowLedger:
    """
    Candle-by-candle template evaluation engine.
    Tracks 100% of signals virtually — independent of user execution.
    """

    def __init__(self, template_manager=None):
        self.config = getattr(cfg, 'SHADOW_LEDGER_CONFIG', {})
        self.ledger_path = self.config.get('ledger_path', 'data/shadow_ledger.json')
        self.tm = template_manager or TemplateManager()
        self.lookahead = self.config.get('lookahead_candles', 20)
        self.cooldown = self.config.get('min_bars_between_signals', 20)
        self.ledger = self._load_ledger()

    def _load_ledger(self):
        """Load existing ledger or initialize empty."""
        return safe_json_read(self.ledger_path, default={
            "metadata": {"last_run": None, "version": "13.4"},
            "template_stats": {},
        })

    def _save_ledger(self):
        """Persist ledger — updates template_stats and metadata only.
        All other keys (attributions, coverage_gaps, disabled_combos) are
        written by their own methods and must not be overwritten here."""
        self.ledger["metadata"]["last_run"] = datetime.now().isoformat()
        data = safe_json_read(self.ledger_path, default={})
        data["metadata"] = self.ledger["metadata"]
        data["template_stats"] = self.ledger.get("template_stats", {})
        safe_json_write(self.ledger_path, data)

    def evaluate_history(self, symbol, df, stock_state_fn=None, max_date=None, timeframe=None):
        """
        Walk through df candle-by-candle, evaluate all templates at each bar.

        Args:
            symbol: Ticker symbol
            df: Full historical DataFrame with features already calculated
            stock_state_fn: Callable(df_slice) → state dict. If None, skips state filtering.
            max_date: Optional str/date — restrict evaluation to bars on or before this date.
                      Used for TRAIN-only mode in the 3-way split pipeline (DDR #14).

        Returns:
            dict with per-template stats for this symbol
        """
        # ═══ TRAIN PERIOD RESTRICTION (DDR #14) ═══
        if max_date is not None:
            original_len = len(df)
            df = df[df.index <= pd.Timestamp(max_date)]
            logger.info(
                f"[{symbol}] Shadow Ledger restricted to max_date={max_date} "
                f"({len(df)}/{original_len} bars)"
            )

        min_candles = self.config.get('min_candles_for_eval', 200)
        if df is None or len(df) < min_candles:
            logger.debug(
                f"[{symbol}] Insufficient data "
                f"({len(df) if df is not None else 0} < {min_candles})"
            )
            return {}

        templates = self.tm.get_enabled()
        if not templates:
            logger.warning("No enabled templates found")
            return {}

        # Per-template tracking for this symbol
        results = {}
        # Cooldown tracker: {template_id: last_signal_bar_index}
        last_signal_bar = {}

        for t in templates:
            results[t.id] = {
                "signal_count": 0, "wins": 0, "losses": 0,
                "total_pnl_pct": 0.0,
            }
            last_signal_bar[t.id] = -self.cooldown  # Allow first signal immediately

        # Coverage gap tracking — reset per symbol call
        cov_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("coverage_gap", {})
        if cov_cfg.get("enabled", False):
            if not hasattr(self, '_coverage_data'):
                self._coverage_data = {}

        # Walk candle-by-candle (skip first min_candles for indicator warmup)
        eval_end = len(df) - self.lookahead
        for i in range(min_candles, eval_end):
            row = df.iloc[i]

            # Optional: classify state at this candle
            state = {}
            if stock_state_fn:
                try:
                    state = stock_state_fn(df.iloc[:i + 1])
                except Exception:
                    state = {}

            # Filter templates by state if we have one; otherwise use all enabled
            matching = self.tm.get_for_state(state, timeframe=timeframe) if state else templates

            # Coverage gap tracking — record per bar
            if cov_cfg.get("enabled", False) and state:
                try:
                    bar_date = df.index[i]
                    self._record_state_coverage(
                        symbol, state, len(matching),
                        [t.name for t in matching], bar_date
                    )
                except Exception as _cov_exc:
                    logger.debug(f"[{symbol}] Coverage tracking error at bar {i}: {_cov_exc}")

            for template in matching:
                try:
                    # ═══ COOLDOWN CHECK ═══
                    if (i - last_signal_bar[template.id]) < self.cooldown:
                        continue

                    # evaluate_conditions returns (bool, list_of_dicts)
                    all_passed, _details = template.evaluate_conditions(row)

                    # Record block-level statistics (P1 #7A)
                    if not all_passed:
                        try:
                            template.record_block_results(
                                _details, symbol=symbol, all_passed=False,
                                outcome=None
                            )
                        except Exception:
                            pass
                        continue

                    # Signal detected — record virtual entry
                    entry_price = float(row.get('close', 0))
                    if entry_price <= 0:
                        continue

                    stop_loss = template.calculate_stop_loss(row)
                    take_profit = template.calculate_take_profit(row)

                    if stop_loss >= entry_price or take_profit <= entry_price:
                        continue

                    # ═══ MARK COOLDOWN ═══
                    last_signal_bar[template.id] = i

                    # Look ahead to determine outcome
                    outcome = self._resolve_outcome(
                        df, i, entry_price, stop_loss, take_profit
                    )

                    # Record block stats WITH outcome for passed signals
                    try:
                        template.record_block_results(
                            _details, symbol=symbol, all_passed=True,
                            outcome=outcome
                        )
                    except Exception:
                        pass

                    # Record result
                    results[template.id]["signal_count"] += 1
                    if outcome["hit"] == "target":
                        results[template.id]["wins"] += 1
                        results[template.id]["total_pnl_pct"] += outcome["pnl_pct"]
                    elif outcome["hit"] == "stop":
                        results[template.id]["losses"] += 1
                        results[template.id]["total_pnl_pct"] += outcome["pnl_pct"]
                    # "neither" = open after lookahead window — not counted as win or loss

                    # Attribution analytics (SPEC §4)
                    try:
                        self._record_signal_attribution(
                            template, symbol, df, i, outcome,
                            entry_price, stop_loss, take_profit
                        )
                    except Exception as _attr_exc:
                        logger.warning(
                            f"[{symbol}] Attribution failed for {template.id}: {_attr_exc}"
                        )

                    # Trust matrix update (CP-2)
                    try:
                        self._update_trust_matrix(template.id, symbol, state, outcome)
                    except Exception as _trust_exc:
                        logger.debug(
                            f"[{symbol}] Trust matrix update failed for {template.id}: {_trust_exc}"
                        )

                except Exception as e:
                    logger.debug(
                        f"[{symbol}] Template {template.id} eval error at bar {i}: {e}"
                    )
                    continue

        # Calculate derived stats
        for tid, stats in results.items():
            sc = stats["signal_count"]
            if sc > 0:
                stats["win_rate"] = round(stats["wins"] / sc * 100, 1)
                stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / sc, 2)
            else:
                stats["win_rate"] = 0.0
                stats["avg_pnl_pct"] = 0.0

        # Store in ledger
        if "template_stats" not in self.ledger:
            self.ledger["template_stats"] = {}
        self.ledger["template_stats"][symbol] = results

        total_signals = sum(r['signal_count'] for r in results.values())
        logger.info(
            f"[{symbol}] Shadow evaluation complete: "
            f"{total_signals} signals across {len(templates)} templates"
        )
        return results

    def _finalize_coverage_gaps(self):
        """
        Called after all symbols have been evaluated.
        Runs gap analysis, saves report, logs results.
        """
        cov_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("coverage_gap", {})
        if not cov_cfg.get("enabled", False):
            return
        if not getattr(self, '_coverage_data', {}):
            return
        try:
            report = self._analyze_coverage_gaps()
            self.ledger["coverage_gaps"] = self._make_serializable(report)
            self._save_coverage_gaps(report)
            self._log_coverage_report(report)
        except Exception as e:
            logger.error(f"[Coverage] Gap analysis failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # CONTEXTUAL TRUST MATRIX  (CP-2)
    # ═══════════════════════════════════════════════════════════════

    def _load_trust_matrix_from_disk(self):
        """Load trust_matrix from shadow_ledger.json."""
        try:
            data = safe_json_read(self.ledger_path, default={})
            return data.get("trust_matrix", {})
        except Exception:
            return {}

    def _save_trust_matrix_to_disk(self, trust_matrix):
        """Persist trust_matrix — writes only the trust_matrix key."""
        try:
            data = safe_json_read(self.ledger_path, default={})
            data["trust_matrix"] = trust_matrix
            safe_json_write(self.ledger_path, data)
        except Exception as e:
            logger.error(f"[Trust] Failed to save trust matrix: {e}")

    def _calculate_decayed_wr_simple(self, signals):
        """Exponentially decayed win rate (oldest→newest, most recent = highest weight)."""
        ct_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("contextual_trust", {})
        decay = ct_cfg.get("decay_rate", 0.95)
        if not signals:
            return 0.5
        n = len(signals)
        total_w, win_w = 0.0, 0.0
        for i, sig in enumerate(signals):
            w = decay ** (n - 1 - i)
            total_w += w
            if sig.get("won", False):
                win_w += w
        return round(win_w / max(total_w, 1e-9), 4)

    def _determine_lifecycle_simple(self, wins, n, decayed_wr, config):
        """Lifecycle classification matching TemplateMatcher._determine_lifecycle."""
        min_signals = config.get("lifecycle_check_min_signals", 10)
        if n < min_signals:
            return "BURN_IN"
        proven_thr = config.get("proven_wr_threshold", 0.50)
        monitoring_thr = config.get("monitoring_wr_threshold", 0.35)
        degraded_thr = config.get("degraded_wr_threshold", 0.20)
        hysteresis = config.get("hysteresis", 0.05)
        min_proven = config.get("min_signals_for_proven", 20)
        if decayed_wr >= proven_thr - hysteresis and n >= min_proven:
            return "PROVEN"
        elif decayed_wr >= monitoring_thr - hysteresis:
            return "MONITORING"
        elif decayed_wr >= degraded_thr - hysteresis:
            return "DEGRADED"
        return "DISABLED"

    def _update_trust_matrix(self, template_id, symbol, stock_state, outcome):
        """Update trust matrix cell for template+symbol+state after a resolved outcome.

        Called after each resolved signal in evaluate_history().
        Only target/stop outcomes are recorded (not "neither" = still open).

        Args:
            template_id: template ID string
            symbol: stock ticker
            stock_state: state dict with trend/structure/volume/volatility
            outcome: dict with keys 'hit' ("target"/"stop"/"neither") and 'pnl_pct'
        """
        ct_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("contextual_trust", {})
        if not ct_cfg.get("enabled", False):
            return

        hit = outcome.get("hit", "neither")
        if hit not in ("target", "stop"):
            return  # "neither" = trade still open — skip

        try:
            trust_matrix = self._load_trust_matrix_from_disk()

            # Build state key
            trend = stock_state.get("trend", "") if stock_state else ""
            structure = stock_state.get("structure", "") if stock_state else ""
            volume = stock_state.get("volume", "") if stock_state else ""
            volatility = stock_state.get("volatility", "") if stock_state else ""
            state_key = f"{trend}:{structure}:{volume}:{volatility}"

            # Navigate / create nested structure
            if template_id not in trust_matrix:
                trust_matrix[template_id] = {}
            if symbol not in trust_matrix[template_id]:
                trust_matrix[template_id][symbol] = {}
            if state_key not in trust_matrix[template_id][symbol]:
                trust_matrix[template_id][symbol][state_key] = {
                    "signals": [], "wins": 0, "total": 0,
                    "decayed_wr": 0.5, "lifecycle": "BURN_IN",
                }

            cell = trust_matrix[template_id][symbol][state_key]
            won = (hit == "target")
            pnl = outcome.get("pnl_pct", 0.0)

            signal_record = {
                "won": won,
                "pnl_pct": round(float(pnl), 4),
                "timestamp": datetime.now().isoformat(),
            }
            cell["signals"].append(signal_record)

            # Rolling cap: keep last 52 signals (~1 year of weekly signals)
            if len(cell["signals"]) > 52:
                cell["signals"] = cell["signals"][-52:]

            cell["total"] += 1
            if won:
                cell["wins"] += 1

            # Update derived stats
            cell["decayed_wr"] = self._calculate_decayed_wr_simple(cell["signals"])
            cell["lifecycle"] = self._determine_lifecycle_simple(
                cell["wins"], cell["total"], cell["decayed_wr"], ct_cfg
            )

            self._save_trust_matrix_to_disk(trust_matrix)
        except Exception as e:
            logger.warning(f"[Trust] _update_trust_matrix error for {template_id}::{symbol}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # COVERAGE GAP DETECTION  (SPEC §4)
    # ═══════════════════════════════════════════════════════════════

    # ── A. Per-bar state accumulator ─────────────────────────────

    def _record_state_coverage(self, symbol, state_dict, templates_matched,
                               template_names, bar_date):
        """Accumulate state coverage data for one bar (in-memory, not disk)."""
        try:
            trend = state_dict.get("trend", "")
            structure = state_dict.get("structure", "")
            volume = state_dict.get("volume", "")
            volatility = state_dict.get("volatility", "")
            state_key = f"{trend}:{structure}:{volume}:{volatility}"

            cd = self._coverage_data
            if state_key not in cd:
                cd[state_key] = {
                    "bar_count": 0,
                    "covered_count": 0,
                    "symbols": {},
                    "templates_seen": set(),
                    "bars_by_year": {},
                }

            entry = cd[state_key]
            entry["bar_count"] += 1
            if templates_matched > 0:
                entry["covered_count"] += 1
                for t in template_names:
                    entry["templates_seen"].add(t)

            if symbol not in entry["symbols"]:
                entry["symbols"][symbol] = {"total": 0, "covered": 0}
            entry["symbols"][symbol]["total"] += 1
            if templates_matched > 0:
                entry["symbols"][symbol]["covered"] += 1

            try:
                ts = pd.Timestamp(bar_date)
                year = str(ts.year)
            except Exception:
                year = str(bar_date)[:4]
            entry["bars_by_year"][year] = entry["bars_by_year"].get(year, 0) + 1
        except Exception as e:
            logger.debug(f"[Coverage] _record_state_coverage error: {e}")

    # ── B. Gap type classifier ────────────────────────────────────

    def _classify_gap_type(self, state_key, templates_matched_ever, disabled_combos):
        """Return "TRUE_GAP" | "EFFECTIVE_GAP" | "COVERED"."""
        try:
            if templates_matched_ever == 0:
                return "TRUE_GAP"

            # If at least one matching template exists but all combos for this state
            # appear in disabled_combos → EFFECTIVE_GAP
            # Simplified: if covered_count==0 but templates_matched_ever>0 → EFFECTIVE_GAP
            cd = self._coverage_data.get(state_key, {})
            covered = cd.get("covered_count", 0)
            if covered == 0:
                return "EFFECTIVE_GAP"

            return "COVERED"
        except Exception:
            return "TRUE_GAP"

    # ── C. Near-miss finder ───────────────────────────────────────

    def _find_near_miss(self, state_key, all_templates):
        """Find the template closest to covering this state (fewest axis mismatches)."""
        try:
            parts = state_key.split(":")
            if len(parts) != 4:
                return None
            actual = {"trend": parts[0], "structure": parts[1],
                      "volume": parts[2], "volatility": parts[3]}
            AXES = ("trend", "structure", "volume", "volatility")

            best = None
            best_matches = -1

            for template in all_templates:
                req = getattr(template, 'required_state', {})
                if not req:
                    continue
                matching = 0
                blocking = []
                for ax in AXES:
                    req_vals = req.get(ax, [])
                    if not req_vals:
                        matching += 1
                        continue
                    if actual.get(ax, "") in req_vals:
                        matching += 1
                    else:
                        blocking.append({
                            "axis": ax,
                            "required": req_vals,
                            "actual": actual.get(ax, ""),
                        })
                if matching > best_matches and matching >= 2:
                    best_matches = matching
                    fix = f"Add {actual.get(blocking[0]['axis'], '')} to {template.name} {blocking[0]['axis']} requirement" if blocking else ""
                    best = {
                        "closest_template": template.name,
                        "matching_axes": matching,
                        "blocking_fields": blocking,
                        "fix_suggestion": fix,
                    }

            return best
        except Exception as e:
            logger.debug(f"[Coverage] _find_near_miss error: {e}")
            return None

    # ── D. Opportunity scorer ─────────────────────────────────────

    def _calc_opportunity_score(self, state_entry, state_key,
                                total_bars_scanned, total_symbols,
                                recent_cutoff_year):
        """Return 0-1 float scoring how valuable a template for this gap would be."""
        try:
            bar_count = state_entry.get("bar_count", 0)
            symbols = state_entry.get("symbols", {})
            bars_by_year = state_entry.get("bars_by_year", {})

            # Volume score from state key
            volume_part = state_key.split(":")[2] if ":" in state_key else ""
            vol_map = {"HEALTHY": 1.0, "SURGING": 0.8, "SURGE": 0.8,
                       "LOW": 0.3, "DRY": 0.3}
            volume_score = vol_map.get(volume_part, 0.5)

            # Recency: bars in recent period / total
            recent_bars = sum(
                v for k, v in bars_by_year.items()
                if k.isdigit() and int(k) >= recent_cutoff_year
            )
            recency_score = min(recent_bars / max(bar_count, 1), 1.0)

            # Frequency
            frequency_score = min(bar_count / max(total_bars_scanned, 1) * 10, 1.0)

            # Diversity
            diversity_score = len(symbols) / max(total_symbols, 1)

            score = (
                volume_score * 0.3
                + recency_score * 0.3
                + frequency_score * 0.2
                + diversity_score * 0.2
            )
            return round(score, 3)
        except Exception:
            return 0.0

    # ── E. Coverage overlap ───────────────────────────────────────

    def _find_coverage_overlap(self):
        """Identify over-covered and single-coverage states."""
        over_covered = []
        single_coverage = []

        for state_key, entry in self._coverage_data.items():
            tmpl_names = list(entry.get("templates_seen", set()))
            n_templates = len(tmpl_names)
            bars = entry.get("bar_count", 0)

            if n_templates > 1:
                over_covered.append({
                    "state": state_key,
                    "templates": n_templates,
                    "template_names": sorted(tmpl_names),
                    "bars": bars,
                })
            elif n_templates == 1:
                single_coverage.append({
                    "state": state_key,
                    "template": tmpl_names[0],
                    "bars": bars,
                    "risk": "HIGH",
                })

        over_covered.sort(key=lambda x: x["templates"], reverse=True)
        return {"over_covered": over_covered, "single_coverage": single_coverage}

    # ── F. Disable-created gaps ───────────────────────────────────

    def _find_disable_created_gaps(self, disabled_combos):
        """Find states/symbols that lost coverage due to auto-disable."""
        result = []
        try:
            for combo_key in disabled_combos:
                parts = combo_key.split("::")
                if len(parts) != 3:
                    continue
                tmpl_id, symbol, trend = parts

                # Find matching state entries for this symbol+trend
                for state_key, entry in self._coverage_data.items():
                    if not state_key.startswith(trend + ":"):
                        continue
                    sym_data = entry.get("symbols", {}).get(symbol)
                    if sym_data is None:
                        continue

                    tmpl_names = list(entry.get("templates_seen", set()))
                    was_only = (len(tmpl_names) == 0 or
                                (len(tmpl_names) == 1 and tmpl_id in tmpl_names))

                    uncovered_bars = sym_data.get("total", 0) - sym_data.get("covered", 0)
                    result.append({
                        "symbol": symbol,
                        "state": state_key,
                        "disabled_template": tmpl_id,
                        "was_only_template": was_only,
                        "bars_now_uncovered": uncovered_bars,
                        "action": "NEEDS_REPLACEMENT" if was_only else "REDUCED_COVERAGE",
                    })
        except Exception as e:
            logger.debug(f"[Coverage] _find_disable_created_gaps error: {e}")
        return result

    # ── G. Main analysis ─────────────────────────────────────────

    def _analyze_coverage_gaps(self):
        """Build full coverage gap report from accumulated _coverage_data."""
        cov_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("coverage_gap", {})
        min_bars = cov_cfg.get("min_bars_to_report", 50)
        warn_pct = cov_cfg.get("min_gap_pct_to_warn", 0.20)
        alert_pct = cov_cfg.get("min_gap_pct_to_alert", 0.50)
        top_n = cov_cfg.get("report_top_n_gaps", 10)
        recent_months = cov_cfg.get("recent_period_months", 12)

        import datetime as dt_module
        now_year = dt_module.datetime.now().year
        recent_cutoff_year = now_year - (recent_months // 12)

        cd = getattr(self, '_coverage_data', {})
        all_templates = self.tm.get_enabled()

        # Aggregate totals
        total_bars = sum(e["bar_count"] for e in cd.values())
        total_covered = sum(e["covered_count"] for e in cd.values())
        total_uncovered = total_bars - total_covered
        cov_pct = round(total_covered / max(total_bars, 1) * 100, 1)
        all_symbols = set()
        for e in cd.values():
            all_symbols.update(e.get("symbols", {}).keys())
        total_symbols = len(all_symbols)

        # Load disabled combos
        try:
            from template_matcher import TemplateMatcher as _TM
            _tm_inst = _TM.__new__(_TM)
            _tm_inst.tm = self.tm
            disabled_combos = _tm_inst._load_disable_list()
        except Exception:
            disabled_combos = set()

        # ── Gaps by state ─────────────────────────────────────────
        gaps_by_state = []
        state_distribution = {}

        for state_key, entry in cd.items():
            bar_count = entry["bar_count"]
            covered = entry["covered_count"]
            tmpl_names = list(entry["templates_seen"])

            # State distribution (all states)
            state_distribution[state_key] = {
                "bars": bar_count,
                "covered": covered > 0,
                "templates": len(tmpl_names),
                "template_names": sorted(tmpl_names),
            }

            # Gap candidates only
            if covered == 0 and bar_count >= min_bars:
                gap_type = self._classify_gap_type(
                    state_key, len(tmpl_names), disabled_combos
                )

                # Temporal
                recent_bars = sum(
                    v for k, v in entry["bars_by_year"].items()
                    if k.isdigit() and int(k) >= recent_cutoff_year
                )
                temporal = dict(entry["bars_by_year"])
                temporal["recent_12m_bars"] = recent_bars
                temporal["recent_12m_pct"] = round(recent_bars / max(bar_count, 1) * 100, 1)

                # Near-miss
                near_miss = None
                if cov_cfg.get("track_near_miss"):
                    near_miss = self._find_near_miss(state_key, all_templates)

                # Opportunity score
                opp_score = self._calc_opportunity_score(
                    entry, state_key, total_bars, total_symbols, recent_cutoff_year
                ) if cov_cfg.get("track_opportunity_score") else 0.0

                # Disabled template info
                disabled_tmpl = None
                if gap_type == "EFFECTIVE_GAP":
                    for ck in disabled_combos:
                        parts = ck.split("::")
                        if len(parts) == 3 and parts[0] in tmpl_names:
                            disabled_tmpl = parts[0]
                            break

                sym_bar_counts = {
                    sym: sdata["total"]
                    for sym, sdata in entry["symbols"].items()
                }

                gaps_by_state.append({
                    "state": state_key,
                    "gap_type": gap_type,
                    "bar_count": bar_count,
                    "pct_of_total": round(bar_count / max(total_bars, 1) * 100, 1),
                    "symbols_affected": sorted(entry["symbols"].keys()),
                    "symbol_bar_counts": sym_bar_counts,
                    "temporal": temporal,
                    "near_miss": near_miss,
                    "opportunity_score": opp_score,
                    "disabled_template": disabled_tmpl,
                    "disabled_wr": None,
                })

        # Sort: opportunity_score desc, then bar_count desc
        gaps_by_state.sort(key=lambda g: (-g["opportunity_score"], -g["bar_count"]))
        gaps_by_state = gaps_by_state[:top_n]

        # ── Gaps by symbol ────────────────────────────────────────
        sym_totals = {}
        sym_uncovered = {}
        sym_state_uncovered = {}

        for state_key, entry in cd.items():
            for sym, sdata in entry["symbols"].items():
                sym_totals[sym] = sym_totals.get(sym, 0) + sdata["total"]
                unc = sdata["total"] - sdata["covered"]
                if unc > 0:
                    sym_uncovered[sym] = sym_uncovered.get(sym, 0) + unc
                    if sym not in sym_state_uncovered:
                        sym_state_uncovered[sym] = []
                    sym_state_uncovered[sym].append({"state": state_key, "bars": unc})

        gaps_by_symbol = []
        for sym in all_symbols:
            total_s = sym_totals.get(sym, 0)
            unc_s = sym_uncovered.get(sym, 0)
            cov_s_pct = round((total_s - unc_s) / max(total_s, 1) * 100, 1)
            unc_pct = 1.0 - (cov_s_pct / 100.0)

            if unc_pct >= alert_pct:
                alert_level = "ALERT"
            elif unc_pct >= warn_pct:
                alert_level = "WARNING"
            else:
                alert_level = "OK"

            states_unc = sorted(
                sym_state_uncovered.get(sym, []),
                key=lambda x: -x["bars"]
            )
            dominant = states_unc[0]["state"] if states_unc else ""

            gaps_by_symbol.append({
                "symbol": sym,
                "total_bars": total_s,
                "uncovered_bars": unc_s,
                "coverage_pct": cov_s_pct,
                "alert_level": alert_level,
                "dominant_uncovered_state": dominant,
                "states_uncovered": states_unc,
            })

        gaps_by_symbol.sort(key=lambda x: -x["uncovered_bars"])

        # ── Overlap + disable gaps ────────────────────────────────
        coverage_overlap = self._find_coverage_overlap() if cov_cfg.get("track_overlap") else {}
        disable_gaps = self._find_disable_created_gaps(disabled_combos) if cov_cfg.get("track_disable_created_gaps") else []

        # ── Recommendations ───────────────────────────────────────
        recommendations = []
        priority = 1

        # REPLACE_DISABLED — urgent
        for dcg in disable_gaps:
            if dcg["was_only_template"]:
                recommendations.append({
                    "priority": priority,
                    "action": "REPLACE_DISABLED",
                    "target_state": dcg["state"],
                    "opportunity_bars": dcg["bars_now_uncovered"],
                    "opportunity_score": 1.0,
                    "symbols": [dcg["symbol"]],
                    "reason": f"Auto-disable of {dcg['disabled_template']} created coverage gap",
                    "near_miss": None,
                })
                priority += 1

        # CREATE/MODIFY from gap analysis
        for gap in gaps_by_state:
            nm = gap.get("near_miss")
            score = gap["opportunity_score"]

            if nm and len(nm.get("blocking_fields", [])) == 1:
                recommendations.append({
                    "priority": priority,
                    "action": "MODIFY_TEMPLATE",
                    "target_state": gap["state"],
                    "opportunity_bars": gap["bar_count"],
                    "opportunity_score": score,
                    "symbols": gap["symbols_affected"],
                    "reason": nm.get("fix_suggestion", "Extend one axis of existing template"),
                    "near_miss": nm,
                })
            elif score > 0.5 and nm:
                recommendations.append({
                    "priority": priority,
                    "action": "CREATE_TEMPLATE",
                    "target_state": gap["state"],
                    "opportunity_bars": gap["bar_count"],
                    "opportunity_score": score,
                    "symbols": gap["symbols_affected"],
                    "reason": f"Feasible: close to {nm['closest_template']} but needs new template",
                    "near_miss": nm,
                })
            elif score > 0.5:
                recommendations.append({
                    "priority": priority,
                    "action": "CREATE_TEMPLATE",
                    "target_state": gap["state"],
                    "opportunity_bars": gap["bar_count"],
                    "opportunity_score": score,
                    "symbols": gap["symbols_affected"],
                    "reason": "High-opportunity gap with no close template match",
                    "near_miss": None,
                })
            priority += 1

        recommendations.sort(key=lambda r: r["priority"])

        report = {
            "last_analysis": datetime.now().date().isoformat(),
            "total_bars_scanned": total_bars,
            "total_bars_covered": total_covered,
            "total_bars_uncovered": total_uncovered,
            "coverage_pct": cov_pct,
            "gaps_by_state": gaps_by_state,
            "gaps_by_symbol": gaps_by_symbol,
            "state_distribution": state_distribution,
            "coverage_overlap": coverage_overlap,
            "disable_created_gaps": disable_gaps,
            "recommendations": recommendations,
            "history": [],
        }
        return report

    # ── H. Save coverage gaps ─────────────────────────────────────

    def _save_coverage_gaps(self, report):
        """Persist coverage gap report to shadow_ledger.json, append history."""
        try:
            data = safe_json_read(self.ledger_path, default={})

            # Append history entry (max 52 weekly entries)
            history = data.get("coverage_gaps", {}).get("history", [])
            history.append({
                "date": report["last_analysis"],
                "coverage_pct": report["coverage_pct"],
                "uncovered_states": len(report["gaps_by_state"]),
            })
            if len(history) > 52:
                history = history[-52:]

            # Convert sets to lists for JSON serialization (safety)
            report_serializable = self._make_serializable(report)
            report_serializable["history"] = history

            data["coverage_gaps"] = report_serializable
            safe_json_write(self.ledger_path, data)
        except Exception as e:
            logger.error(f"[Coverage] _save_coverage_gaps failed: {e}")

    @staticmethod
    def _make_serializable(obj):
        """Recursively convert sets → sorted lists for JSON serialization."""
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, dict):
            return {k: ShadowLedger._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ShadowLedger._make_serializable(i) for i in obj]
        return obj

    # ── I. Log coverage report ────────────────────────────────────

    def _log_coverage_report(self, report):
        """Emit structured log lines for the coverage gap report."""
        total = report.get("total_bars_scanned", 0)
        covered = report.get("total_bars_covered", 0)
        uncovered = report.get("total_bars_uncovered", 0)
        cov_pct = report.get("coverage_pct", 0.0)
        gaps = report.get("gaps_by_state", [])
        top = gaps[0] if gaps else {}

        logger.info(
            f"[COVERAGE-SUMMARY] total_bars={total} | "
            f"covered={covered} ({cov_pct / 100:.1%}) | "
            f"uncovered={uncovered} | "
            f"gap_states={len(gaps)} | "
            f"top_gap={top.get('state', 'N/A')} "
            f"({top.get('bar_count', 0)} bars, score={top.get('opportunity_score', 0):.2f})"
        )

        for gap in gaps:
            nm = gap.get("near_miss") or {}
            logger.info(
                f"[COVERAGE-GAP] state={gap['state']} | "
                f"type={gap['gap_type']} | "
                f"bars={gap['bar_count']} | "
                f"pct={gap['pct_of_total']:.1f}% | "
                f"symbols={gap['symbols_affected']} | "
                f"recent_12m={gap['temporal'].get('recent_12m_bars', 0)} | "
                f"score={gap['opportunity_score']:.2f} | "
                f"near_miss={nm.get('closest_template', 'NONE')}"
            )

        for sym in report.get("gaps_by_symbol", []):
            if sym["alert_level"] != "OK":
                unc = sym["uncovered_bars"]
                tot = sym["total_bars"]
                logger.warning(
                    f"[COVERAGE-GAP] symbol={sym['symbol']} | "
                    f"uncovered={unc}/{tot} ({100 - sym['coverage_pct']:.1f}%) | "
                    f"dominant_state={sym['dominant_uncovered_state']} | "
                    f"status={sym['alert_level']}"
                )

        for dcg in report.get("disable_created_gaps", []):
            if dcg.get("was_only_template"):
                logger.warning(
                    f"[COVERAGE-GAP] DISABLE_CREATED | "
                    f"symbol={dcg['symbol']} | state={dcg['state']} | "
                    f"disabled={dcg['disabled_template']} | "
                    f"was_only_template=True | "
                    f"bars_uncovered={dcg['bars_now_uncovered']} | "
                    f"action=NEEDS_REPLACEMENT"
                )

        for rec in report.get("recommendations", [])[:5]:
            logger.info(
                f"[COVERAGE-RECOMMEND] priority={rec['priority']} | "
                f"action={rec['action']} | "
                f"state={rec['target_state']} | "
                f"bars={rec['opportunity_bars']} | "
                f"score={rec['opportunity_score']:.2f} | "
                f"symbols={rec['symbols']} | "
                f"reason={rec['reason']}"
            )

    def _resolve_outcome(self, df, entry_idx, entry_price, stop_loss, take_profit):
        """
        Look ahead from entry candle to determine if target or stop was hit first.
        Stop is checked FIRST per conservative evaluation (worst-case intra-bar assumption).
        """
        end_idx = min(entry_idx + self.lookahead + 1, len(df))
        for j in range(entry_idx + 1, end_idx):
            candle = df.iloc[j]
            low = float(candle.get('low', entry_price))
            high = float(candle.get('high', entry_price))

            # Check stop first (conservative)
            if low <= stop_loss:
                pnl = (stop_loss - entry_price) / entry_price * 100
                return {"hit": "stop", "pnl_pct": round(pnl, 2), "bars": j - entry_idx}

            # Then check target
            if high >= take_profit:
                pnl = (take_profit - entry_price) / entry_price * 100
                return {"hit": "target", "pnl_pct": round(pnl, 2), "bars": j - entry_idx}

        return {"hit": "neither", "pnl_pct": 0.0, "bars": self.lookahead}

    # ═══════════════════════════════════════════════════════════════
    # ATTRIBUTION ANALYTICS  (SPEC §4)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_float(val, ndigits=4):
        """Return float rounded to ndigits, or None for NaN/Inf/invalid."""
        try:
            v = float(val)
            return None if (math.isnan(v) or math.isinf(v)) else round(v, ndigits)
        except (TypeError, ValueError):
            return None

    # ── A. Kill candle classification ─────────────────────────────

    def _classify_kill_type(self, prev_close, open_price, high, low, close, stop_price):
        """Return "gap_down" | "wick" | "drift" | "reversal"."""
        try:
            sf = self._safe_float
            pc = sf(prev_close); op = sf(open_price)
            lo = sf(low); cl = sf(close); sp = sf(stop_price)
            if None in (pc, op, lo, cl, sp):
                return "reversal"

            gap_pct = (op - pc) / pc * 100 if pc != 0 else 0
            if gap_pct < -0.5 and op <= sp:
                return "gap_down"

            body = cl - op
            tail = min(op, cl) - lo
            if abs(body) > 0 and tail > abs(body) * 2:
                return "wick"

            if op != 0 and abs(cl - op) / op * 100 < 0.3:
                return "drift"

            return "reversal"
        except Exception:
            return "reversal"

    def _build_kill_candle_data(self, bars, entry_idx, exit_idx, entry_price, stop_price):
        """Build kill candle analytics dict for a losing trade."""
        try:
            sf = self._safe_float
            if exit_idx is None or exit_idx >= len(bars):
                return None

            kb = bars.iloc[exit_idx]
            pb = bars.iloc[exit_idx - 1] if exit_idx > 0 else kb

            op = sf(kb.get('open')); cl = sf(kb.get('close'))
            hi = sf(kb.get('high')); lo = sf(kb.get('low'))
            pc = sf(pb.get('close'))
            vol = sf(kb.get('volume')); va = sf(kb.get('vol_avg_20'))

            kill_type = self._classify_kill_type(pc, op, hi, lo, cl, stop_price)

            body_pct = sf((cl - op) / op * 100) if op else None
            wick_pct = sf((hi - max(op or 0, cl or 0)) / op * 100) if op and hi and cl else None
            tail_pct = sf((min(op or 0, cl or 0) - lo) / op * 100) if op and lo and cl else None
            gap_pct = sf((op - pc) / pc * 100) if pc and pc != 0 and op else None
            vol_ratio = sf(vol / va) if vol and va and va != 0 else None

            max_fav_pct = None
            ep = self._safe_float(entry_price)
            if ep:
                highs = [sf(bars.iloc[j].get('high')) for j in range(entry_idx + 1, exit_idx + 1)
                         if sf(bars.iloc[j].get('high')) is not None]
                if highs:
                    max_fav_pct = sf((max(highs) - ep) / ep * 100)

            bars_in_trade = exit_idx - entry_idx if entry_idx is not None else None
            stop_dist_pct = sf((stop_price - entry_price) / entry_price * 100) if entry_price else None

            return {
                "kill_type": kill_type,
                "candle_body_pct": body_pct,
                "candle_wick_pct": wick_pct,
                "candle_tail_pct": tail_pct,
                "gap_pct": gap_pct,
                "volume_ratio": vol_ratio,
                "phase_at_death": None,
                "bars_in_trade": bars_in_trade,
                "max_favorable_pct": max_fav_pct,
                "stop_distance_pct": stop_dist_pct,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_kill_candle_data failed: {e}")
            return None

    # ── C. Entry quality ──────────────────────────────────────────

    def _build_entry_quality(self, bars, entry_idx, entry_price):
        """Score the quality of entry timing."""
        try:
            sf = self._safe_float
            eb = bars.iloc[entry_idx]
            ep = sf(entry_price)
            bar_low = sf(eb.get('low'))
            bar_open = sf(eb.get('open'))

            entry_vs_low = sf((ep - bar_low) / bar_low * 100) if ep and bar_low else None
            entry_vs_open = sf((ep - bar_open) / bar_open * 100) if ep and bar_open else None

            # Bars to first profitable close
            bars_to_profit = None
            for j in range(entry_idx + 1, min(entry_idx + 21, len(bars))):
                cl = sf(bars.iloc[j].get('close'))
                if cl and ep and cl > ep:
                    bars_to_profit = j - entry_idx
                    break

            # Max drawdown in first 3 bars
            imm_dd = None
            lows = [sf(bars.iloc[j].get('low')) for j in range(entry_idx + 1, min(entry_idx + 4, len(bars)))
                    if sf(bars.iloc[j].get('low')) is not None]
            if lows and ep:
                imm_dd = sf((min(lows) - ep) / ep * 100)

            return {
                "entry_vs_low_pct": entry_vs_low,
                "entry_vs_open_pct": entry_vs_open,
                "bars_to_first_profit": bars_to_profit,
                "immediate_drawdown_pct": imm_dd,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_entry_quality failed: {e}")
            return None

    # ── D. Volume profile ─────────────────────────────────────────

    def _build_volume_profile(self, bars, entry_idx, exit_idx):
        """Volume ratios at entry/exit and trend across the trade."""
        try:
            sf = self._safe_float
            clamp_exit = min(exit_idx, len(bars) - 1)
            eb = bars.iloc[entry_idx]
            xb = bars.iloc[clamp_exit]

            va_e = sf(eb.get('vol_avg_20'))
            vol_at_entry = sf(sf(eb.get('volume')) / va_e) if sf(eb.get('volume')) and va_e else None

            va_x = sf(xb.get('vol_avg_20')) or va_e
            vol_at_exit = sf(sf(xb.get('volume')) / va_x) if sf(xb.get('volume')) and va_x else None

            ratios = []
            for j in range(entry_idx, clamp_exit + 1):
                v = sf(bars.iloc[j].get('volume'))
                va = sf(bars.iloc[j].get('vol_avg_20')) or va_e
                if v and va:
                    ratios.append(v / va)

            avg_vol = sf(sum(ratios) / len(ratios)) if ratios else None

            trend = "flat"
            if len(ratios) >= 2:
                mid = len(ratios) // 2
                f_avg = sum(ratios[:mid]) / mid if mid else 0
                s_avg = sum(ratios[mid:]) / (len(ratios) - mid) if (len(ratios) - mid) else 0
                if f_avg and s_avg > f_avg * 1.2:
                    trend = "increasing"
                elif f_avg and s_avg < f_avg * 0.8:
                    trend = "decreasing"

            return {
                "volume_at_entry": vol_at_entry,
                "volume_at_exit": vol_at_exit,
                "avg_volume_during_trade": avg_vol,
                "volume_trend": trend,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_volume_profile failed: {e}")
            return None

    # ── E. Market context (SPY) ───────────────────────────────────

    def _build_market_context(self, spy_bars, entry_idx, exit_idx):
        """SPY return on exit day and during trade. Returns None if no SPY data."""
        if spy_bars is None:
            return None
        try:
            sf = self._safe_float
            clamp = min(exit_idx, len(spy_bars) - 1)
            if entry_idx >= len(spy_bars):
                return None

            eb = spy_bars.iloc[entry_idx]
            xb = spy_bars.iloc[clamp]

            spy_open_x = sf(xb.get('open'))
            spy_close_x = sf(xb.get('close'))
            spy_day = sf((spy_close_x - spy_open_x) / spy_open_x * 100) if spy_open_x else None

            spy_entry_cl = sf(eb.get('close'))
            spy_trade = sf((spy_close_x - spy_entry_cl) / spy_entry_cl * 100) if spy_entry_cl else None

            spy_trend = "BULLISH" if (spy_trade is not None and spy_trade > 0) else "BEARISH"

            return {
                "spy_return_on_day": spy_day,
                "spy_return_during_trade": spy_trade,
                "spy_trend": spy_trend,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_market_context failed: {e}")
            return None

    # ── F. Indicator snapshot ─────────────────────────────────────

    def _build_indicator_snapshot(self, bars, entry_idx, exit_idx):
        """RSI/ER/ATR/BB/ADX at entry, exit, and delta between them."""
        try:
            sf = self._safe_float
            COLS = ['rsi', 'er_fast', 'er_slow', 'atr', 'bb_width', 'volume_ratio', 'adx']
            clamp = min(exit_idx, len(bars) - 1)
            eb = bars.iloc[entry_idx]
            xb = bars.iloc[clamp]

            def read(row):
                return {c: sf(row.get(c)) for c in COLS}

            at_entry = read(eb)
            at_exit = read(xb)
            delta = {
                c: sf(at_entry[c] - at_exit[c]) if at_entry[c] is not None and at_exit[c] is not None else None
                for c in COLS
            }
            return {"at_entry": at_entry, "at_exit": at_exit, "delta": delta}
        except Exception as e:
            logger.warning(f"[Attribution] _build_indicator_snapshot failed: {e}")
            return None

    # ── G. Weakest block ──────────────────────────────────────────

    def _compute_block_margin(self, block_name, row, params):
        """Margin between actual indicator value and block threshold (larger = safer)."""
        sf = self._safe_float

        if block_name == 'rsi_between' and len(params) >= 2:
            rsi = sf(row.get('rsi'))
            if rsi is None: return None
            return min(rsi - params[0], params[1] - rsi)

        if block_name in ('er_slow_above', 'er_fast_above') and params:
            col = 'er_slow' if block_name == 'er_slow_above' else 'er_fast'
            v = sf(row.get(col))
            return (v - params[0]) if v is not None else None

        if block_name == 'close_above_sma' and params:
            close = sf(row.get('close'))
            sma = sf(row.get(f'sma_{int(params[0])}'))
            return (close - sma) if close is not None and sma is not None else None

        if block_name == 'volume_surge' and params:
            vol = sf(row.get('volume')); va = sf(row.get('vol_avg_20'))
            if vol is None or va is None or va == 0: return None
            return (vol / va) - params[0]

        if block_name == 'macd_above_signal':
            macd = sf(row.get('macd')); sig = sf(row.get('macd_signal'))
            return (macd - sig) if macd is not None and sig is not None else None

        if block_name == 'adx_above' and params:
            adx = sf(row.get('adx'))
            return (adx - params[0]) if adx is not None else None

        if block_name == 'sma_above_sma' and len(params) >= 2:
            fast = sf(row.get(f'sma_{int(params[0])}'))
            slow = sf(row.get(f'sma_{int(params[1])}'))
            return (fast - slow) if fast is not None and slow is not None else None

        return None

    def _build_weakest_block(self, template, bars, entry_idx):
        """Find the condition block with the smallest margin to its threshold."""
        try:
            row = bars.iloc[entry_idx]
            min_margin = float('inf')
            weakest = None

            for block_spec in template.conditions:
                block_name = block_spec.get('block', '')
                params = block_spec.get('params', [])
                try:
                    margin = self._compute_block_margin(block_name, row, params)
                    if margin is None or math.isnan(float(margin)):
                        continue
                    margin = float(margin)
                    if margin < min_margin:
                        min_margin = margin
                        threshold = params[0] if params else 0
                        weakest = {
                            "block_name": block_name,
                            "value_at_entry": self._safe_float(margin + float(threshold)) if isinstance(threshold, (int, float)) else None,
                            "threshold": self._safe_float(threshold) if isinstance(threshold, (int, float)) else None,
                            "margin": self._safe_float(margin),
                        }
                except Exception:
                    continue

            return weakest
        except Exception as e:
            logger.warning(f"[Attribution] _build_weakest_block failed: {e}")
            return None

    # ── H. Risk/Reward ────────────────────────────────────────────

    def _build_risk_reward(self, entry_price, stop_price, target_price,
                           exit_price, bars, entry_idx, exit_idx):
        """Planned vs realized R:R and max favorable excursion."""
        try:
            sf = self._safe_float
            ep = sf(entry_price); sp = sf(stop_price)
            tp = sf(target_price); xp = sf(exit_price)

            risk = (ep - sp) if ep and sp else None
            reward = (tp - ep) if tp and ep else None

            planned_rr = sf(reward / risk) if risk and risk != 0 else None
            realized_rr = sf((xp - ep) / risk) if xp and ep and risk and risk != 0 else None
            tgt_dist = sf((tp - ep) / ep * 100) if tp and ep else None
            stop_dist = sf((ep - sp) / ep * 100) if ep and sp else None

            clamp = min(exit_idx, len(bars) - 1)
            highs = [sf(bars.iloc[j].get('high')) for j in range(entry_idx + 1, clamp + 1)
                     if sf(bars.iloc[j].get('high')) is not None]
            max_fav = None; max_fav_rr = None
            if highs and ep:
                max_fav = sf((max(highs) - ep) / ep * 100)
                if risk and risk != 0:
                    max_fav_rr = sf((max(highs) - ep) / risk)

            return {
                "planned_rr": planned_rr,
                "realized_rr": realized_rr,
                "target_distance_pct": tgt_dist,
                "stop_distance_pct": stop_dist,
                "max_favorable_pct": max_fav,
                "max_favorable_rr": max_fav_rr,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_risk_reward failed: {e}")
            return None

    # ── I. Time context ───────────────────────────────────────────

    def _build_time_context(self, bars, entry_idx, exit_idx):
        """Day-of-week, dates, and bars/calendar days in trade."""
        try:
            idx = bars.index
            entry_ts = idx[entry_idx]
            clamp = min(exit_idx, len(bars) - 1)
            exit_ts = idx[clamp]

            entry_dt = pd.Timestamp(entry_ts)
            exit_dt = pd.Timestamp(exit_ts)
            cal_days = (exit_dt - entry_dt).days

            return {
                "entry_day_of_week": entry_dt.day_name(),
                "entry_date": str(entry_dt.date()),
                "exit_date": str(exit_dt.date()),
                "bars_in_trade": clamp - entry_idx,
                "calendar_days_in_trade": cal_days,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_time_context failed: {e}")
            return None

    # ── J. Preceding candles ──────────────────────────────────────

    def _build_preceding_candles(self, bars, entry_idx):
        """Multi-window price action analysis before entry bar."""
        try:
            sf = self._safe_float
            evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("attribution", {})
            windows = evo_cfg.get("preceding_candle_windows", [3, 5, 10])
            result = {"windows": windows}

            for w in windows:
                start = entry_idx - w
                if start < 0:
                    result[f"window_{w}"] = None
                    continue

                slice_df = bars.iloc[start:entry_idx]
                if len(slice_df) == 0:
                    result[f"window_{w}"] = None
                    continue

                opens = [sf(r.get('open')) for _, r in slice_df.iterrows()]
                closes = [sf(r.get('close')) for _, r in slice_df.iterrows()]
                highs = [sf(r.get('high')) for _, r in slice_df.iterrows()]
                lows = [sf(r.get('low')) for _, r in slice_df.iterrows()]
                vols = [sf(r.get('volume')) for _, r in slice_df.iterrows()]
                vas = [sf(r.get('vol_avg_20')) for _, r in slice_df.iterrows()]

                # Valid pairs only
                valid = [(o, c) for o, c in zip(opens, closes) if o and c]
                green_count = sum(1 for o, c in valid if c > o)
                green_pct = green_count / len(valid) if valid else 0.5

                if green_pct > 0.7:
                    pattern = "bullish"
                elif green_pct < 0.3:
                    pattern = "bearish"
                elif all(o and c and abs(c - o) / o * 100 < 0.2 for o, c in valid if o):
                    pattern = "doji_sequence"
                else:
                    pattern = "mixed"

                first_close = next((sf(r.get('close')) for _, r in slice_df.iterrows()
                                    if sf(r.get('close'))), None)
                last_close = next((sf(r.get('close')) for _, r in reversed(list(slice_df.iterrows()))
                                   if sf(r.get('close'))), None)
                momentum_pct = sf((last_close - first_close) / first_close * 100) if first_close and last_close else None

                # Volume trend
                vol_ratios = [v / va for v, va in zip(vols, vas) if v and va and va != 0]
                v_trend = "flat"
                if len(vol_ratios) >= 2:
                    mid = len(vol_ratios) // 2
                    fa = sum(vol_ratios[:mid]) / mid if mid else 0
                    sa = sum(vol_ratios[mid:]) / (len(vol_ratios) - mid) if (len(vol_ratios) - mid) else 0
                    if fa and sa > fa * 1.2:
                        v_trend = "increasing"
                    elif fa and sa < fa * 0.8:
                        v_trend = "decreasing"

                all_highs = [h for h in highs if h]
                all_lows = [l for l in lows if l]
                ep = sf(bars.iloc[entry_idx].get('close'))
                hh_dist = sf((ep - max(all_highs)) / max(all_highs) * 100) if all_highs and ep else None
                ll_dist = sf((ep - min(all_lows)) / min(all_lows) * 100) if all_lows and ep else None

                bodies = [abs(c - o) / o * 100 for o, c in valid if o]
                avg_body = sf(sum(bodies) / len(bodies)) if bodies else None

                result[f"window_{w}"] = {
                    "pattern": pattern,
                    "momentum_pct": momentum_pct,
                    "volume_trend": v_trend,
                    "highest_high_dist_pct": hh_dist,
                    "lowest_low_dist_pct": ll_dist,
                    "green_candle_pct": sf(green_pct * 100),
                    "avg_body_pct": avg_body,
                }

            return result
        except Exception as e:
            logger.warning(f"[Attribution] _build_preceding_candles failed: {e}")
            return None

    # ── K. Key levels ─────────────────────────────────────────────

    def _build_key_levels(self, bars, entry_idx, entry_price):
        """Distance to SMA50/200 and recent swing high/low (20-bar lookback)."""
        try:
            sf = self._safe_float
            ep = sf(entry_price)
            row = bars.iloc[entry_idx]

            sma50 = sf(row.get('sma_50'))
            sma200 = sf(row.get('sma_200'))
            dist_sma50 = sf((ep - sma50) / sma50 * 100) if ep and sma50 else None
            dist_sma200 = sf((ep - sma200) / sma200 * 100) if ep and sma200 else None

            lookback = min(20, entry_idx)
            swing_slice = bars.iloc[max(0, entry_idx - lookback):entry_idx]
            resist = support = None
            if len(swing_slice):
                highs = [sf(r.get('high')) for _, r in swing_slice.iterrows() if sf(r.get('high'))]
                lows = [sf(r.get('low')) for _, r in swing_slice.iterrows() if sf(r.get('low'))]
                resist = max(highs) if highs else None
                support = min(lows) if lows else None

            dist_res = sf((ep - resist) / resist * 100) if ep and resist else None
            dist_sup = sf((ep - support) / support * 100) if ep and support else None

            return {
                "distance_to_resistance_pct": dist_res,
                "distance_to_support_pct": dist_sup,
                "distance_to_sma200_pct": dist_sma200,
                "distance_to_sma50_pct": dist_sma50,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_key_levels failed: {e}")
            return None

    # ── L. Concurrent signals ─────────────────────────────────────

    def _build_concurrent_signals(self, template_name, symbol, signal_date, all_signals_cache):
        """Count signals fired on the same date from the cache."""
        if all_signals_cache is None:
            return None
        try:
            day_signals = all_signals_cache.get(signal_date, [])
            wins = sum(1 for s in day_signals if s.get("outcome") == "win")
            losses = sum(1 for s in day_signals if s.get("outcome") == "loss")
            same_tmpl = sum(1 for s in day_signals if s.get("template") == template_name)
            return {
                "signals_same_day": len(day_signals),
                "wins_same_day": wins,
                "losses_same_day": losses,
                "same_template_same_day": same_tmpl,
            }
        except Exception as e:
            logger.warning(f"[Attribution] _build_concurrent_signals failed: {e}")
            return None

    # ── M. Record attribution ─────────────────────────────────────

    def _record_attribution(self, template_name, symbol, attribution_data):
        """Append attribution record to shadow_ledger.json under attributions key."""
        try:
            evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("attribution", {})
            max_records = evo_cfg.get("max_attribution_records", 500)

            data = safe_json_read(self.ledger_path, default={})
            if "attributions" not in data:
                data["attributions"] = {}
            if template_name not in data["attributions"]:
                data["attributions"][template_name] = {}
            if symbol not in data["attributions"][template_name]:
                data["attributions"][template_name][symbol] = []

            records = data["attributions"][template_name][symbol]
            records.append(attribution_data)

            # Rolling window — keep newest max_records
            if len(records) > max_records:
                data["attributions"][template_name][symbol] = records[-max_records:]

            safe_json_write(self.ledger_path, data)
        except Exception as e:
            logger.error(f"[Attribution] _record_attribution failed for {template_name}:{symbol}: {e}")

    # ── Orchestrator ──────────────────────────────────────────────

    def _record_signal_attribution(self, template, symbol, df, entry_idx,
                                   outcome, entry_price, stop_loss, take_profit):
        """Build and persist all attribution fields for one virtual signal."""
        evo_cfg = getattr(cfg, 'TEMPLATE_EVOLUTION_CONFIG', {}).get("attribution", {})
        if not evo_cfg.get("enabled", False):
            return

        exit_idx = min(entry_idx + outcome.get("bars", 1), len(df) - 1)
        target_hit = outcome["hit"] == "target"
        pnl = outcome.get("pnl_pct", 0.0)
        exit_price = stop_loss if outcome["hit"] == "stop" else (
            take_profit if target_hit else self._safe_float(df.iloc[exit_idx].get('close', entry_price))
        )

        try:
            entry_ts = df.index[entry_idx]
            current_date_iso = str(pd.Timestamp(entry_ts).date())
        except Exception:
            current_date_iso = datetime.now().date().isoformat()

        attribution = {
            "date": current_date_iso,
            "outcome": "win" if target_hit else "loss",
            "pnl_pct": round(pnl, 4),
        }

        def _try(key, fn, *args, **kwargs):
            try:
                attribution[key] = fn(*args, **kwargs)
            except Exception as exc:
                logger.warning(f"[Attribution] {key} builder failed: {exc}")
                attribution[key] = None

        if evo_cfg.get("track_kill_candle") and not target_hit:
            _try("kill_candle", self._build_kill_candle_data,
                 df, entry_idx, exit_idx, entry_price, stop_loss)

        if evo_cfg.get("track_entry_quality"):
            _try("entry_quality", self._build_entry_quality, df, entry_idx, entry_price)

        if evo_cfg.get("track_volume_profile"):
            _try("volume_profile", self._build_volume_profile, df, entry_idx, exit_idx)

        if evo_cfg.get("track_market_context"):
            _try("market_context", self._build_market_context, None, entry_idx, exit_idx)

        if evo_cfg.get("track_indicator_snapshot"):
            _try("indicator_snapshot", self._build_indicator_snapshot, df, entry_idx, exit_idx)

        if evo_cfg.get("track_weakest_block"):
            _try("weakest_block", self._build_weakest_block, template, df, entry_idx)

        if evo_cfg.get("track_risk_reward"):
            _try("risk_reward", self._build_risk_reward,
                 entry_price, stop_loss, take_profit, exit_price, df, entry_idx, exit_idx)

        if evo_cfg.get("track_time_context"):
            _try("time_context", self._build_time_context, df, entry_idx, exit_idx)

        if evo_cfg.get("track_preceding_candles"):
            _try("preceding_candles", self._build_preceding_candles, df, entry_idx)

        if evo_cfg.get("track_key_levels"):
            _try("key_levels", self._build_key_levels, df, entry_idx, entry_price)

        if evo_cfg.get("track_concurrent_signals"):
            _try("concurrent_signals", self._build_concurrent_signals,
                 template.name, symbol, current_date_iso, None)

        self._record_attribution(template.name, symbol, attribution)

        kc = attribution.get("kill_candle") or {}
        tc = attribution.get("time_context") or {}
        vp = attribution.get("volume_profile") or {}
        mc = attribution.get("market_context") or {}
        wb = attribution.get("weakest_block") or {}
        logger.info(
            f"[ATTRIBUTION] {template.name}:{symbol} | "
            f"outcome={attribution['outcome']} | pnl={pnl:.2%} | "
            f"kill_type={kc.get('kill_type', 'N/A')} | "
            f"phase={kc.get('phase_at_death', 'N/A')} | "
            f"bars={tc.get('bars_in_trade', 'N/A')} | "
            f"vol_exit={vp.get('volume_at_exit', 'N/A')} | "
            f"spy_day={mc.get('spy_return_on_day', 'N/A')} | "
            f"weakest={wb.get('block_name', 'N/A')}"
        )

    def get_template_stats(self, symbol=None):
        """
        Get per-template statistics.

        Args:
            symbol: If given → returns per-stock stats for that symbol.
                    If None → returns global stats aggregated across all symbols.

        Returns:
            dict of {template_id: {signal_count, wins, losses, win_rate, avg_pnl_pct}}
        """
        all_stats = self.ledger.get("template_stats", {})

        if symbol:
            return all_stats.get(symbol, {})

        # Aggregate across all symbols
        global_stats = {}
        for sym_stats in all_stats.values():
            for tid, stats in sym_stats.items():
                if tid not in global_stats:
                    global_stats[tid] = {
                        "signal_count": 0, "wins": 0,
                        "losses": 0, "total_pnl_pct": 0.0,
                    }
                global_stats[tid]["signal_count"] += stats.get("signal_count", 0)
                global_stats[tid]["wins"] += stats.get("wins", 0)
                global_stats[tid]["losses"] += stats.get("losses", 0)
                global_stats[tid]["total_pnl_pct"] += stats.get("total_pnl_pct", 0.0)

        for tid, stats in global_stats.items():
            sc = stats["signal_count"]
            stats["win_rate"] = round(stats["wins"] / sc * 100, 1) if sc > 0 else 0.0
            stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / sc, 2) if sc > 0 else 0.0

        return global_stats

    def apply_decay(self):
        """
        SPEC v13.4 §4: Apply vectorized decay to all stored template stats.

        Recent signals are weighted more; old signals fade at a rate that depends
        on template category. VSA/institutional templates retain memory longer
        than momentum templates.

        Adds 'decayed_win_rate', 'decay_weight', and 'decay_category' to each
        template's stats entry. template_matcher reads decayed_win_rate for
        confidence scoring.

        Decay formula: decayed_win_rate = raw_wr * weight + 50.0 * (1 - weight)
        As weight → min_weight, win_rate regresses to 50% (neutral/unknown),
        NOT to 0% — an old template should be uncertain, not condemned.
        """
        decay_config = getattr(cfg, 'VECTORIZED_DECAY_CONFIG', {})
        if not decay_config.get('enabled', True):
            return

        decay_rates = decay_config.get('decay_rates', {})
        period_days = decay_config.get('decay_period_days', 7)
        min_weight = decay_config.get('min_weight', 0.05)

        last_run = self.ledger.get("metadata", {}).get("last_run")
        if not last_run:
            logger.debug("No previous run timestamp — skipping decay")
            return

        now = datetime.now()
        try:
            last_dt = datetime.fromisoformat(last_run)
            days_since = (now - last_dt).days
        except (ValueError, TypeError):
            logger.warning(f"Invalid last_run timestamp: {last_run}")
            return

        if days_since <= 0:
            return  # Same day — no decay

        periods = days_since / period_days

        # Build id → template map for category lookup
        templates_by_id = {t.id: t for t in self.tm.get_enabled()}

        for symbol, sym_stats in self.ledger.get("template_stats", {}).items():
            for tid, stats in sym_stats.items():
                template = templates_by_id.get(tid)
                if template and hasattr(template, 'get_category'):
                    category = template.get_category()
                else:
                    category = getattr(template, 'category', 'default') if template else 'default'

                rate = decay_rates.get(category, decay_rates.get('default', 0.95))
                weight = max(rate ** periods, min_weight)

                raw_wr = stats.get("win_rate", 50.0)
                stats["decayed_win_rate"] = round(raw_wr * weight + 50.0 * (1 - weight), 1)
                stats["decay_weight"] = round(weight, 4)
                stats["decay_category"] = category

        logger.info(
            f"Vectorized decay applied: {days_since}d since last run, "
            f"{periods:.1f} periods (rates by category)"
        )

    def run_full_evaluation(self, data_source_manager, symbols=None, feature_engine=None, max_date=None, stock_state_fn=None, timeframe=None):
        """
        Batch evaluation: run candle-by-candle for all symbols.
        Intended for OFFLINE/weekend execution per DDR Part C.

        Args:
            data_source_manager: DSM instance for fetching data
            symbols: List of symbols. Defaults to DEFAULT_TRAINING_SYMBOLS from config.
            feature_engine: FeatureEngine instance for indicator calculation. Optional.
            max_date: Optional str — restrict evaluation to bars on or before this date.
                      Used for TRAIN-only mode in the 3-way split pipeline (DDR #14).
            stock_state_fn: Callable(df_slice) → state dict for coverage gap detection.
        """
        if symbols is None:
            symbols = list(getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS', []))

        days_back = self.config.get('eval_days_back', 1095)
        min_candles = self.config.get('min_candles_for_eval', 200)

        # Reset coverage accumulator for this batch run
        self._coverage_data = {}

        logger.info(
            f"Shadow Ledger: Starting full evaluation for "
            f"{len(symbols)} symbols, {days_back} days back"
        )
        if max_date:
            logger.info(f"Shadow Ledger: TRAIN restriction active — max_date={max_date}")

        evaluated = 0
        skipped = 0

        for symbol in symbols:
            try:
                df = data_source_manager.get_stock_data(
                    symbol, days_back=days_back, interval=timeframe or '1d'
                )
                if df is None or len(df) < min_candles:
                    logger.debug(f"[{symbol}] Skipped — insufficient data")
                    skipped += 1
                    continue

                if feature_engine is not None:
                    df = feature_engine.calculate_features(df)

                self.evaluate_history(symbol, df, stock_state_fn=stock_state_fn, max_date=max_date, timeframe=timeframe)
                evaluated += 1
                # Log per-symbol summary
                sym_stats = self.ledger.get("template_stats", {}).get(symbol, {})
                sym_total = sum(s.get("signal_count", 0) for s in sym_stats.values())
                logger.info(f"[{symbol}] Evaluation complete: {sym_total} total signals")

            except Exception as e:
                logger.error(f"[{symbol}] Shadow evaluation failed: {e}")
                skipped += 1
                continue

        # Apply vectorized decay before saving (SPEC v13.4 §4)
        self.apply_decay()

        # Log per-symbol per-template detail for simulator compatibility
        for symbol in symbols:
            sym_stats = self.ledger.get("template_stats", {}).get(symbol, {})
            for tid, stats in sym_stats.items():
                sc = stats.get("signal_count", 0)
                if sc > 0:
                    wr = stats.get("win_rate", 0.0)
                    avg_pnl = stats.get("avg_pnl_pct", 0.0)
                    logger.debug(
                        f"[{symbol}] {tid}: {sc} signals, "
                        f"WR={wr:.1f}%, AvgPnL={avg_pnl:+.2f}%"
                    )

        # Save updated block_stats back to template JSON files
        try:
            self.tm.save_all()
            logger.info("Template block_stats saved to disk")
        except Exception as e:
            logger.warning(f"Failed to save template block_stats: {e}")

        # Coverage gap analysis — runs once across all evaluated symbols
        self._finalize_coverage_gaps()

        self._save_ledger()
        logger.info(
            f"Shadow Ledger: Complete. Evaluated: {evaluated}, Skipped: {skipped}. "
            f"Saved to {self.ledger_path}"
        )


# ════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════
# Usage:
#   python shadow_ledger.py
#   python shadow_ledger.py --symbols AAPL MSFT NVDA
#   python shadow_ledger.py --symbols AAPL --days-back 365
#
# Default: runs on DEFAULT_TRAINING_SYMBOLS from system_config.py
# Intended for offline/weekend execution (DDR Part C).
# Output: data/shadow_ledger.json (used by template_matcher for
#         DDR #1 Asset-Specific win rates)
# ════════════════════════════════════════════════════════════════

def _print_summary(sl):
    """Print human-readable summary of evaluation results to stdout."""
    all_stats = sl.ledger.get("template_stats", {})
    if not all_stats:
        print("\n[ShadowLedger] No evaluation results to summarize.")
        return

    # Aggregate across all symbols
    global_stats = {}
    symbols_evaluated = list(all_stats.keys())

    for sym_stats in all_stats.values():
        for tid, stats in sym_stats.items():
            if tid not in global_stats:
                global_stats[tid] = {
                    "signal_count": 0, "wins": 0, "losses": 0,
                    "total_pnl_pct": 0.0,
                }
            global_stats[tid]["signal_count"] += stats.get("signal_count", 0)
            global_stats[tid]["wins"] += stats.get("wins", 0)
            global_stats[tid]["losses"] += stats.get("losses", 0)
            global_stats[tid]["total_pnl_pct"] += stats.get("total_pnl_pct", 0.0)

    total_signals = sum(s["signal_count"] for s in global_stats.values())

    print(f"\n{'=' * 55}")
    print(f" Shadow Ledger Evaluation Complete")
    print(f"{'=' * 55}")
    print(f" Symbols evaluated: {len(symbols_evaluated)}")
    print(f" Total signals:     {total_signals}")
    print(f"{'-' * 55}")
    print(f" {'Template':<25} {'Signals':>8} {'Wins':>6} {'WR%':>7} {'AvgPnL':>8}")
    print(f"{'-' * 55}")

    for tid in sorted(global_stats.keys()):
        s = global_stats[tid]
        sc = s["signal_count"]
        wr = round(s["wins"] / sc * 100, 1) if sc > 0 else 0.0
        avg_pnl = round(s["total_pnl_pct"] / sc, 2) if sc > 0 else 0.0
        print(f" {tid:<25} {sc:>8} {s['wins']:>6} {wr:>6.1f}% {avg_pnl:>+7.2f}%")

    print(f"{'-' * 55}")

    # Per-symbol breakdown (top 10 by signal count)
    print(f"\n Per-symbol signal counts (top contributors):")
    sym_signals = []
    for sym, sym_stats in all_stats.items():
        sym_total = sum(s.get("signal_count", 0) for s in sym_stats.values())
        sym_signals.append((sym, sym_total))
    sym_signals.sort(key=lambda x: x[1], reverse=True)

    for sym, count in sym_signals[:10]:
        print(f"   {sym:<8} {count:>5} signals")

    # Block-level statistics summary
    print(f"\n{'-' * 55}")
    print(f" Top Blockers (blocks that most often kill signals):")
    print(f"{'-' * 55}")

    all_block_stats = {}
    try:
        from setup_templates import TemplateManager
        tm = TemplateManager()
        for template in tm.get_enabled():
            bstats = template.statistics.get("block_stats", {})
            for block_name, bs in bstats.items():
                if block_name not in all_block_stats:
                    all_block_stats[block_name] = {
                        "evaluated": 0, "passed": 0, "failed": 0,
                        "was_the_blocker": 0,
                        "when_passed_trades": 0, "when_passed_wins": 0,
                    }
                agg = all_block_stats[block_name]
                agg["evaluated"] += bs.get("evaluated", 0)
                agg["passed"] += bs.get("passed", 0)
                agg["failed"] += bs.get("failed", 0)
                agg["was_the_blocker"] += bs.get("was_the_blocker", 0)
                wp = bs.get("when_passed", {})
                agg["when_passed_trades"] += wp.get("total_trades", 0)
                agg["when_passed_wins"] += wp.get("wins", 0)
    except Exception:
        pass

    if all_block_stats:
        print(f" {'Block':<28} {'Eval':>6} {'Pass%':>6} {'Blkr':>5} {'WR%':>6}")
        print(f"{'-' * 55}")
        sorted_blocks = sorted(
            all_block_stats.items(),
            key=lambda x: x[1]["was_the_blocker"],
            reverse=True
        )
        for block_name, bs in sorted_blocks:
            ev = bs["evaluated"]
            pr = round(bs["passed"] / ev * 100, 1) if ev > 0 else 0
            blkr = bs["was_the_blocker"]
            wp_trades = bs["when_passed_trades"]
            wr = round(bs["when_passed_wins"] / wp_trades * 100, 1) if wp_trades > 0 else 0
            print(f" {block_name:<28} {ev:>6} {pr:>5.1f}% {blkr:>5} {wr:>5.1f}%")
    else:
        print(" (no block stats collected yet)")

    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    import argparse
    import sys
    import time

    # ── Parse CLI arguments ──────────────────────────────────
    parser = argparse.ArgumentParser(
        description="StockWise Shadow Ledger — offline candle-by-candle template evaluation",
        epilog="Output: data/shadow_ledger.json (feeds DDR #1 Asset-Specific win rates)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to evaluate (default: DEFAULT_TRAINING_SYMBOLS from config)"
    )
    parser.add_argument(
        "--days-back", type=int, default=None,
        help="Days of history to evaluate (default: from SHADOW_LEDGER_CONFIG.eval_days_back)"
    )
    parser.add_argument(
        "--max-date", type=str, default=None,
        help="Restrict evaluation to bars on or before this date (YYYY-MM-DD). "
             "Used for train-only mode in the 3-way split pipeline."
    )
    args = parser.parse_args()

    # ── Setup logging to console ─────────────────────────────
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # ── Resolve symbols ──────────────────────────────────────
    symbols = args.symbols
    if symbols is None:
        symbols = list(getattr(cfg, 'DEFAULT_TRAINING_SYMBOLS', []))
    if not symbols:
        print("[ShadowLedger] ERROR: No symbols provided and DEFAULT_TRAINING_SYMBOLS is empty.")
        print("Usage: python shadow_ledger.py --symbols AAPL MSFT NVDA")
        sys.exit(1)

    # ── Override days_back if provided ───────────────────────
    sl = ShadowLedger()
    if args.days_back is not None:
        sl.config['eval_days_back'] = args.days_back

    days_back = sl.config.get('eval_days_back', 1095)

    print(f"[ShadowLedger] Starting evaluation:")
    print(f"  Symbols:   {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"  Days back: {days_back}")
    if args.max_date:
        print(f"  Max date: {args.max_date} (TRAIN period restriction)")
    print(f"  Output:    {sl.ledger_path}")
    print()

    # ── Initialize dependencies ──────────────────────────────
    try:
        from data_source_manager import DataSourceManager
        from feature_engine import FeatureEngine
        dsm = DataSourceManager()
        fe = FeatureEngine()
    except Exception as e:
        print(f"[ShadowLedger] ERROR: Failed to initialize dependencies: {e}")
        sys.exit(1)

    # ── Create stock_state_fn for coverage gap detection ─────
    stock_state_fn = None
    try:
        from stock_hunter import StockHunter
        class _MockDM:
            stock_client = None
        hunter = StockHunter(_MockDM())
        stock_state_fn = lambda df_slice: hunter.classify_stock_state(df_slice)
        print(f"  State fn: StockHunter.classify_stock_state")
    except Exception as e:
        print(f"  State fn: UNAVAILABLE ({e}) — coverage gaps won't be detected")

    # ── Run evaluation ───────────────────────────────────────
    start_time = time.time()

    sl.run_full_evaluation(
        data_source_manager=dsm,
        symbols=symbols,
        feature_engine=fe,
        max_date=args.max_date,
        stock_state_fn=stock_state_fn
    )

    elapsed = time.time() - start_time

    # ── Print summary ────────────────────────────────────────
    _print_summary(sl)

    print(f"[ShadowLedger] Duration: {elapsed:.1f}s")
    print(f"[ShadowLedger] Results saved to: {sl.ledger_path}")
    print(f"[ShadowLedger] template_matcher will now use per-stock win rates (DDR #1)")
