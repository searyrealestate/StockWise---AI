# setup_templates.py

"""
StockWise Gen-13 Setup Template Engine
======================================
Defines the data model for trading templates and provides
load/save/validate operations.

A Template is a reusable pattern that describes:
- WHEN to enter (conditions on indicators)
- WHERE to place stop-loss and take-profit
- WHAT market state it works best in

Templates are stored as JSON in data/templates/ directory.
They can be:
  1. Seed templates (manually defined, ship with the system)
  2. Discovered templates (found by backtesting historical data)
"""

import os
import json
import logging
import time
from datetime import datetime
import system_config as cfg

logger = logging.getLogger("TemplateEngine")


class SetupTemplate:
    """
    A single trading setup template.

    Structure:
    {
        "id": "TREND_PULLBACK_EMA",
        "name": "Trend Pullback to EMA",
        "description": "Buy on pullback to EMA_12 in confirmed uptrend",
        "version": 1,
        "source": "seed",          # "seed" or "discovered"
        "enabled": true,

        "required_state": {
            "trend": ["BULLISH"],          # Which trend directions this works in
            "structure": ["OPEN_FIELD", "NEAR_SUPPORT"],  # Acceptable structures
            "volume": ["HEALTHY", "SURGING"],              # Required volume states
            "volatility": ["NORMAL", "COMPRESSED"]         # Acceptable volatility
        },

        "conditions": [
            {"indicator": "rsi", "operator": "between", "value": [40, 65]},
            {"indicator": "close", "operator": ">", "reference": "ema_12"},
            {"indicator": "macd", "operator": ">", "reference": "macd_signal"},
            {"indicator": "volume", "operator": ">", "reference": "vol_avg_20", "multiplier": 1.2}
        ],

        "entry": {
            "type": "close",               # Enter at close of signal candle
            "confirmation_candles": 1       # Wait 1 candle to confirm
        },

        "stop_loss": {
            "method": "atr",               # "atr", "swing_low", "sma", "fixed_pct"
            "atr_multiplier": 1.5,
            "fallback_pct": 0.02           # 2% fallback if method fails
        },

        "take_profit": {
            "method": "atr",               # "atr", "resistance", "fixed_pct"
            "atr_multiplier": 3.0,
            "use_runner_mode": true         # Let Phase 4 Runner handle exit
        },

        "statistics": {
            "total_activations": 0,
            "wins": 0,
            "losses": 0,
            "avg_profit_pct": 0.0,
            "avg_loss_pct": 0.0,
            "win_rate": 0.0,
            "last_activated": null,
            "best_tickers": {},            # {"AAPL": {"wins": 5, "losses": 1}}
            "worst_tickers": {},
            "best_conditions": [],         # ["earnings_season", "low_vix"]
            "created_at": null,
            "updated_at": null
        }
    }
    """

    # Required fields that every template must have
    REQUIRED_FIELDS = ['id', 'name', 'conditions', 'stop_loss', 'take_profit']

    # Valid operators for conditions
    VALID_OPERATORS = ['>', '<', '>=', '<=', '==', '!=', 'between', 'crosses_above', 'crosses_below']

    def __init__(self, data):
        """Initialize from a dictionary (loaded from JSON)."""
        self.data = data
        self.id = data.get('id', 'UNKNOWN')
        self.name = data.get('name', 'Unnamed Template')
        self.enabled = data.get('enabled', True)
        self.source = data.get('source', 'seed')
        self.required_state = data.get('required_state', {})
        self.conditions = data.get('conditions', [])
        self.entry = data.get('entry', {"type": "close", "confirmation_candles": 0})
        self.stop_loss = data.get('stop_loss', {"method": "atr", "atr_multiplier": 2.0, "fallback_pct": 0.02})
        self.take_profit = data.get('take_profit', {"method": "atr", "atr_multiplier": 3.0, "use_runner_mode": True})
        self.statistics = data.get('statistics', self._empty_stats())

    def _empty_stats(self):
        """Returns a fresh statistics block."""
        return {
            "total_activations": 0, "wins": 0, "losses": 0,
            "avg_profit_pct": 0.0, "avg_loss_pct": 0.0, "win_rate": 0.0,
            "last_activated": None,
            "best_tickers": {}, "worst_tickers": {},
            "best_conditions": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

    def validate(self):
        """
        Validates the template structure. Returns (is_valid, errors_list).
        """
        errors = []

        for field in self.REQUIRED_FIELDS:
            if field not in self.data:
                errors.append(f"Missing required field: {field}")

        # Validate conditions
        for i, cond in enumerate(self.conditions):
            if 'indicator' not in cond:
                errors.append(f"Condition {i}: missing 'indicator'")
            if 'operator' not in cond:
                errors.append(f"Condition {i}: missing 'operator'")
            elif cond['operator'] not in self.VALID_OPERATORS:
                errors.append(f"Condition {i}: invalid operator '{cond['operator']}'")
            if cond.get('operator') == 'between' and not isinstance(cond.get('value'), list):
                errors.append(f"Condition {i}: 'between' operator requires list value [min, max]")

        # Validate stop_loss
        if self.stop_loss.get('method') not in ['atr', 'swing_low', 'sma', 'fixed_pct']:
            errors.append(f"Invalid stop_loss method: {self.stop_loss.get('method')}")

        # Validate take_profit
        if self.take_profit.get('method') not in ['atr', 'resistance', 'fixed_pct']:
            errors.append(f"Invalid take_profit method: {self.take_profit.get('method')}")

        return len(errors) == 0, errors

    def get_win_rate(self):
        """Returns win rate as a percentage."""
        total = self.statistics.get('total_activations', 0)
        if total == 0:
            return 0.0
        return (self.statistics.get('wins', 0) / total) * 100.0

    def record_result(self, ticker, profit_pct, won):
        """
        Records the outcome of a template activation.
        Updates running statistics.
        """
        stats = self.statistics
        stats['total_activations'] = stats.get('total_activations', 0) + 1
        stats['last_activated'] = datetime.now().isoformat()
        stats['updated_at'] = datetime.now().isoformat()

        if won:
            stats['wins'] = stats.get('wins', 0) + 1
            # Running average profit
            old_avg = stats.get('avg_profit_pct', 0.0)
            n_wins = stats['wins']
            stats['avg_profit_pct'] = old_avg + (profit_pct - old_avg) / n_wins
        else:
            stats['losses'] = stats.get('losses', 0) + 1
            old_avg = stats.get('avg_loss_pct', 0.0)
            n_losses = stats['losses']
            stats['avg_loss_pct'] = old_avg + (profit_pct - old_avg) / n_losses

        # Update win rate
        total = stats['total_activations']
        stats['win_rate'] = (stats['wins'] / total) * 100.0 if total > 0 else 0.0

        # Track per-ticker performance
        best = stats.get('best_tickers', {})
        if ticker not in best:
            best[ticker] = {"wins": 0, "losses": 0}
        if won:
            best[ticker]["wins"] += 1
        else:
            best[ticker]["losses"] += 1
        stats['best_tickers'] = best

    def to_dict(self):
        """Serialize back to dictionary for JSON storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.data.get('description', ''),
            "version": self.data.get('version', 1),
            "source": self.source,
            "enabled": self.enabled,
            "required_state": self.required_state,
            "conditions": self.conditions,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "statistics": self.statistics,
        }


class TemplateManager:
    """
    Loads, saves, and manages the library of trading templates.
    Templates are stored as individual JSON files in data/templates/.
    """

    def __init__(self):
        self.templates_dir = os.path.join(cfg.DB_DIR, "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        self.templates = {}  # id -> SetupTemplate
        self.load_all()

    def load_all(self):
        """Load all template JSON files from the templates directory."""
        self.templates = {}

        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return

        for filename in os.listdir(self.templates_dir):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(self.templates_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                template = SetupTemplate(data)
                is_valid, errors = template.validate()
                if is_valid:
                    self.templates[template.id] = template
                    logger.debug(f"Loaded template: {template.id} ({template.name})")
                else:
                    logger.warning(f"Invalid template {filename}: {errors}")
            except Exception as e:
                logger.error(f"Failed to load template {filename}: {e}")

        logger.info(f"Template library loaded: {len(self.templates)} templates")

    def save_template(self, template):
        """Save a single template to disk."""
        filepath = os.path.join(self.templates_dir, f"{template.id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(template.to_dict(), f, indent=2)
            logger.debug(f"Saved template: {template.id}")
        except Exception as e:
            logger.error(f"Failed to save template {template.id}: {e}")

    def save_all(self):
        """Save all templates to disk (useful after updating statistics)."""
        for template in self.templates.values():
            self.save_template(template)
        logger.info(f"Saved {len(self.templates)} templates to disk")

    def get_enabled(self):
        """Return list of all enabled templates."""
        return [t for t in self.templates.values() if t.enabled]

    def get_for_state(self, stock_state):
        """
        Return templates that match the stock's current state.
        Filters enabled templates by required_state compatibility.
        """
        matching = []
        for template in self.get_enabled():
            if self._state_matches(template.required_state, stock_state):
                matching.append(template)
        return matching

    def _state_matches(self, required_state, stock_state):
        """
        Check if a stock's state matches a template's requirements.
        Each required_state field is a list of acceptable values.
        If a field is missing from required_state, any value is accepted.
        """
        for key, acceptable_values in required_state.items():
            actual_value = stock_state.get(key, '')
            if actual_value not in acceptable_values:
                return False
        return True

    def get_template_by_id(self, template_id):
        """Get a specific template by ID."""
        return self.templates.get(template_id)

    def add_template(self, template_data):
        """Add a new template from a dictionary. Validates before adding."""
        template = SetupTemplate(template_data)
        is_valid, errors = template.validate()
        if not is_valid:
            logger.error(f"Cannot add invalid template: {errors}")
            return False
        self.templates[template.id] = template
        self.save_template(template)
        logger.info(f"Added new template: {template.id} ({template.name})")
        return True

    def get_statistics_summary(self):
        """Return a summary of all template statistics for logging/reporting."""
        summary = []
        for t in self.templates.values():
            summary.append({
                "id": t.id,
                "name": t.name,
                "enabled": t.enabled,
                "source": t.source,
                "total": t.statistics.get('total_activations', 0),
                "win_rate": t.get_win_rate(),
                "avg_profit": t.statistics.get('avg_profit_pct', 0.0),
            })
        return sorted(summary, key=lambda x: x['win_rate'], reverse=True)
