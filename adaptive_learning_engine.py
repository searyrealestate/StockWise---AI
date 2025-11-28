# adaptive_learning_engine.py

"""
Adaptive Learning Engine
========================

This script implements a persistent "brain" for the trading system that allows it
to adapt its risk tolerance based on its recent performance.

The `AdaptiveLearner` class tracks the historical accuracy of the system's
signals. If the system has been performing poorly (low accuracy), it automatically
tightens the entry criteria ("Hard Mode"). If the system is performing well,
it relaxes the criteria to capture more opportunities ("Easy Mode").

Key Features:
-------------
-   **Dynamic Thresholding**: Calculates a strictness level (0-100) based on
    a rolling window of recent trade outcomes.
-   **Persistence**: Saves and loads its state (history and pending trades) to
    a JSON file (`system_brain_state.json`), ensuring learning continuity
    across application restarts.
-   **Feedback Loop**: Provides methods (`register_trade`, `record_feedback`)
    to track live trades and update the internal history once outcomes are known.

Usage:
------
    learner = AdaptiveLearner(window_size=10)
    current_threshold = learner.get_threshold()
    learner.record_feedback(was_correct=True)
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

STATE_FILE = "system_brain_state.json"


class AdaptiveLearner:
    """
    Persistent learning engine.
    Tracks historical accuracy and adjusts the buy threshold dynamically.
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []  # List of 1 (Hit) or 0 (Miss)
        self.pending_trades = []  # List of dicts: {'date': '...', 'type': 'UP', 'entry_price': 100}

        self.load_state()

    def get_threshold(self):
        """Returns the current strictness level based on recent performance."""
        # Calculate recent accuracy
        if not self.history:
            return 50  # Neutral start

        recent = self.history[-self.window_size:]
        accuracy = sum(recent) / len(recent)

        # --- ADAPTIVE LOGIC ---
        if accuracy < 0.4:
            return 65  # Penalized: Hard mode (High threshold)
        elif accuracy > 0.7:
            return 40  # Boosted: Easy mode (Low threshold)
        else:
            return 50  # Neutral

    def register_trade(self, date, direction, price):
        """Call this when the system signals a BUY."""
        self.pending_trades.append({
            'date': str(date),
            'direction': direction,  # 'UP'
            'entry_price': float(price)
        })
        self.save_state()

    def update_outcomes(self, current_date, current_price):
        """
        Checks pending trades to see if they resolved (T+5 days passed).
        Updates history and removes resolved trades.
        """
        # Placeholder for live loop logic
        pass

    def record_feedback(self, was_correct: bool):
        """Manually record a result (used by backtester/debugger)."""
        self.history.append(1 if was_correct else 0)
        # Keep history size manageable
        if len(self.history) > 50:
            self.history.pop(0)
        self.save_state()

    def save_state(self):
        state = {
            'history': self.history,
            'pending_trades': self.pending_trades
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save brain state: {e}")

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.history = state.get('history', [])
                    self.pending_trades = state.get('pending_trades', [])
            except Exception:
                logger.warning("Could not load brain state. Starting fresh.")