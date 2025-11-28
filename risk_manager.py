# risk_manager.py
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class RiskManager:

    def __init__(self, portfolio_value, global_risk_pct=1.0):
        """
        Initialize the risk manager with total portfolio value.

        :param portfolio_value: Total value of the trading account (e.g., 100000)
        :param global_risk_pct: Max % of portfolio to risk on one trade (e.g., 1.0)
        """
        self.portfolio_value = portfolio_value
        self.global_risk_pct = global_risk_pct
        self.max_risk_dollars_per_trade = self.portfolio_value * (self.global_risk_pct / 100.0)
        logger.info(f"RiskManager initialized. Max risk per trade: ${self.max_risk_dollars_per_trade:.2f}")

    def update_portfolio_value(self, new_value):
        """Allows updating the portfolio value as it grows/shrinks."""
        self.portfolio_value = new_value
        self.max_risk_dollars_per_trade = self.portfolio_value * (self.global_risk_pct / 100.0)
        logger.info(
            f"Portfolio value updated to ${new_value:.2f}. New max risk: ${self.max_risk_dollars_per_trade:.2f}")

    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Calculates position size based on the dollar risk model.

        :param entry_price: The price at which the asset will be bought.
        :param stop_loss_price: The price at which the trade will be exited for a loss.
        :return: (int) The number of shares to purchase.
        """
        if entry_price <= 0 or stop_loss_price >= entry_price:
            logger.warning(f"Invalid position size calculation: Entry ${entry_price}, SL ${stop_loss_price}")
            return 0  # Invalid parameters

        # 1. Calculate risk per share
        risk_per_share = entry_price - stop_loss_price

        # 2. Calculate number of shares based on max allowed dollar risk
        num_shares = self.max_risk_dollars_per_trade / risk_per_share

        # 3. Ensure we don't use more than the total portfolio value (safety check)
        investment_amount = num_shares * entry_price
        if investment_amount > self.portfolio_value:
            num_shares = self.portfolio_value / entry_price
            logger.warning("Position size capped by total portfolio value.")

        logger.info(f"Position size calculated: {np.floor(num_shares)} shares.")
        return np.floor(num_shares)  # Can only buy whole shares

    def manage_open_position(self, current_day_data: pd.Series, position_data: dict):
        """
        Updates the stop-loss for an open position and checks all exit rules.
        This function should be called on every new data bar (day or live tick).

        :param current_day_data: A pandas Series (row) for the current day.
                                 MUST contain 'low', 'high', 'close', 'atr_14', 'sma_150'.
        :param position_data: A dict holding the state of the open trade.
            {
                'entry_price': 100,
                'current_stop_loss': 95, # This value will be updated
                'use_trailing_stop': True,
                'atr_multiplier': 2.0
            }

        :return: (str, dict) A tuple of (signal, updated_position_data)
                 Signal: "HOLD" or "EXIT_SIGNAL"
        """
        try:
            current_low = current_day_data['low']
            current_close = current_day_data['close']

            # --- 1. NEW: Structural Stop-Loss (150-day SMA) ---
            # This is the new rule you requested.
            # Check if price CLOSED below the 150 SMA
            if 'sma_150' in current_day_data:
                current_sma_150 = current_day_data['sma_150']
                if current_close < current_sma_150:
                    logger.info(
                        f"EXIT_SIGNAL: Structural stop hit. Close ({current_close:.2f}) < 150-day SMA ({current_sma_150:.2f}).")
                    return "EXIT_SIGNAL", position_data

            # --- 2. Volatility Stop-Loss (Your existing logic, now integrated) ---

            # 2a. Standard (non-trailing) Stop-Loss Check
            if not position_data.get('use_trailing_stop', False):
                if current_low <= position_data['current_stop_loss']:
                    logger.info(f"EXIT_SIGNAL: Static stop-loss hit at ${position_data['current_stop_loss']}.")
                    return "EXIT_SIGNAL", position_data
                return "HOLD", position_data

            # 2b. Trailing Stop-Loss Logic
            current_high = current_day_data['high']
            # Make sure 'atr_14' is in your data; it's added in results_analyzer.py
            atr_value = current_day_data.get('atr_14', 0)
            if atr_value == 0:
                logger.warning("ATR value is 0, trailing stop will not work correctly.")
                return "HOLD", position_data  # Hold as a failsafe

            atr_mult = position_data.get('atr_multiplier', 2.5)

            new_potential_stop = current_high - (atr_value * atr_mult)
            new_stop_loss = max(position_data['current_stop_loss'], new_potential_stop)

            # Check if the *current low* has breached the *new* stop
            if current_low <= new_stop_loss:
                logger.info(f"EXIT_SIGNAL: Trailing stop-loss hit at ${new_stop_loss:.2f}.")
                position_data['current_stop_loss'] = new_stop_loss  # Update before exiting
                return "EXIT_SIGNAL", position_data

            # 2d. No exit, update the position with the (potentially) higher stop-loss
            if new_stop_loss > position_data['current_stop_loss']:
                logger.debug(f"Trailing stop raised to ${new_stop_loss:.2f}")
                position_data['current_stop_loss'] = new_stop_loss

            return "HOLD", position_data

        except KeyError as e:
            logger.error(f"Missing expected data in current_day_data: {e}. Holding position as failsafe.",
                         exc_info=True)
            return "HOLD", position_data
        except Exception as e:
            logger.error(f"Error in manage_open_position: {e}. Holding position as failsafe.", exc_info=True)
            return "HOLD", position_data