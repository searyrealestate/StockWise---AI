# trading_environment.py

"""
Stock Trading Environment for Reinforcement Learning
==================================================

This module defines a custom environment using `gymnasium` (OpenAI Gym).
It allows an RL agent (like PPO from stable-baselines3) to learn a
trading strategy by interacting with historical stock data.

The environment uses the `FeatureCalculator` for observations and the
`RiskManager` for realistic position sizing and P/L calculations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from risk_manager import RiskManager
from stockwise_simulation import FeatureCalculator  # Assumes FeatureCalculator is in this file

logger = logging.getLogger(__name__)


class StockTradingEnv(gym.Env):
    """
    A custom stock trading environment for reinforcement learning.

    Action Space:
        - 0: HOLD
        - 1: BUY
        - 2: SELL
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, feature_calculator: FeatureCalculator,
                 initial_balance=100000, risk_pct=1.0):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.feature_calculator = feature_calculator
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct

        logger.info("Calculating features for RL environment...")
        self.features_df = self.feature_calculator.calculate_all_features(self.df)

        # Drop rows where features couldn't be calculated
        self.df = self.df.iloc[self.df.index.isin(self.features_df.index)]

        if self.features_df.empty:
            raise ValueError("Feature calculation resulted in an empty DataFrame. Cannot create environment.")

        # --- Define Spaces ---
        self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: BUY, 2: SELL

        # Observation space is the set of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.features_df.columns),),
            dtype=np.float32
        )

        self.render_mode = 'human'
        self.reset()

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        self.position = None  # {'entry_price': float, 'shares': int, 'stop_loss': float}

        self.risk_manager = RiskManager(
            portfolio_value=self.initial_balance,
            global_risk_pct=self.risk_pct
        )

        logger.info(f"Environment reset. Initial balance: ${self.balance:.2f}")

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self):
        """Returns the observation for the current time step."""
        return self.features_df.iloc[self.current_step].values.astype(np.float32)

    def _get_info(self):
        """Returns auxiliary info."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'portfolio_value': self.portfolio_value,
            'position': self.position
        }

    def step(self, action):
        """Executes one time step within the environment."""

        terminated = False
        reward = 0.0

        current_data = self.df.iloc[self.current_step]
        current_price = current_data['close']

        # --- 1. Update Risk Manager & Portfolio Value ---
        # Portfolio value = cash balance + value of open position
        if self.position:
            self.portfolio_value = self.balance + (self.position['shares'] * current_price)
        else:
            self.portfolio_value = self.balance

        self.risk_manager.update_portfolio_value(self.portfolio_value)

        # --- 2. Check for Stop-Loss (if a position is open) ---
        if self.position:
            # We need to pass the full data row to the upgraded risk manager
            # This requires adding 'atr_14' and 'sma_150' to our features_df
            # For simplicity in this demo, we'll fake a simple stop-loss check

            # --- Simple Stop-Loss (replace with full RiskManager call later) ---
            if current_price < self.position['stop_loss']:
                logger.info(f"RL: Stop-loss hit at step {self.current_step}.")
                action = 2  # Force a SELL

        # --- 3. Handle Action ---
        if action == 1(BUY):
            if not self.position:
                # Buy signal. Calculate position size.
                stop_loss_price = current_price * 0.95  # Simple 5% stop
                shares = self.risk_manager.calculate_position_size(
                    entry_price=current_price,
                    stop_loss_price=stop_loss_price
                )
                if shares > 0:
                    cost = shares * current_price
                    if self.balance >= cost:
                        self.balance -= cost
                        self.position = {
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss_price
                        }
                        logger.debug(f"RL: BUY {shares} at ${current_price:.2f}")
                        reward = -0.01  # Small penalty for transaction cost

        elif action == 2(SELL):
            if self.position:
                # Sell signal. Close position.
                proceeds = self.position['shares'] * current_price
                profit = proceeds - (self.position['shares'] * self.position['entry_price'])

                self.balance += proceeds
                logger.debug(f"RL: SELL {self.position['shares']} at ${current_price:.2f}. Profit: ${profit:.2f}")

                # Reward is the profit/loss from the trade
                reward = profit / self.initial_balance  # Normalize reward

                self.position = None

        # --- 4. Calculate Reward for HOLD ---
        if self.position:
            # Reward is the unrealized P/L for this step
            unrealized_pl = (self.position['shares'] * current_price) - \
                            (self.position['shares'] * self.df.iloc[self.current_step - 1]['close'])
            reward = unrealized_pl / self.initial_balance

        # --- 5. Advance Time ---
        self.current_step += 1
        if self.current_step >= len(self.features_df) - 1:
            terminated = True

        # --- 6. Finalize ---
        if terminated and self.position:
            # Liquidate at the end
            proceeds = self.position['shares'] * current_price
            self.balance += proceeds
            self.position = None

        obs = self._get_obs()
        info = self._get_info()
        truncated = False  # We don't use truncated

        return obs, reward, terminated, truncated, info

    def render(self):
        """Renders the environment (e.g., in a plot) - optional."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Position: {self.position}")