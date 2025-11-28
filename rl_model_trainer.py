# rl_model_trainer.py

"""
Reinforcement Learning Model Trainer
====================================

This script trains a PPO (Proximal Policy Optimization) agent from
stable-baselines3 using the custom StockTradingEnv.
"""

import pandas as pd
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data_source_manager import DataSourceManager
from stockwise_simulation import FeatureCalculator, load_contextual_data
from trading_environment import StockTradingEnv
from logging_setup import setup_json_logging

# Setup logging
setup_json_logging()
logger = logging.getLogger(__name__)


def train_agent():
    """
    Main function to load data, create env, train, and save the model.
    """
    logger.info("--- Starting RL Model Training ---")

    # --- 1. Load and Prepare Data ---
    logger.info("Loading training data (SPY)...")
    dm = DataSourceManager(use_ibkr=False)  # Use yfinance

    # Load contextual data first
    context_data = load_contextual_data(dm)

    # Create feature calculator
    fc = FeatureCalculator(
        data_manager=dm,
        contextual_data=context_data,
        is_cloud=False  # Assume local training
    )

    # Load 5 years of SPY data for training
    spy_data = dm.get_stock_data("SPY", days_back=365 * 5)

    if spy_data.empty:
        logger.error("Failed to load training data. Exiting.")
        return

    # --- 2. Create the Environment ---
    logger.info("Creating StockTradingEnv...")
    try:
        # Wrap the environment for stable-baselines
        env = DummyVecEnv([lambda: StockTradingEnv(
            df=spy_data,
            feature_calculator=fc,
            initial_balance=100000,
            risk_pct=1.0
        )])
        logger.info("Environment created successfully.")
    except ValueError as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred creating env: {e}", exc_info=True)
        return

    # --- 3. Train the PPO Agent ---
    logger.info("Initializing PPO model (MlpPolicy)...")
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log="./ppo_stock_tensorboard/")

    logger.info("Starting model training (100,000 timesteps)...")
    # Note: 100k steps is very small and will just check if it runs.
    # A real model will need millions of steps.
    try:
        model.learn(total_timesteps=100000)
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return

    # --- 4. Save the Model ---
    model_save_path = "ppo_stock_trader_v1"
    logger.info(f"Training complete. Saving model to {model_save_path}.zip")
    model.save(model_save_path)
    logger.info("--- RL Model Training Finished ---")


if __name__ == "__main__":
    train_agent()