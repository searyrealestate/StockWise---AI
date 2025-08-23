import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from itertools import product
import yfinance as yf
from tqdm import tqdm
import warnings
import traceback
import os
import time
import logging
import sys
from stockwise_simulation import ProfessionalStockAdvisor

# Attempt to reconfigure stdout and stderr to UTF-8 for better console emoji/Unicode support.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
except Exception as e:
    # If reconfiguration fails (e.g., in some IDE consoles or environments that don't allow it),
    # print a warning to the original stderr and proceed. Emojis might still be an issue.
    print(f"Warning: Could not reconfigure console encoding to UTF-8. Emojis may appear as '?' or cause errors: {e}",
          file=sys.__stderr__)

warnings.filterwarnings('ignore')


class StockWiseAutoCalibrator:
    """
    Auto-calibration system using Walk-Forward Analysis
    Optimizes StockWise algorithm parameters for maximum performance
    """

    def __init__(self, advisor_instance, config=None):
        self.advisor = advisor_instance
        self.config = config or self.get_default_config()
        self.results = {}
        self.best_parameters = {}
        self.validation_windows = []
        self.configuration_files = 'configuration/'

        os.makedirs(self.configuration_files, exist_ok=True)
        logging.info("⭐ Initializing StockWiseAutoCalibrator...")
        logging.info(f"Configuration files will be saved to: {self.configuration_files}")

    def _normalize_numeric_values(self, obj):
        """
        Recursively converts numpy numeric types to standard Python types for JSON serialization.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._normalize_numeric_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_numeric_values(elem) for elem in obj]
        else:
            return obj

    def save_config_for_all_strategies(self, best_params_overall):
        """
        Saves the optimal parameters for each strategy into separate JSON files
        and a combined calibration report.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save individual production parameter files for each strategy
        for strategy, data in best_params_overall.items():
            if 'optimal_parameters' in data:
                # Construct the production config for the strategy
                prod_config = {
                    "strategy_multipliers": {
                        strategy: {
                            "profit": data['optimal_parameters'].get('profit_threshold', 1.0),
                            "risk": data['optimal_parameters'].get('stop_loss_threshold', 1.0),
                            # Use the correct confidence_req key based on strategy
                            "confidence_req": data['optimal_parameters'].get(
                                f'confidence_req_{strategy.lower().replace(" ", "_")}', 75)
                        }
                    },
                    "signal_weights": {
                        "trend": data['optimal_parameters'].get('weight_trend', 0.45),
                        "momentum": data['optimal_parameters'].get('weight_momentum', 0.30),
                        "volume": data['optimal_parameters'].get('weight_volume', 0.10),
                        "sr": data['optimal_parameters'].get('weight_support_resistance', 0.05),
                        "model": data['optimal_parameters'].get('weight_ai_model', 0.10)
                    },
                    "thresholds": {
                        # Use the correct buy/sell_threshold keys based on strategy
                        "buy_threshold": data['optimal_parameters'].get(
                            f'buy_threshold_{strategy.lower().replace(" ", "_")}', 0.9),
                        "sell_threshold": data['optimal_parameters'].get(
                            f'sell_threshold_{strategy.lower().replace(" ", "_")}', -0.9)
                    },
                    "confidence_params": {
                        "base_multiplier": data['optimal_parameters'].get('confidence_base_multiplier', 1.0),
                        "confluence_weight": data['optimal_parameters'].get('confidence_confluence_weight', 1.0),
                        "penalty_strength": data['optimal_parameters'].get('confidence_penalty_strength', 1.0)
                    }
                }

                # Normalize any numpy types before saving
                prod_config_normalized = self._normalize_numeric_values(prod_config)

                file_name = f"stockwise_production_params_{strategy.lower().replace(' ', '_')}_{timestamp}.json"
                file_path = os.path.join(self.configuration_files, file_name)
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(prod_config_normalized, f, indent=2)
                    logging.info(f"✅ Saved production parameters for '{strategy}' to {file_path}")
                except Exception as e:
                    logging.error(f"❌ Failed to save production parameters for '{strategy}' to {file_path}: {e}")

        # Save the comprehensive calibration report
        report_file_name = f"stockwise_calibration_{timestamp}.json"
        report_file_path = os.path.join(self.configuration_files, report_file_name)

        full_report = {
            "metadata": {
                "calibration_date": timestamp,
                "config": self._normalize_numeric_values(self.config)  # Save the config used for this run
            },
            "best_parameters_overall_summary": self._normalize_numeric_values(best_params_overall)
        }

        try:
            with open(report_file_path, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, indent=2)
            logging.info(f"✅ Saved comprehensive calibration report to {report_file_path}")
        except Exception as e:
            logging.error(f"❌ Failed to save calibration report to {report_file_path}: {e}")

    def get_default_config(self):
        """
        Provides a flexible default configuration for calibration.
        """
        return {
            "stock_universe": "NASDAQ_100",
            "market_cap_min": 1_000_000_000,
            "price_min": 5.0,
            "volume_min": 1_000_000,
            "start_date": "2023-08-17", # Start date for historical data fetch
            "end_date": "2025-07-17",   # End date for historical data fetch
            "training_period": 365,     # Days for initial training data in each walk-forward window
            "validation_period": 90,    # Days for testing/validation in each walk-forward window
            "num_walk_forward_windows": 4, # Number of walk-forward validation windows

            # Add walk_forward configuration
            "walk_forward": {
                "train_months": 12,
                "test_months": 3,
                "step_months": 3
            },

            "advisor_params": {
                # These are the *ranges* for the parameters to be optimized
                "investment_days": [7, 14, 21, 30, 60],
                "profit_threshold": [0.03, 0.04, 0.05, 0.06], # As percentage, e.g., 0.03 = 3%
                "stop_loss_threshold": [0.04, 0.06, 0.08, 0.10], # As percentage, e.g., 0.04 = 4%
                # Strategy-specific thresholds
                "confidence_req_balanced": [70, 75, 80],
                "confidence_req_conservative": [80, 85, 90],
                "confidence_req_aggressive": [60, 65, 70],
                "confidence_req_swing_trading": [65, 70, 75],

                "buy_threshold_balanced": [0.8, 0.9, 1.0, 1.2], # Score thresholds for BUY/SELL
                "sell_threshold_balanced": [-0.8, -0.9, -1.0, -1.2],

                "buy_threshold_conservative": [1.0, 1.5, 2.0],
                "sell_threshold_conservative": [-1.0, -1.5, -2.0],

                "buy_threshold_aggressive": [0.5, 0.6, 0.7, 0.8],
                "sell_threshold_aggressive": [-0.5, -0.6, -0.7, -0.8],

                "buy_threshold_swing_trading": [0.7, 0.8, 0.9],
                "sell_threshold_swing_trading": [-0.7, -0.8, -0.9],

                # Global signal weights (can be optimized if desired, sum to 1.0)
                "weight_trend": [0.45], # Example: Fixed for now, can be a range
                "weight_momentum": [0.30],
                "weight_volume": [0.10],
                "weight_support_resistance": [0.05],
                "weight_ai_model": [0.10],

                # Confidence parameters
                "confidence_base_multiplier": [0.85, 1.0, 1.15],
                "confidence_confluence_weight": [0.8, 1.0, 1.2],
                "confidence_penalty_strength": [0.8, 0.9, 1.0]
            },
            "test_config": {
                "sanity": {"stocks": 2, "test_points": 10, "param_samples": 2},
                "small": {"stocks": 10, "test_points": 100, "param_samples": 10},
                "medium": {"stocks": 12, "test_points": 200, "param_samples": 20},
                "full": {"stocks": 20, "test_points": 300, "param_samples": 24}
            },
            "prediction_window_days": 7, # How many days into the future to check actual return
            "evaluation_thresholds": {
                "buy_profit_min": 3.0,     # Min % return for a 'BUY' to be considered profitable
                "sell_loss_min": -2.0,     # Max % loss for a 'SELL/AVOID' to be considered successful avoidance
                "wait_max_change": 2.0     # Max % change for a 'WAIT' to be considered correct (sideways)
            }
        }

    # From ChatGPT
    # def _generate_param_grid(self, advisor_params, strategy_type):
    #     """
    #     Generate parameter grid properly for a given strategy type.
    #     """
    #     strategy_lower = strategy_type.lower().replace(" ", "_")
    #
    #     param_space = {
    #         "investment_days": advisor_params["investment_days"],
    #         "profit_threshold": advisor_params["profit_threshold"],
    #         "stop_loss_threshold": advisor_params["stop_loss_threshold"],
    #         "buy_threshold": advisor_params.get(f"buy_threshold_{strategy_lower}", [0.9]),
    #         "sell_threshold": advisor_params.get(f"sell_threshold_{strategy_lower}", [-0.9]),
    #         "confidence_req": advisor_params.get(f"confidence_req_{strategy_lower}", [75]),
    #         "weight_trend": advisor_params["weight_trend"],
    #         "weight_momentum": advisor_params["weight_momentum"],
    #         "weight_volume": advisor_params["weight_volume"],
    #         "weight_support_resistance": advisor_params["weight_support_resistance"],
    #         "weight_ai_model": advisor_params["weight_ai_model"],
    #         "confidence_base_multiplier": advisor_params["confidence_base_multiplier"],
    #         "confidence_confluence_weight": advisor_params["confidence_confluence_weight"],
    #         "confidence_penalty_strength": advisor_params["confidence_penalty_strength"],
    #     }
    #
    #     return self.generate_parameter_combinations(param_space)

    def _generate_param_grid(self, advisor_params, strategy_type):
        """
        Generate parameter grid properly for a given strategy type.
        This method constructs the specific parameter space for a single strategy
        and then uses `generate_parameter_combinations` to get all combinations.
        """
        strategy_lower = strategy_type.lower().replace(" ", "_")

        param_space = {
            "investment_days": advisor_params["investment_days"],
            "profit_threshold": advisor_params["profit_threshold"],
            "stop_loss_threshold": advisor_params["stop_loss_threshold"],
            # Ensure these are correctly pulling from the strategy-specific keys
            "buy_threshold": advisor_params.get(f"buy_threshold_{strategy_lower}", [0.9]),
            "sell_threshold": advisor_params.get(f"sell_threshold_{strategy_lower}", [-0.9]),
            "confidence_req": advisor_params.get(f"confidence_req_{strategy_lower}", [75]),
            "weight_trend": advisor_params["weight_trend"],
            "weight_momentum": advisor_params["weight_momentum"],
            "weight_volume": advisor_params["weight_volume"],
            "weight_support_resistance": advisor_params["weight_support_resistance"],
            "weight_ai_model": advisor_params["weight_ai_model"],
            "confidence_base_multiplier": advisor_params["confidence_base_multiplier"],
            "confidence_confluence_weight": advisor_params["confidence_confluence_weight"],
            "confidence_penalty_strength": advisor_params["confidence_penalty_strength"],
        }

        # Now, use the generic generator to create the full grid for this specific strategy
        return self.generate_parameter_combinations(param_space)

    def evaluate_performance(self, backtest_results, optimization_metric='net_profit'):
        """
        Evaluates the performance of the algorithm based on backtest results.
        Enhanced to consider multiple metrics for a more robust evaluation.
        :param backtest_results: List of dictionaries, each containing results for a symbol.
        :param optimization_metric: The primary metric to optimize for ('net_profit', 'win_rate', 'sharpe_ratio').
        """
        if not backtest_results:
            logging.warning("No backtest results to evaluate. Returning -inf.")
            return -float('inf')

        total_net_profit_pct = 0
        total_trades = 0
        winning_trades = 0
        sharpe_ratios = []
        daily_returns = []

        for result in backtest_results:
            total_net_profit_pct += result.get('net_profit_pct', 0)
            total_trades += result.get('total_trades', 0)
            winning_trades += result.get('winning_trades', 0)
            if 'daily_returns' in result and result['daily_returns'] is not None:
                daily_returns.extend(result['daily_returns'])

            if 'sharpe_ratio' in result and result['sharpe_ratio'] is not None:
                sharpe_ratios.append(result['sharpe_ratio'])

        avg_net_profit_pct = total_net_profit_pct / len(backtest_results) if backtest_results else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        overall_sharpe_ratio = -float('inf')
        if daily_returns:
            returns_series = pd.Series(daily_returns)
            if not returns_series.empty and returns_series.std() > 0:
                overall_sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)
            else:
                logging.warning("Cannot calculate Sharpe Ratio: Insufficient or zero-variance daily returns.")

        if optimization_metric == 'net_profit':
            metric_value = avg_net_profit_pct
        elif optimization_metric == 'win_rate':
            metric_value = win_rate
        elif optimization_metric == 'sharpe_ratio':
            if sharpe_ratios:
                metric_value = np.mean([s for s in sharpe_ratios if s is not None and np.isfinite(s)])
            else:
                metric_value = overall_sharpe_ratio
        else:
            logging.warning(f"Unknown optimization metric: {optimization_metric}. Defaulting to net_profit.")
            metric_value = avg_net_profit_pct

        if not np.isfinite(metric_value):
            return -float('inf')

        return metric_value

    def run_full_calibration_pipeline(self, max_duration_hours=None):
        """
        Runs the full auto-calibration pipeline, now with an optional time limit.
        :param max_duration_hours: Maximum duration in hours for the calibration to run.
        """
        start_time = time.time()
        logging.info("🚀 Starting full calibration pipeline...")
        if max_duration_hours:
            logging.info(f"⏳ Calibration will run for a maximum of {max_duration_hours} hours.")

        try:
            logging.info("Generating walk-forward validation windows...")
            # FIX: Corrected call to the existing method
            self.validation_windows = self.create_walk_forward_windows()
            logging.info(f"Generated {len(self.validation_windows)} validation windows.")

            logging.info("Fetching initial stock universe for symbol selection...")
            # Assuming _get_nasdaq_100_symbols and _filter_symbols are external or handled differently
            # For this pipeline, we'll use a placeholder for selected_symbols for demonstration.
            # In a real scenario, these would come from your stock universe functions.
            all_symbols = self.get_stock_universe(self.config['num_symbols_to_test'])
            # Filtering logic would go here if not handled in get_stock_universe directly
            selected_symbols = all_symbols # Placeholder

            if len(selected_symbols) > self.config['num_symbols_to_test']:
                selected_symbols = np.random.choice(
                    selected_symbols,
                    min(len(selected_symbols), self.config['num_symbols_to_test']),
                    replace=False
                ).tolist()
                logging.info(f"Selected a subset of {len(selected_symbols)} symbols for calibration.")
            else:
                logging.info(f"Using {len(selected_symbols)} eligible symbols for calibration.")

            # This method (run_full_calibration_pipeline) would typically orchestrate calls
            # to `optimize_strategy` for each strategy type.
            # The current setup in `run_calibration` already handles this.
            # For this reason, the loop for parameter combinations is removed here,
            # as `optimize_strategy` is responsible for its own grid search.

            best_overall_metric = -float('inf')
            best_overall_params = {} # This will be populated by `run_calibration` or if this method does the full loop.

            # If this method is used as the primary entry point for a "full" run,
            # it would iterate through strategies and call optimize_strategy for each.
            # However, `run_calibration` serves as the main entry in this script.
            # So, this function is mostly a placeholder for a different workflow or a high-level summary.

            # The `self.best_parameters` is populated by `run_calibration`, not this `pipeline` directly.
            # So, the final reporting will use `self.best_parameters` from the calibrator instance.

            logging.info("Optimization orchestration handled by `run_calibration` method.")


        except Exception as e:
            logging.critical(f"❌ Critical error during calibration pipeline: {e}")
            logging.critical(traceback.format_exc())
        finally:
            end_time = time.time()
            total_time = (end_time - start_time) / 60
            logging.info(f"⏱️ Total calibration time: {total_time:.2f} minutes.")
            logging.info("Calibration pipeline finished.")

    def get_test_size_config(self, size):
        """Configure test parameters based on size"""
        logging.info(f"⚙️ Configuring test parameters for size: '{size}'")
        configs = {
            'sanity': {'stocks': 2, 'test_points': 10, 'param_samples': 2},
            'small': {'stocks': 10, 'test_points': 100, 'param_samples': 10},
            'medium': {'stocks': 12, 'test_points': 200, 'param_samples': 20},
            'full': {'stocks': 20, 'test_points': 300, 'param_samples': 24},
        }
        selected_config = configs.get(size, configs['medium'])
        logging.info(
            f"Selected config: Stocks={selected_config['stocks']}, Test Points={selected_config['test_points']}, Param Samples={selected_config['param_samples']}")
        return selected_config

    def run_calibration(self, test_size='medium', strategies=['balanced']):
        """
        Main function to run the auto-calibration process.
        """
        logging.info("🚀 Starting StockWise Auto-Calibration ({test_size} test)")
        logging.info("============================================================")
        logging.info(f"Strategies to optimize: {', '.join(strategies)}")

        # Adjust configuration based on test size
        test_config = self.config['test_config'].get(test_size, self.config['test_config']['medium'])
        num_stocks = test_config['stocks']
        num_test_points = test_config['test_points']
        param_samples = test_config['param_samples']

        # Store original walk_forward config to restore later
        original_wf_config = self.config['walk_forward'].copy()

        # Temporarily adjust walk-forward config for smaller tests if specified
        if test_size == 'sanity':
            self.config['walk_forward']['train_months'] = 2  # 2 months train
            self.config['walk_forward']['test_months'] = 1  # 1 month test
            self.config['walk_forward']['step_months'] = 1  # 1 month step
            self.config['num_walk_forward_windows'] = 2  # Fewer windows for sanity
        elif test_size == 'small':
            self.config['walk_forward']['train_months'] = 6  # 6 months train
            self.config['walk_forward']['test_months'] = 2  # 2 months test
            self.config['walk_forward']['step_months'] = 2  # 2 month step
            self.config['num_walk_forward_windows'] = 3  # More windows for small

        logging.info(f"⚙️ Configuring test parameters for size: '{test_size}'")
        logging.info(
            f"Selected config: Stocks={num_stocks}, Test Points={num_test_points}, Param Samples={param_samples}")

        logging.info("📊 Step 1: Preparing data...")
        stock_universe = self.get_stock_universe(num_stocks=num_stocks)
        test_timestamps = self.generate_test_timestamps(num_test_points=num_test_points)
        validation_windows = self.create_walk_forward_windows()

        # Ensure that validation_windows actually contains windows
        if not validation_windows:
            logging.error("❌ No walk-forward validation windows could be created. Aborting calibration.")
            return self.get_empty_metrics()  # Return empty if no windows to process

        logging.info("✅ Data Preparation Complete: Loaded {} stocks, {} timestamps, {} validation windows".format(
            len(stock_universe), len(test_timestamps), len(validation_windows)
        ))

        # Perform optimization for each selected strategy
        for strategy in strategies:
            self.optimize_strategy(strategy, stock_universe, test_timestamps, validation_windows, param_samples)

        # Restore original walk_forward config
        self.config['walk_forward'] = original_wf_config
        self.config['num_walk_forward_windows'] = original_wf_config.get('num_walk_forward_windows',
                                                                         4)  # Restore default if not explicitly set

        return self.best_parameters

    def get_parameter_space(self, strategy_type):
        """
        Dynamically generates the parameter space for a given strategy type.
        This ensures only relevant parameters are combined for each strategy.
        """
        param_space = {}
        advisor_params = self.config['advisor_params']

        # Global parameters applicable to all strategies
        param_space['investment_days'] = advisor_params['investment_days']
        param_space['profit_threshold'] = advisor_params['profit_threshold']
        param_space['stop_loss_threshold'] = advisor_params['stop_loss_threshold']

        # Signal weights (assuming these are global for now, but can be strategy-specific)
        param_space['weight_trend'] = advisor_params['weight_trend']
        param_space['weight_momentum'] = advisor_params['weight_momentum']
        param_space['weight_volume'] = advisor_params['weight_volume']
        param_space['weight_support_resistance'] = advisor_params['weight_support_resistance']
        param_space['weight_ai_model'] = advisor_params['weight_ai_model']

        # Confidence parameters (assuming global for now, but can be strategy-specific)
        param_space['confidence_base_multiplier'] = advisor_params['confidence_base_multiplier']
        param_space['confidence_confluence_weight'] = advisor_params['confidence_confluence_weight']
        param_space['confidence_penalty_strength'] = advisor_params['confidence_penalty_strength']

        # Strategy-specific thresholds
        strategy_lower = strategy_type.lower().replace(' ', '_')

        # Ensure the correct key exists before adding to param_space
        if f'confidence_req_{strategy_lower}' in advisor_params:
            param_space[f'confidence_req_{strategy_lower}'] = advisor_params[f'confidence_req_{strategy_lower}']

        if f'buy_threshold_{strategy_lower}' in advisor_params:
            param_space[f'buy_threshold_{strategy_lower}'] = advisor_params[f'buy_threshold_{strategy_lower}']

        if f'sell_threshold_{strategy_lower}' in advisor_params:
            param_space[f'sell_threshold_{strategy_lower}'] = advisor_params[f'sell_threshold_{strategy_lower}']

        param_space['strategy_type'] = [strategy_type]  # Add the strategy type itself to the parameters

        return param_space

    def get_stock_universe(self, num_stocks=None):
        """
        Retrieves a list of NASDAQ-100 symbols, filtering by market cap, price, and volume.
        """
        logging.info(f"Retrieving stock universe. Desired count: {num_stocks if num_stocks else 'All'}")

        # This list is a static approximation for testing. In a real scenario,
        # you'd fetch this from a reliable source like a financial API.
        # Ensure these symbols generally have enough historical data for the test period.
        nasdaq_100_symbols = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA", "TSLA", "META", "ADBE", "NFLX",
            "CMCSA", "PEP", "COST", "AMD", "INTC", "CSCO", "QCOM", "AMGN", "SBUX", "MDLZ",
            "PYPL", "INTU", "TXN", "AMAT", "CHTR", "GILD", "VRTX", "BKNG", "FISV", "ADP",
            "ISRG", "REGN", "ATVI", "TMUS", "CERN", "KHC", "ADSK", "ILMN", "LRCX", "WBA",
            "MNST", "JD", "BIDU", "NTES", "MELI", "SNPS", "CRWD", "MAR", "LULU", "ROST",
            "EXC", "XEL", "AEP", "PCAR", "DLTR", "SWKS", "ASML", "ORLY", "CDNS", "CTAS",
            "MCHP", "IDXX", "FAST", "CPRT", "MRNA", "SPLK", "VRSK", "SGEN", "DXCM", "ANSS",
            "WDAY", "TCOM", "BIIB", "ODFL", "PDD", "TEAM", "KLAC", "PAYX", "PTON", "MRVL",
            "AZN", "ZM", "OKTA", "PENN", "DDOG", "ZS", "DOCU", "NFLX", "SPLG", "SQ",
            "ZI", "CRSP", "VEEV", "OKTA", "DDOG", "ZS", "FSLR", "ENPH", "PLUG", "ALGN"
        ]

        # Filter symbols based on a very simplified criteria for a demo
        # In a real scenario, you'd fetch current market data for these filters.
        # For testing purposes, we'll just simulate a selection.
        selected_symbols = nasdaq_100_symbols  # Start with all

        if num_stocks and len(selected_symbols) > num_stocks:
            # For consistent testing, use a fixed random seed or just take the first N
            selected_symbols = sorted(selected_symbols)[:num_stocks]  # Deterministic subset
            logging.info(f"Selected a subset of {num_stocks} stocks for testing.")
        else:
            logging.info(f"Using all {len(selected_symbols)} available stocks.")

        logging.info(f"Loaded stock universe: {len(selected_symbols)} symbols.")
        return selected_symbols

    def generate_test_timestamps(self, num_test_points):
        """
        Generates a list of evenly spaced test timestamps within the calibration range.
        """
        start = datetime.strptime(self.config['start_date'], '%Y-%m-%d').date()
        end = datetime.strptime(self.config['end_date'], '%Y-%m-%d').date()
        logging.info(f"Generating {num_test_points} evenly spaced test timestamps.")

        business_days = pd.bdate_range(start=start, end=end)
        logging.info(f"Found {len(business_days)} business days between {start} and {end}.")

        if len(business_days) < num_test_points:
            logging.warning(f"Not enough business days ({len(business_days)}) to generate {num_test_points} test points. Using all available business days.")
            selected_dates = business_days.tolist()
        else:
            step = max(1, len(business_days) // num_test_points)
            selected_dates = business_days[::step].tolist()
            # Ensure we have exactly num_test_points and it ends near the desired end
            if len(selected_dates) > num_test_points:
                selected_dates = selected_dates[:num_test_points]
            elif len(selected_dates) < num_test_points:
                # Add more dates if needed, taking from the end
                missing = num_test_points - len(selected_dates)
                if len(business_days) > len(selected_dates) + missing:
                    selected_dates.extend(business_days[-missing:].tolist())
                selected_dates = sorted(list(set(selected_dates))) # Remove duplicates and sort

        logging.info(f"Selected {len(selected_dates)} timestamps with a step of {step} days.")
        logging.info(f"Generated {len(selected_dates)} test timestamps between {selected_dates[0]} and {selected_dates[-1]}.")
        return selected_dates

    def get_timestamps_in_window(self, start_date, end_date):
        """Get timestamps within a specific window"""
        business_days = pd.bdate_range(start=start_date, end=end_date)
        step = max(1, len(business_days) // 20)
        selected_dates = business_days[::step]
        logging.debug(f"Selected {len(selected_dates)} timestamps in window {start_date.date()} to {end_date.date()}.")
        return [date.date() for date in selected_dates]

    def apply_parameters_to_advisor(self, parameters):
        """Apply a parameter set to the advisor; normalize strategy-scoped keys."""
        logging.debug(f"Applying parameters to advisor: {parameters}")

        # Determine current strategy
        strategy_type = parameters.get('strategy_type')
        if isinstance(strategy_type, list):  # when coming from param_space
            strategy_type = strategy_type[0]
        if not strategy_type:
            raise ValueError("Parameter set missing 'strategy_type'.")

        self.advisor.current_strategy = strategy_type

        strategy_lower = strategy_type.lower().replace(' ', '_')

        # Normalize thresholds (accept both generic and strategy-specific keys)
        buy = parameters.get(f"buy_threshold_{strategy_lower}",
                             parameters.get("buy_threshold"))
        sell = parameters.get(f"sell_threshold_{strategy_lower}",
                              parameters.get("sell_threshold"))
        min_conf = parameters.get(f"confidence_req_{strategy_lower}",
                                  parameters.get("min_confidence"))

        # Push thresholds into advisor (if present)
        if buy is not None:
            setattr(self.advisor, "current_buy_threshold", float(buy))
        if sell is not None:
            setattr(self.advisor, "current_sell_threshold", float(sell))

        # Update strategy multipliers/confidence in advisor.strategy_settings
        if hasattr(self.advisor, "strategy_settings") and strategy_type in self.advisor.strategy_settings:
            s = self.advisor.strategy_settings[strategy_type]
            if "profit_threshold" in parameters:
                s["profit"] = float(parameters["profit_threshold"])
            if "stop_loss_threshold" in parameters:
                s["risk"] = float(parameters["stop_loss_threshold"])
            if min_conf is not None:
                s["confidence_req"] = float(min_conf)

        # Update signal weights if provided
        if hasattr(self.advisor, "signal_weights"):
            sw = dict(self.advisor.signal_weights)
            sw["trend"] = parameters.get("weight_trend", sw.get("trend"))
            sw["momentum"] = parameters.get("weight_momentum", sw.get("momentum"))
            sw["volume"] = parameters.get("weight_volume", sw.get("volume"))
            sw["support_resistance"] = parameters.get("weight_support_resistance", sw.get("support_resistance"))
            sw["ai_model"] = parameters.get("weight_ai_model", sw.get("ai_model"))
            # strip Nones
            self.advisor.signal_weights = {k: v for k, v in sw.items() if v is not None}

        # Confidence shaping parameters (global)
        for k_cfg, k_param in [
            ("confidence_base_multiplier", "confidence_base_multiplier"),
            ("confidence_confluence_weight", "confidence_confluence_weight"),
            ("confidence_penalty_strength", "confidence_penalty_strength"),
        ]:
            if parameters.get(k_param) is not None:
                setattr(self.advisor, k_cfg, float(parameters[k_param]))

    def calculate_actual_return(self, symbol, start_date, holding_days):
        """
        Calculates the actual return for a stock over a given holding period.
        """
        end_date = start_date + timedelta(days=holding_days)

        try:
            # Download data starting slightly before to ensure start_date is included
            df = yf.download(symbol, start=start_date - timedelta(days=5), end=end_date + timedelta(days=5),
                             progress=False, show_errors=False)
            if df.empty:
                logging.warning(f"No data from yfinance for {symbol} from {start_date} to {end_date}.")
                return None

            # Find the closest actual market date for start_date
            start_dt = pd.to_datetime(start_date)
            if start_dt in df.index:
                start_price_date = start_dt
            elif not df.index.empty:
                start_price_date = df.index[np.argmin(np.abs(df.index - start_dt))]
            else:
                logging.warning(f"No data points for {symbol} around {start_date}.")
                return None

            start_price = df['Close'].loc[start_price_date]

            # Find the closest actual market date for end_date within the holding period
            end_price_candidate_dates = df.loc[
                df.index >= pd.to_datetime(start_date) + timedelta(days=holding_days)].index
            if end_price_candidate_dates.empty:
                logging.warning(
                    f"No future data for {symbol} after {start_date} for {holding_days} days holding period.")
                return None

            end_price_date = end_price_candidate_dates[0]
            end_price = df['Close'].loc[end_price_date]

            actual_return = ((end_price - start_price) / start_price) * 100
            logging.debug(
                f"Calculated actual return for {symbol} from {start_price_date.date()} to {end_price_date.date()}: {actual_return:.2f}%")
            return actual_return
        except Exception as e:
            logging.error(f"Error calculating actual return for {symbol} from {start_date}: {e}")
            return None

    def calculate_performance_metrics(self, results):
        """
        Calculates key performance metrics from a list of evaluation results.
        """
        if not results:
            return self.get_empty_metrics()

        total_correct = 0
        total_direction_correct = 0
        total_profitable_trades = 0
        total_buy_signals = 0
        total_sell_signals = 0
        total_wait_signals = 0
        total_trades = len(results)

        returns = []
        confidences = []

        buy_success_trades = 0

        for res in results:
            # Ensure 'net_actual_return' is present and is not None before appending
            if res['performance'].get('net_actual_return') is not None:
                returns.append(res['performance']['net_actual_return'])  # Use net actual return for performance

            if 'confidence' in res['recommendation']:
                confidences.append(res['recommendation']['confidence'])

            # Access performance flags directly from the nested 'performance' dictionary
            if res['performance'].get('correct', False):
                total_correct += 1
            if res['performance'].get('direction_correct', False):
                total_direction_correct += 1
            if res['performance'].get('profitable', False):
                total_profitable_trades += 1

            action = res['recommendation']['action']
            if action == "BUY":
                total_buy_signals += 1
                if res['performance'].get('profitable', False):
                    buy_success_trades += 1
            elif action == "SELL/AVOID":
                total_sell_signals += 1
            elif action == "WAIT":
                total_wait_signals += 1

        overall_accuracy = (total_correct / total_trades) * 100 if total_trades > 0 else 0
        direction_accuracy = (total_direction_correct / total_trades) * 100 if total_trades > 0 else 0
        buy_success_rate = (buy_success_trades / total_buy_signals) * 100 if total_buy_signals > 0 else 0

        avg_return = np.mean(returns) if returns else 0
        volatility = np.std(returns) if len(returns) > 1 else 0

        sharpe_ratio = avg_return / volatility if volatility != 0 else 0

        if returns:
            returns_decimal = np.array(returns) / 100.0
            cumulative_returns_product = np.cumprod(1 + returns_decimal)
            cumulative_returns_product = np.insert(cumulative_returns_product, 0, 1)

            peak = np.maximum.accumulate(cumulative_returns_product)
            drawdown_ratio = np.where(peak != 0, (cumulative_returns_product - peak) / peak, 0)
            max_drawdown = np.min(drawdown_ratio) * 100
        else:
            max_drawdown = 0

        confidence_avg = np.mean(confidences) if confidences else 0

        signal_distribution = {
            "BUY": (total_buy_signals / total_trades) * 100 if total_trades > 0 else 0,
            "SELL/AVOID": (total_sell_signals / total_trades) * 100 if total_trades > 0 else 0,
            "WAIT": (total_wait_signals / total_trades) * 100 if total_trades > 0 else 0,
        }

        return {
            "overall_accuracy": overall_accuracy,
            "direction_accuracy": direction_accuracy,
            "buy_success_rate": buy_success_rate,
            "avg_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "confidence_avg": confidence_avg,
            "signal_distribution": signal_distribution
        }

    def evaluate_prediction(self, recommendation, actual_return):
        """
        Evaluates the accuracy and profitability of a recommendation.
        """
        eval_thresholds = self.config['evaluation_thresholds']
        buy_profit_min = eval_thresholds['buy_profit_min']
        sell_loss_min = eval_thresholds['sell_loss_min']
        wait_max_change = eval_thresholds['wait_max_change']

        action = recommendation['action']

        correct = False
        direction_correct = False
        profitable = False

        if actual_return is None:
            return {'correct': False, 'direction_correct': False, 'profitable': False, 'actual_return': None}

        # Apply Israeli fees and tax to the actual return for evaluation
        net_actual_return = self.advisor.apply_israeli_fees_and_tax(actual_return)

        if action == "BUY":
            direction_correct = net_actual_return > 0
            profitable = net_actual_return >= buy_profit_min
            correct = profitable
        elif action == "SELL/AVOID":
            direction_correct = net_actual_return < 0  # Price went down
            profitable = net_actual_return <= sell_loss_min  # Avoided loss (or made profit by shorting/inverse)
            correct = profitable
        elif action == "WAIT":
            direction_correct = abs(net_actual_return) <= wait_max_change
            profitable = True  # Waiting is considered 'profitable' if it avoids significant loss/gain, or is flat
            correct = direction_correct

        return {
            'correct': correct,
            'direction_correct': direction_correct,
            'profitable': profitable,
            'actual_return': actual_return,
            'net_actual_return': net_actual_return  # Store net return for more accurate performance metrics
        }

    def aggregate_walk_forward_results(self, window_results):
        """Aggregate results across all walk-forward windows"""
        logging.info(f"Aggregating {len(window_results)} walk-forward window results.")
        if not window_results:
            logging.warning("No window results to aggregate. Returning empty metrics.")
            return self.get_empty_metrics()

        all_metrics = [wr['performance'] for wr in window_results if wr['performance']]

        if not all_metrics:
            logging.warning(
                "No valid performance metrics found across all windows after filtering. Returning empty metrics.")
            return self.get_empty_metrics()

        overall_accuracies = [m['overall_accuracy'] for m in all_metrics if m['overall_accuracy'] is not None]
        direction_accuracies = [m['direction_accuracy'] for m in all_metrics if m['direction_accuracy'] is not None]
        buy_success_rates = [m['buy_success_rate'] for m in all_metrics if m['buy_success_rate'] is not None]
        avg_returns = [m['avg_return'] for m in all_metrics if m['avg_return'] is not None]
        volatilities = [m['volatility'] for m in all_metrics if m['volatility'] is not None]
        sharpe_ratios = [m['sharpe_ratio'] for m in all_metrics if m['sharpe_ratio'] is not None]
        max_drawdowns = [m['max_drawdown'] for m in all_metrics if m['max_drawdown'] is not None]
        total_trades = sum([m['total_trades'] for m in all_metrics])
        confidence_avgs = [m['confidence_avg'] for m in all_metrics if m['confidence_avg'] is not None]

        aggregated = {
            'overall_accuracy': np.mean(overall_accuracies) if overall_accuracies else 0,
            'direction_accuracy': np.mean(direction_accuracies) if direction_accuracies else 0,
            'buy_success_rate': np.mean(buy_success_rates) if buy_success_rates else 0,
            'avg_return': np.mean(avg_returns) if avg_returns else 0,
            'volatility': np.mean(volatilities) if volatilities else 0,
            'sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            'total_trades': total_trades,
            'confidence_avg': np.mean(confidence_avgs) if confidence_avgs else 50
        }
        logging.info(f"Aggregated metrics calculated: Overall Accuracy={aggregated['overall_accuracy']:.2f}%")

        signal_dist_buy = [m['signal_distribution']['BUY'] for m in all_metrics if
                           'BUY' in m.get('signal_distribution', {}) and m['signal_distribution']['BUY'] is not None]
        signal_dist_sell = [m['signal_distribution']['SELL/AVOID'] for m in all_metrics if
                            'SELL/AVOID' in m.get('signal_distribution', {}) and m['signal_distribution']['SELL/AVOID'] is not None]
        signal_dist_wait = [m['signal_distribution']['WAIT'] for m in all_metrics if
                            'WAIT' in m.get('signal_distribution', {}) and m['signal_distribution']['WAIT'] is not None]

        signal_dist = {
            'BUY': np.mean(signal_dist_buy) if signal_dist_buy else 0,
            'SELL/AVOID': np.mean(signal_dist_sell) if signal_dist_sell else 0,
            'WAIT': np.mean(signal_dist_wait) if signal_dist_wait else 100
        }

        aggregated['signal_distribution'] = signal_dist
        logging.info(f"Aggregated signal distribution: {signal_dist}")

        return aggregated

    def get_empty_metrics(self):
        """Returns a dictionary with empty/zero metrics."""
        return {
            "overall_accuracy": 0.0,
            "direction_accuracy": 0.0,
            "buy_success_rate": 0.0,
            "avg_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "confidence_avg": 0.0,
            "signal_distribution": {"BUY": 0.0, "SELL/AVOID": 0.0, "WAIT": 0.0}
        }

    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns"""
        if not returns:
            logging.debug("No returns data for drawdown calculation. Returning 0.")
            return 0

        valid_returns = [r for r in returns if r is not None]
        if not valid_returns:
            logging.debug("No valid returns data for drawdown calculation. Returning 0.")
            return 0

        cumulative = np.cumprod(1 + np.array(valid_returns) / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100

        logging.debug(f"Calculated max drawdown: {np.min(drawdown):.2f}%")
        return np.min(drawdown)

    def generate_parameter_combinations(self, param_space):
        """
        Generates all possible combinations of parameters from a given parameter space.
        This is a generic utility function.

        Args:
            param_space (dict): A dictionary where keys are parameter names and values
                                are lists of possible values for that parameter.

        Returns:
            list: A list of dictionaries, each representing a unique combination of parameters.
        """
        keys = param_space.keys()
        values = param_space.values()

        # Use itertools.product to get all combinations
        combinations = list(product(*values))

        # Convert combinations back into a list of dictionaries
        param_grid = []
        for combo in combinations:
            param_grid.append(dict(zip(keys, combo)))
        return param_grid

    def create_performance_summary(self):
        """Create performance summary for report."""
        logging.info("Creating overall performance summary.")
        summary = {}

        for strategy, results in self.best_parameters.items():
            params = results.get('parameters', {})
            s_lower = strategy.lower().replace(' ', '_')

            buy = params.get(f"buy_threshold_{s_lower}", params.get("buy_threshold"))
            sell = params.get(f"sell_threshold_{s_lower}", params.get("sell_threshold"))
            min_conf = params.get(f"confidence_req_{s_lower}", params.get("min_confidence"))

            summary[strategy] = {
                'fitness_score': results['fitness_score'],
                'performance_metrics': results['performance'],
                'optimal_parameters': {
                    'buy_threshold': buy,
                    'sell_threshold': sell,
                    'min_confidence': min_conf,
                    'profit_threshold': params.get('profit_threshold'),
                    'stop_loss_threshold': params.get('stop_loss_threshold'),
                    'signal_weights': {
                        'trend': params.get('weight_trend'),
                        'momentum': params.get('weight_momentum'),
                        'volume': params.get('weight_volume'),
                        'support_resistance': params.get('weight_support_resistance'),
                        'ai_model': params.get('weight_ai_model'),
                    },
                    'confidence_params': {
                        'confidence_base_multiplier': params.get('confidence_base_multiplier'),
                        'confidence_confluence_weight': params.get('confidence_confluence_weight'),
                        'confidence_penalty_strength': params.get('confidence_penalty_strength'),
                    }
                }
            }
        return summary

    def create_walk_forward_windows(self):
        """
        Creates walk-forward validation windows based on the configuration.
        """
        logging.info("Generating walk-forward validation windows.")
        train_months = self.config['walk_forward']['train_months']
        test_months = self.config['walk_forward']['test_months']
        step_months = self.config['walk_forward']['step_months']
        num_windows = self.config['num_walk_forward_windows']

        windows = []
        current_train_end = datetime.strptime(self.config['start_date'], '%Y-%m-%d') + timedelta(
            days=train_months * 30)  # Approximate

        for i in range(num_windows):
            if current_train_end.date() >= datetime.strptime(self.config['end_date'], '%Y-%m-%d').date():
                logging.warning(f"Reached end date. Stopping walk-forward window generation at {len(windows)} windows.")
                break

            train_start = datetime.strptime(self.config['start_date'], '%Y-%m-%d') + timedelta(
                days=i * step_months * 30)
            train_end = train_start + timedelta(days=train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_months * 30)

            # Adjust to actual business days if needed (simple approximation for now)
            # Find the closest business day for each boundary if exact dates are required

            # Ensure the test_end does not exceed the overall end_date
            overall_end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
            if test_end > overall_end_date:
                test_end = overall_end_date

            window = {
                'train_start': train_start.date(),
                'train_end': train_end.date(),
                'test_start': test_start.date(),
                'test_end': test_end.date(),
                'window_id': i + 1
            }
            windows.append(window)
            logging.info(
                f"Generated window {i + 1}: Train ({window['train_start']} - {window['train_end']}), Test ({window['test_start']} - {window['test_end']})")

            current_train_end = train_end + timedelta(days=step_months * 30)  # Move to the next step

        logging.info(f"Finished generating {len(windows)} walk-forward windows.")
        return windows

    def validate_parameters_walk_forward(self, parameters, stocks, windows):
        """Validate parameters using walk-forward analysis"""
        window_results = []

        for window in tqdm(windows, desc="Walk-Forward Validation"):
            logging.info(
                f"Processing window {window['window_id']}: Test period {window['test_start'].date()} to {window['test_end'].date()}")
            test_timestamps = self.get_timestamps_in_window(
                window['test_start'],
                window['test_end']
            )
            logging.debug(f"Window {window['window_id']} has {len(test_timestamps)} timestamps for testing.")

            window_performance = self.test_parameter_set(
                parameters, stocks, test_timestamps, window
            )

            window_results.append({
                'window_id': window['window_id'],
                'test_period': f"{window['test_start'].date()} to {window['test_end'].date()}",
                'performance': window_performance
            })
            logging.info(
                f"Window {window['window_id']} performance: Overall Accuracy={window_performance['overall_accuracy']:.2f}%, Sharpe={window_performance['sharpe_ratio']:.2f}")

        aggregated_performance = self.aggregate_walk_forward_results(window_results)
        logging.info(
            f"Aggregated performance for this parameter set across all windows: Fitness Score={self.calculate_fitness_score(aggregated_performance):.2f}")
        return aggregated_performance

    def test_parameter_set(self, parameters, stocks, timestamps, window):
        """Test a specific parameter set"""
        logging.debug(
            f"Testing parameter set on {len(stocks)} stocks for {len(timestamps)} timestamps in window {window['window_id']}.")

        self.apply_parameters_to_advisor(parameters)
        logging.debug("Parameters applied to advisor instance.")

        results = []
        for stock in stocks:
            for timestamp in timestamps:
                try:
                    # Run analysis with current parameters
                    # This call will now use the parameters just applied by apply_parameters_to_advisor
                    recommendation = self.advisor.analyze_stock_enhanced(stock, timestamp)

                    if recommendation:
                        logging.debug(
                            f"  {stock} on {timestamp}: Recommendation: {recommendation['action']} (Confidence: {recommendation['confidence']:.1f}%)")
                        actual_return = self.calculate_actual_return(
                            stock, timestamp, self.config['prediction_window_days']
                        )

                        performance = self.evaluate_prediction(recommendation, actual_return)

                        normalized_recommendation = self._normalize_numeric_values(recommendation)
                        normalized_performance = self._normalize_numeric_values(performance)
                        normalized_actual_return = self._normalize_numeric_values(actual_return)

                        actual_return_str = f"{normalized_actual_return:.2f}" if normalized_actual_return is not None else "N/A"

                        results.append({
                            'stock': stock,
                            'timestamp': timestamp,
                            'recommendation': normalized_recommendation,
                            'actual_return': normalized_actual_return,
                            'performance': normalized_performance
                        })
                        logging.debug(
                            f"  {stock} on {timestamp}: Actual Return={actual_return_str}%, Evaluation={normalized_performance['correct']}")

                except Exception as e:
                    logging.error(
                        f"❌ Critical Error during testing {stock} on {timestamp} in window {window['window_id']}: {e}")
                    logging.error(f"  Full traceback: {traceback.format_exc()}")
                    results.append({
                        'stock': stock,
                        'timestamp': timestamp,
                        'recommendation': self._normalize_numeric_values(
                            {'action': 'WAIT', 'confidence': 50, 'expected_profit_pct': 0}),
                        'actual_return': self._normalize_numeric_values(None),
                        'performance': self._normalize_numeric_values(self.get_empty_metrics())
                    })
                    continue

        calculated_metrics = self.calculate_performance_metrics(results)
        logging.debug(f"Performance metrics calculated for this parameter set in window {window['window_id']}.")
        return calculated_metrics

    def aggregate_window_metrics(self, window_performances):
        """
        Aggregates performance metrics from multiple walk-forward windows.
        This provides an overall picture for a given parameter set.
        """
        if not window_performances:
            return self.get_empty_metrics()

        # Sum up total trades and average others
        total_trades_sum = sum(wp.get('total_trades', 0) for wp in window_performances)

        # For metrics that are percentages or ratios, we can average them.
        # However, for avg_return and Sharpe, a more rigorous aggregation might be needed
        # (e.g., compounding returns or considering standard deviation of all trades).
        # For simplicity, we'll average here.

        agg_metrics = {k: [] for k in self.get_empty_metrics().keys() if
                       k not in ['total_trades', 'signal_distribution']}
        agg_metrics['signal_distribution'] = {k: [] for k in self.get_empty_metrics()['signal_distribution'].keys()}

        for wp in window_performances:
            for k, v in wp.items():
                if k == 'total_trades':
                    continue  # Handled separately
                elif k == 'signal_distribution':
                    for signal_type, percentage in v.items():
                        agg_metrics['signal_distribution'][signal_type].append(percentage)
                else:
                    agg_metrics[k].append(v)

        final_agg_metrics = {}
        for k, v_list in agg_metrics.items():
            if k == 'signal_distribution':
                final_agg_metrics[k] = {signal_type: np.mean(percentages) for signal_type, percentages in
                                        v_list.items()}
            else:
                final_agg_metrics[k] = np.mean(v_list) if v_list else 0.0

        final_agg_metrics['total_trades'] = total_trades_sum

        # Recalculate sharpe ratio based on aggregated returns and volatility (if needed)
        # For now, relying on averaged sharpe from windows

        return final_agg_metrics

    def optimize_strategy(self, strategy_type, test_stocks, test_timestamps, validation_windows, param_samples):
        """
        Optimizes parameters for a single strategy type using walk-forward validation.
        """
        logging.info(f"\n🎯 Step 2: Optimizing {strategy_type.upper()} strategy...")
        logging.info(f"🔍 Optimizing {strategy_type} strategy...")


        # Generate parameter combinations for this specific strategy
        # parameter_combinations = self.generate_parameter_combinations(strategy_type, max_combinations=param_samples)
        # ✅ Build parameter combinations properly for this strategy
        parameter_space = self.get_parameter_space(strategy_type)
        parameter_combinations = self.generate_parameter_combinations(parameter_space, max_combinations=param_samples)

        logging.info(f"Testing {len(parameter_combinations)} parameter combinations for {strategy_type} strategy...")

        best_fitness_score = -np.inf
        best_params_for_strategy = None
        best_performance_for_strategy = None

        for i, params in enumerate(tqdm(parameter_combinations, desc=f"Optimizing {strategy_type}")):
            try:
                logging.info(f"--- Testing Parameter Set {i + 1}/{len(parameter_combinations)} for {strategy_type} ---")
                logging.info(f"  Parameters: {params}")

                # Apply current parameter set to the advisor
                self.apply_parameters_to_advisor(params)

                window_performances = []
                all_window_results = []

                # Perform walk-forward validation for this parameter set
                for window in validation_windows:
                    logging.info(
                        f"Processing window {window['window_id']}: Test period {window['test_start']} to {window['test_end']}")

                    # Temporarily update advisor's data fetching range for this window
                    original_advisor_start_date = self.advisor.config_file['start_date'] if hasattr(self.advisor,
                                                                                                    'config_file') and self.advisor.config_file else None
                    original_advisor_end_date = self.advisor.config_file['end_date'] if hasattr(self.advisor,
                                                                                                'config_file') and self.advisor.config_file else None

                    # If the advisor loads config, update it or pass dates directly
                    # For current setup, advisor fetches data based on 'end_date' and 'days_back' for analyze_stock_enhanced
                    # So, we pass 'end_date' directly to analyze_stock_enhanced

                    window_results = []
                    # Filter test timestamps to be within the current window's test period
                    current_window_timestamps = [
                        ts for ts in test_timestamps
                        if window['test_start'] <= ts.date() <= window['test_end']  # FIX IS HERE: ts.date()
                    ]

                    # Shuffle stock symbols and timestamps for more robust testing per window
                    current_test_symbols = list(test_stocks)  # Copy to shuffle
                    np.random.shuffle(current_test_symbols)
                    np.random.shuffle(current_window_timestamps)

                    for symbol in current_test_symbols:
                        for timestamp in current_window_timestamps:
                            rec = self.advisor.analyze_stock_enhanced(symbol, timestamp.strftime('%Y-%m-%d'))

                            if rec:
                                actual_return = self.calculate_actual_return(
                                    symbol,
                                    timestamp,  # Use the original timestamp for return calculation
                                    self.config['prediction_window_days']  # Holding period
                                )
                                eval_result = self.evaluate_prediction(rec, actual_return)

                                # Add recommendation and actual_return to eval_result for full context
                                eval_result['recommendation'] = rec
                                eval_result['actual_return_raw'] = actual_return  # Store original actual return
                                window_results.append({
                                    'symbol': symbol,
                                    'timestamp': timestamp.strftime('%Y-%m-%d'),
                                    'recommendation': rec,
                                    'actual_return': actual_return,  # This will be raw return
                                    'evaluation': eval_result
                                })
                            else:
                                logging.warning(f"Skipping {symbol} on {timestamp}: No recommendation generated.")

                    if window_results:
                        window_performance = self.calculate_performance_metrics(
                            [r['evaluation'] for r in window_results])
                        window_performances.append(window_performance)
                        all_window_results.extend(window_results)  # Collect all detailed results
                        logging.info(
                            f"Window {window['window_id']} performance: Overall Accuracy={window_performance['overall_accuracy']:.2f}%, Sharpe={window_performance['sharpe_ratio']:.2f}")
                    else:
                        logging.warning(
                            f"No results for window {window['window_id']}. Skipping performance calculation for this window.")

            except Exception as e:
                logging.error(f"❌ Error during optimization of parameter set {i + 1} for {strategy_type}: {e}")
                logging.error(f"Traceback for parameter set error:\n{traceback.format_exc()}")
                continue  # Skip to next parameter set

        if window_performances:
            logging.info(f"Aggregating {len(window_performances)} walk-forward window results.")
            # Aggregate performance across all windows for this parameter set
            aggregated_metrics = self.aggregate_window_metrics(window_performances)
            current_fitness_score = self.calculate_fitness_score(aggregated_metrics)
            logging.info(
                f"Aggregated metrics calculated: Overall Accuracy={aggregated_metrics['overall_accuracy']:.2f}%")
            logging.info(f"Aggregated signal distribution: {aggregated_metrics['signal_distribution']}")
            logging.info(
                f"Aggregated performance for this parameter set across all windows: Fitness Score={current_fitness_score:.2f}")

            if current_fitness_score > best_fitness_score:
                best_fitness_score = current_fitness_score
                best_params_for_strategy = params  # This parameter set is the best so far
                best_performance_for_strategy = aggregated_metrics
                logging.info(f"🎯 New best parameter set for {strategy_type}: Score {best_fitness_score:.2f}")
                logging.info(f"   Accuracy: {best_performance_for_strategy['overall_accuracy']:.1f}%")
                logging.info(f"   BUY Success: {best_performance_for_strategy['buy_success_rate']:.1f}%")
                logging.info(f"   Sharpe: {best_performance_for_strategy['sharpe_ratio']:.2f}")
        else:
            logging.warning(f"No valid performance data generated for any parameter set for {strategy_type}.")

        if best_params_for_strategy:
            self.best_parameters[strategy_type] = {
                "optimal_parameters": best_params_for_strategy,
                "performance": best_performance_for_strategy,
                "fitness_score": best_fitness_score
            }
            logging.info(
                f"✅ Optimization for {strategy_type} complete. Final Best Fitness Score: {best_fitness_score:.2f}")
        else:
            self.best_parameters[strategy_type] = {
                "optimal_parameters": {},
                "performance": self.get_empty_metrics(),
                "fitness_score": -np.inf
            }
            logging.warning(f"❌ Optimization for {strategy_type} completed but no optimal parameters were found.")

        logging.info(
            f"🏆 Optimization for {strategy_type.upper()} completed. Best Fitness Score: {best_fitness_score:.2f}")
        return self.best_parameters[strategy_type]

    def _save_intermediate_strategy_results(self, strategy_type, results_data, failed_index, error_message):
        """
        Saves intermediate results when an error occurs during strategy optimization.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"stockwise_partial_calibration_{strategy_type}_{timestamp}_failed_at_set_{failed_index}.json"
        file_location = os.path.join(self.configuration_files, filename)

        partial_report = {
            'metadata': {
                'calibration_date': timestamp,
                'strategy_type': strategy_type,
                'error_at_parameter_set_index': failed_index,
                'error_message': error_message,
                'status': 'partial_results_due_to_error'
            },
            'tested_parameters_results': self._normalize_numeric_values(results_data)
        }

        try:
            with open(file_location, 'w') as f:
                json.dump(partial_report, f, indent=2, default=str)
            logging.warning(f"⚠️ Partial results saved to: {file_location}")
        except Exception as e:
            logging.error(f"❌ Failed to save partial results: {e}. Error: {e}")

        # In algo_configurator.py, within the StockWiseAutoCalibrator class:

    def calculate_fitness_score(self, performance_metrics):
        """
        Calculates a single fitness score to evaluate parameter sets.
        Higher score = better performance. This is where you define 'optimal'.
        Adjust weights based on what you prioritize (e.g., accuracy, Sharpe).
        """
        # Ensure that required keys exist to prevent KeyError
        overall_accuracy = performance_metrics.get('overall_accuracy', 0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        buy_success_rate = performance_metrics.get('buy_success_rate', 0)
        avg_return = performance_metrics.get('avg_return', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0)
        total_trades = performance_metrics.get('total_trades', 0)

        # Penalize for very few trades as it might lead to unstable metrics
        trade_penalty = 0
        if total_trades < 50:  # Example threshold for minimum trades
            trade_penalty = (50 - total_trades) * 0.1  # Small penalty per missing trade

        # Normalize metrics to a similar scale if they vary widely (e.g., accuracy 0-100, Sharpe typically < 5)
        # Here, we assume a base target for each, and reward exceeding it.

        # Prioritize Sharpe Ratio and Overall Accuracy
        score = (overall_accuracy * 0.4) + \
                (max(0, sharpe_ratio) * 10) + \
                (buy_success_rate * 0.2) + \
                (max(0, avg_return) * 0.5)

        # Penalize for high drawdown, but ensure it's not double-penalized if Sharpe is already low
        # Only apply drawdown penalty if it's significantly negative
        if max_drawdown < -15:  # Example: penalize drawdowns worse than -15%
            score += max_drawdown * 0.5  # max_drawdown is negative, so adding it reduces score

        # Add a bonus for a healthy number of trades if it's not excessively penalized by accuracy
        if total_trades > 100:
            score += min(5, total_trades / 1000)  # Max 5 point bonus for high trade count

        score -= trade_penalty

        logging.debug(f"Fitness score calculated: {score:.2f}")
        return score

    def generate_final_report(self, execution_time_hours):
        """
        Generates a concise final report of the calibration results.
        """
        logging.info("\n" + "=" * 80)
        logging.info("📊 StockWise Auto-Calibration Final Report 📊")
        logging.info("=" * 80)
        logging.info(f"Calibration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total Execution Time: {execution_time_hours:.2f} hours")
        logging.info("-" * 80)

        if not self.best_parameters:
            logging.warning("No optimal parameters found or saved during this calibration run.")
            return

        for strategy, data in self.best_parameters.items():
            logging.info(f"\n✨ Optimal Parameters for Strategy: {strategy} ✨")

            # Unpack optimal parameters and performance metrics
            optimal_params = data.get('optimal_parameters', {})
            performance_metrics = data.get('performance', self.get_empty_metrics())
            fitness_score = data.get('fitness_score', 0.0)

            logging.info(f"  Best Fitness Score: {fitness_score:.2f}")
            logging.info("  Optimal Parameters:")
            for k, v in optimal_params.items():
                logging.info(f"    - {k}: {v}")

            logging.info("  Aggregated Performance Metrics:")
            logging.info(f"    - Overall Accuracy: {performance_metrics.get('overall_accuracy', 0.0):.2f}%")
            logging.info(f"    - Direction Accuracy: {performance_metrics.get('direction_accuracy', 0.0):.2f}%")
            logging.info(f"    - BUY Success Rate: {performance_metrics.get('buy_success_rate', 0.0):.2f}%")
            logging.info(f"    - Average Return: {performance_metrics.get('avg_return', 0.0):.2f}%")
            logging.info(f"    - Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0.0):.2f}")
            logging.info(f"    - Max Drawdown: {performance_metrics.get('max_drawdown', 0.0):.2f}%")
            logging.info(f"    - Total Trades: {performance_metrics.get('total_trades', 0)}")
            logging.info(f"    - Average Confidence: {performance_metrics.get('confidence_avg', 0.0):.2f}%")
            logging.info(
                f"    - Signal Distribution: BUY={performance_metrics.get('signal_distribution', {}).get('BUY', 0.0):.1f}%, SELL/AVOID={performance_metrics.get('signal_distribution', {}).get('SELL/AVOID', 0.0):.1f}%, WAIT={performance_metrics.get('signal_distribution', {}).get('WAIT', 0.0):.1f}%")
            logging.info("-" * 40)

        logging.info("=" * 80)
        logging.info("Calibration report and production parameters saved to 'configuration/' directory.")
        logging.info("Detailed logs available in 'logs/' directory.")
        logging.info(
            "Run `python your_script.py --compare` to compare historical runs.")  # Placeholder for future feature
        logging.info("=" * 80)

    def print_results_summary(self):
        """Print human-readable results summary"""

        logging.info("\n" + "=" * 80)
        logging.info("🎯 STOCKWISE AUTO-CALIBRATION RESULTS SUMMARY")
        logging.info("=" * 80)

        if not self.best_parameters:
            logging.info("No calibration results available to summarize.")
            logging.info("=" * 80)
            return

        for strategy, results in self.best_parameters.items():
            perf = results['performance']
            logging.info(f"\n📈 {strategy.upper()} STRATEGY:")
            logging.info(f"   Overall Accuracy: {perf['overall_accuracy']:.1f}%" if perf[
                                                                                        'overall_accuracy'] is not None else "   Overall Accuracy: N/A")
            logging.info(f"   Direction Accuracy: {perf['direction_accuracy']:.1f}%" if perf[
                                                                                            'direction_accuracy'] is not None else "   Direction Accuracy: N/A")
            logging.info(f"   BUY Success Rate: {perf['buy_success_rate']:.1f}%" if perf[
                                                                                        'buy_success_rate'] is not None else "   BUY Success Rate: N/A")
            logging.info(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f}" if perf[
                                                                               'sharpe_ratio'] is not None else "   Sharpe Ratio: N/A")
            logging.info(f"   Fitness Score: {results['fitness_score']:.1f}" if results[
                                                                                    'fitness_score'] is not None else "   Fitness Score: N/A")

            params = results.get('parameters', {})
            logging.info(f"   Optimal Parameters:")

            # Direct parameters
            logging.info(f"     Investment Days: {params.get('investment_days', 'N/A')}")
            logging.info(f"     Profit Threshold: {params.get('profit_threshold', 'N/A'):.2f}")
            logging.info(f"     Stop Loss Threshold: {params.get('stop_loss_threshold', 'N/A'):.2f}")
            logging.info(f"     Min Confidence: {params.get('min_confidence', 'N/A'):.0f}%")
            logging.info(f"     Buy Threshold: {params.get('buy_threshold', 'N/A'):.2f}")
            logging.info(f"     Sell Threshold: {params.get('sell_threshold', 'N/A'):.2f}")

            # Nested signal weights
            signal_weights = params.get('signal_weights', {})
            if signal_weights:
                logging.info("     Signal Weights:")
                for weight_name, weight_val in signal_weights.items():
                    logging.info(f"       {weight_name.replace('_', ' ').title()}: {weight_val:.2f}")
            else:
                logging.info("     Signal Weights: N/A")

            # Nested confidence parameters
            confidence_params = params.get('confidence_params', {})
            if confidence_params:
                logging.info("     Confidence Parameters:")
                for conf_param_name, conf_param_val in confidence_params.items():
                    logging.info(f"       {conf_param_name.replace('_', ' ').title()}: {conf_param_val:.2f}")
            else:
                logging.info("     Confidence Parameters: N/A")

        logging.info("\n" + "=" * 80)

    def export_parameters_for_production(self, strategy_type):
        """Export optimized parameters for production use."""
        logging.info(f"📦 Attempting to export production parameters for {strategy_type}.")

        if strategy_type not in self.best_parameters or not self.best_parameters[strategy_type].get('parameters'):
            logging.warning(f"⚠️ No optimized parameters found for {strategy_type}. Skipping export.")
            return None

        params = dict(self.best_parameters[strategy_type]['parameters'])
        s_lower = strategy_type.lower().replace(' ', '_')

        # Normalize
        buy = params.get(f"buy_threshold_{s_lower}", params.get("buy_threshold"))
        sell = params.get(f"sell_threshold_{s_lower}", params.get("sell_threshold"))
        min_conf = params.get(f"confidence_req_{s_lower}", params.get("min_confidence"))

        production_config = self._normalize_numeric_values({
            'strategy_multipliers': {
                strategy_type: {
                    'profit': params.get('profit_threshold', 1.0),
                    'risk': params.get('stop_loss_threshold', 1.0),
                    'confidence_req': min_conf if min_conf is not None else 75
                }
            },
            'signal_weights': {
                'trend': params.get('weight_trend', 0.45),
                'momentum': params.get('weight_momentum', 0.30),
                'volume': params.get('weight_volume', 0.10),
                'support_resistance': params.get('weight_support_resistance', 0.05),
                'ai_model': params.get('weight_ai_model', 0.10)
            },
            'thresholds': {
                'buy_threshold': buy if buy is not None else 0.0,
                'sell_threshold': sell if sell is not None else 0.0
            },
            'confidence_params': {
                'confidence_base_multiplier': params.get('confidence_base_multiplier', 1.0),
                'confidence_confluence_weight': params.get('confidence_confluence_weight', 1.0),
                'confidence_penalty_strength': params.get('confidence_penalty_strength', 1.0),
            }
        })

        filename = f"stockwise_production_params_{strategy_type}_{datetime.now().strftime('%Y%m%d')}.json"
        file_location = os.path.join(self.configuration_files, filename)
        with open(file_location, 'w') as f:
            json.dump(production_config, f, indent=2)
        logging.info(f"✅ Exported production parameters to: {file_location}")
        return file_location

    def load_latest_production_config(self, strategy_type, directory=None):
        """
        Returns the parsed JSON of the most recently saved production config for the given strategy_type.
        """
        import glob, os, json
        directory = directory or self.configuration_files
        pattern = f"stockwise_production_params_{strategy_type.lower().replace(' ', '_')}_*.json"
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        if not files:
            logging.warning(f"No production config files found for {strategy_type} in {directory}.")
            return None
        latest = files[-1]
        try:
            with open(latest, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load production config {latest}: {e}")
            return None

    def apply_production_config_to_advisor(self, production_config: dict):
        """
        Calls the advisor-side method if available; otherwise maps fields directly.
        """
        if not production_config:
            return
        if hasattr(self.advisor, "apply_production_config"):
            self.advisor.apply_production_config(production_config)
            return

        # Fallback direct mapping (mirrors advisor.apply_production_config)
        strategy = production_config.get("strategy")
        if strategy:
            self.advisor.current_strategy = strategy

        adv = production_config.get("advisor_settings", {})
        if "investment_days" in adv:
            self.advisor.investment_days = int(adv["investment_days"])

        if "global_signal_weights" in adv and hasattr(self.advisor, "signal_weights"):
            gw = adv["global_signal_weights"]
            self.advisor.signal_weights.update({
                "trend": gw.get("trend", self.advisor.signal_weights.get("trend")),
                "momentum": gw.get("momentum", self.advisor.signal_weights.get("momentum")),
                "volume": gw.get("volume", self.advisor.signal_weights.get("volume")),
                "support_resistance": gw.get("support_resistance",
                                             self.advisor.signal_weights.get("support_resistance")),
                "ai_model": gw.get("ai_model", self.advisor.signal_weights.get("ai_model")),
            })

        stg_all = production_config.get("strategy_multipliers", {})
        stg_cfg = stg_all.get(self.advisor.current_strategy, {})
        if self.advisor.current_strategy in self.advisor.strategy_settings:
            self.advisor.strategy_settings[self.advisor.current_strategy].update({
                "profit": stg_cfg.get("profit",
                                      self.advisor.strategy_settings[self.advisor.current_strategy].get("profit")),
                "risk": stg_cfg.get("risk", self.advisor.strategy_settings[self.advisor.current_strategy].get("risk")),
                "confidence_req": stg_cfg.get("confidence_req",
                                              self.advisor.strategy_settings[self.advisor.current_strategy].get(
                                                  "confidence_req")),
                "buy_threshold": stg_cfg.get("buy_threshold",
                                             self.advisor.strategy_settings[self.advisor.current_strategy].get(
                                                 "buy_threshold")),
                "sell_threshold": stg_cfg.get("sell_threshold",
                                              self.advisor.strategy_settings[self.advisor.current_strategy].get(
                                                  "sell_threshold")),
            })

        conf = production_config.get("confidence_params", {})
        if hasattr(self.advisor, "confidence_params") and conf:
            self.advisor.confidence_params.update({
                "base_multiplier": conf.get("base_multiplier", self.advisor.confidence_params.get("base_multiplier")),
                "confluence_weight": conf.get("confluence_weight",
                                              self.advisor.confidence_params.get("confluence_weight")),
                "penalty_strength": conf.get("penalty_strength",
                                             self.advisor.confidence_params.get("penalty_strength")),
            })


def create_and_visualize_performance_summary(best_parameters_dict, display_type='table'):
    """
    Creates a summary table or chart of the best performance metrics for each strategy.

    Args:
        best_parameters_dict (dict): The dictionary containing best parameters and performance
                                     for each strategy, as returned by StockWiseAutoCalibrator.
        display_type (str): 'table' to print a formatted table, 'chart' to show a bar chart.
    """
    print("\n" + "=" * 80)
    print("📈 VISUALIZING CALIBRATION PERFORMANCE")
    print("=" * 80)

    if not best_parameters_dict:
        print("No best parameters available to visualize.")
        return

    summary_data = []
    for strategy_type, results in best_parameters_dict.items():
        if not results or not results.get('performance'):
            print(f"Skipping visualization for {strategy_type}: No performance data.")
            continue

        perf = results['performance']

        # Collect relevant metrics
        summary_data.append({
            'Strategy': strategy_type,
            'Overall Accuracy (%)': round(perf.get('overall_accuracy', 0.0), 2),
            'Direction Accuracy (%)': round(perf.get('direction_accuracy', 0.0), 2),
            'BUY Success Rate (%)': round(perf.get('buy_success_rate', 0.0), 2),
            'Avg Return (%)': round(perf.get('avg_return', 0.0), 2),
            'Sharpe Ratio': round(perf.get('sharpe_ratio', 0.0), 2),
            'Max Drawdown (%)': round(perf.get('max_drawdown', 0.0), 2),
            'Total Trades': int(perf.get('total_trades', 0)),
            'Avg Confidence (%)': round(perf.get('confidence_avg', 0.0), 2),
            'Fitness Score': round(results.get('fitness_score', 0.0), 2)
        })

    if not summary_data:
        print("No valid summary data generated for visualization.")
        return

    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.set_index('Strategy')  # Set strategy as index for better display

    if display_type == 'table':
        print("\n📊 Performance Metrics Summary Table:\n")
        print(df_summary.to_string())  # Use to_string() for better console formatting
        print("\n" + "=" * 80)
    elif display_type == 'chart':
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("❌ Matplotlib not found. Cannot generate charts. Please install it (`pip install matplotlib`).")
            return

        print("\n📊 Generating Performance Charts...\n")

        # Define metrics to plot
        metrics_to_plot = [
            'Overall Accuracy (%)',
            'Direction Accuracy (%)',
            'BUY Success Rate (%)',
            'Avg Return (%)',
            'Sharpe Ratio',
            'Fitness Score'
        ]

        # Filter out metrics that are all zero or NaN to avoid empty charts
        metrics_to_plot_filtered = [
            metric for metric in metrics_to_plot
            if not df_summary[metric].apply(lambda x: np.isclose(x, 0) or np.isnan(x)).all()
        ]

        if not metrics_to_plot_filtered:
            print("No non-zero/non-NaN metrics to plot. Skipping chart generation.")
            return

        fig, axes = plt.subplots(nrows=len(metrics_to_plot_filtered), ncols=1,
                                 figsize=(10, 4 * len(metrics_to_plot_filtered)))

        if len(metrics_to_plot_filtered) == 1:  # Handle single subplot case
            axes = [axes]

        for i, metric in enumerate(metrics_to_plot_filtered):
            ax = axes[i]
            colors = ['skyblue' if val >= 0 else 'salmon' for val in df_summary[metric]]
            bars = ax.bar(df_summary.index, df_summary[metric], color=colors)
            ax.set_title(metric)
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)  # Rotate labels if needed

            # Add value labels on top of bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        yval + (0.01 * np.max(df_summary[metric]) if np.max(
                            df_summary[metric]) > 0 else 0.01) if yval >= 0 else yval - (
                            0.05 * np.min(df_summary[metric]) if np.min(df_summary[metric]) < 0 else 0.05),
                        # Offset label for negative bars
                        round(yval, 2), ha='center', va='bottom' if yval >= 0 else 'top', color='black', fontsize=9)

        plt.tight_layout()
        plt.show()
        print("\n" + "=" * 80)
    else:
        print("Invalid display_type. Please choose 'table' or 'chart'.")


def calculate_fitness_score(metrics, strategy_name):
    """
    Calculates a comprehensive fitness score based on various performance metrics.
    🎯 Optimized for risk-adjusted returns and consistent performance,
    penalizing high drawdown and negative Sharpe Ratios.
    """
    overall_accuracy = metrics.get('overall_accuracy', 0)
    buy_success_rate = metrics.get('buy_success_rate', 0)
    avg_return = metrics.get('avg_return', 0) # Average return of trades for the period
    sharpe_ratio = metrics.get('sharpe_ratio', -100) # Default to a very low Sharpe if unavailable
    max_drawdown = metrics.get('max_drawdown', 0) # Max drawdown (expected to be negative or zero)
    total_trades = metrics.get('total_trades', 0)
    # direction_accuracy = metrics.get('direction_accuracy', 0) # Can be used, but focusing on core performance

    # Convert drawdown to a positive value for easier penalty calculation
    abs_max_drawdown = abs(max_drawdown)

    # Define weights for each component of the fitness score
    # These weights determine the importance of each metric in the overall fitness
    w_overall_accuracy = 0.20  # Reduced slightly to give more weight to risk-adjusted return
    w_buy_success_rate = 0.20  # Reduced slightly
    w_avg_return = 0.30        # Main driver of profitability
    w_sharpe_ratio = 0.20      # Increased significantly for risk-adjusted performance
    w_drawdown_penalty = 0.10  # Explicit penalty for large drawdowns
    w_trade_volume_bonus = 0.05 # Small bonus for sufficient trade volume

    # --- Component Scoring & Penalties ---

    # Base scores for accuracy and return
    score_overall_accuracy = overall_accuracy # Already a percentage (e.g., 29.6)
    score_buy_success_rate = buy_success_rate # Already a percentage (e.g., 21.3)
    score_avg_return = avg_return * 100 # Convert avg_return (decimal) to percentage (e.g., 0.05 -> 5.0)

    # Sharpe Ratio contribution: Penalize negative Sharpe heavily, reward positive Sharpe
    # Scale Sharpe to make it contribute meaningfully to the score (e.g., Sharpe of 1.0 -> 10 points)
    score_sharpe_ratio = sharpe_ratio * 10 if sharpe_ratio > 0 else sharpe_ratio * 20 # Stronger penalty for negative Sharpe

    # Drawdown penalty: Larger absolute drawdown results in a higher penalty
    # Normalize drawdown penalty if needed, but direct subtraction is fine for now
    penalty_drawdown = abs_max_drawdown * 0.5 # Example: 50% of absolute drawdown value as penalty

    # Bonus for sufficient number of trades (ensures strategies are active enough)
    trade_bonus = 0
    if total_trades > 50: # If a reasonable number of trades occurred
        trade_bonus = min(total_trades / 100, 10) # Max 10 bonus points, scales with trades

    # Penalty for very poor overall accuracy (if it falls below a minimum acceptable level)
    penalty_low_accuracy = 0
    if overall_accuracy < 20: # If overall accuracy is below 20%
        penalty_low_accuracy = (20 - overall_accuracy) * 1.0 # Significant penalty

    # --- Combined Fitness Score Calculation ---
    # Sum weighted scores and subtract penalties
    fitness = (
        score_overall_accuracy * w_overall_accuracy +
        score_buy_success_rate * w_buy_success_rate +
        score_avg_return * w_avg_return +
        score_sharpe_ratio * w_sharpe_ratio +
        trade_bonus * w_trade_volume_bonus -
        penalty_drawdown * w_drawdown_penalty -
        penalty_low_accuracy
    )

    # Ensure the fitness score doesn't become overly negative due to extreme penalties,
    # though negative scores are expected for poor-performing strategies.
    return round(fitness, 2) # Round for cleaner output


# --- MAIN EXECUTION BLOCK ---
def main_calibration_run():
    log_directory = 'logs/'
    config_directory = 'configuration/'

    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(config_directory, exist_ok=True)

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f'calibration_progress_{timestamp_str}.log'
    log_file_path = os.path.join(log_directory, log_file_name)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            handler.close()

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)

    logging.info("✨ Initializing Stock Advisor for calibration.")
    # Ensure debug=True is passed to see detailed logs from advisor
    advisor = ProfessionalStockAdvisor(debug=True, use_ibkr=False, download_log=True)
    logging.info("✅ Stock Advisor initialized.")

    calibrator = StockWiseAutoCalibrator(advisor)
    logging.info("✅ Auto-Calibrator instance created.")

    prod_cfg = calibrator.load_latest_production_config(strategy_type="balanced")
    calibrator.apply_production_config_to_advisor(prod_cfg)

    logging.info(f"✅ load latest production config. {prod_cfg}")

    # User input for test size
    print("\n--- Select Calibration Test Size ---")
    print("1 = Sanity (Quick test, ~12 minutes)")
    print("2 = Small (Limited stocks/timestamps/params, ~5 hours)")
    print("3 = Medium (Balanced test, ~12 hours)")
    print("4 = Full (Comprehensive, ~24 hours)")
    test_type_input = input("Test Size (1 = sanity, 2 = small, 3 = medium, 4 = full): ")

    selected_test_size = 'medium'  # Default
    selected_strategies = ['balanced']  # Default

    if test_type_input == '1':
        selected_test_size = 'sanity'
        selected_strategies = ['balanced', 'aggressive']
        logging.info(f"User selected SANITY test (1) with strategies: {', '.join(selected_strategies)}")
    elif test_type_input == '2':
        selected_test_size = 'small'
        selected_strategies = ['balanced', 'aggressive']
        logging.info(f"User selected SMALL test (2) with strategies: {', '.join(selected_strategies)}")
    elif test_type_input == '3':
        selected_test_size = 'medium'
        selected_strategies = ['balanced', 'conservative', 'aggressive']
        logging.info(f"User selected MEDIUM test (3) with strategies: {', '.join(selected_strategies)}")
    elif test_type_input == '4':
        selected_test_size = 'full'
        selected_strategies = ['balanced', 'aggressive', 'conservative', 'swing trading']
        logging.info(f"User selected FULL test (4) with strategies: {', '.join(selected_strategies)}")
    else:
        logging.error("Invalid input. Please enter a number between 1 and 4. Exiting.")
        sys.exit(1)  # Use sys.exit() for cleaner exit

    starting_time = time.time()

    best_params = run_stockwise_calibration(
        test_size=selected_test_size,
        strategies=selected_strategies
    )

    end_time = time.time()
    total_execution_seconds = end_time - starting_time
    execution_time_hours = total_execution_seconds / 3600

    calibrator.save_config_for_all_strategies(best_params)
    calibrator.generate_final_report(execution_time_hours=execution_time_hours)

    logging.info(f"Total Execution Time: {total_execution_seconds:.2f} seconds")
    logging.info(f"\n✅ StockWise Auto-Calibration completed in {execution_time_hours:.2f} hours.")

    # Ensure all handlers are properly closed at the end of the main run
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)


def run_stockwise_calibration(test_size='medium', strategies=['balanced']):
    """
    Example of how to run the calibration with configurable test size and strategies.

    Args:
        test_size (str): The size of the test ('small', 'medium', 'full', 'sanity').
        strategies (list): A list of strategy types to optimize (e.g., ['balanced', 'aggressive']).
    """

    # --- LOGGING SETUP moved here for controlled execution ---
    log_directory = 'logs/'  # New dedicated directory for log files
    config_directory = 'configuration/'  # New dedicated directory for JSON config files

    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(config_directory, exist_ok=True) # Ensure config directory is also created

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = f'calibration_progress_{timestamp_str}.log'
    log_file_path = os.path.join(log_directory, log_file_name)

    root_logger = logging.getLogger()
    # Remove existing handlers to prevent duplicate output if this function is called multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        if isinstance(handler, logging.FileHandler):
            try:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
            except Exception as e:
                pass  # Ignore errors if handler is already closed or invalid

    root_logger.setLevel(logging.INFO)  # Set overall logging level to INFO

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Suppress specific noisy library logs
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    # --- END LOGGING SETUP ---

    starting_time = time.time()
    logging.info("✨ Initializing Stock Advisor for calibration.")

    advisor = ProfessionalStockAdvisor(debug=True,
                                       download_log=False)
    logging.info("✅ Stock Advisor initialized.")

    calibrator = StockWiseAutoCalibrator(advisor)
    logging.info("✅ Auto-Calibrator instance created.")

    best_params = calibrator.run_calibration(
        test_size=test_size,
        strategies=strategies
    )
    logging.info("📈 Calibration process finished.")

    logging.info("\n📦 Exporting best parameters for production...")
    for strategy in best_params.keys():
        calibrator.export_parameters_for_production(strategy)

    end_time = time.time()
    total_execution_seconds = end_time - starting_time
    execution_time_hours = total_execution_seconds / 3600

    calibrator.generate_final_report(execution_time_hours=execution_time_hours)

    logging.info(f"Total Execution Time: {total_execution_seconds:.2f} seconds")
    logging.info(f"\n✅ StockWise Auto-Calibration completed in {execution_time_hours:.2f} hours.")

    # Ensure all file handlers are closed before returning
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                handler.close()
                root_logger.removeHandler(handler)
            except Exception as e:
                logging.warning(f"Error closing final log handler in run_stockwise_calibration: {e}")

    return best_params


def run_stockwise_calibration(test_size='medium', strategies=['balanced']):
    # This function is now simplified as setup is done in main_calibration_run
    # Re-initialize advisor and calibrator as they might be new instances per run
    advisor = ProfessionalStockAdvisor(debug=True, use_ibkr=False, download_log=True)
    calibrator = StockWiseAutoCalibrator(advisor)

    # Adjust calibration config based on test_size and run optimization
    best_params = calibrator.run_calibration(
        test_size=test_size,
        strategies=strategies
    )

    # Call to display performance summary AFTER calibration completes
    create_and_visualize_performance_summary(best_params, display_type='table')
    create_and_visualize_performance_summary(best_params, display_type='chart')

    return best_params

if __name__ == "__main__":
    main_calibration_run()
    # # The logging setup is now inside run_stockwise_calibration, so this only
    # # affects any messages *before* run_stockwise_calibration is called.
    # # It's good practice to keep it consistent.
    # root_logger = logging.getLogger()
    # root_logger.setLevel(logging.INFO)  # Ensure initial messages are visible
    #
    # logging.info("--- Starting StockWise Auto-Calibration Script ---")
    # print("Select a test size and strategies to optimize:")
    # print("Estimate run time based on selected test size and strategies:")
    # print("Sanity Test = ~12 minutes ; Small Test = ~5 hours ; Medium Test = ~12 hours ; Full Test = ~24 hours")
    # test_type_input = input("Test Size (1 = sanity, 2 = small, 3 = medium, 4 = full): ")
    #
    # selected_test_size = 'medium'
    # selected_strategies = ['Balanced']
    #
    # if test_type_input == '1':
    #     selected_test_size = 'sanity'
    #     selected_strategies = ['Balanced', 'Aggressive']
    #     logging.info(f"User selected SANITY test (1) with strategies: {', '.join(selected_strategies)}")
    # elif test_type_input == '2':
    #     selected_test_size = 'small'
    #     selected_strategies = ['Balanced', 'Aggressive']
    #     logging.info(f"User selected SMALL test (2) with strategies: {', '.join(selected_strategies)}")
    # elif test_type_input == '3':
    #     selected_test_size = 'medium'
    #     selected_strategies = ['Balanced', 'Conservative', 'Aggressive']
    #     logging.info(f"User selected MEDIUM test (3) with strategies: {', '.join(selected_strategies)}")
    # elif test_type_input == '4':
    #     selected_test_size = 'full'
    #     selected_strategies = ['Balanced', 'Aggressive', 'Conservative', 'Swing Trading']
    #     logging.info(f"User selected FULL test (4) with strategies: {', '.join(selected_strategies)}")
    # else:
    #     logging.error("Invalid input. Please enter a number between 1 and 4. Exiting.")
    #     sys.exit(1)
    #
    # run_stockwise_calibration(
    #     test_size=selected_test_size,
    #     strategies=selected_strategies
    # )
    # logging.info("--- Script execution completed ---")
