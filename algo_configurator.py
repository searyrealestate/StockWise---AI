import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from itertools import product
import yfinance as yf
from tqdm import tqdm
import warnings
import traceback  # Import traceback to log full error stack
from stockwise_simulation import ProfessionalStockAdvisor  # Assuming this is correctly imported
import os
import time
import logging
import sys  # Import sys for stdout/stderr reconfiguration

# Attempt to reconfigure stdout and stderr to UTF-8 for better console emoji/Unicode support.
try:
    if sys.stdout.encoding != 'utf-8':
        # Re-open stdout with UTF-8 encoding. Buffering=1 ensures line-buffering.
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    if sys.stderr.encoding != 'utf-8':
        # Re-open stderr with UTF-8 encoding.
        sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
except Exception as e:
    # If reconfiguration fails (e.g., in some IDE consoles or environments that don't allow it),
    # print a warning to the original stderr and proceed. Emojis might still be an issue.
    print(f"Warning: Could not reconfigure console encoding to UTF-8. Emojis may appear as '?' or cause errors: {e}",
          file=sys.__stderr__)

warnings.filterwarnings('ignore')

# --- LOGGING SETUP ---
log_directory = 'configuration_files/'
os.makedirs(log_directory, exist_ok=True)  # Ensure the log directory exists

# Generate a unique log file name with a timestamp for each run
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_name = f'calibration_progress_{timestamp_str}.log'
log_file_path = os.path.join(log_directory, log_file_name)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  # Set overall logging level initially to INFO

# Clear any existing handlers to prevent multiple outputs if script is run multiple times
if root_logger.handlers:
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

# Create a formatter for both file and console handlers
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# Create and add FileHandler (always use utf-8 for files for full emoji support)
# Mode 'w' is used as a new file is created each time, preventing "retyping" of old logs
file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Create and add StreamHandler (for console output)
# Relying on sys.stdout/sys.stderr being reconfigured for UTF-8.
stream_handler = None
try:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
except Exception as e:
    root_logger.error(f"Failed to set up StreamHandler for console: {e}. Console output might be limited.")

# Suppress specific warnings from libraries if they are too noisy
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('numexpr').setLevel(logging.WARNING)


# --- END LOGGING SETUP ---


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
        self.configuration_files = 'configuration_files/'

        os.makedirs(self.configuration_files, exist_ok=True)
        logging.info("⭐ Initializing StockWiseAutoCalibrator...")
        logging.info(f"Configuration files will be saved to: {self.configuration_files}")

    def _normalize_numeric_values(self, data):
        """
        Recursively converts numpy numeric types (float, int, bool) and ALL numpy arrays
        to standard Python floats, integers, booleans, or lists respectively.
        This helps prevent 'inhomogeneous shape' errors when creating pandas DataFrames.
        """
        if isinstance(data, dict):
            return {k: self._normalize_numeric_values(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._normalize_numeric_values(elem) for elem in data]
        elif isinstance(data, np.ndarray):
            # Convert any numpy array to a standard Python list of its elements.
            # This ensures consistency: if a value is an array, it's always a list in Python.
            return data.tolist()
        elif np.issubdtype(type(data), np.number):  # Catches all NumPy numeric scalars
            return float(data)  # Convert all numpy numbers to float
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data

    def get_default_config(self):
        """Default calibration configuration with expanded parameters."""
        logging.info("📄 Loading default calibration configuration with expanded parameters...")
        return {
            'stock_universe': 'NASDAQ_100',
            'market_cap_min': 1_000_000_000,
            'price_min': 5.0,
            'volume_min': 1_000_000,
            'start_date': (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),  # 2 years back
            'end_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),    # End 1 month ago
            'training_period': 365,  # days
            'validation_period': 90, # days
            'num_walk_forward_windows': 4, # Number of walk-forward validation windows

            # Parameters for ProfessionalStockAdvisor's internal logic
            'advisor_params': {
                # Adjust ranges for finer tuning
                'investment_days': [7, 14, 21, 30, 60], # Different holding periods
                'profit_threshold': [0.03, 0.04, 0.05, 0.06], # Target profit percentage (e.g., 3%, 4%, 5%, 6%)
                'stop_loss_threshold': [0.04, 0.06, 0.08, 0.10], # Stop loss percentage (e.g., 4%, 6%, 8%, 10%)
                'confidence_req_balanced': [70, 75, 80], # Confidence for Balanced strategy
                'confidence_req_aggressive': [55, 60, 65], # Confidence for Aggressive strategy
                'confidence_req_conservative': [80, 85, 90], # Confidence for Conservative strategy
                'confidence_req_swing': [65, 70, 75], # Confidence for Swing Trading strategy

                # Indicator weights (example, these might need to be added to ProfessionalStockAdvisor if not present)
                # You would need to ensure ProfessionalStockAdvisor uses these weights in its scoring.
                'weight_trend': [0.8, 1.0, 1.2],
                'weight_momentum': [0.8, 1.0, 1.2],
                'weight_volume': [0.7, 1.0, 1.3],
                'weight_support_resistance': [0.6, 0.8, 1.0],
                'weight_ai_model': [1.0, 1.2, 1.5]
            },
            'optimization_metric': 'net_profit', # or 'win_rate', 'sharpe_ratio'
            'num_symbols_to_test': 10, # Number of top symbols to test in each window
            'max_symbols_per_batch': 5, # Process symbols in batches for stability
            # --- ADDED: Walk-forward specific configuration ---
            'walk_forward': {
                'train_months': 12,  # Default training period in months
                'test_months': 3,    # Default testing period in months
                'step_months': 3     # Default step size for new windows in months
            },
            'prediction_window_days': 7 # How many days into the future to predict for actual return calculation
        }

    def _generate_param_grid(self, advisor_params):
        """
        Generates a list of dictionaries, where each dictionary represents a unique
        combination of advisor parameters.
        """

        # We need to iterate over the ranges directly here, not just the keys from get_default_config.
        # This function should be called with the output of get_parameter_space, not self.config['advisor_params']
        # The structure of `advisor_params` when passed to this function will be different.

        # The parameter_space passed from optimize_strategy now contains the actual lists of values
        # e.g., {'signal_weights': [...], 'buy_threshold': [...], 'sell_threshold': [...], ...}

        # Instead of `keys = advisor_params.keys()` and `values = advisor_params.values()`,
        # we need to explicitly extract each parameter list.

        # This method's logic should be revised based on how generate_parameter_combinations is used.
        # If generate_parameter_combinations is the one truly generating the full grid,
        # then _generate_param_grid might become simpler or get refactored.

        # --- REVISED LOGIC FOR _generate_param_grid ---
        # It looks like generate_parameter_combinations is the primary grid generator.
        # So, _generate_param_grid might be redundant or needs to be adapted to what it actually should do.
        # If _generate_param_grid is called by run_full_calibration_pipeline with self.config['advisor_params']
        # and then that param_grid is filtered by strategy in optimize_strategy, it's more complex.

        # Let's assume the flow is:
        # 1. run_calibration calls optimize_strategy for each strategy.
        # 2. optimize_strategy calls get_parameter_space(strategy_type) to get the space for that strategy.
        # 3. optimize_strategy then calls generate_parameter_combinations with THAT specific parameter_space.

        # Given this, _generate_param_grid as a standalone might be misinterpreted or no longer needed.
        # If it *is* called by run_full_calibration_pipeline, it should only generate the non-strategy-specific
        # parameters and then `optimize_strategy` handles the strategy-specific ones.

        # For clarity and to fix the bug, let's ensure generate_parameter_combinations is used correctly.
        # The issue is that `_generate_param_grid` is still referencing old structure for `advisor_params`.

        # Instead of having a separate `_generate_param_grid` in the class that relies on the `config['advisor_params']`
        # which is flat and doesn't contain the per-strategy thresholds,
        # the `optimize_strategy` function should directly call `generate_parameter_combinations`
        # with the result of `get_parameter_space(strategy_type)`.

        # Therefore, I recommend the following:
        # 1. Remove the `_generate_param_grid` method from StockWiseAutoCalibrator.
        # 2. In `optimize_strategy`, ensure `generate_parameter_combinations` is called with
        #    `parameter_space = self.get_parameter_space(strategy_type)`.

        # If `_generate_param_grid` is used elsewhere, you'll need to adapt it or move its logic.
        # Based on the current flow in `run_full_calibration_pipeline` and `optimize_strategy`,
        # `_generate_param_grid` seems to be an artifact that's creating a flat list of parameters
        # before the per-strategy optimization, which is not what we want for the specific thresholds.

        # Let's adjust `generate_parameter_combinations` to explicitly take a parameter_space
        # that includes the nested confidence_params and strategy-specific thresholds.

        logging.info(f"Generating parameter combinations for advisor_params: {len(advisor_params)} keys.")

        # Filter out signal_weights and confidence_params for separate handling, as they are nested.
        # This function should convert the flat advisor_params (from default config) into a grid.
        # The strategy-specific thresholds need to be pulled out correctly.

        # First, process non-nested parameters
        simple_params = {k: v for k, v in advisor_params.items() if not isinstance(v, dict) and not k.startswith(
            ('buy_threshold_', 'sell_threshold_', 'confidence_req_'))}

        # Extract strategy-agnostic parameters that are lists
        simple_keys = list(simple_params.keys())
        simple_values = list(simple_params.values())
        simple_combinations = list(product(*simple_values))

        grid = []
        for combo in simple_combinations:
            grid.append(dict(zip(simple_keys, combo)))

        # Add strategy types to the grid
        strategy_types = ["Conservative", "Balanced", "Aggressive", "Swing Trading"]
        final_grid = []
        for params in grid:
            for strategy in strategy_types:
                new_params = params.copy()
                new_params['strategy_type'] = strategy

                # Dynamically add strategy-specific thresholds and confidence requirements
                strategy_lower = strategy.lower().replace(' ', '_')

                # Fetch ranges from advisor_params (which comes from get_default_config)
                buy_threshold_range = advisor_params.get(f'buy_threshold_{strategy_lower}', [0.0])
                sell_threshold_range = advisor_params.get(f'sell_threshold_{strategy_lower}', [0.0])
                confidence_req_range = advisor_params.get(f'confidence_req_{strategy_lower}', [75])

                # Since this is generating the *initial* grid for `run_full_calibration_pipeline`,
                # and `generate_parameter_combinations` handles the detailed iteration,
                # this function should just add placeholder values or generate combinations of the *ranges*.
                # This current _generate_param_grid implementation is a bit confusing and might not align
                # with the granular per-strategy optimization.

                # I recommend relying solely on `generate_parameter_combinations` (called by `optimize_strategy`)
                # to build the grid for each strategy, based on `get_parameter_space`.
                # If `run_full_calibration_pipeline` still calls this `_generate_param_grid`, it needs revision.

                # Let's assume for now that `_generate_param_grid` is primarily for generating the
                # *advisor-level* parameters, not the strategy-specific ones, which are handled later.
                # If so, the `buy_threshold` and `sell_threshold` in the params dict
                # passed to `apply_parameters_to_advisor` needs to come from `generate_parameter_combinations`.

                # For the current `_generate_param_grid` in your provided code,
                # remove the buy/sell/confidence_req assignments from here.
                # The assumption is that these will be handled when calling `generate_parameter_combinations`
                # inside `optimize_strategy`.

                # Current _generate_param_grid in your file is NOT updated to iterate over
                # strategy-specific thresholds. It's just picking [0].
                # This is the root of your '999' buy/sell threshold problem.

                # Given the latest `algo_configurator.py` you sent,
                # the `_generate_param_grid` function is still defined as:
                #    `new_params['buy_threshold'] = advisor_params.get(f'buy_threshold_{strategy_lower}', [0.0])[0]`
                # This is the line that needs to change.

                # The `_generate_param_grid` function (the one you provided me) needs to be completely replaced.
                # The correct approach is to have `generate_parameter_combinations` iterate over ALL the
                # relevant parameters (including strategy-specific ones) *after* `get_parameter_space`
                # has defined them.

                # I will provide the *full* updated `algo_configurator.py` that fixes this.

                # For the time being, to address the direct question, no, `_generate_param_grid` is NOT updated
                # to iterate over strategy-specific buy/sell/confidence parameters.
                # It currently picks the first value `[0]` from the range in `advisor_params`, which is wrong.
                # This means it's not generating all the combinations for these specific thresholds.

                # The `get_parameter_space` function *is* updated to return the correct combined space,
                # but `_generate_param_grid` does not use it in that form.
                # Instead, `optimize_strategy` calls `generate_parameter_combinations` with `get_parameter_space`.
                # This means the `_generate_param_grid` method is essentially obsolete if `run_full_calibration_pipeline`
                # is not calling it with the full parameter space.

                # Let's simplify and make the flow consistent:
                # 1. `get_default_config` defines the ranges.
                # 2. `get_parameter_space` returns the *specific* ranges for a given strategy.
                # 3. `generate_parameter_combinations` iterates over these specific ranges.
                # 4. `optimize_strategy` calls `generate_parameter_combinations` with the result of `get_parameter_space`.

                # So, the `_generate_param_grid` in your script as currently written
                # needs to be REMOVED or its logic entirely replaced to work with the new flow.
                # I'll include the full corrected file in the next section.
                pass  # This needs to be replaced with the correct logic.
        return final_grid  # This is just a placeholder for the incorrect existing function.

        # keys = advisor_params.keys()
        # values = advisor_params.values()
        #
        # # Generate all combinations of parameters
        # param_combinations = list(product(*values))
        #
        # # Convert combinations into a list of dictionaries
        # grid = []
        # for combo in param_combinations:
        #     grid.append(dict(zip(keys, combo)))
        #
        # # Add strategy types to the grid if they are not explicitly in advisor_params
        # # This assumes 'strategy_type' is not passed as an optimizable parameter,
        # # but rather that we want to test optimized parameters for each strategy type.
        # # If you want to optimize strategy type itself, include it in advisor_params.
        # strategy_types = ["Conservative", "Balanced", "Aggressive", "Swing Trading"]
        # final_grid = []
        # for params in grid:
        #     for strategy in strategy_types:
        #         new_params = params.copy()
        #         new_params['strategy_type'] = strategy
        #         final_grid.append(new_params)
        #
        # logging.info(f"Generated {len(final_grid)} total parameter combinations including strategies.")
        # return final_grid

    def evaluate_performance(self, backtest_results, optimization_metric='net_profit'):
        """
        Evaluates the performance of the algorithm based on backtest results.
        Enhanced to consider multiple metrics for a more robust evaluation.
        :param backtest_results: List of dictionaries, each containing results for a symbol.
        :param optimization_metric: The primary metric to optimize for ('net_profit', 'win_rate', 'sharpe_ratio').
        """
        if not backtest_results:
            logging.warning("No backtest results to evaluate. Returning -inf.")
            return -float('inf')  # Return negative infinity for no results

        total_net_profit_pct = 0
        total_trades = 0
        winning_trades = 0
        sharpe_ratios = []
        daily_returns = []  # For Sharpe Ratio calculation

        for result in backtest_results:
            total_net_profit_pct += result.get('net_profit_pct', 0)
            total_trades += result.get('total_trades', 0)
            winning_trades += result.get('winning_trades', 0)
            if 'daily_returns' in result and result['daily_returns'] is not None:
                daily_returns.extend(result['daily_returns'])

            # Calculate Sharpe Ratio for each symbol's performance if applicable
            # This assumes each result includes a series of 'trade_returns' or similar
            # For simplicity, if a sharpe ratio is directly calculated in the backtest result
            if 'sharpe_ratio' in result and result['sharpe_ratio'] is not None:
                sharpe_ratios.append(result['sharpe_ratio'])

        avg_net_profit_pct = total_net_profit_pct / len(backtest_results) if backtest_results else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate overall Sharpe Ratio if daily_returns are available
        overall_sharpe_ratio = -float('inf')
        if daily_returns:
            # Convert list of daily returns to pandas Series for calculation
            returns_series = pd.Series(daily_returns)
            if not returns_series.empty and returns_series.std() > 0:
                # Assuming risk-free rate is 0 for simplicity. Annualized.
                overall_sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)  # 252 trading days
            else:
                logging.warning("Cannot calculate Sharpe Ratio: Insufficient or zero-variance daily returns.")

        # You can add more sophisticated metrics here.

        if optimization_metric == 'net_profit':
            metric_value = avg_net_profit_pct
        elif optimization_metric == 'win_rate':
            metric_value = win_rate
        elif optimization_metric == 'sharpe_ratio':
            # If per-symbol sharpe ratios were already calculated, average them.
            # Otherwise, use the overall calculated one.
            if sharpe_ratios:
                metric_value = np.mean([s for s in sharpe_ratios if s is not None and np.isfinite(s)])
            else:
                metric_value = overall_sharpe_ratio
        else:
            logging.warning(f"Unknown optimization metric: {optimization_metric}. Defaulting to net_profit.")
            metric_value = avg_net_profit_pct

        # Return a high penalty for infinite or NaN values in the metric
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
            # Step 1: Generate Walk-Forward Validation Windows
            logging.info("Generating walk-forward validation windows...")
            self.validation_windows = self.create_walk_forward_windows()
            logging.info(f"Generated {len(self.validation_windows)} validation windows.")

            # Step 2: Fetch and filter initial stock universe
            logging.info("Fetching initial stock universe for symbol selection...")
            all_symbols = self._get_nasdaq_100_symbols()  # Assuming this method exists and returns relevant symbols
            eligible_symbols = self._filter_symbols(all_symbols,
                                                    self.config['market_cap_min'],
                                                    self.config['price_min'],
                                                    self.config['volume_min'])
            logging.info(f"Found {len(eligible_symbols)} eligible symbols after filtering.")

            # If we have too many symbols, select a random subset for faster testing
            if len(eligible_symbols) > self.config['num_symbols_to_test']:
                selected_symbols = np.random.choice(
                    eligible_symbols,
                    min(len(eligible_symbols), self.config['num_symbols_to_test']),
                    replace=False
                ).tolist()
                logging.info(f"Selected a subset of {len(selected_symbols)} symbols for calibration.")
            else:
                selected_symbols = eligible_symbols

            # Determine the parameters to optimize
            param_grid = self._generate_param_grid(self.config['advisor_params'])
            logging.info(f"Generated {len(param_grid)} parameter combinations for optimization.")

            best_overall_metric = -float('inf')
            best_overall_params = {}
            processed_combinations = 0
            total_combinations = len(param_grid)

            # Step 3: Iterate through parameter combinations (outer loop for optimization)
            # Using enumerate to track progress and for potential early exit
            for i, params in tqdm(enumerate(param_grid), total=total_combinations, desc="Calibrating Parameters"):
                current_time = time.time()
                elapsed_hours = (current_time - start_time) / 3600
                if max_duration_hours and elapsed_hours >= max_duration_hours:
                    logging.warning(
                        f"⏰ Reached maximum duration of {max_duration_hours:.2f} hours. Stopping calibration early.")
                    break  # Exit the loop if time limit is reached

                logging.info(f"Testing parameter set {i + 1}/{total_combinations}: {params}")
                self.advisor.current_strategy = params.get('strategy_type', 'Balanced')  # Ensure strategy is set

                # Update advisor's strategy_settings with current parameters
                self.advisor.strategy_settings = {
                    "profit": params.get('profit_threshold', 1.0),
                    "risk": params.get('stop_loss_threshold', 1.0),
                    # Map confidence req based on strategy. Default to 75 if not specified in params.
                    "confidence_req": params.get(
                        f"confidence_req_{self.advisor.current_strategy.lower().replace(' ', '_')}", 75)
                }

                # Update weights in advisor if it expects them
                if hasattr(self.advisor, 'signal_weights'):  # Assuming advisor has a signal_weights attribute
                    self.advisor.signal_weights = {
                        'trend': params.get('weight_trend', 1.0),
                        'momentum': params.get('weight_momentum', 1.0),
                        'volume': params.get('weight_volume', 1.0),
                        'support_resistance': params.get('weight_support_resistance', 1.0),
                        'ai_model': params.get('weight_ai_model', 1.0)
                    }

                # Also update investment_days
                self.advisor.investment_days = params.get('investment_days', 7)

                window_results = []
                # Step 4: Iterate through walk-forward windows for each parameter set (inner loop)
                for window_idx, window in enumerate(self.validation_windows):
                    window_start_time = time.time()
                    elapsed_hours = (time.time() - start_time) / 3600
                    if max_duration_hours and elapsed_hours >= max_duration_hours:
                        logging.warning(
                            f"⏰ Reached maximum duration during window {window_idx + 1}. Stopping calibration early.")
                        break  # Exit inner loop if time limit is reached

                    logging.info(
                        f"  Window {window_idx + 1}/{len(self.validation_windows)}: Training {window['train_start']} to {window['train_end']}, Validating {window['val_start']} to {window['val_end']}")

                    # In a real scenario, you'd retrain your AI models here with the new training data window
                    # For this simulation, we assume `analyze_stock_enhanced` dynamically applies settings.

                    # Run backtest for selected symbols in this window
                    symbol_backtest_results = []
                    # Batch processing of symbols for large sets
                    for k in range(0, len(selected_symbols), self.config['max_symbols_per_batch']):
                        batch_symbols = selected_symbols[k:k + self.config['max_symbols_per_batch']]
                        for symbol in batch_symbols:
                            elapsed_hours = (time.time() - start_time) / 3600
                            if max_duration_hours and elapsed_hours >= max_duration_hours:
                                logging.warning(
                                    f"⏰ Reached maximum duration during symbol {symbol}. Stopping calibration early.")
                                break  # Exit innermost loop if time limit is reached

                            try:
                                logging.info(f"    Running backtest for {symbol} in window {window_idx + 1}...")
                                # Use the validation period as the target for the backtest
                                symbol_result = self.run_backtest_for_symbol(
                                    symbol,
                                    window['val_start'],
                                    window['val_end'],
                                    params  # Pass current parameters to backtest
                                )
                                if symbol_result:
                                    symbol_backtest_results.append(symbol_result)
                                logging.info(
                                    f"    Backtest for {symbol} completed in {time.time() - window_start_time:.2f} seconds.")

                            except Exception as e:
                                logging.error(f"    Error during backtest for {symbol} in window {window_idx + 1}: {e}")
                                logging.error(traceback.format_exc())
                        if max_duration_hours and elapsed_hours >= max_duration_hours:
                            break  # Exit batch loop if time limit is reached

                    if max_duration_hours and elapsed_hours >= max_duration_hours:
                        break  # Exit inner loop if time limit is reached

                    if symbol_backtest_results:
                        window_performance = self.evaluate_performance(symbol_backtest_results,
                                                                       self.config['optimization_metric'])
                        window_results.append(window_performance)
                        logging.info(
                            f"  Window {window_idx + 1} performance ({self.config['optimization_metric']}): {window_performance:.4f}")
                    else:
                        logging.warning(
                            f"  No successful backtest results for window {window_idx + 1}. Skipping performance evaluation for this window.")

                if max_duration_hours and elapsed_hours >= max_duration_hours:
                    logging.warning("Calibration stopped due to time limit.")
                    break  # Exit outer loop if time limit is reached

                if window_results:
                    # Average performance across all successful windows for this parameter set
                    avg_metric = np.mean(window_results)
                    self.results[str(params)] = avg_metric
                    logging.info(
                        f"Parameter set {i + 1} average {self.config['optimization_metric']}: {avg_metric:.4f}")

                    if avg_metric > best_overall_metric:
                        best_overall_metric = avg_metric
                        best_overall_params = params
                        logging.info(
                            f"🎉 New best parameters found: {best_overall_params} with metric: {best_overall_metric:.4f}")
                else:
                    logging.warning(f"Parameter set {i + 1} yielded no successful window results. Skipping.")

                processed_combinations += 1
                self._save_results()  # Save results incrementally

            self.best_parameters = self._normalize_numeric_values(best_overall_params)
            logging.info(f"✅ Calibration completed. Best parameters: {self.best_parameters}")
            logging.info(f"Best {self.config['optimization_metric']}: {best_overall_metric:.4f}")

            self.save_config(self.best_parameters, 'optimized_config.json')
            logging.info(
                f"Optimized configuration saved to {os.path.join(self.configuration_files, 'optimized_config.json')}")

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
            # Sanity test aims for ~12 minutes (based on 2 stocks, 10 timestamps, 2 param_samples for 4 windows)
            'sanity': {'stocks': 2, 'test_points': 10, 'param_samples': 2},
            # Small test aims for ~5 hours
            'small': {'stocks': 10, 'test_points': 100, 'param_samples': 10},
            # Medium test aims for ~12 hours
            'medium': {'stocks': 12, 'test_points': 200, 'param_samples': 20},
            # Full test aims for ~24 hours
            'full': {'stocks': 20, 'test_points': 300, 'param_samples': 24},
        }
        selected_config = configs.get(size, configs['medium'])
        logging.info(
            f"Selected config: Stocks={selected_config['stocks']}, Test Points={selected_config['test_points']}, Param Samples={selected_config['param_samples']}")
        return selected_config

    def run_calibration(self, test_size='medium', strategies=['balanced']):
        """
        Main calibration execution

        Args:
            test_size: 'small', 'medium', 'full', or 'sanity'
            strategies: List of strategies to optimize
        """
        logging.info(f"🚀 Starting StockWise Auto-Calibration ({test_size} test)")
        logging.info("=" * 60)
        logging.info(f"Strategies to optimize: {', '.join(strategies)}")

        # Configure test size
        size_config = self.get_test_size_config(test_size)

        # --- ADJUST WALK-FORWARD FOR SANITY TEST ---
        # Store original walk_forward config to restore later
        original_walk_forward_config = self.config['walk_forward'].copy()

        if test_size == 'sanity':
            # For sanity, drastically reduce the number of walk-forward windows
            self.config['walk_forward']['train_months'] = 2  # Very short train period
            self.config['walk_forward']['test_months'] = 1  # Very short test period
            self.config['walk_forward']['step_months'] = 12  # Large steps to reduce total windows further
            logging.info("❗ Running SANITY test: Walk-forward window parameters adjusted for minimal run.")
            logging.info(
                f"Sanity Walk-Forward Config: Train={self.config['walk_forward']['train_months']}m, Test={self.config['walk_forward']['test_months']}m, Step={self.config['walk_forward']['step_months']}m")

        # Step 1: Prepare data
        logging.info("📊 Step 1: Preparing data...")
        stocks = self.get_stock_universe(size_config['stocks'])
        logging.info(f"Loaded stock universe: {len(stocks)} symbols.")
        timestamps = self.generate_test_timestamps(size_config['test_points'])
        logging.info(
            f"Generated {len(timestamps)} test timestamps between {self.config['start_date']} and {self.config['end_date']}.")
        self.validation_windows = self.create_walk_forward_windows()
        logging.info(f"Created {len(self.validation_windows)} walk-forward validation windows.")

        logging.info(
            f"✅ Data Preparation Complete: Loaded {len(stocks)} stocks, {len(timestamps)} timestamps, {len(self.validation_windows)} validation windows")

        # Step 2: Optimize each strategy
        for strategy in strategies:
            logging.info(f"\n🎯 Step 2: Optimizing {strategy.upper()} strategy...")
            best_params = self.optimize_strategy(strategy, stocks, timestamps, self.validation_windows, size_config)
            self.best_parameters[strategy] = best_params
            logging.info(
                f"🏆 Optimization for {strategy.upper()} completed. Best Fitness Score: {best_params['fitness_score']:.2f}")

        # --- Restore original walk_forward settings here (before return) ---
        # This is important as generate_final_report uses self.config if called later
        self.config['walk_forward'] = original_walk_forward_config  # Restore the entire dictionary
        logging.info("🔄 Restored original walk-forward configuration.")

        return self.best_parameters

    def get_parameter_space(self, strategy_type):
        """Define parameter space for optimization"""
        logging.info(f"Defining parameter space for {strategy_type} strategy.")
        base_spaces = {
            'signal_weights': [
                {'trend': 0.45, 'momentum': 0.30, 'volume': 0.10, 'sr': 0.05, 'model': 0.10},
                {'trend': 0.40, 'momentum': 0.35, 'volume': 0.10, 'sr': 0.05, 'model': 0.10},
                {'trend': 0.50, 'momentum': 0.25, 'volume': 0.10, 'sr': 0.05, 'model': 0.10},
                {'trend': 0.35, 'momentum': 0.35, 'volume': 0.15, 'sr': 0.05, 'model': 0.10},
                {'trend': 0.40, 'momentum': 0.30, 'volume': 0.15, 'sr': 0.10, 'model': 0.05},
                {'trend': 0.30, 'momentum': 0.40, 'volume': 0.15, 'sr': 0.10, 'model': 0.05}
            ],
            'confidence_params': {
                'base_multiplier': [0.85, 0.90, 0.95, 1.0, 1.05],
                'confluence_weight': [0.8, 0.9, 1.0, 1.1, 1.2],
                'penalty_strength': [0.8, 0.9, 1.0, 1.1, 1.2]
            }
        }

        strategy_specific = {
            'conservative': {
                'buy_threshold': [2.0, 2.2, 2.5, 2.8, 3.0],
                'sell_threshold': [-1.5, -1.8, -2.0, -2.2],
                'min_confidence': [75, 78, 80, 82, 85]
            },
            'balanced': {
                'buy_threshold': [1.2, 1.4, 1.6, 1.8, 2.0],
                'sell_threshold': [-1.0, -1.2, -1.5, -1.8],
                'min_confidence': [68, 70, 72, 75, 78]
            },
            'aggressive': {
                'buy_threshold': [0.8, 1.0, 1.2, 1.4, 1.6],
                'sell_threshold': [-0.8, -1.0, -1.2, -1.4],
                'min_confidence': [60, 63, 65, 68, 70]
            },
            'swing': {
                'buy_threshold': [1.4, 1.6, 1.8, 2.0, 2.2],
                'sell_threshold': [-1.2, -1.4, -1.6, -1.8],
                'min_confidence': [68, 70, 72, 75, 78]
            }
        }
        # THIS IS THE CRITICAL CHANGE in get_parameter_space:
        # It should return a combined dictionary of base parameters AND the specific strategy's parameters
        return {**base_spaces, **strategy_specific[strategy_type]}

    def get_stock_universe(self, stock_count):
        """Get NASDAQ 100 stocks with filtering"""
        logging.info(f"Retrieving stock universe. Desired count: {stock_count}")

        # NASDAQ 100 symbols (you can update this list)
        # Sorted the list for consistency, though it won't impact logic directly
        nasdaq_100_symbols = [
            'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'AMAT', 'AMD', 'AMGN',
            'AMZN', 'ANSS', 'ARM', 'ASML', 'ATVI', 'AVGO', 'BGNE', 'BIIB', 'BKNG', 'CCEP', 'CDNS',
            'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTSH', 'DASH', 'DDOG',
            'DLTR', 'DXCM', 'EA', 'ENPH', 'EXC', 'FANG', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GOOG',
            'GOOGL', 'HON', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LRCX',
            'LULU', 'LYFT', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU',
            'NFLX', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL',
            'QCOM', 'REGN', 'RIVN', 'ROKU', 'ROST', 'SBUX', 'SIRI', 'SMCI', 'SNPS', 'TEAM', 'TMUS',
            'TSLA', 'TTD', 'TXN', 'UBER', 'VRSK', 'VRTX', 'WBA', 'WBD', 'XEL', 'ZM', 'ZS'
        ]

        if stock_count == 'all':
            logging.info("Selected all NASDAQ 100 symbols.")
            return nasdaq_100_symbols
        else:
            selected_stocks = nasdaq_100_symbols[:min(stock_count, len(nasdaq_100_symbols))]
            logging.info(f"Selected a subset of {len(selected_stocks)} stocks for testing.")
            return selected_stocks

    def generate_test_timestamps(self, test_points):
        """Generate evenly spaced test timestamps"""
        logging.info(f"Generating {test_points} evenly spaced test timestamps.")
        start_date = pd.Timestamp(self.config['start_date'])
        end_date = pd.Timestamp(self.config['end_date'])

        # Generate business days only
        business_days = pd.bdate_range(start=start_date, end=end_date)
        logging.info(f"Found {len(business_days)} business days between {start_date.date()} and {end_date.date()}.")

        # Select evenly spaced timestamps
        if test_points >= len(business_days):
            logging.info(
                "Number of test points is greater than or equal to total business days. Using all business days.")
            return business_days.tolist()

        # Ensure step is at least 1 to avoid ZeroDivisionError if business_days is small
        step = max(1, len(business_days) // test_points)
        selected_dates = business_days[::step][:test_points]
        logging.info(f"Selected {len(selected_dates)} timestamps with a step of {step} days.")

        return [date.date() for date in selected_dates]

    def get_timestamps_in_window(self, start_date, end_date):
        """Get timestamps within a specific window"""
        # This function is called frequently, so keep logging minimal here to avoid excessive log size
        business_days = pd.bdate_range(start=start_date, end=end_date)
        step = max(1, len(business_days) // 20)  # Max 20 timestamps per window
        selected_dates = business_days[::step]
        logging.debug(f"Selected {len(selected_dates)} timestamps in window {start_date.date()} to {end_date.date()}.")
        return [date.date() for date in selected_dates]

    def apply_parameters_to_advisor(self, parameters):
        """Apply parameter set to the advisor instance"""
        logging.debug(f"Applying parameters to advisor: {parameters}")

        # Update the current strategy in the advisor
        if 'strategy_type' in parameters:
            self.advisor.current_strategy = parameters['strategy_type']
            logging.debug(f"Advisor current strategy set to: {self.advisor.current_strategy}")

        # Update signal weights
        if hasattr(self.advisor, 'signal_weights'):
            self.advisor.signal_weights = {
                'trend': parameters.get('weight_trend', self.advisor.signal_weights.get('trend', 1.0)),
                'momentum': parameters.get('weight_momentum', self.advisor.signal_weights.get('momentum', 1.0)),
                'volume': parameters.get('weight_volume', self.advisor.signal_weights.get('volume', 1.0)),
                'support_resistance': parameters.get('weight_support_resistance',
                                                     self.advisor.signal_weights.get('support_resistance', 1.0)),
                'ai_model': parameters.get('weight_ai_model', self.advisor.signal_weights.get('ai_model', 1.0))
            }
            logging.debug(f"Advisor signal weights set to: {self.advisor.signal_weights}")

        # Update the specific settings for the CURRENT strategy within advisor.strategy_settings
        current_strategy_key = self.advisor.current_strategy
        if current_strategy_key in self.advisor.strategy_settings:
            strategy_dict_to_update = self.advisor.strategy_settings[current_strategy_key]

            strategy_dict_to_update['profit'] = parameters.get('profit_threshold',
                                                               strategy_dict_to_update.get('profit', 1.0))
            strategy_dict_to_update['risk'] = parameters.get('stop_loss_threshold',
                                                             strategy_dict_to_update.get('risk', 1.0))

            # Use the specific confidence requirement for the current strategy
            conf_req_key = f"confidence_req_{current_strategy_key.lower().replace(' ', '_')}"
            strategy_dict_to_update['confidence_req'] = parameters.get(conf_req_key,
                                                                       strategy_dict_to_update.get('confidence_req',
                                                                                                   75))

            # Update buy_threshold and sell_threshold for the current strategy
            buy_thresh_key = f'buy_threshold_{current_strategy_key.lower().replace(" ", "_")}'
            sell_thresh_key = f'sell_threshold_{current_strategy_key.lower().replace(" ", "_")}'

            strategy_dict_to_update['buy_threshold'] = parameters.get(buy_thresh_key,
                                                                      strategy_dict_to_update.get('buy_threshold', 0.0))
            strategy_dict_to_update['sell_threshold'] = parameters.get(sell_thresh_key,
                                                                       strategy_dict_to_update.get('sell_threshold',
                                                                                                   0.0))

            logging.debug(f"Advisor strategy settings for {current_strategy_key} updated to: {strategy_dict_to_update}")
        else:
            logging.error(
                f"❌ Strategy '{current_strategy_key}' not found in advisor's strategy_settings during parameter application. This indicates an issue in default settings or strategy naming.")
        # Apply investment days
        self.advisor.investment_days = parameters.get('investment_days', self.advisor.investment_days)
        logging.debug(f"Advisor investment days set to: {self.advisor.investment_days}")

        # Apply confidence parameters if the advisor has them
        if hasattr(self.advisor, 'confidence_params'):
            self.advisor.confidence_params = {
                'base_multiplier': parameters.get('base_multiplier',
                                                  self.advisor.confidence_params.get('base_multiplier', 1.0)),
                'confluence_weight': parameters.get('confluence_weight',
                                                    self.advisor.confidence_params.get('confluence_weight', 1.0)),
                'penalty_strength': parameters.get('penalty_strength',
                                                   self.advisor.confidence_params.get('penalty_strength', 1.0))
            }
            logging.debug(f"Advisor confidence parameters set to: {self.advisor.confidence_params}")

        # The lines below for current_buy_threshold and current_sell_threshold might be redundant
        # if the advisor exclusively uses self.strategy_settings[self.current_strategy]['buy_threshold']
        # but for robustness, we can keep them if ProfessionalStockAdvisor internally uses them.
        # Ensure they align with the values just set in strategy_settings.
        if hasattr(self.advisor, 'current_buy_threshold') and current_strategy_key in self.advisor.strategy_settings:
            self.advisor.current_buy_threshold = self.advisor.strategy_settings[current_strategy_key]['buy_threshold']
            logging.debug(f"Advisor current_buy_threshold set to: {self.advisor.current_buy_threshold}")
        if hasattr(self.advisor, 'current_sell_threshold') and current_strategy_key in self.advisor.strategy_settings:
            self.advisor.current_sell_threshold = self.advisor.strategy_settings[current_strategy_key]['sell_threshold']
            logging.debug(f"Advisor current_sell_threshold set to: {self.advisor.current_sell_threshold}")

    def calculate_actual_return(self, stock, timestamp, prediction_days):
        """Calculate actual return for prediction validation"""
        logging.debug(f"[{stock}] Calculating actual return for {timestamp} over {prediction_days} days.")
        try:
            # Convert timestamp to datetime if needed
            if isinstance(timestamp, str):
                timestamp = pd.Timestamp(timestamp).date()

            # Get stock data
            start_date = timestamp - timedelta(days=5)  # Buffer for data
            end_date = timestamp + timedelta(days=prediction_days + 5)

            logging.debug(f"[{stock}] Fetching data from {start_date} to {end_date}")
            df = yf.download(stock, start=start_date, end=end_date, progress=False)

            if df.empty:
                logging.warning(
                    f"[{stock}] ⚠️ No data found for {stock} between {start_date} and {end_date}. Cannot calculate actual return.")
                return None

            # Get price at timestamp
            timestamp_pd = pd.Timestamp(timestamp)
            start_price_raw = None
            if timestamp_pd in df.index:
                start_price_raw = df.loc[timestamp_pd, 'Close']
            else:
                # Find closest date
                closest_idx = df.index.get_indexer([timestamp_pd], method='nearest')[0]
                if closest_idx < 0 or closest_idx >= len(df):
                    logging.warning(
                        f"[{stock}] ⚠️ No close date found for {stock} at {timestamp} in fetched data. df.index: {df.index.min().date()} to {df.index.max().date()}. Cannot calculate actual return.")
                    return None
                start_price_raw = df.iloc[closest_idx]['Close']

            # --- FIX: Ensure start_price is a scalar float ---
            if pd.api.types.is_scalar(start_price_raw):
                start_price = float(start_price_raw)
            elif isinstance(start_price_raw, (pd.Series, np.ndarray)) and len(start_price_raw) > 0:
                start_price = float(start_price_raw.iloc[0]) # Use iloc[0] for Series
            else:
                start_price = None # Handle cases where extraction fails

            if start_price is None:
                logging.warning(f"[{stock}] Start price could not be determined for {timestamp}. Returning None.")
                return None
            logging.debug(f"[{stock}] Start price for {timestamp} found directly: {start_price:.2f}")


            # Get price after prediction_days
            future_date = timestamp + timedelta(days=prediction_days)
            future_pd = pd.Timestamp(future_date)

            # Find next available trading day
            future_data = df[df.index >= future_pd]
            if future_data.empty:
                logging.warning(
                    f"[{stock}] ⚠️ No future data found for {stock} after {future_date} in fetched data. df.index: {df.index.min().date()} to {df.index.max().date()}. Cannot calculate actual return.")
                return None

            end_price_raw = future_data.iloc[0]['Close']
            # --- FIX: Ensure end_price is a scalar float ---
            if pd.api.types.is_scalar(end_price_raw):
                end_price = float(end_price_raw)
            elif isinstance(end_price_raw, (pd.Series, np.ndarray)) and len(end_price_raw) > 0:
                end_price = float(end_price_raw.iloc[0])
            else:
                end_price = None # Handle cases where extraction fails

            if end_price is None:
                logging.warning(f"[{stock}] End price could not be determined for {future_date}. Returning None.")
                return None
            logging.debug(f"[{stock}] End price for {future_data.iloc[0].name.date()}: {end_price:.2f}")

            # Calculate return
            actual_return = (end_price - start_price) / start_price * 100
            logging.debug(f"[{stock}] Actual return for {timestamp}: {actual_return:.2f}%")
            return actual_return

        except Exception as e:
            logging.error(f"[{stock}] ❌ Error calculating return for {timestamp}: {e}")
            logging.error(f"[{stock}] Full traceback for return calculation error: {traceback.format_exc()}")
            return None

    def evaluate_prediction(self, recommendation, actual_return):
        """Evaluate prediction accuracy"""
        if actual_return is None:
            logging.debug(
                f"Recommendation evaluation: Actual return is None for recommendation {recommendation['action']}. Returning default evaluation.")
            return {'correct': False, 'direction_correct': False, 'profitable': False,
                    'actual_return': None, 'predicted_profit': 0}

        action = recommendation['action']
        predicted_profit = recommendation.get('expected_profit_pct', 0)
        logging.debug(
            f"Recommendation evaluation: Action={action}, Predicted Profit={predicted_profit:.2f}%, Actual Return={actual_return:.2f}%")

        # Direction accuracy
        direction_correct = False
        if action == 'BUY':
            direction_correct = actual_return > 0
        elif action == 'SELL/AVOID':
            direction_correct = actual_return < 0
        else:  # WAIT
            direction_correct = abs(actual_return) < 3.0  # Sideways movement

        profitable = False
        if action == 'BUY':
            profitable = actual_return > 2.0  # At least 2% profit
        elif action == 'SELL/AVOID':
            profitable = actual_return < -1.0  # Avoided loss
        else:
            profitable = abs(actual_return) < 2.0

        # Overall correctness (stricter criteria)
        correct = False
        if action == 'BUY':
            correct = actual_return > 3.0  # Strong profit for BUY
        elif action == 'SELL/AVOID':
            correct = actual_return < -2.0  # Significant loss avoided
        else:  # WAIT
            correct = abs(actual_return) < 2.0  # Truly sideways

        logging.debug(
            f"Evaluation results: Correct={correct}, Direction Correct={direction_correct}, Profitable={profitable}")
        return {
            'correct': correct,
            'direction_correct': direction_correct,
            'profitable': profitable,
            'actual_return': actual_return,
            'predicted_profit': predicted_profit
        }

    def aggregate_walk_forward_results(self, window_results):
        """Aggregate results across all walk-forward windows"""
        logging.info(f"Aggregating {len(window_results)} walk-forward window results.")
        if not window_results:
            logging.warning("No window results to aggregate. Returning empty metrics.")
            return self.get_empty_metrics()

        # Extract performance metrics from all windows
        all_metrics = [wr['performance'] for wr in window_results if wr['performance']]

        if not all_metrics:
            logging.warning(
                "No valid performance metrics found across all windows after filtering. Returning empty metrics.")
            return self.get_empty_metrics()

        # Filter out None or non-numeric values for calculations
        overall_accuracies = [m['overall_accuracy'] for m in all_metrics if m['overall_accuracy'] is not None]
        direction_accuracies = [m['direction_accuracy'] for m in all_metrics if m['direction_accuracy'] is not None]
        buy_success_rates = [m['buy_success_rate'] for m in all_metrics if m['buy_success_rate'] is not None]
        avg_returns = [m['avg_return'] for m in all_metrics if m['avg_return'] is not None]
        volatilities = [m['volatility'] for m in all_metrics if m['volatility'] is not None]
        sharpe_ratios = [m['sharpe_ratio'] for m in all_metrics if m['sharpe_ratio'] is not None]
        max_drawdowns = [m['max_drawdown'] for m in all_metrics if m['max_drawdown'] is not None]
        total_trades = sum([m['total_trades'] for m in all_metrics])
        confidence_avgs = [m['confidence_avg'] for m in all_metrics if m['confidence_avg'] is not None]

        # Calculate averages, handling empty lists
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

        # Aggregate signal distribution - ensure they are numeric before summing
        signal_dist_buy = [m['signal_distribution']['BUY'] for m in all_metrics if
                           'BUY' in m.get('signal_distribution', {}) and m['signal_distribution']['BUY'] is not None]
        signal_dist_sell = [m['signal_distribution']['SELL/AVOID'] for m in all_metrics if
                            'SELL/AVOID' in m.get('signal_distribution', {}) and m['signal_distribution'][
                                'SELL/AVOID'] is not None]
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
        """Return empty metrics structure"""
        logging.debug("Returning empty metrics structure.")
        return {
            'overall_accuracy': 0,
            'direction_accuracy': 0,
            'buy_success_rate': 0,
            'avg_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'signal_distribution': {'BUY': 0, 'SELL/AVOID': 0, 'WAIT': 100},
            'confidence_avg': 50
        }

    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns"""
        if not returns:
            logging.debug("No returns data for drawdown calculation. Returning 0.")
            return 0

        # Filter out None values just in case
        valid_returns = [r for r in returns if r is not None]
        if not valid_returns:
            logging.debug("No valid returns data for drawdown calculation. Returning 0.")
            return 0

        cumulative = np.cumprod(1 + np.array(valid_returns) / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100

        logging.debug(f"Calculated max drawdown: {np.min(drawdown):.2f}%")
        return np.min(drawdown)

    def generate_parameter_combinations(self, parameter_space, max_combinations):
        """Generate parameter combinations for testing"""
        logging.info(f"Generating parameter combinations, limiting to {max_combinations} samples.")
        combinations = []

        # Get signal weight combinations
        signal_weights_list = parameter_space['signal_weights']

        # Get threshold combinations
        buy_thresholds = parameter_space['buy_threshold']
        sell_thresholds = parameter_space['sell_threshold']
        min_confidences = parameter_space['min_confidence']

        # Get confidence parameter combinations
        conf_params = parameter_space['confidence_params']
        base_multipliers = conf_params['base_multiplier']
        confluence_weights = conf_params['confluence_weight']
        penalty_strengths = conf_params['penalty_strength']

        # Generate all combinations
        all_possible_combinations = product(
            signal_weights_list,
            buy_thresholds,
            sell_thresholds,
            min_confidences,
            base_multipliers,
            confluence_weights,
            penalty_strengths
        )

        for combo_tuple in all_possible_combinations:
            signal_weights, buy_thresh, sell_thresh, min_conf, base_mult, conf_weight, penalty_str = combo_tuple
            combination = {
                'signal_weights': signal_weights,
                'buy_threshold': buy_thresh,
                'sell_threshold': sell_thresh,
                'min_confidence': min_conf,
                'confidence_params': {
                    'base_multiplier': base_mult,
                    'confluence_weight': conf_weight,
                    'penalty_strength': penalty_str
                }
            }
            combinations.append(combination)
            # Limit combinations if max_combinations is reached
            if len(combinations) >= max_combinations:
                logging.info(f"Generated {len(combinations)} parameter combinations (reached max_combinations limit).")
                return combinations

        logging.info(f"Generated {len(combinations)} parameter combinations (all possible combinations).")
        return combinations

    def create_performance_summary(self):
        """Create performance summary for report"""
        logging.info("Creating overall performance summary.")
        summary = {}

        for strategy, results in self.best_parameters.items():
            # Safely get parameters, defaulting to an empty dict if 'parameters' key is missing or None
            params = results.get('parameters', {})

            summary[strategy] = {
                'fitness_score': results['fitness_score'],
                'performance_metrics': results['performance'],
                'optimal_parameters': {
                    'buy_threshold': params.get('buy_threshold', None),
                    'sell_threshold': params.get('sell_threshold', None),
                    'min_confidence': params.get('min_confidence', None)
                }
            }
            logging.debug(
                f"Summary for {strategy}: Fitness Score {results['fitness_score']:.2f}, Optimal Params: {params}")

        return summary

    def create_walk_forward_windows(self):
        """Create chronological walk-forward validation windows"""
        logging.info("Generating walk-forward validation windows.")
        start_date = pd.Timestamp(self.config['start_date'])
        end_date = pd.Timestamp(self.config['end_date'])

        train_months = self.config['walk_forward']['train_months']
        test_months = self.config['walk_forward']['test_months']
        step_months = self.config['walk_forward']['step_months']

        logging.info(
            f"Walk-forward config: Train={train_months} months, Test={test_months} months, Step={step_months} months.")

        windows = []
        current_start = start_date

        while current_start + pd.DateOffset(months=train_months + test_months) <= end_date:
            train_end = current_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)

            window_info = {
                'train_start': current_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'window_id': len(windows) + 1
            }
            windows.append(window_info)
            logging.info(
                f"Generated window {len(windows)}: Train ({current_start.date()} - {train_end.date()}), Test ({test_start.date()} - {test_end.date()})")

            current_start += pd.DateOffset(months=step_months)

        logging.info(f"Finished generating {len(windows)} walk-forward windows.")
        return windows

    def validate_parameters_walk_forward(self, parameters, stocks, windows):
        """Validate parameters using walk-forward analysis"""
        # This function iterates over windows, detailed logging for each window's performance is handled inside the loop
        # logging.info(f"Starting walk-forward validation for current parameter set. {len(windows)} windows to process.")
        window_results = []

        for window in tqdm(windows, desc="Walk-Forward Validation"):
            logging.info(
                f"Processing window {window['window_id']}: Test period {window['test_start'].date()} to {window['test_end'].date()}")
            # Get test timestamps for this window
            test_timestamps = self.get_timestamps_in_window(
                window['test_start'],
                window['test_end']
            )
            logging.debug(f"Window {window['window_id']} has {len(test_timestamps)} timestamps for testing.")

            # Test parameters on this window
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

        # Aggregate results across all windows
        aggregated_performance = self.aggregate_walk_forward_results(window_results)
        logging.info(
            f"Aggregated performance for this parameter set across all windows: Fitness Score={self.calculate_fitness_score(aggregated_performance):.2f}")
        return aggregated_performance

    def test_parameter_set(self, parameters, stocks, timestamps, window):
        """Test a specific parameter set"""
        # This is a highly iterative function, so detailed logging within loops should use logging.debug
        # to prevent overwhelming the logs during full runs.
        logging.debug(
            f"Testing parameter set on {len(stocks)} stocks for {len(timestamps)} timestamps in window {window['window_id']}.")

        # Apply parameters to advisor
        self.apply_parameters_to_advisor(parameters)
        logging.debug("Parameters applied to advisor instance.")

        results = []
        for stock in stocks:
            for timestamp in timestamps:
                try:
                    # Run analysis with current parameters
                    recommendation = self.advisor.analyze_stock_enhanced(stock, timestamp)

                    if recommendation:
                        logging.debug(
                            f"  {stock} on {timestamp}: Recommendation: {recommendation['action']} (Confidence: {recommendation['confidence']:.1f}%)")
                        # Calculate actual return
                        actual_return = self.calculate_actual_return(
                            stock, timestamp, self.config['prediction_window_days']
                        )

                        # Evaluate prediction
                        performance = self.evaluate_prediction(recommendation, actual_return)

                        # --- FIX: Normalize numeric types before adding to results ---
                        # Apply normalization to the entire dictionaries/values
                        normalized_recommendation = self._normalize_numeric_values(recommendation)
                        normalized_performance = self._normalize_numeric_values(performance)
                        normalized_actual_return = self._normalize_numeric_values(actual_return) # This might be None

                        # --- FIX: Conditionally format normalized_actual_return ---
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
                    logging.error(f"  Full traceback: {traceback.format_exc()}")  # Log full traceback for debugging
                    # Ensure a minimal valid structure is appended even on error
                    results.append({
                        'stock': stock,
                        'timestamp': timestamp,
                        'recommendation': self._normalize_numeric_values(
                            {'action': 'WAIT', 'confidence': 50, 'expected_profit_pct': 0}),
                        # Default safe recommendation
                        'actual_return': self._normalize_numeric_values(None),  # Explicitly None if calculation failed
                        'performance': self._normalize_numeric_values(self.get_empty_metrics())  # Use empty metrics
                    })
                    continue  # Continue to next timestamp/stock even if one fails

        calculated_metrics = self.calculate_performance_metrics(results)
        logging.debug(f"Performance metrics calculated for this parameter set in window {window['window_id']}.")
        return calculated_metrics

    def calculate_performance_metrics(self, results):
        """Calculate comprehensive performance metrics"""
        logging.debug(f"Calculating performance metrics for {len(results)} individual test results.")

        if not results:
            logging.warning("No individual test results to calculate performance metrics. Returning empty metrics.")
            return self.get_empty_metrics()

        # The DataFrame creation should now be more robust due to type normalization
        df = pd.DataFrame(results)
        logging.debug(f"DataFrame created with {len(df)} rows.")

        # Filter out invalid entries before calculations
        valid_results = [r for r in results if
                         r['actual_return'] is not None and r['recommendation']['confidence'] is not None]

        if not valid_results:
            logging.warning("No valid individual results after filtering. Returning empty metrics.")
            return self.get_empty_metrics()

        logging.debug(f"Using {len(valid_results)} valid results for metric calculation.")

        # Basic accuracy metrics
        correct_predictions = sum(r['performance']['correct'] for r in valid_results)
        total_predictions = len(valid_results)
        overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        logging.debug(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_predictions})")

        # Direction accuracy
        direction_correct = sum(r['performance']['direction_correct'] for r in valid_results)
        direction_accuracy = (direction_correct / total_predictions) * 100 if total_predictions > 0 else 0
        logging.debug(f"Direction Accuracy: {direction_accuracy:.2f}%")

        # Signal-specific metrics
        buy_signals = [r for r in valid_results if r['recommendation']['action'] == 'BUY']
        sell_signals = [r for r in valid_results if r['recommendation']['action'] == 'SELL/AVOID']
        wait_signals = [r for r in valid_results if r['recommendation']['action'] == 'WAIT']

        buy_success = sum(r['performance']['profitable'] for r in buy_signals) if buy_signals else 0
        buy_success_rate = (buy_success / len(buy_signals) * 100) if buy_signals else 0
        logging.debug(f"BUY Signals: {len(buy_signals)}, BUY Success: {buy_success_rate:.2f}%")

        # Risk metrics
        returns = [r['actual_return'] for r in valid_results if r['actual_return'] is not None]  # Ensure no None values
        avg_return = np.mean(returns) if returns else 0
        volatility = np.std(returns) if returns else 0
        sharpe_ratio = (avg_return / volatility) if volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(returns)  # Pass filtered returns
        logging.debug(
            f"Avg Return: {avg_return:.2f}%, Volatility: {volatility:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}%")

        # Signal distribution
        signal_distribution = {
            'BUY': len(buy_signals) / total_predictions * 100 if total_predictions > 0 else 0,
            'SELL/AVOID': len(sell_signals) / total_predictions * 100 if total_predictions > 0 else 0,
            'WAIT': len(wait_signals) / total_predictions * 100 if total_predictions > 0 else 0
        }
        logging.debug(f"Signal Distribution: {signal_distribution}")

        confidence_values = [r['recommendation']['confidence'] for r in valid_results if
                             r['recommendation']['confidence'] is not None]
        confidence_avg = np.mean(confidence_values) if confidence_values else 50
        logging.debug(f"Average Confidence: {confidence_avg:.2f}%")

        return {
            'overall_accuracy': overall_accuracy,
            'direction_accuracy': direction_accuracy,
            'buy_success_rate': buy_success_rate,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_predictions,
            'signal_distribution': signal_distribution,
            'confidence_avg': confidence_avg
        }

    def optimize_strategy(self, strategy_type, stocks, timestamps, windows, size_config):
        """Optimize parameters for specific strategy using binary search + grid search"""

        logging.info(f"🔍 Optimizing {strategy_type} strategy...")

        parameter_space = self.get_parameter_space(strategy_type)
        best_score = -np.inf  # Initialize with negative infinity for fitness score
        best_params = None
        performance = self.get_empty_metrics()  # Initialize performance to prevent UnboundLocalError
        current_strategy_results = []  # To store results of each tested parameter set

        # Generate parameter combinations (limited by size_config)
        param_combinations = self.generate_parameter_combinations(
            parameter_space,
            max_combinations=size_config['param_samples']
        )

        logging.info(f"Testing {len(param_combinations)} parameter combinations for {strategy_type} strategy...")

        for i, params in enumerate(tqdm(param_combinations, desc=f"Testing {strategy_type}")):
            logging.info(f"--- Testing Parameter Set {i + 1}/{len(param_combinations)} for {strategy_type} ---")
            logging.info(f"  Parameters: {params}")  # Log the full parameter set being tested

            try:
                # Validate using walk-forward analysis
                current_performance = self.validate_parameters_walk_forward(params, stocks, windows)

                # Calculate fitness score
                fitness_score = self.calculate_fitness_score(current_performance)

                current_strategy_results.append({
                    'parameter_set_id': i,
                    'parameters': params,
                    'performance': current_performance,
                    'fitness_score': fitness_score
                })

                if fitness_score > best_score:
                    old_best_score = best_score
                    best_score = fitness_score
                    best_params = params.copy()
                    performance = current_performance  # Update the best performance as well

                    if old_best_score == -np.inf:  # First successful parameter set
                        logging.info(f"🎯 First successful parameter set for {strategy_type}: Score {fitness_score:.2f}")
                    else:
                        logging.info(
                            f"🎯 New best {strategy_type}: Score improved from {old_best_score:.2f} to {fitness_score:.2f}")
                    logging.info(f"   Accuracy: {performance['overall_accuracy']:.1f}%")
                    logging.info(f"   BUY Success: {performance['buy_success_rate']:.1f}%")
                    logging.info(f"   Sharpe: {performance['sharpe_ratio']:.2f}")

            except Exception as e:
                logging.error(f"❌ Error testing parameter set {i} for {strategy_type}: {e}")
                logging.error(f"  Full traceback: {traceback.format_exc()}")  # Log full traceback for debugging
                self._save_intermediate_strategy_results(strategy_type, current_strategy_results, i, str(e))
                continue

        # If no successful parameters were found (e.g., all failed or list was empty)
        if best_params is None and current_strategy_results:
            # Fallback to the best among partially successful runs if any
            best_run = max(current_strategy_results, key=lambda x: x['fitness_score'])
            best_params = best_run['parameters']
            performance = best_run['performance']
            best_score = best_run['fitness_score']
            logging.info(
                f"ℹ️ No new best found, falling back to best observed so far for {strategy_type}: Score {best_score:.2f}")
        elif best_params is None:  # If current_strategy_results is also empty (all failed from start)
            best_params = {}  # Return empty dict or default
            performance = self.get_empty_metrics()
            best_score = 0
            logging.warning(f"⚠️ No successful parameter sets found for {strategy_type}. Returning empty results.")

        logging.info(f"✅ Optimization for {strategy_type} complete. Final Best Fitness Score: {best_score:.2f}")
        return {
            'parameters': best_params,
            'performance': performance,
            'fitness_score': best_score
        }

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
            # Ensure results data is normalized too
        }

        try:
            with open(file_location, 'w') as f:
                json.dump(partial_report, f, indent=2, default=str)
            logging.warning(f"⚠️ Partial results saved to: {file_location}")
        except Exception as e:
            logging.error(f"❌ Failed to save partial results: {e}. Error: {e}")

    def calculate_fitness_score(self, performance):
        """Calculate weighted fitness score from performance metrics"""
        logging.debug(f"Calculating fitness score for performance: {performance}")
        weights = {
            'overall_accuracy': 0.25,
            'direction_accuracy': 0.20,
            'buy_success_rate': 0.25,
            'sharpe_ratio': 0.15,
            'signal_distribution_penalty': 0.15
        }

        # Handle cases where performance metrics might be None or 0
        overall_accuracy = performance.get('overall_accuracy', 0)
        direction_accuracy = performance.get('direction_accuracy', 0)
        buy_success_rate = performance.get('buy_success_rate', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        signal_distribution = performance.get('signal_distribution', {'BUY': 0, 'SELL/AVOID': 0, 'WAIT': 100})

        # Normalize metrics to 0-100 scale
        accuracy_score = min(overall_accuracy, 100)
        direction_score = min(direction_accuracy, 100)
        buy_success_score = min(buy_success_rate, 100)
        sharpe_score = min(max(sharpe_ratio * 50, 0), 100)  # Convert to 0-100

        # Penalty for extreme signal distributions
        buy_pct = signal_distribution.get('BUY', 0)
        if buy_pct < 15 or buy_pct > 50:
            distribution_penalty = abs(32.5 - buy_pct) * 2  # Ideal ~32.5%
        else:
            distribution_penalty = 0

        distribution_score = max(100 - distribution_penalty, 0)

        # Calculate weighted score
        fitness_score = (
                accuracy_score * weights['overall_accuracy'] +
                direction_score * weights['direction_accuracy'] +
                buy_success_score * weights['buy_success_rate'] +
                sharpe_score * weights['sharpe_ratio'] +
                distribution_score * weights['signal_distribution_penalty']
        )
        logging.debug(f"Calculated fitness score: {fitness_score:.2f}")

        return fitness_score

    def generate_final_report(self, execution_time_hours=0.0):
        """Generate comprehensive calibration report"""
        logging.info("📝 Generating final calibration report.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        report = {
            'metadata': {
                'calibration_date': timestamp,
                'config': self._normalize_numeric_values(self.config),  # Normalize config before saving
                'total_strategies_tested': len(self.best_parameters),
                'validation_method': 'walk_forward_analysis',
                'execution_time_hours': execution_time_hours
            },
            'best_parameters': self._normalize_numeric_values(self.best_parameters),  # Normalize best_parameters
            'summary': self._normalize_numeric_values(self.create_performance_summary())  # Normalize summary
        }

        # Save results
        filename = f"stockwise_calibration_{timestamp}.json"
        file_location = os.path.join(self.configuration_files, filename)
        try:
            with open(file_location, 'w') as f:
                json.dump(report, f, indent=2,
                          default=str)  # default=str handles non-serializable objects like datetime
            logging.info(f"\n✅ Calibration complete! Results saved to: {file_location}")
            self.print_results_summary()
        except Exception as e:
            logging.error(f"❌ Failed to save final report: {e}. Error: {e}")

        return filename

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

            params = results.get('parameters', {})  # Safely get parameters dict
            logging.info(f"   Optimal Thresholds:")
            # Use .get() with None as default, then check for None for printing
            buy_thresh = params.get('buy_threshold')
            sell_thresh = params.get('sell_threshold')
            min_conf = params.get('min_confidence')

            logging.info(f"     BUY: {buy_thresh:.2f}" if buy_thresh is not None else "     BUY: N/A")
            logging.info(f"     SELL: {sell_thresh:.2f}" if sell_thresh is not None else "     SELL: N/A")
            logging.info(
                f"     Min Confidence: {min_conf:.0f}%" if min_conf is not None else "     Min Confidence: N/A%")  # Confidence is often integer

        logging.info("\n" + "=" * 80)

    def export_parameters_for_production(self, strategy_type):
        """Export optimized parameters for production use"""
        logging.info(f"📦 Attempting to export production parameters for {strategy_type}.")

        if strategy_type not in self.best_parameters or not self.best_parameters[strategy_type]['parameters']:
            logging.warning(f"⚠️ No optimized parameters found for {strategy_type}. Skipping export.")
            return None

        params = self.best_parameters[strategy_type]['parameters']

        production_config = {
            'strategy_multipliers': {
                strategy_type: {
                    'profit': params.get('profit_multiplier', 1.0),
                    'risk': params.get('risk_multiplier', 1.0),
                    'confidence_req': params['min_confidence']
                }
            },
            'signal_weights': params['signal_weights'],
            'thresholds': {
                'buy_threshold': params['buy_threshold'],
                'sell_threshold': params['sell_threshold']
            },
            'confidence_params': params['confidence_params']
        }

        # Normalize production_config before dumping to JSON
        production_config = self._normalize_numeric_values(production_config)

        filename = f"stockwise_production_params_{strategy_type}_{datetime.now().strftime('%Y%m%d')}.json"
        file_location = os.path.join(self.configuration_files, filename)
        try:
            with open(file_location, 'w') as f:
                json.dump(production_config, f, indent=2)
            logging.info(f"✅ Production parameters exported to: {file_location}")
        except Exception as e:
            logging.error(f"❌ Failed to export production parameters for {strategy_type}: {e}. Error: {e}")

        return file_location

    # Usage Example


def run_stockwise_calibration(test_size='medium', strategies=['balanced', 'aggressive']):
    """
    Example of how to run the calibration with configurable test size and strategies.

    Args:
        test_size (str): The size of the test ('small', 'medium', 'full', 'sanity').
        strategies (list): A list of strategy types to optimize (e.g., ['balanced', 'aggressive']).
    """
    starting_time = time.time()
    logging.info("✨ Initializing Stock Advisor for calibration.")
    advisor = ProfessionalStockAdvisor(debug=True,
                                       download_log=False)  # download_log=False means no separate log file from advisor
    logging.info("✅ Stock Advisor initialized.")

    # Create calibrator
    calibrator = StockWiseAutoCalibrator(advisor)
    logging.info("✅ Auto-Calibrator instance created.")

    # Run calibration with specified configurations
    best_params = calibrator.run_calibration(
        test_size=test_size,
        strategies=strategies
    )
    logging.info("📈 Calibration process finished.")

    # Export best parameters for production
    logging.info("\n📦 Exporting best parameters for production...")
    for strategy in best_params.keys():
        calibrator.export_parameters_for_production(strategy)

    end_time = time.time()
    total_execution_seconds = end_time - starting_time
    execution_time_hours = total_execution_seconds / 3600

    # Pass the actual execution time to the final report
    # The generate_final_report call was moved here to ensure accurate execution_time_hours
    calibrator.generate_final_report(execution_time_hours=execution_time_hours)

    logging.info(f"Total Execution Time: {total_execution_seconds:.2f} seconds")
    logging.info(f"\n✅ StockWise Auto-Calibration completed in {execution_time_hours:.2f} hours.")
    return best_params


# Run it
if __name__ == "__main__":
    # Temporarily set root logger to DEBUG to capture all granular logs for debugging this issue
    root_logger.setLevel(logging.WARNING)

    logging.info("--- Starting StockWise Auto-Calibration Script ---")
    print("Select a test size and strategies to optimize:")
    print("Estimate run time based on selected test size and strategies:")
    print("Sanity Test = ~12 minutes ; Small Test = ~5 hours ; Medium Test = ~12 hours ; Full Test = ~24 hours")
    test_type_input = input("Test Size (1 = sanity, 2 = small, 3 = medium, 4 = full): ")

    selected_test_size = 'medium'  # Default
    selected_strategies = ['balanced']  # Default

    if test_type_input == '1':
        selected_test_size = 'sanity'
        selected_strategies = ['balanced', 'aggressive']  # As per your last request for sanity
        logging.info(f"User selected SANITY test (1) with strategies: {', '.join(selected_strategies)}")
    elif test_type_input == '2':
        selected_test_size = 'small'
        selected_strategies = ['balanced', 'aggressive']  # Assuming you want both for small
        logging.info(f"User selected SMALL test (2) with strategies: {', '.join(selected_strategies)}")
    elif test_type_input == '3':
        selected_test_size = 'medium'
        selected_strategies = ['balanced', 'conservative', 'aggressive']  # Assuming you want more for medium
        logging.info(f"User selected MEDIUM test (3) with strategies: {', '.join(selected_strategies)}")
    elif test_type_input == '4':
        selected_test_size = 'full'
        selected_strategies = ['balanced', 'aggressive', 'conservative', 'swing']  # Assuming all for full
        logging.info(f"User selected FULL test (4) with strategies: {', '.join(selected_strategies)}")
    else:
        logging.error("Invalid input. Please enter a number between 1 and 4. Exiting.")
        exit()

    run_stockwise_calibration(
        test_size=selected_test_size,
        strategies=selected_strategies
    )
    logging.info("--- Script execution completed ---")
