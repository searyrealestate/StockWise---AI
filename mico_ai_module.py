# mico_ai_module.py

import pandas as pd
import logging
from typing import Dict, Any


class AI_ParamPredictor:
    """
    Predicts optimal parameters for the Micha (MICO) Rule-Based Advisor.
    This class enables the 'B' (AI-Adaptive) scenario in the A/B testing framework.
    """

    def __init__(self, model_version: str = "v1.0"):
        self.model = None  # Placeholder for a deep learning or RL model
        self.model_version = model_version
        self.log = logging.getLogger(type(self).__name__)
        self.log.info(f"AI_ParamPredictor initialized (Version: {model_version}).")

        # --- START: NEW METHODS FOR AI_ParamPredictor (in mico_ai_module.py) ---

    def get_adaptive_state(self) -> Dict[str, Any]:
        """Returns the current state learned from past performance."""
        # This state should be unique to the session or application run if not persisted externally.
        if not hasattr(self, '_adaptive_state'):
            self._adaptive_state = {
                # Initialize base adaptive parameter
                'last_stop_loss_multiplier': 2.0,
                'avg_win_rate': 0.50,
                'last_finetune_date': 'N/A'
            }
        return self._adaptive_state

        # קובץ: mico_ai_module.py
    def run_finetuning_simulation(self,
                                  trade_results_summary: Dict[str, Any],
                                  finetune_date: str) -> str:
        """
        Simulates the AI learning process by updating its internal adaptive state
        based on historical trade results (P&L) from a backtest.

        ASSUMES trade_results_summary NOW CONTAINS: 'win_rate', 'total_trades', 'max_drawdown_pct'

        :param trade_results_summary: A dictionary containing metrics
        :return: A status message indicating parameter adjustment.
        """

        if trade_results_summary.get('total_trades', 0) < 10:  # Increased trades required for better data
            return "Finetuning skipped: Insufficient trade history (< 10 trades)."

        state = self.get_adaptive_state()
        win_rate = trade_results_summary['win_rate']
        max_drawdown_pct = trade_results_summary.get('max_drawdown_pct', 0)  # Assumed new metric

        # Get current multiplier
        current_sl_mult = state['last_stop_loss_multiplier']
        new_sl_mult = current_sl_mult

        # --- ENHANCED ADAPTIVE LEARNING LOGIC (Focus on Drawdown/Stability) ---
        status_msg = ""

        # Rule 1: High Drawdown Penalty (Focus on risk stability)
        # If the system had a high drawdown, tighten the stop to prevent catastrophic loss.
        if max_drawdown_pct > 15.0:
            new_sl_mult = max(1.5, current_sl_mult - 0.5)  # Tighten by 0.5x, minimum 1.5x
            status_msg = f"CRITICAL: Max Drawdown ({max_drawdown_pct:.1f}%) > 15%. **TIGHTENING** adaptive SL."

        # Rule 2: Low Win Rate, but Drawdown is low (Suggests entries are bad, not exits)
        elif win_rate < 0.40 and max_drawdown_pct < 10.0:
            # We don't change SL much, maybe slightly widen to avoid noise
            new_sl_mult = min(3.0, current_sl_mult + 0.1)
            status_msg = f"Win rate low ({win_rate:.2f}), but stable. SL slightly **WIDENED** to filter noise."

        # Rule 3: High Win Rate (Reward stability)
        elif win_rate > 0.70:
            # Tighten the stop to reduce average loss on the few losing trades
            new_sl_mult = max(1.5, current_sl_mult - 0.1)
            status_msg = f"Win rate high ({win_rate:.2f}). **TIGHTENING** adaptive SL for efficiency."

        else:
            status_msg = "Performance is stable (40-70% Win Rate, Drawdown < 15%). SL multiplier unchanged."

        # Apply limits
        state['last_stop_loss_multiplier'] = np.clip(new_sl_mult, 1.5, 3.0)

        # Update tracking metrics
        state['avg_win_rate'] = win_rate
        state['last_finetune_date'] = finetune_date

        self.log.info(f"Finetuning complete. New SL Multiplier: {state['last_stop_loss_multiplier']}")
        self._adaptive_state = state

        return status_msg

    def predict_optimal_params(self,
                               symbol: str,
                               default_params: Dict[str, Any],
                               df_slice: pd.DataFrame,
                               recent_trade_history: pd.DataFrame = None) -> Dict[str, Any]:

        adjusted_params = default_params.copy()
        state = self.get_adaptive_state()  # Get the current adaptive state
        features = self._engineer_contextual_features(df_slice)

        current_atr = features['atr_14']
        atr_threshold = features['volatility_75th_percentile']

        # --- ADAPTIVE ADJUSTMENT LOGIC (Using Learned State) ---
        learned_sl_mult = state['last_stop_loss_multiplier']

        # Rule 1: Use the learned multiplier as the base stop-loss.
        final_sl_mult = learned_sl_mult

        # Rule 2: Apply a temporary high-volatility safeguard (overrides the learned state if current ATR is extremely high)
        if current_atr > atr_threshold * 1.5:
            # If current market is extreme, default to a conservative 2.5x stop if learned stop is tighter
            final_sl_mult = max(learned_sl_mult, 2.5)
            adjusted_params['ai_adjustment_made'] = True
            prediction_reason = f"High Volatility Override -> Stop at {final_sl_mult:.1f}x ATR."

        else:
            adjusted_params['ai_adjustment_made'] = learned_sl_mult != default_params['atr_mult_stop']
            prediction_reason = f"Adaptive Stop (Learned: {final_sl_mult:.1f}x ATR) used. Last Finetune: {state['last_finetune_date']}."

        adjusted_params['atr_mult_stop'] = final_sl_mult
        # --- END ADAPTIVE ADJUSTMENT LOGIC ---

        # 2. Attach Debugging Indicators
        adjusted_params['ai_predictor_reason'] = prediction_reason
        adjusted_params['ai_model_version'] = f"{self.model_version} (SL={final_sl_mult:.1f})"

        self.log.info(f"[{symbol}] AI predicted ATR Multiplier: {adjusted_params['atr_mult_stop']}")

        return adjusted_params

    def _engineer_contextual_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Helper function to calculate features specifically for the AI predictor.
        Keeps the main prediction function small and focuses on memory efficiency
        by only processing the needed indicators.
        """
        if df.empty:
            return {'latest_close': 0.0, 'atr_14': 0.0, 'rsi_14': 50.0}

        # Ensure latest data is calculated (assuming df_slice comes with basics + indicators)
        latest = df.iloc[-1]

        atr_col = 'atrr_14'  # Name based on pandas-ta default naming convention

        # We must check if the column exists (it should have been added by the advisor)
        # If it doesn't exist, we must calculate it first.
        if atr_col not in df.columns:
            # Calculate ATR (assuming default length 14)
            df.ta.atr(length=14, append=True)
            df.columns = [col.lower() for col in df.columns]
            df.dropna(inplace=True)
            latest = df.iloc[-1]
            if atr_col not in df.columns:
                self.log.warning(f"ATR column '{atr_col}' not found even after calculation.")
                # Return safe defaults if calculation fails completely
                return {'latest_close': latest.get('close', 0.0), 'atr_14': 0.0, 'rsi_14': latest.get('rsi_14', 50.0),
                        'volatility_75th_percentile': 0.0}

        atr_quantile = df[atr_col].quantile(0.75) if atr_col in df.columns else 0

        features = {
            'latest_close': latest['close'],
            'atr_14': latest.get(atr_col, 0.0),
            'rsi_14': latest.get('rsi_14', 50.0),
            'volatility_75th_percentile': atr_quantile,
            # Add other features your internal AI models might need (e.g., trend strength)
        }
        return features