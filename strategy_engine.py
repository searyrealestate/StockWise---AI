# strategy_engine.py

"""
StockWise Gen-12 Strategy Engine (The Orchestra)
================================================
The Decision Core.
This module implements the Mixture of Experts (MoE) architecture.
Instead of one giant script trying to do everything, we have 3 specialized Agents 
and 1 Conductor who coordinates them.

- Agent 1 (Regime Router): The Gatekeeper. Looks at the DSP math to decide if the market is safe.
- Agent 2 (Tactical Sniper): The Strategist. Combines AI + Technicals + Taxes to find the edge.
- Agent 3 (Risk Actuary): The Accountant. Calculates exactly how many shares to buy to stay safe.
- Conductor (StrategyEngine): The Public Interface that connects everything.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import joblib
import time
import system_config as cfg
from feature_engine import FeatureEngine

# Initialize Logger (Rule 2 Compliance: Uses the custom format from system_config)
logger = cfg.LoggerSetup.setup_logger("StrategyEngine")

class RegimeRouter:
    """
    AGENT 1: The Gatekeeper.
    Responsibility: Determine the mathematical state of the asset using Digital Signal Processing.
    Output: 'TREND', 'CHOP', or 'HALT'.
    """
    def __init__(self):
        """
        StockWise Gen-13 Strategy Engine (The Conductor)
        Initializes the Sub-Agents for the Mixture of Experts architecture.
        """
        
        # [GEN-13 FIX]: The following line caused the cascading failure and has been removed:
        # self.orchestra = AgentOrchestra() 
        
        # Restored classification thresholds (Do not delete these)
        self.trend_thr = 0.6
        self.chop_thr = 0.4    

    def classify_regime(self, df):
        """
        Analyzes the Efficiency Ratio (ER) to route the stock to the correct AI.
        """
        if df.empty: 
            return "HALT"
        
        # We look at the very last candle to see the current state
        last = df.iloc[-1]
        er_slow = last.get('er_slow', 0)
        er_fast = last.get('er_fast', 0)
        
        # --- STORY: The Whipsaw Protection Logic ---
        # Imagine a car speeding up a hill (High Slow ER). Suddenly, the driver slams the brakes (Low Fast ER).
        # Even though the car is still high up the hill, it is about to roll backward.
        # If we see the Fast Signal collapse (< 0.2) while the Slow Signal is still strong (> 0.6),
        # we shout "HALT!" to prevent buying the exact top of a crash.
        if er_slow > 0.6 and er_fast < 0.2:
            logger.debug(f"Regime HALT: Velocity Divergence detected (Slow: {er_slow:.2f}, Fast: {er_fast:.2f}).")
            return "HALT"
            
        # --- STORY: The Routing Logic ---
        # If the signal is clean and efficient, we send it to the TREND engine.
        # If the signal is messy and noisy, we send it to the CHOP engine.
        # If it's somewhere in the middle (The Dead Zone), we do nothing.
        if er_slow >= self.trend_thr:
            logger.debug(f"Regime: TREND | er_slow={er_slow:.2f} | er_fast={er_fast:.2f}")
            return "TREND"
        elif er_slow <= self.chop_thr:
            logger.debug(f"Regime: CHOP | er_slow={er_slow:.2f} | er_fast={er_fast:.2f}")
            return "CHOP"
        else:
            logger.debug(f"Regime: NEUTRAL | er_slow={er_slow:.2f} | er_fast={er_fast:.2f}")
            return "NEUTRAL"

class TacticalSniper:
    """
    AGENT 2: The Strategist.
    Responsibility: 
    1. Load the specialized AI Model (Trend vs Chop).
    2. Calculate the Technical Score.
    3. Generate the Master Consensus Score.
    4. Enforce the Friction-Adjusted Alpha (Taxes/Fees) Veto.
    """
    
    def __init__(self):
        self.models_dir = cfg.MODELS_DIR
        
        # We load the "Brains" we trained in Phase 4.
        # One brain is an expert at Trends, the other is an expert at Chop.
        self.trend_model = self._load_model("Trend_Master_Model.pkl")
        self.chop_model = self._load_model("Chop_Master_Model.pkl")

        # We also load the "Eyes" (Feature List) so we know exactly what data
        # each brain needs to see to make a decision.
        self.trend_features = self._load_json(os.path.join(self.models_dir, "Trend_Master_Model_features.json"))
        self.chop_features = self._load_json(os.path.join(self.models_dir, "Chop_Master_Model_features.json"))

        # Load the Technical DAGs (The filter sequences we optimized in Phase 3)
        self.dag_file = os.path.join(cfg.DB_DIR, "best_params.json")
        self.dags = self._load_json(self.dag_file)


    def _load_model(self, filename):
        """Safely loads a .pkl model file."""
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                logger.error(f"Failed to load AI Model {filename}: {e}")
                return None
        return None

    def _load_json(self, path):
        """Safely loads a .json file."""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load JSON {path}: {e}")
                return {}
        return []

    def get_ai_probability(self, df, regime):
        """
        Asks the specialized AI model: "What is the probability of profit?"
        Returns a score from 0 to 100.
        """
        # Select the correct Brain and Eyes based on the Regime
        model = self.trend_model if regime == "TREND" else self.chop_model
        features = self.trend_features if regime == "TREND" else self.chop_features
        
        if model is None:
            return 50.0 # Neutral score if AI is missing
            
        try:
            # --- STORY: The Blindfold Check ---
            # We must filter the dataframe to show the AI *only* the columns it was trained on.
            # If we show the Trend AI a column it doesn't recognize (like 'rsi' if it was masked), it will crash.
            if not features:
                return 50.0
                
            # Double check that we actually have the data needed
            missing_cols = [c for c in features if c not in df.columns]
            if missing_cols:
                logger.warning(f"AI Input Mismatch. Missing cols: {missing_cols}")
                return 50.0

            # Ask the Brain for a prediction on the very last row of data
            X = df[features].iloc[[-1]]

            # Safe prediction: supports both Classifier (predict_proba) and Regressor (predict)
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[0][1]  # Probability of Class 1 (Profit)
                return round(prob * 100.0, 1)
            else:
                # Regressor fallback: clamp output to 0-100 range
                raw = model.predict(X)[0]
                return round(float(np.clip(raw * 100.0, 0.0, 100.0)), 1)
            
        except Exception as e:
            logger.debug(f"AI Prediction failed for {regime}: {e}")
            return 50.0

    def evaluate_friction_adjusted_alpha(self, price, stop_loss, target):
        """
        The Hurdle Rate Check.
        This function simulates the trade including all taxes and fees to see if it's actually worth it.
        """
        costs = cfg.COSTS_CONFIG
        alpha = cfg.FRICTION_AND_ALPHA
        
        # We use a placeholder quantity of 100 shares just to do the math.
        qty = 100 
        
        # 1. Calculate Gross Outcomes (Paper Money)
        gross_profit = (target - price) * qty
        gross_loss = (price - stop_loss) * qty
        
        # 2. Calculate Friction (The Cost of Business)
        # We pay slippage on entry, commission on entry, and commission on exit.
        friction = (price * costs["slippage_pct"] * qty) + (costs["commission_per_share"] * qty * 2)
        
        # 3. Calculate Net Outcomes (Real Money)
        net_profit = gross_profit - friction
        # The government only takes taxes if we win.
        net_profit_after_tax = net_profit * (1 - costs["tax_rate"])
        
        # If we lose, we pay the gross loss PLUS the friction fees.
        net_loss = gross_loss + friction 
        
        # 4. Calculate the Ratios
        net_profit_pct = net_profit_after_tax / (price * qty)
        net_rr = net_profit_after_tax / net_loss if net_loss > 0 else 0
        
        # 5. The Veto Logic
        # If the trade makes less than 1.5% profit after everything, we walk away.
        if net_profit_pct < alpha["min_net_profit_pct"]:
            logger.debug(f"Alpha VETO: Net Profit {net_profit_pct:.2%} < Threshold {alpha['min_net_profit_pct']:.2%}")
            return False
            
        # If the reward isn't at least 1.5x the risk, we walk away.
        if net_rr < alpha["min_net_rr"]:
            logger.debug(f"Alpha VETO: Net R:R {net_rr:.2f} < Threshold {alpha['min_net_rr']}")
            return False
            
        return True

    def analyze(self, symbol, df, regime):
        """
        The Tactical Sniper (Agent 2) - DYNAMIC SETUP MODE.
        Instead of generic indicators, this engine scans for specific 
        High-Probability Setups based on the 63+ available features.
        
        Logic:
        1. Scan for specific setups (DSP Trend, Volatility Squeeze, VSA, Candle Patterns).
        2. Accumulate 'Technical Weight' based on detected setups.
        3. Use AI Score only as a validator/multiplier, not the primary driver.
        """
        # Get the latest data point (Current Candle)
        last = df.iloc[-1]
        price = last['close']
        atr = last.get('atr', price * 0.01)

        # --- DETAILED LOGGING: Indicator Snapshot ---
        _rsi_raw = last.get('rsi', None)
        _macd_raw = last.get('macd', None)
        _bbw_raw = last.get('bb_width', None)
        _rvol_raw = last.get('rvol', None)
        logger.debug(f"[{symbol}] INDICATORS: "
                     f"RSI={f'{_rsi_raw:.1f}' if _rsi_raw is not None else 'N/A'} | "
                     f"MACD={f'{_macd_raw:.4f}' if _macd_raw is not None else 'N/A'} | "
                     f"ATR={atr:.2f} | "
                     f"ER_slow={last.get('er_slow', 0):.2f} | "
                     f"ER_fast={last.get('er_fast', 0):.2f} | "
                     f"BB_width={f'{_bbw_raw:.3f}' if _bbw_raw is not None else 'N/A'} | "
                     f"RVOL={f'{_rvol_raw:.2f}' if _rvol_raw is not None else 'N/A'} | "
                     f"Volume={last.get('volume', 0):.0f}")

        # Initialize Setup Tracking
        active_setups = []
        technical_weight = 0.0
        
        # --- SETUP 1: DSP SUPER TREND (The King of Trends) ---
        # Checks if the Digital Signal Processing (DSP) Engine confirms a noise-free trend.
        # Requires both the Slow Wave (Trend) and Fast Wave (Cycle) to align.
        er_slow_val = last.get('er_slow', 0)
        er_threshold = getattr(cfg, 'DSP_CONFIG', {}).get('threshold_coherent_trend', 0.55)
        if er_slow_val >= er_threshold and last.get('trend_alignment', 0) == 1:
            active_setups.append("DSP_SUPER_TREND")
            technical_weight += 35  # Very strong signal
            logger.debug(f"[{symbol}] SETUP_FOUND: DSP_SUPER_TREND | weight=35 | er_slow={er_slow_val:.2f} | trend_align=1")
            
        # --- SETUP 2: VOLATILITY SQUEEZE (TTM Squeeze Logic) ---
        # Checks if Bollinger Bands are inside Keltner Channels (Energy building up).
        # We look for narrow BB Width (< 0.15) and squeeze flag.
        if last.get('bb_width', 1.0) < 0.15:
            # Check for expansion out of squeeze
            if last.get('squeeze_on', 0) == 1:
                active_setups.append("VOLATILITY_SQUEEZE_PREP")
                technical_weight += 20
                logger.debug(f"[{symbol}] SETUP_FOUND: VOLATILITY_SQUEEZE_PREP | weight=20 | bb_width={last.get('bb_width', 0):.3f}")
            elif last.get('mom_sqz', 0) > 0: # Momentum firing up
                active_setups.append("SQUEEZE_FIRING_LONG")
                technical_weight += 30
                logger.debug(f"[{symbol}] SETUP_FOUND: SQUEEZE_FIRING_LONG | weight=30 | bb_width={last.get('bb_width', 0):.3f} | mom_sqz={last.get('mom_sqz', 0)}")

        # --- SETUP 3: VSA SMART MONEY (Volume Spread Analysis) ---
        # Detects Institutional Buying: High Volume + Price Increase.
        # We compare current volume to the 20-day average.
        vol_avg = last.get('vol_avg_20', 1.0)
        curr_vol = last.get('volume', 0)
        
        if curr_vol > (vol_avg * 1.5) and last['close'] > last['open']:
            active_setups.append("VSA_INSTITUTIONAL_BUYING")
            technical_weight += 25
            logger.debug(f"[{symbol}] SETUP_FOUND: VSA_INSTITUTIONAL_BUYING | weight=25 | vol={curr_vol:.0f} | vol_avg={vol_avg:.0f} | ratio={curr_vol/vol_avg:.1f}x")
            
        # --- SETUP 4: CANDLESTICK PATTERNS (Price Action) ---
        # Scans all CDL_ columns generated by the Feature Engine (Ta-Lib / Pandas-TA).
        candle_score = 0
        for col in last.index:
            if str(col).startswith('CDL_') and last[col] != 0:
                pattern_name = col.replace('CDL_', '')
                if last[col] > 0: # Bullish Pattern
                    active_setups.append(f"CANDLE_{pattern_name}")
                    candle_score += 15
                    logger.debug(f"[{symbol}] SETUP_FOUND: CANDLE_{pattern_name} | weight=15 | value={last[col]}")
        
        # Cap candle score impact (we don't want 5 weak candles to override a trend)
        technical_weight += min(candle_score, 25)
        
        # --- SETUP 5: MOMENTUM BREAKOUT (The RSI sweet spot) ---
        # Fixes the "Mid-Range Blind Spot". Recognizes RSI 50-75 as bullish momentum.
        rsi = last.get('rsi', 50.0)
        macd = last.get('macd', 0)
        signal = last.get('macd_signal', 0)
        
        if 50 < rsi < 75 and macd > signal:
            active_setups.append("MOMENTUM_BREAKOUT")
            technical_weight += 20
            logger.debug(f"[{symbol}] SETUP_FOUND: MOMENTUM_BREAKOUT | weight=20 | rsi={rsi:.1f} | macd={macd:.4f} | signal={signal:.4f}")

        # --- SETUP 6: DIP BUY (Oversold Bounce) ---
        # Only valid in CHOP regime. Buying when blood is in the streets.
        if regime == "CHOP" and rsi < 30:
            active_setups.append("OVERSOLD_BOUNCE")
            technical_weight += 30
            logger.debug(f"[{symbol}] SETUP_FOUND: OVERSOLD_BOUNCE | weight=30 | rsi={rsi:.1f} | regime={regime}")

        # --- SCORE AGGREGATION ---
        # Cap technical score at 100
        tech_score = min(technical_weight, 100.0)
        logger.debug(f"[{symbol}] TECH_SCORE: {tech_score:.1f} | raw_weight={technical_weight:.1f} | setups_count={len(active_setups)}")

        # --- AI CONSENSUS (Secondary Validator) ---
        # The AI score is now used to Confirm or Deny the Technical Setup.
        ai_prob = self.get_ai_probability(df, regime)
        logger.debug(f"[{symbol}] AI_SCORE: {ai_prob:.1f} | regime={regime} | model={'Trend' if regime == 'TREND' else 'Chop'}")
        
        master_score = 0.0
        
        # LOGIC: Technicals LEAD, AI Follows.
        if tech_score >= 50:
            # Strong Technical Setup Found.
            if ai_prob >= 40:
                # AI confirms (or is at least neutral). Boost the score.
                # Formula: 70% Tech / 30% AI
                master_score = (tech_score * 0.7) + (ai_prob * 0.3)
            else:
                # AI strongly disagrees (< 40). Penalty applied.
                master_score = (tech_score * 0.5) + (ai_prob * 0.5)
        else:
            # Weak Technical Setup. Even if AI is high, we are cautious.
            master_score = (tech_score * 0.4) + (ai_prob * 0.6)

        logger.debug(f"[{symbol}] MASTER_SCORE: {master_score:.1f} | formula={'70T/30A' if tech_score >= 50 and ai_prob >= 40 else '50T/50A' if tech_score >= 50 else '40T/60A'}")

        # Log the detailed findings for the user
        setups_str = ", ".join(active_setups) if active_setups else "None"
        logger.debug(f"[{symbol}] SETUPS: [{setups_str}] | Tech: {tech_score} | AI: {ai_prob:.1f} | Master: {master_score:.1f}")

        # --- RISK MANAGEMENT (Stop Loss & Targets) ---
        # Architectural Fix: Prevent 0 ATR from causing immediate stop-outs
        if atr == 0 or pd.isna(atr):
            atr = price * 0.02
            
        # Calculate dynamic stops based on ATR (Volatility)
        if regime == "TREND":
            stop_loss = price - (atr * 2.0) # Looser stop for trends
            target = price + (atr * 4.0) # 1:2 Risk/Reward
        else:
            stop_loss = price - (atr * 1.5) # Tighter stop for chop
            target = price + (atr * 2.5)

        _risk = price - stop_loss
        _reward = target - price
        logger.debug(f"[{symbol}] RISK_CALC: price={price:.2f} | stop={stop_loss:.2f} | target={target:.2f} | risk={_risk:.2f} | reward={_reward:.2f} | RR={_reward/_risk:.2f}" if _risk > 0 else f"[{symbol}] RISK_CALC: price={price:.2f} | stop={stop_loss:.2f} | target={target:.2f} | risk=0 | RR=N/A")

        # --- AGENT 3: FRICTION CHECK ---
        # Ensures the trade has positive expectancy after fees
        is_viable = self.evaluate_friction_adjusted_alpha(price, stop_loss, target)
        
        if not is_viable:
            logger.debug(f"[{symbol}] FRICTION_FAIL: price={price:.2f} | stop={stop_loss:.2f} | target={target:.2f}")
            master_score = 0
            logger.debug(f"[{symbol}] Trade killed by Friction/Alpha Check.")
            
        # --- RETURN DECISION TICKET ---
        action = "BUY" if master_score > 60 else "WAIT"
        logger.debug(f"[{symbol}] DECISION: {action} | master={master_score:.1f} | threshold=60")
        return {
            "action": action,
            "master_score": master_score,
            "ai_score": ai_prob,
            "tech_score": tech_score,
            "setups_found": active_setups, # Passing setups to Live Engine for logging
            "stop_loss": stop_loss,
            "target_price": target
        }

class RiskActuary:
    """
    AGENT 3: The Accountant.
    Responsibility: Position Sizing and Capital Preservation.
    Enforces the '1% Volumetric Cap' so we don't become the whale that splashes the water.
    """
    def __init__(self):
        # We only risk half of our daily loss limit on a single trade.
        self.risk_per_trade_pct = cfg.RISK_CONFIG["max_daily_loss_pct"] / 2 
        self.equity = cfg.RISK_CONFIG["starting_capital"]
        self.vol_limit_pct = cfg.VOLUMETRIC_LIMITS["max_adv_participation_pct"]
        
    def calculate_size(self, price, stop_loss, volume_avg):
        # 1. Risk-Based Sizing (How much can I afford to lose?)
        risk_per_share = price - stop_loss
        if risk_per_share <= 0: return 0
        
        # Example: $25,000 equity * 0.75% risk = $187.50 max allowable loss
        risk_capital = self.equity * self.risk_per_trade_pct 
        shares_risk = risk_capital / risk_per_share
        
        # 2. Volumetric Cap (How much can the market handle?)
        # We never want to buy more than 1% of the daily volume, or we will cause slippage.
        max_vol_shares = volume_avg * self.vol_limit_pct
        
        # 3. Final Decision
        # We take the smaller of the two numbers to be safe.
        final_shares = int(min(shares_risk, max_vol_shares))
        
        logger.debug(f"Sizing: Risk-Based={int(shares_risk)} | Vol-Based={int(max_vol_shares)} | Final={final_shares}")
        return final_shares

class StrategyEngine:
    """
    Gen-13 Strategy Engine (Dual-Core).
    1. Technical Score (85+ Models)
    2. AI Score (Orchestra)
    3. Alpha Entry Equation (Strict 0.5% Net Profit Check)
    4. Cooldown Wash-Trade Prevention
    """
    def __init__(self, db_manager=None):
        """
        Gen-13 Strategy Engine (Dual-Core).
        """
        # 1. Initialize the DSP Gatekeeper
        self.router = RegimeRouter()
        
        # 2. Initialize the AI & Technical Evaluator
        self.sniper = TacticalSniper()
        
        # 3. Core Engine Components
        self.features = FeatureEngine()
        self.db = db_manager
        self.cooldown_file = getattr(cfg, 'COOLDOWN_FILE_PATH', 'data/cooldown_list.json')

    def apply_checklist_bonus(self, row, raw_tech_score):
        """
        Gen-13: Technical Score Calibration via Integration Checklist
        Adds a mathematical bonus to the raw technical score based on microstructure.
        """
        bonus = 0.0
        
        # 1. Institutional Support (VSA): Relative Volume surge
        rvol = float(row.get('rvol', 1.0) if pd.notna(row.get('rvol')) else 1.0)
        if rvol > 1.3:
            bonus += 10.0
            
        # 2. Trend Alignment
        close = float(row.get('close', 0.0))
        sma_50 = float(row.get('sma_50', 0.0))
        sma_200 = float(row.get('sma_200', 0.0))
        if close > sma_50 > sma_200 > 0:
            bonus += 10.0
            
        # 3. Market Regime (DSP): Coherent trend verification
        er_slow = float(row.get('er_slow', 0.0) if pd.notna(row.get('er_slow')) else 0.0)
        if er_slow > 0.40:
            bonus += 8.0
            
        # 4. Momentum Potential: Ideal RSI band (not overbought, not dead)
        rsi_14 = float(row.get('rsi', 50.0) if pd.notna(row.get('rsi')) else 50.0)
        if 40 <= rsi_14 <= 65:
            bonus += 7.0
            
        # 5. Distance from Resistance: Room to grow to the upper Bollinger Band
        bb_upper = float(row.get('bb_upper', 0.0))
        if close > 0 and bb_upper > 0 and ((bb_upper - close) / close) > 0.03:
            bonus += 5.0
            
        # 6. Volatility Contraction (Squeeze): Historic low width
        is_squeeze = row.get('squeeze_on', 0) == 1
        if is_squeeze:
            bonus += 10.0
            
        new_score = raw_tech_score + bonus
        return min(new_score, 100.0)

    def _track_missed_opportunity(self, verdict):
        """
        Gen-13: Forward Testing Ledger.
        Logs premium setups (High Score) that were killed by the Friction/Alpha equation.
        """
        import os, json
        from datetime import datetime
        import system_config as cfg
        
        try:
            path = getattr(cfg, 'MISSED_OPPORTUNITIES_PATH', os.path.join(cfg.DATA_DIR, "missed_opportunities.json"))
            data = []
            if os.path.exists(path):
                with open(path, 'r') as f:
                    try:
                        data = json.load(f)
                    except:
                        data = []
                        
            record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": verdict.get('symbol', 'UNKNOWN'),
                "master_score": verdict.get('master_score', 0),
                "reason": verdict.get('reason', 'Unknown Veto'),
                "setups": verdict.get('setups_found', [])
            }
            data.append(record)
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
                
            import logging
            logging.getLogger("StrategyEngine").info(f"[{record['symbol']}] Premium Trade Vetoed -> Logged to Missed Opportunities Ledger (Score: {record['master_score']:.1f})")
        except Exception as e:
            pass

    def evaluate_ticker(self, symbol, df_features):
        """
        Gen-13 Facade Pattern: Bridges legacy calls to the Dual-Core Agentic Architecture, 
        builds execution tickets, and calculates Checklist Bonuses & Missed Opportunities.
        """
        try:
            import system_config as cfg
            
            # 0. Feature Calculation
            strategy_config = self.load_strategy_for_ticker(symbol)
            df = self.features.calculate_features(df_features, strategy_config)

            # 1. Identify Regime (Trend vs Chop)
            regime = self.router.classify_regime(df)

            # [Bug 2.3 Fix] Block analysis for HALT and NEUTRAL regimes
            if regime == "HALT":
                logger.debug(f"[{symbol}] Regime HALT -- velocity divergence detected. Skipping analysis.")
                return {"symbol": symbol, "action": "WAIT", "master_score": 0,
                        "reason": "Regime HALT: Velocity Divergence (er_slow/er_fast conflict)"}

            if regime == "NEUTRAL":
                logger.debug(f"[{symbol}] Regime NEUTRAL -- dead zone. Skipping analysis.")
                return {"symbol": symbol, "action": "WAIT", "master_score": 0,
                        "reason": "Regime NEUTRAL: No clear trend or chop signal"}

            # 2. RUN FULL STRATEGY (Agent 2 - The Sniper)
            verdict = self.sniper.analyze(symbol, df, regime)
            
            # Inject the symbol into the payload for live engine compatibility
            verdict['symbol'] = symbol
            
            # --- GEN-13: Apply Technical Checklist Bonus ---
            last_row = df.iloc[-1]
            raw_tech = verdict.get('tech_score', 50.0)
            new_tech = self.apply_checklist_bonus(last_row, raw_tech)
            verdict['tech_score'] = new_tech
            
            # Recalculate Master Score with the inflated technical score
            ai_score = verdict.get('ai_score', 50.0)
            new_master = (new_tech + ai_score) / 2.0
            verdict['master_score'] = new_master
            verdict['scores'] = {'master': new_master, 'tech': new_tech, 'ai': ai_score}
            
            # --- GEN-13: Threshold Calibration & Missed Opportunities ---
            min_approval = getattr(cfg, 'MIN_MASTER_SCORE_APPROVAL', 80.0)
            premium_threshold = getattr(cfg, 'PREMIUM_TRADE_THRESHOLD', 75.0)
            
            if verdict.get('action') == 'WAIT':
                # If a premium trade was vetoed strictly by the friction/alpha logic
                reason = verdict.get('reason', '')
                if new_master >= premium_threshold and ('Friction' in reason or 'Alpha' in reason or 'R:R' in reason):
                    self._track_missed_opportunity(verdict)
            elif verdict.get('action') == 'BUY':
                # Enforce the new calibrated master score threshold
                if new_master < min_approval:
                    verdict['action'] = 'WAIT'
                    verdict['reason'] = f'Score {new_master:.1f} < Calibrated Threshold {min_approval}'
            
            # 3. Execution Ticket Builder (Inject missing limits for live_trading_engine)
            if verdict.get('action') == 'BUY':
                last_row = df.iloc[-1]
                current_price = float(last_row['close'])
                
                # Dynamically calculate volatility-based boundaries (ATR)
                atr = float(last_row.get('atr', current_price * 0.02))
                
                # Architectural Fix: Prevent ATR from being 0 due to NaN filling
                if atr == 0 or pd.isna(atr):
                    atr = current_price * 0.02
                
                if 'limit_price' not in verdict:
                    verdict['limit_price'] = current_price
                if 'stop_loss' not in verdict:
                    verdict['stop_loss'] = current_price - (2.5 * atr) # Gen-13 Rule: 2.5 ATR Breathing Room
                if 'take_profit' not in verdict:
                    verdict['take_profit'] = current_price + (3.0 * atr)
                if 'qty' not in verdict:
                    verdict['qty'] = 10 # Default fallback quantity for paper trading
            
            return verdict
        except Exception as e:
            import logging
            logging.getLogger("StrategyEngine").error(f"Facade evaluation failed for {symbol}: {e}")
            return {"symbol": symbol, "action": "WAIT", "master_score": 0, "reason": str(e)}


    def _is_in_cooldown(self, ticker):
        """
        [Cooldown Gate] Checks if a ticker hit a Stop-Loss recently and is currently blacklisted.
        Returns True if the asset is restricted, forcing the scanner to ignore it.
        """
        try:
            if not os.path.exists(self.cooldown_file):
                return False
                
            with open(self.cooldown_file, 'r', encoding='utf-8') as f:
                cooldown_data = json.load(f)
                
            if ticker in cooldown_data:
                timestamp = cooldown_data[ticker].get("timestamp", 0)
                cooldown_period = getattr(cfg, 'COOLDOWN_PERIOD_HOURS', 24) * 3600
                
                if (time.time() - timestamp) < cooldown_period:
                    return True
        except Exception as e:
            logger.error(f"Failed to evaluate cooldown state for {ticker}: {str(e)}")
            
        return False

    def load_strategy_for_ticker(self, ticker):
        """[Strategy Genome] Loads specific config from JSON."""
        try:
            strategy_file = getattr(cfg, 'STRATEGY_MAP_FILE', 'ticker_strategies.json')
            if os.path.exists(strategy_file):
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    strategies = json.load(f)
                return strategies.get(ticker, strategies.get("DEFAULT"))
        except Exception as e:
            logger.warning(f"Failed to load strategy map: {str(e)}")
        return None

    def calculate_entry_equation(self, ticker, final_score, atr_pct):
        """
        [Alpha Entry Equation]
        Calculates expected probabilistic price movement against market volatility.
        Strict Condition: (Expected_Rise - Total_Friction) >= 0.5% Net Profit.
        """
        # Load risk parameters dynamically from system_config
        total_friction = getattr(cfg, 'BASE_FRICTION', 0.003) 
        min_net_profit = getattr(cfg, 'MIN_NET_PROFIT', 0.005)
        
        # Base statistical expectancy calculation
        expected_rise = (final_score / 100.0) * atr_pct
        
        # The ultimate boolean gate: must exceed threshold strictly mathematically
        is_profitable = (expected_rise - total_friction) >= min_net_profit
        
        return is_profitable, expected_rise, total_friction

    async def decide_action(self, ticker, dataframe, news_context=None):
        """
        Evaluates the asset based on Fail-Fast architecture, implementing strict survival rules,
        dynamic thresholds, and separate logging streams (INFO vs DEBUG).
        """
        # --- 0. THE COOLDOWN GATE (Wash Trade Prevention) ---
        if self._is_in_cooldown(ticker):
            # INFO level: User needs to know a trade was killed by the system explicitly
            logger.info(f"[{ticker}] VETO: Trade aborted by Cooldown Protocol (Recent Stop-Loss hit).")
            return "WAIT", 0

        # --- 1. DATA GUARD (Fail Fast) ---
        MIN_ROWS = 200 
        if dataframe.empty or len(dataframe) < MIN_ROWS:
            logger.error(f"[{ticker}] INSUFFICIENT DATA: {len(dataframe)} rows detected. Veto executed.")
            return "WAIT", 0

        # --- 2. LOAD STRATEGY GENOME ---
        strategy_config = self.load_strategy_for_ticker(ticker)
        
        # --- 3. FEATURE CALCULATION ---
        df = self.features.calculate_features(dataframe, strategy_config)
        row = df.iloc[-1]

        # Extract current price for precise logging and expectancy calculations
        close_price = float(row.get('close', 0.0))

        # --- 4. STRUCTURAL VETO GATES ---
        if row.get('volume', 0) < 1: 
            return "WAIT", 0

        if row.get('death_cross'):
            logger.info(f"[{ticker}] VETO: Structural Death Cross detected | Price: ${close_price:.2f}")
            return "WAIT", 0

        if row.get('vsa_squat_bar'):
            logger.info(f"[{ticker}] VETO: Institutional Manipulation Trap (Squat Bar) | Price: ${close_price:.2f}")
            return "WAIT", 0

        # --- 5. DUAL-CORE SCORING ---
        tech_score, patterns = self.features.calculate_technical_score(df, strategy_config)
        ai_score = await self.orchestra.get_ai_score(ticker, row)
        
        # The Master Score calculation will be replaced dynamically in Phase B
        final_score = (tech_score + ai_score) / 2

        # Logging separation architecture
        logger.info(f"[{ticker}] Scan Complete | Price: ${close_price:.2f} | Master Score: {final_score:.1f}")
        logger.debug(f"[{ticker}] Forensic Math | Tech: {tech_score:.1f} | AI: {ai_score:.1f} | SETUPS: {patterns}")
        
        # --- 6. THE ENTRY EQUATION (Net Profit Hurdle) ---
        atr_value = row.get('atr', 0)
        atr_pct = (atr_value / close_price) if close_price > 0 else 0
        
        is_profitable, expected_rise, friction = self.calculate_entry_equation(ticker, final_score, atr_pct)
        
        if not is_profitable:
            logger.debug(f"[{ticker}] Alpha VETO: Expected Return ({expected_rise:.2%}) minus Friction ({friction:.2%}) fails the 0.5% Net Profit test.")
            return "WAIT", final_score

        # --- 7. EXECUTION DECISION ---
        buy_threshold = strategy_config.get("buy_threshold", 75) if strategy_config else 75
        
        if row.get('trend_alignment', 0) == 1.0:
            buy_threshold -= 5 

        if final_score >= buy_threshold:
            logger.info(f"[{ticker}] BUY SIGNAL TRIGGERED | Price: ${close_price:.2f} | Exp.Rise: {expected_rise:.2%} (Passes Alpha Equation)")
            return "BUY", final_score
            
        return "WAIT", final_score

# Compatibility Alias
StrategyOrchestra = StrategyEngine
