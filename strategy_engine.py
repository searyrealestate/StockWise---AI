# # strategy_engine.py

# import logging
# import numpy as np
# import system_config as cfg

# # Logger setup
# logger = logging.getLogger("StrategyEngine")


# class MarketRegimeDetector:
#     @staticmethod
#     def detect_regime(latest_row):
#         close = latest_row['close']
#         sma_short = latest_row.get('sma_short', close)
#         sma_long = latest_row.get('sma_long', close)
#         adx = latest_row.get('adx', latest_row.get('ADX', latest_row.get('ADX_14', latest_row.get('adx_14', 0))))

#         if close < sma_long:
#             return "BEARISH"

#         # 2. RANGING_SHUFFLE (Consolidation)
#         # if sma_short > 0:
#         #     price_near_sma_short = abs(close - sma_short) / sma_short < 0.01
#         # else:
#         #     price_near_sma_short = False
#         #
#         # if adx < cfg.STRATEGY_PARAMS.get('adx_threshold', 25) and price_near_sma_short:
#         #     return "RANGING_SHUFFLE"  # <-- NEW SHUFFLE REGIME

#         # 3. TRENDING_UP (Strong Bull)
#         if close > sma_long:
#             if adx >= cfg.STRATEGY_PARAMS.get('adx_threshold', 25) and close > sma_short:
#                 return "TRENDING_UP"

#         # 4. RANGING_BULL (Default / Mild Uptrend)
#         return "RANGING_BULL"


# class StrategyOrchestra:
#     """
#         Gen-7 Strategy Orchestra - 'PREDATOR' Edition
#         Optimized for High Win Rate & Dip Buying													
#     """

#     fundamentals = None # Gen-8 Global State

#     @classmethod
#     def set_fundamentals(cls, data):
#         cls.fundamentals = data

#     @staticmethod
#     def decide_action(ticker, row, analysis):
#         """
#         Sniper Decision Engine (Predator Edition)
#         Updates:
#         1. Reduced AI dependence (60% -> 40%) to allow Technicals to shine.
#         2. Added 'Dip Buy' logic for Oversold conditions.
#         3. Dynamic Thresholds: Strong Technicals lowers AI requirement.
#         """

#         # --- 1. ROBUST DATA EXTRACTION (The Fix) ---
#         def get_val(row, keys, default=0.0):
#             for k in keys:
#                 if k in row: return float(row[k])
#             return default
        
#         # # --- 1. EXTRACT RAW DATA ---
#         # ai_prob = analysis.get('AI_Probability', 0.0) 
#         # fund_score = analysis.get('Fundamental_Score', 50)
        
#         current_rsi = get_val(row, ['rsi_14', 'RSI', 'rsi', 'RSI_14'], 50.0)
#         current_adx = get_val(row, ['adx_14', 'ADX_14', 'ADX', 'adx'], 0.0)
#         current_vol = get_val(row, ['volume', 'Volume', 'VOL'], 0.0)
#         vol_ma = get_val(row, ['vol_ma', 'Vol_MA', 'VOL_MA'], current_vol)

#         ai_prob = analysis.get('AI_Probability', 0.0) 
#         fund_score = analysis.get('Fundamental_Score', 50)
#         price = row.get('close', 0)
#         ema_200 = row.get('EMA_200', 0)

#         # --- 2. CALCULATE TECHNICAL SCORE ---
#         tech_raw_score = 0
#         tech_log = []
        
#         # A. Trend Strength (ADX) OR Reversal Strength
#         if current_adx > 25: 
#             tech_raw_score += 20
#             tech_log.append("StrongTrend (+20)")

#             # ** NEW: GOLDEN ZONE BONUS **
#             # This specific combo had 60.9% Win Rate in your logs
#             if 50 < current_rsi < 70:
#                 tech_raw_score += 15
#                 tech_log.append("GoldenZone (+15)")

#         # B. DIPS (The Losing Strategy - Fixed)
#         # Old logic (RSI < 40) had 20% WR. We tightened it to < 30 (Extreme only).
#         elif current_rsi < 30:
#             tech_raw_score += 25
#             tech_log.append("DeepValue (+25)")
        
#         # C. RSI HEALTH
#         if 45 < current_rsi < 65: 
#             tech_raw_score += 10
#             tech_log.append("RSI_Prime (+10)")
            
#         # D. MACRO TREND
#         if price > ema_200: 
#             tech_raw_score += 20
#             tech_log.append("MacroTrend (+20)")
            
#         # E. VOLUME
#         if current_vol > vol_ma:
#             tech_raw_score += 10
#             tech_log.append("High_Vol (+10)")

#         # Cap Technical Score
#         tech_raw_score = min(tech_raw_score, 100)

#         # --- 3. CALCULATE COMMITTEE SCORES ---
#         # Weights: AI (40%), Fund (15%), Tech (45%)
#         w_ai = 0.40
#         w_fund = 0.15
#         w_tech = 0.45
        
#         score_ai_contribution = (ai_prob * 100) * w_ai
#         score_fund_contribution = fund_score * w_fund
#         score_tech_contribution = tech_raw_score * w_tech
        
#         final_score = score_ai_contribution + score_fund_contribution + score_tech_contribution

#         # --- 4. DETERMINE VERDICT (BEFORE VETO) ---
#         # Raised Threshold to 60 to stop the "Machine Gun"
#         if final_score >= 60:
#             verdict = "BUY"
#         elif final_score <= 30:
#             verdict = "SELL"
#         else:
#             verdict = "WAIT"

#         # --- 5. DETAILED LOGGING (The Report Card) ---
#         # Log if score is decent (>40) or if it's a BUY signal
#         if final_score > 40:
#             log_msg = (
#                 f"\n📋 REPORT CARD: [{ticker}] @ {row.name if hasattr(row, 'name') else 'Now'}\n"
#                 f"--------------------------------------------------\n"
#                 f"   RSI: {current_rsi:.1f} | ADX: {current_adx:.1f}\n" # Log the ACTUAL values seen
#                 f"1. 🧠 AI BRAIN       (Prob {ai_prob:.2f}): +{score_ai_contribution:.1f}\n"
#                 f"2. 📈 TECHNICALS     (Raw {tech_raw_score}): +{score_tech_contribution:.1f}\n"
#                 f"   L Details: {', '.join(tech_log)}\n"
#                 f"--------------------------------------------------\n"
#                 f"🏆 FINAL SCORE: {final_score:.1f} / 100\n"
#                 f"⚖️ VERDICT: {verdict}\n"
#             )
#             logger.info(log_msg)

#         # --- 6. VETO GATES (DYNAMIC) ---
#         # We check vetoes last so we can see the "Report Card" first        
#         # Check Sniper Lock (Perfect Market Structure)
#         # is_sniper_lock = (current_adx > 25 and current_rsi < 70 and current_vol > vol_ma)
#         required_confidence = cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD

#         # DYNAMIC THRESHOLD: If Technicals are perfect (>80), AI can be dumber (0.45)
#         if tech_raw_score > 80:
#             required_confidence = 0.45
#             logger.info(f"⚡ OVERRIDE: Perfect Technicals! Lowering AI Threshold to {required_confidence}")
        
        
#         if verdict == "BUY":
#             # 1. HARD RSI CEILING (Stop Buying Tops)
#             if current_rsi > 70:
#                 logger.info(f"🛑 SAFETY VETO: RSI {current_rsi:.1f} is dangerously high. Abort.")
#                 return "WAIT"

#             # 2. DIRECTIONAL VETO (With Dip Exception)
#             if price < ema_21:
#                 # ALLOW if it is a Deep Dip (RSI < 30)
#                 if current_rsi < 30: 
#                     logger.info(f"⚡ FALLING KNIFE CATCH: Price < EMA21 but RSI {current_rsi:.1f} is oversold. ALLOWED.")
#                 else:
#                     logger.info(f"🛑 TREND VETO: Price {price:.2f} < EMA21 {ema_21:.2f} and not oversold.")
#                     return "WAIT"

#             # 2. TOXIC DIP PROTECTION
#             # If trend is weak (ADX < 20) and RSI is not extreme (< 30), don't buy.
#             if current_adx < 20 and current_rsi > 30:
#                 logger.info(f"🛑 CHOP VETO: Weak Trend (ADX {current_adx:.1f}) & Not Deep Value.")
#                 return "WAIT"

#             # 3. AI Veto
#             if ai_prob < required_confidence:
#                 logger.info(f"🚫 AI Veto: {ai_prob:.2f} < {required_confidence:.2f}")
#                 return "WAIT"

#             # 4. Fundamental Veto
#             if fund_score < 50:
#                 return "WAIT"

#             # 5. Context-Aware Trend/Dip Logic
#             if current_adx < 30:
#                 # WEAK TREND: Only buy DEEP Dips or High AI
#                 if current_rsi < 40:
#                      logger.info(f"⚡ VALUE ENTRY: Buying Deep Dip (RSI {current_rsi:.1f}) in weak trend.")
#                 elif ai_prob > 0.75:
#                     logger.info(f"⚡ AI SNIPE: Buying Weak Trend on High AI Confidence ({ai_prob:.2f}).")
#                 else:
#                     logger.info(f"🚫 Trend Veto: Weak Trend (ADX {current_adx:.1f}) & Not Deep Value.")
#                     return "WAIT"
#             else:
#                 # STRONG TREND: Buy Shallow Dips or Breakouts
#                 if current_rsi < 60:
#                      logger.info(f"🚀 TREND ENTRY: Buying healthy pullback/breakout (RSI {current_rsi:.1f}).")

#             # 4. Fundamental Veto
#             if fund_score < 50:
#                 return "WAIT"
                
#             # ENTRY OPTIMIZATION
#             if current_adx > 30:
#                 logger.info(f"🚀 MOMENTUM: Buy MARKET.")
#             elif current_rsi < 40:
#                 logger.info(f"⚓ DIP SNIPE: Buy MARKET.")
#             else:
#                 limit_target = price * 0.998 
#                 logger.info(f"🔫 STANDARD: Limit at {limit_target:.2f}")

#         return verdict

#     @staticmethod
#     def get_adaptive_targets(row, entry_price):
#         """
#         Orchestra Stop-Loss Engine (High Win Rate Edition).
#         Adjusts R:R to favor 'Hitting the Target' more often.
#         """

#         # Robust Fetch
#         def get_val(row, keys, default=0.0):
#             for k in keys:
#                 if k in row: return float(row[k])
#             return default

#         # 1. Get Volatility Data (ROBUST CHECK)
#         # Check all possible column names for ATR
#         atr = get_val(row, ['atr', 'ATRr_14', 'atr_14'], entry_price * 0.02)
#         natr = (atr / entry_price) * 100
        
#         # Stop Multipliers (Volatility Based)
#         if natr >= 4.0:
#             stop_mult = 3.0  # Extreme Volatility (Crypto/Biotech mode)
#             regime = "EXTREME"
#         elif natr >= 2.5:
#             stop_mult = 2.5  # High Volatility (Wild Tech: NVDA, TSLA)
#             regime = "HIGH"
#         else:
#             stop_mult = 2.0  # Normal Volatility (Standard: GOOGL, AAPL)
#             regime = "NORMAL"
        
            
#         # 4. Calculate Targets
#         # We aim for 2.0 R:R (Risk/Reward), so Target is 2x the Stop distance
#         # BUT for crazy stocks, we might cap the target or scale it differently.
#         # Here we stick to 2.0x for consistency.
#         # --- CRITICAL CHANGE FOR WIN RATE ---
#         # Old: target_mult = stop_mult * 2.0 (1:2 Risk Reward) -> ~40-50% Win Rate
#         # New: target_mult = stop_mult * 1.3 (1:1.3 Risk Reward) -> Aiming for 65%+ Win Rate												
#         target_mult = stop_mult * 1.3
        
#         stop_loss = entry_price - (atr * stop_mult)
#         target_price = entry_price + (atr * target_mult)
        
#         return stop_loss, target_price, f"{regime} (Stop {stop_mult}x / Target {target_mult:.1f}x)"

#     @staticmethod
#     def _agent_fundamentals(f, p):
#         """
#         Agent 8: Fundamental Analysis.
#         Uses static class variable 'fundamentals' set by external engine.
#         """
#         score = 0
#         details = {}
        
#         fund = StrategyOrchestra.fundamentals
#         if not fund:
#             return 0, {}
            
#         # PE Ratio Logic
#         pe = fund.get('trailingPE')
#         if pe and pe > 0:
#             if pe < 20: 
#                 score += 10
#                 # details['Fund_LowPE'] = 10
#             elif pe > 50:
#                 score -= 5 # Overvalued penalty
                
#         # # Revenue Growth
#         # rev_growth = fund.get('revenueGrowth')
#         # if rev_growth and rev_growth > 0.20: # >20% growth
#         #     score += 10
#         #     details['Fund_HighGrowth'] = 10
            
#         return score, details

#     @staticmethod
#     def _agent_machine_learning(f, p):
#         """Gen-8: ML Confirmation (Lorentzian/KNN) with Veto Power"""
#         score = 0
#         details = {}
#         ml_signal = f.get('ml_signal', 0)
        
#         if ml_signal == 1:
#             score += 25 # Confirmation
#             # details["ML_Lorentzian_Bull"] = 25
#         else:
#             score -= 50 # VETO: If ML says Down, we kill the trade.
#             # details["ML_Lorentzian_Bear_VETO"] = -50
            
#         return score, details

#     @staticmethod
#     def _agent_deep_learning(f, p):
#         """Gen-9: Deep Learning 'Sniper' Agent"""
#         score = 0
#         details = {}
        
#         # We need historical context (DataFrame) for LSTM
#         history_window = f.get('history_window')
        
#         if history_window is None or history_window.empty:
#             return 0, {}
            
#         try:
#             # Lazy Import to avoid overhead if not used
#             from stockwise_ai_core import StockWiseAI
            
#             # Singleton storage could be improved, but for now instantiate/load
#             # In production, this should be initialized once in the class __init__
#             ai = StockWiseAI() 
            
#             # Predict
#             # AI expects exactly 60 bars.
#             if len(history_window) >= 60:
#                 # Take last 60
#                 window = history_window.iloc[-60:].copy()
#                 action, confidence = ai.predict(window)
                
#                 if action == 2 and confidence > 0.95:
#                     score += 100 # INSTANT TRIGGER
#                     details["AI_SNIPER_BUY"] = 100
#                     # details["AI_Confidence"] = float(f"{confidence:.4f}")
#         except:
#             # logger.error(f"AI Agent Error: {e}")
#             pass

#         return score, details


#     @staticmethod
#     def get_score(features, regime, params=None):
#         """
#         Main dispatch method to select the correct agent based on regime.

#         Gen-7 Logic Override: Calculates potential score from all relevant agents
#         and selects the highest positive score, allowing high-confidence Dip Buyer
#         signals to override conservative regime filters (Falling Knife Paradox fix).
#         """
#         if params is None: params = cfg.STRATEGY_PARAMS

#         # score_details = {
#         #     "Trigger_Breakout": 0, "Conf_Volume": 0, "Conf_Slope": 0, "Conf_EMA_Short": 0,
#         #     "Conf_PSAR_Bull": 0, "Conf_Kalman_Positive": 0, "Conf_Wavelet_LowNoise": 0,
#         #     "Penalty_Wavelet_Noise": 0, "Conf_Sector_Strong": 0, "Penalty_Sector_Weak": 0,
#         #     "Penalty_RSI_Overbought": 0, "Feature_SlopeAngle": features.get('slope_angle', 0.0),
#         #     "Feature_KalmanSmooth": features.get('kalman_smooth', 0.0),
#         #     "Feature_WaveletNoise": features.get('wavelet_std_ratio', 99.0),
#         #     "Feature_RSI": features.get('rsi_14', 50),
#         #     "Regime": regime
#         # }
        
#         score_details = {"Regime": regime}

#         # --- Gen-7 Logic Override: Calculate potential score from all relevant agents ---
#         # We calculate everything upfront so we can compare them
#         # Simplified dispatch for robustness
#         breakout_score, breakout_details = StrategyOrchestra._agent_breakout(features, params)
#         dip_score, dip_details = StrategyOrchestra._agent_dip_buyer(features, params)
#         # bear_score, bear_details = StrategyOrchestra._agent_bear_defense(features)

#         # Default initialization
#         score = 0
#         details = {}
#         # active_agent = "None"
        
#         # --- GEN-8: AGENT CALCULATIONS ---
        
#         # ml_score, ml_details = StrategyOrchestra._agent_machine_learning(features, params)
#         # fund_score, fund_details = StrategyOrchestra._agent_fundamentals(features, params)
#         # dl_score, dl_details = StrategyOrchestra._agent_deep_learning(features, params)
        
#         # # BASE SCORE BOOST (Add to whatever agent is selected)
#         # gen8_boost = ml_score + fund_score + dl_score
#         # gen8_details = {}
#         # gen8_details.update(ml_details)
#         # gen8_details.update(fund_details)
#         # gen8_details.update(dl_details)

#         # if regime == "BEARISH":
#         #     # In Bear market, Breakouts are harder (Penalty), Dips are preferred.
#         #     # But if a breakout is HUGE (Score > 80), we take it (Turnaround play).
#         #     breakout_score -= 10
#         #     breakout_details["Penalty_Regime_Bear"] = -10

#         #     if dip_score > breakout_score:
#         #         score = dip_score
#         #         details = dip_details
#         #         active_agent = "DipBuyer_Bear"
#         #     else:
#         #         score = breakout_score
#         #         details = breakout_details
#         #         active_agent = "Breakout_CounterTrend"
        
#         # Prioritize Dip in ranges, Breakout in trends
#         # A. Prioritize Breakout if it has the highest score in Trending/Bullish regimes
#         if regime == "TRENDING_UP":
#             # In Bull market, Breakouts are preferred.
#             if breakout_score >= dip_score:
#                 score, details = breakout_score, breakout_details
#             else:
#                 score, details = dip_score, dip_details
#         else:  # RANGING / SHUFFLE
#             # In chop/bear, we ONLY want dip buys
#             score, details = dip_score, dip_details

#         # Merge the score breakdown for logging
#         score_details.update(details)

#         # --- CAP THE SCORE AT 100 ---
#         score = min(score, 100)

#         return score, score_details

#     @staticmethod
#     def _agent_breakout(f, p):
#         """
#         Gen-7 Breakout Agent (Ablation-Ready)
#         """
#         score = 0
#         details = {}

#         # --- Breakout Trigger (The Base) ---
#         if f.get('recent_high', 0) > 0 and f['high'] > f['recent_high']:
#             score += 20
#         #     score += cfg.SCORE_TRIGGER_BREAKOUT
#         #     details["Trigger_Breakout"] = cfg.SCORE_TRIGGER_BREAKOUT

#         # # --- EMA Trend Crossover ---
#         # if "ema_crossover" not in disabled_features:
#         #     if f.get('ema_9', 0) > f.get('ema_21', 0):
#         #         score += 10      # EMA crossover bonus - ENABLED for Gen-7
#         #         details["Conf_EMA_Crossover"] = 10

#         # # --- ADX Trend Strength ---
#         # if "adx" not in disabled_features:
#         #     if f.get('ADX_14', 0) > 20:
#         #         score += 5
#         #         details["Conf_ADX_Strong"] = 5

#         # # --- 8. Volume Confirmation ---
#         # if "volume" not in disabled_features:
#         #     if f['volume'] > f.get('vol_ma', 0) * 1.1:
#         #         score += cfg.SCORE_CONFIRM_VOLUME
#         #         details["Conf_Volume"] = cfg.SCORE_CONFIRM_VOLUME

#         return score, details

#     @staticmethod
#     def _agent_dip_buyer(f, p):
#         """
#         Gen-7 Dip Buyer
#         Updates: Added 'Falling Knife' filter (200 EMA).
#         """
#         # # Get disabled features list
#         # disabled_features = getattr(cfg, 'DISABLED_FEATURES', [])

#         score = 0
#         details = {}
        
#         # wt1 = f.get('wt1', 0)
#         # wt2 = f.get('wt2', 0)

#         # # 1. Rising Knife Filter (Wait, Falling Knife) with Gen-7 OVERRIDE
#         # long_trend = f.get('sma_200', f.get('sma_long', 0))
        
#         # # Calculate Signals FIRST
#         # # Dip Trigger
#         # wt_score = 0
#         # is_deep_value = False
        
#         # # Gen-7 Spec: Trigger if WT1 < -60 (Deep Oversold) AND Crossover
#         # if wt1 < -60 and wt1 > wt2:
#         #     wt_score = 45  # High Trigger Bonus (The override logic)
#         #     # is_deep_value = True
#         # elif wt1 < -53 and wt1 > wt2:
#         #      wt_score = 30 # Normal trigger
#         # elif f.get('rsi_14', 50) < 35:
#         #     # Fallback Trigger (Weak)
#         #     wt_score = 25

#         # # # Now Apply Filter with Override Logic
#         # # if long_trend > 0 and f['close'] < long_trend * 0.85:
#         # #     if is_deep_value:
#         # #         # OVERRIDE: Allowed because it's a "Deep Value" play
#         # #         details["Info_Filter_Override"] = "Deep_Value_Catch"
#         # #     else:
#         # #         # Block trade
#         # #         # print(f"DEBUG: DipBuyer Filtered. Close: {f['close']} < LongTrend: {long_trend}") # DEBUG
#         # #         return 0, {"Status": "Filter_Bear_Crash"}

#         # score += wt_score
#         # details["Trigger_Dip"] = wt_score

#         # # # 2. Support Confirmation: Price near Keltner Channel Lower Band [cite: 174]
#         # # kc_score = 0
#         # # kc_lower = f.get('kc_lower', 0)
#         # # # Logic: If price is dipping into or below the Lower Band
#         # # if 0 < kc_lower and f['close'] <= kc_lower * 1.01: 
#         # #     kc_score = 15
#         # #     details["Conf_KC_Lower"] = 15
#         # # score += kc_score

#         # # 3. Trend Support Bonus (+10)
#         # # If we are dipping BUT still above the 200 EMA, it's a "Bullish Pullback".
#         # if "trend_support" not in disabled_features:
#         #     if f['close'] > long_trend:
#         #         score += 10
#         #         details["Conf_Trend_Support"] = 10

#         def get_val(row, keys, default=0.0):
#             for k in keys:
#                 if k in row: return float(row[k])
#             return default
            
#         rsi = get_val(f, ['rsi_14', 'RSI', 'rsi'], 50)
        
#         if rsi < 30:
#             score += 40

#         return score, details

#     @staticmethod
#     def _agent_bear_defense(f):
#         score = 0
#         details = {}

#         # Existing RSI-based defense
#         if f.get('rsi14', 50) < 20:
#             score += 25
#             details['DefenseRSIOversold'] = 25

#         return score, details


# strategy_engine.py

# strategy_engine.py

# --- IMPORTS ---
import logging  # For logging system events and decisions
import numpy as np  # For numerical operations and array handling
import system_config as cfg  # Import global system configuration settings

# --- LOGGER SETUP ---
# Create a logger specific to this module for easier debugging
logger = logging.getLogger("StrategyEngine")

class RegimeConfig:
    """
    Adaptive Parameter Store.
    This class acts as a database of rules. It returns different trading parameters 
    (Risk, Targets, Thresholds) depending on which 'Regime' (Bull, Bear, Chop) the market is in.
    """
    @staticmethod
    def get_params(regime):
        # 1. Define the DEFAULT / NEUTRAL parameters (The baseline)
        params = {
            'adx_threshold': 25,       # Minimum ADX to consider a trend 'strong'
            'rsi_buy_min': 30,         # Minimum RSI to consider 'oversold'
            'rsi_buy_max': 70,         # Maximum RSI to allow a buy (prevent buying tops)
            'stop_loss_mult': 2.0,     # Stop Loss distance (Multiplier of ATR)
            'profit_target_mult': 1.5, # Take Profit distance (Multiplier of Stop Loss)
            'description': "Neutral"   # Label for logging
        }

        # 2. Adjust for TRENDING BULL Market (Aggressive Buying)
        if regime == "TRENDING_BULL":
            params.update({
                'adx_threshold': 20,       # Lower barrier: We want to enter trends early
                'rsi_buy_min': 40,         # Buy shallow dips (don't wait for <30, it won't happen)
                'rsi_buy_max': 75,         # Allow buying even when slightly overbought (momentum)
                'stop_loss_mult': 2.5,     # Widen stop loss to avoid getting shaken out by noise
                'profit_target_mult': 2.0, # Aim for bigger wins (Home Runs)
                'description': "TREND HUNTER"
            })
        
        # 3. Adjust for TRENDING BEAR Market (Defensive Buying)
        elif regime == "TRENDING_BEAR":
            params.update({
                'adx_threshold': 30,       # Require VERY strong trend to even consider acting
                'rsi_buy_min': 20,         # Only buy extreme crashes (Capitulation)
                'rsi_buy_max': 35,         # Never buy bounces/rallies
                'stop_loss_mult': 1.5,     # Tight leash: If it drops more, get out immediately
                'profit_target_mult': 1.0, # Quick scalps only (Base Hits)
                'description': "BEAR SNIPER"
            })
            
        # 4. Adjust for CHOPPY / RANGE Market (Oscillator Mode)
        elif regime == "CHOPPY_RANGE":
            params.update({
                'adx_threshold': 999,      # Ignore Trend Strength (ADX is useless in chop)
                'rsi_buy_min': 25,         # Buy support levels (low RSI)
                'rsi_buy_max': 45,         # Don't buy mid-range
                'stop_loss_mult': 2.0,     # Standard stops
                'profit_target_mult': 1.2, # Modest targets (don't expect a breakout)
                'description': "RANGE TRADER"
            })
            
        # Return the final dictionary of parameters
        return params

class MarketRegimeDetector:
    """
    The 'Eyes' of the system.
    Analyzes technical indicators to classify the market into one of 3 states.
    """
    @staticmethod
    def detect_regime(row):
        """
        Classifies the current market state.
        :param row: A dictionary or Series containing current price/indicators.
        :return: String ("TRENDING_BULL", "TRENDING_BEAR", "CHOPPY_RANGE")
        """
        # Helper function to safely get values from the row (handling missing keys)
        def g(k, default=0.0): return float(row.get(k, default))

        # Extract Key Indicators
        close = g('close')
        ema_50 = g('sma_50', close)   # Medium-term trend proxy
        ema_200 = g('sma_200', close) # Long-term trend proxy (The 200 SMA)
        adx = g('adx', g('ADX_14', 0)) # Trend Strength
        
        # 1. Determine Trend Direction based on Moving Averages
        trend_direction = "NEUTRAL"
        # Bullish: Price > 200 AND 50 > 200 (Golden Cross alignment)
        if close > ema_200 and ema_50 > ema_200:
            trend_direction = "BULL"
        # Bearish: Price < 200 AND 50 < 200 (Death Cross alignment)
        elif close < ema_200 and ema_50 < ema_200:
            trend_direction = "BEAR"
            
        # 2. Determine Trend Strength using ADX
        # If ADX is high (>25), we trust the trend direction
        if adx > 25:
            if trend_direction == "BULL": return "TRENDING_BULL"
            if trend_direction == "BEAR": return "TRENDING_BEAR"
        
        # 3. If ADX is low or MAs are messy, assume it's Ranging/Choppy
        return "CHOPPY_RANGE"


class StrategyOrchestra:
    """
    Gen-10 Strategy Orchestra - 'ADAPTIVE' Edition.
    This class is the 'Brain'. It uses the Regime Detector to load the right rules,
    then evaluates the trade using AI, Fundamentals, and Technicals.
    """

    fundamentals = None # Class variable to store fundamental data (PE, Revenue, etc.)

    @classmethod
    def set_fundamentals(cls, data):
        """Sets the fundamental data for the current stock."""
        cls.fundamentals = data

    @staticmethod
    def decide_action(ticker, row, analysis):
        """
        Main Decision Function.
        1. Detects Regime.
        2. Loads Adaptive Parameters.
        3. Scores trade based on specific Regime rules.
        """
        # --- 1. DATA EXTRACTION ---
        # Helper to get value from row, checking multiple potential column names
        def get_val(keys, default=0.0):
            for k in keys:
                if k in row: return float(row[k])
            return default
        
        # Extract core indicators
        current_rsi = get_val(['rsi_14', 'RSI', 'rsi'], 50.0)
        current_adx = get_val(['adx_14', 'ADX', 'adx'], 0.0)
        price = get_val(['close'], 0.0)
        
        # Extract inputs from the AI Core
        ai_prob = analysis.get('AI_Probability', 0.0) 
        fund_score = analysis.get('Fundamental_Score', 50)

        # --- 2. DETECT REGIME & LOAD PARAMS ---
        # Ask the Detector: "What is the market doing?"
        regime = MarketRegimeDetector.detect_regime(row)
        # Load the rules for that specific market
        params = RegimeConfig.get_params(regime)
        
        # --- 3. CALCULATE SCORE ---
        final_score = 0
        log_reasons = [] # Keep a list of reasons for logging
        
        # A. Base AI Score (Weighted 40%)
        # If AI is 90% confident, this contributes 36 points (90 * 0.4)
        ai_score = ai_prob * 100
        final_score += ai_score * 0.40
        
        # B. Fundamental Score (Weighted 10%)
        # Good fundamentals give a small boost
        final_score += fund_score * 0.10
        
        # C. Technical Score (Weighted 50% - Adaptive)
        tech_score = 0
        
        # --- LOGIC BRANCHING BASED ON REGIME ---
        
        # CASE 1: BULL TREND (We want to buy!)
        if regime == "TRENDING_BULL":
            # Rule: Buy Pullbacks (RSI 40-60)
            if current_rsi < 60 and current_rsi > 40:
                tech_score += 30; log_reasons.append("Bull_Pullback")
            # Rule: Reward strong trends
            if current_adx > 25:
                tech_score += 20; log_reasons.append("Strong_Trend")
            # Rule: Price above short-term EMA (Momentum)
            if price > get_val(['ema_21'], 0):
                tech_score += 20; log_reasons.append("Above_EMA21")
            # Bonus: "Golden Zone" setup
            if 50 < current_rsi < 65:
                tech_score += 15; log_reasons.append("Golden_Zone")

        # CASE 2: BEAR TREND (Be very careful!)
        elif regime == "TRENDING_BEAR":
            # Rule: Only buy Deep Oversold (Mean Reversion)
            if current_rsi < 30:
                tech_score += 50; log_reasons.append("Oversold_Bear")
            elif current_rsi < 20:
                tech_score += 80; log_reasons.append("Capitulation_Buy")
            else:
                # Penalty: Don't buy if RSI is normal in a bear market
                tech_score -= 50; log_reasons.append("Bear_Trend_Penalty")

        # CASE 3: CHOPPY RANGE (Buy Support)
        elif regime == "CHOPPY_RANGE":
            # Rule: Buy near the bottom of the range
            if current_rsi < 35:
                tech_score += 40; log_reasons.append("Range_Bottom")
            # Rule: Penalize buying near the top
            elif current_rsi > 60:
                tech_score -= 30; log_reasons.append("Range_Top_Penalty")
            else:
                # Mid-range is "Dead Money" - avoid
                tech_score -= 10; log_reasons.append("Mid_Range_Noise")

        # Normalize Technical Score to 0-100 and add to Final Score
        tech_score = max(0, min(100, tech_score))
        final_score += tech_score * 0.50

        # --- 4. VERDICT GENERATION ---
        # Set the bar for buying based on difficulty
        buy_threshold = 60
        if regime == "TRENDING_BEAR": buy_threshold = 75 # Harder to buy in bear
        if regime == "CHOPPY_RANGE": buy_threshold = 65  # Harder to buy in chop
        
        verdict = "WAIT"
        if final_score >= buy_threshold:
            verdict = "BUY"
        
        # --- 5. LOGGING ---
        # Log if the score is interesting (close to buying)
        if final_score > 45:
            logger.info(
                f"\n ORCHESTRA REPORT [{ticker}]\n"
                f"   Mode: {params['description']} ({regime})\n"
                f"   RSI: {current_rsi:.1f} | ADX: {current_adx:.1f}\n"
                f"   Scores: AI({ai_score:.0f}) Fund({fund_score}) Tech({tech_score})\n"
                f"   Reasons: {', '.join(log_reasons)}\n"
                f"   Total: {final_score:.1f} / {buy_threshold} -> {verdict}"
            )

        # --- 6. SAFETY OVERRIDES (The "Veto" System) ---
        if verdict == "BUY":
            # Global Ceiling: Never buy if RSI is extreme (>80)
            if current_rsi > 80:
                logger.info("SAFETY: RSI > 80. Kill Switch.")
                return "WAIT"
            
            # Regime Specific Vetoes
            if regime == "TRENDING_BULL" and current_rsi > 75:
                return "WAIT" # Too hot even for a bull
                
            if regime == "CHOPPY_RANGE" and current_adx > 40:
                # If ADX spikes in a range, it might be a breakout against us
                logger.info("SAFETY: Range Volatility Spike. Wait for clarity.")
                return "WAIT"

        return verdict

    @staticmethod
    def get_adaptive_targets(row, entry_price):
        """
        Calculates Stop Loss and Take Profit based on the Regime.
        """
        # 1. Detect Regime to know which Multipliers to use
        regime = MarketRegimeDetector.detect_regime(row)
        params = RegimeConfig.get_params(regime)
        
        # 2. Get ATR (Average True Range) for volatility-based sizing
        def get_val(keys, default=0.0):
            for k in keys:
                if k in row: return float(row[k])
            return default
        atr = get_val(['atr', 'ATRr_14', 'atr_14'], entry_price * 0.02)
        
        # 3. Calculate Dynamic Levels
        # Stop Loss distance = ATR * Multiplier (e.g., 2.0x)
        stop_dist = atr * params['stop_loss_mult']
        # Target distance = Stop distance * R:R ratio (e.g., 1.5x)
        target_dist = stop_dist * params['profit_target_mult'] 
        
        stop_loss = entry_price - stop_dist
        target_price = entry_price + target_dist
        
        return stop_loss, target_price, f"{params['description']} (ATR x{params['stop_loss_mult']})"

    # --- AGENT STUBS ---
    # These empty functions are kept to prevent errors if other files try to import them.
    @staticmethod
    def get_score(features, regime, params=None): return 0, {}
    @staticmethod
    def _agent_breakout(f, p): return 0, {}
    @staticmethod
    def _agent_dip_buyer(f, p): return 0, {}
    @staticmethod
    def _agent_bear_defense(f): return 0, {}
    @staticmethod
    def _agent_fundamentals(f, p): return 0, {}
    @staticmethod
    def _agent_machine_learning(f, p): return 0, {}
    @staticmethod
    def _agent_deep_learning(f, p): return 0, {}