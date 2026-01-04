# strategy_engine.py

import logging
import numpy as np
import system_config as cfg

# Logger setup
logger = logging.getLogger("StrategyEngine")


class MarketRegimeDetector:
    @staticmethod
    def detect_regime(latest_row):
        close = latest_row['close']
        sma_short = latest_row.get('sma_short', close)
        sma_long = latest_row.get('sma_long', close)
        adx = latest_row.get('ADX_14', 0)

        if close < sma_long:
            return "BEARISH"

        # 2. RANGING_SHUFFLE (Consolidation)
        # if sma_short > 0:
        #     price_near_sma_short = abs(close - sma_short) / sma_short < 0.01
        # else:
        #     price_near_sma_short = False
        #
        # if adx < cfg.STRATEGY_PARAMS.get('adx_threshold', 25) and price_near_sma_short:
        #     return "RANGING_SHUFFLE"  # <-- NEW SHUFFLE REGIME

        # 3. TRENDING_UP (Strong Bull)
        if close > sma_long:
            if adx >= cfg.STRATEGY_PARAMS.get('adx_threshold', 25) and close > sma_short:
                return "TRENDING_UP"

        # 4. RANGING_BULL (Default / Mild Uptrend)
        return "RANGING_BULL"


class StrategyOrchestra:
    """
        Gen-7 Strategy Orchestra
        Orchestrates specialized agents (Breakout, DipBuyer) incorporating
        advanced indicators: EMA, PSAR, WaveTrend, Keltner Channels.
    """
    fundamentals = None # Gen-8 Global State

    @classmethod
    def set_fundamentals(cls, data):
        cls.fundamentals = data

    # @staticmethod
    # def decide_action(ticker, row, analysis):
    #     """
    #     New Weighted Decision Engine (Committee Vote).
    #     Replaces the old 'All or Nothing' logic.
    #     """
    #     ai_prob = analysis.get('AI_Probability', 0.0) # 0.0 to 1.0
    #     fund_score = analysis.get('Fundamental_Score', 50) # 0 to 100
    #     adx = row.get('ADX_14', 0)
        
    #     # --- NEW SCORING ENGINE (WITH SNIPER VETO) ---

    #     # [STEP 4 INTEGRATION] Check for Sniper Lock (Perfect Market Structure)
    #     # If perfect structure exists, we slightly lower the AI confidence requirement.
    #     # We check both 'adx' and 'ADX_14' to be safe with column naming.
    #     current_adx = row.get('adx', row.get('ADX_14', 0))
    #     current_rsi = row.get('rsi_14', 50)
    #     current_vol = row.get('volume', 0)
    #     vol_ma = row.get('vol_ma', current_vol)

    #     is_sniper_lock = (current_adx > 25 and current_rsi < 70 and current_vol > vol_ma)
        
    #     # Determine Dynamic Threshold
    #     required_confidence = cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD
    #     if is_sniper_lock:
    #         # Reward perfect technicals by allowing slightly lower AI confidence
    #         required_confidence *= 0.85 
    #         logger.info(f"🎯 Sniper Lock Active: Lowering AI Threshold to {required_confidence:.2f}")

    #     # 1. Hard Veto: If AI is confused, DO NOT TRADE.
    #     if ai_prob < required_confidence:
    #         logger.info(f"🚫 AI Veto: Confidence {ai_prob:.2f} < Threshold {required_confidence:.2f}")
    #         return "WAIT"

    #     # 2. Hard Veto: If Fundamentals are trash, DO NOT TRADE.
    #     if fund_score < 50:
    #         logger.info(f"🚫 Fundamental Veto: Score {fund_score} < 50")
    #         return "WAIT"

    #     # 3. Only calculate score if Vetoes pass
    #     # Normalize Inputs to 0-100 scale
    #     score_ai = ai_prob * 100
    #     score_fund = fund_score
    #     score_tech = 0
        
    #     # Technical Bonus
    #     if adx > 25: score_tech += 20
    #     # Check RSI (Safety: Not Overbought, Not Oversold)
    #     if row.get('RSI_14', 50) < 70 and row.get('RSI_14', 50) > 40: score_tech += 20
    #     # Check Trend (Price above 200 EMA)
    #     if row['close'] > row.get('EMA_200', 0): score_tech += 30
        
    #     # 2. Weighted Sum (The Committee Vote)
    #     # AI Opinion: 40% | Fundamentals: 30% | Technicals: 30%
    #     final_score = (score_ai * 0.4) + (score_fund * 0.3) + (score_tech * 0.3)
        
    #     # Log the breakdown so we can debug later
    #     logger.info(f"🔍 {ticker} Score breakdown: AI({int(score_ai)}) Fund({score_fund}) Tech({score_tech}) -> FINAL: {final_score:.1f}")

    #     # 3. Dynamic Threshold
    #     # If score >= 60, we take the trade.
    #     if final_score >= 50:
    #         return "BUY"
    #     elif final_score <= 30:
    #         return "SELL"
    #     else:
    #         return "WAIT"

    @staticmethod
    def decide_action(ticker, row, analysis):
        """
        Sniper Decision Engine (Verbose Logging Edition)
        """
        # --- 1. EXTRACT RAW DATA ---
        ai_prob = analysis.get('AI_Probability', 0.0) 
        fund_score = analysis.get('Fundamental_Score', 50)
        
        current_adx = row.get('adx', row.get('ADX_14', 0))
        current_rsi = row.get('rsi_14', 50)
        current_vol = row.get('volume', 0)
        vol_ma = row.get('vol_ma', current_vol)
        price = row.get('close', 0)
        ema_200 = row.get('EMA_200', 0)
        ema_21 = row.get('ema_21', row.get('sma_short', 0)) # Short term trend

        # --- 2. CALCULATE TECHNICAL SCORE ---
        tech_raw_score = 0
        tech_log = []
        
        # A. Trend Strength (ADX)
        if current_adx > 20: 
            tech_raw_score += 20
            tech_log.append("ADX>25 (+20)")
        
        # B. RSI Safety (Not Overbought)
        if 40 < current_rsi < 70: 
            tech_raw_score += 20
            tech_log.append("RSI_Safe (+20)")
            
        # C. Trend Alignment (Above 200 EMA)
        if price > ema_200: 
            tech_raw_score += 30
            tech_log.append("Price>EMA200 (+30)")
            
        # D. Volume Confirmation (Sniper Lock Component)
        if current_vol > vol_ma:
            tech_raw_score += 10
            tech_log.append("High_Vol (+10)")

        # Cap Technical Score
        tech_raw_score = min(tech_raw_score, 100)

        # --- 3. CALCULATE COMMITTEE SCORES ---
        # Weights: AI (60%), Fund (10%), Tech (30%)
        w_ai = 0.6
        w_fund = 0.1
        w_tech = 0.3
        
        score_ai_contribution = (ai_prob * 100) * w_ai
        score_fund_contribution = fund_score * w_fund
        score_tech_contribution = tech_raw_score * w_tech
        
        final_score = score_ai_contribution + score_fund_contribution + score_tech_contribution

        # --- 4. DETERMINE VERDICT (BEFORE VETO) ---
        # Raised Threshold to 60 to stop the "Machine Gun"
        if final_score >= 60:
            verdict = "BUY"
        elif final_score <= 30:
            verdict = "SELL"
        else:
            verdict = "WAIT"

        # --- 5. DETAILED LOGGING (The Report Card) ---
        # Log if score is decent (>40) or if it's a BUY signal
        if final_score > 45:
            log_msg = (
                f"\n📋 REPORT CARD: [{ticker}] @ {row.name if hasattr(row, 'name') else 'Now'}\n"
                f"--------------------------------------------------\n"
                f"1. 🧠 AI BRAIN       (Prob {ai_prob:.2f} | W {w_ai*100}%): +{score_ai_contribution:.1f}\n"
                f"2. 📊 FUNDAMENTALS   (Raw {fund_score}   | W {w_fund*100}%): +{score_fund_contribution:.1f}\n"
                f"3. 📈 TECHNICALS     (Raw {tech_raw_score}   | W {w_tech*100}%): +{score_tech_contribution:.1f}\n"
                f"   L Details: {', '.join(tech_log)}\n"
                f"--------------------------------------------------\n"
                f"🏆 FINAL SCORE: {final_score:.1f} / 100  [Req: 60]\n"
                f"⚖️ VERDICT: {verdict}\n"
            )
            logger.info(log_msg)

        # --- 6. VETO GATES (After Logging) ---
        # We check vetoes last so we can see the "Report Card" first
        
        # Check Sniper Lock (Perfect Market Structure)
        is_sniper_lock = (current_adx > 25 and current_rsi < 70 and current_vol > vol_ma)
        required_confidence = cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD
        
        if verdict == "BUY":
            # 1. AI Veto
            if ai_prob < required_confidence:
                logger.info(f"🚫 AI Veto: {ai_prob:.2f} < {required_confidence:.2f}")
                return "WAIT"

            # 2. Trend Strength Veto (With Accumulation Override)
            # Bottleneck Fix: Allow buying in "Chop" (ADX < 20) IF AI is highly confident (Accumulation)
            if current_adx < 20:
                if ai_prob > 0.70:
                    logger.info(f"⚡ ACCUMULATION ENTRY: Low ADX ({current_adx:.1f}) overridden by High AI ({ai_prob:.2f})")
                else:
                    logger.info(f"🚫 Trend Veto: ADX {current_adx:.1f} < 20 (Chop) & AI Neutral")
                    return "WAIT"

            # 3. DIRECTIONAL VETO (CRITICAL FIX)
            # Never buy if price is below short-term trend (EMA 21).
            # This prevents buying "Strong Downtrends".
            if price < ema_21:
                if ai_prob > 0.75:
                    logger.info(f"🪂 DIP BUY: Price below EMA21 overridden by Massive AI Confidence ({ai_prob:.2f})")
                else:
                    logger.info(f"🚫 Directional Veto: Price {price:.2f} < EMA21 {ema_21:.2f} (Downtrend)")
                    return "WAIT"

            # 4. Fundamental Veto
            if fund_score < 50:
                return "WAIT"
                
            # # --- 7. ADAPTIVE ENTRY ENGINE (The Fix for "Hot" Markets) ---
            # if verdict == "BUY":
            
            # Scenario A: THE ROCKET (High Momentum -> Buy Market)
            # If ADX > 40 or RSI > 65, the stock is flying. Don't wait for a dip.
            if current_adx > 40 or current_rsi > 65:
                logger.info(f"🚀 ROCKET DETECTED (ADX {current_adx:.1f}): Buying at MARKET immediately.")
            
            # Scenario B: THE SNIPER (Strong Trend -> Slight Discount)
            # If ADX is healthy (25-40), try to save 0.3% (standard liquidity dip).
            elif current_adx > 25:
                discount = price * 0.003 
                limit_target = price - discount
                logger.info(f"🔫 SNIPER ENTRY: Market is solid. Setting Limit at {limit_target:.2f} (-0.3%)")

            # Scenario C: THE BOTTOM FISHER (Choppy/Reversal -> Deep Discount)
            # If ADX < 25, the trend is weak. We have leverage to demand a better price.
            else:
                # Use ATR for volatility if available, else 1%
                atr = row.get('atr', price * 0.01)
                limit_target = price - (atr * 0.5) 
                logger.info(f"⚓ BOTTOM FISHING: Market is choppy. Waiting for {limit_target:.2f} (Deep Pullback)")


        return verdict

    @staticmethod
    def get_adaptive_targets(row, entry_price):
        """
        Orchestra Stop-Loss Engine (Auto-Adaptive).
        Calculates dynamic stops based on Normalized ATR (Volatility %).
        """
        # 1. Get Volatility Data (ROBUST CHECK)
        # Check all possible column names for ATR
        atr = row.get('atr', row.get('ATRr_14', row.get('atr_14', 0.0)))
        
        # Fallback if ATR is missing or 0
        if atr <= 0:
            # logger.warning(f"⚠️ Missing ATR for adaptive stop. Defaulting to 2.5%.")
            atr = entry_price * 0.025
        
        # 2. Calculate "Wildness" (NATR)
        # Example: ATR 5.00 / Price 100.00 = 5% Volatility
        natr = (atr / entry_price) * 100
        
        # 3. Assign Dynamic Multiplier
        if natr >= 4.0:
            stop_mult = 3.5  # Extreme Volatility (Crypto/Biotech mode)
            regime = "EXTREME"
        elif natr >= 2.5:
            stop_mult = 3.0  # High Volatility (Wild Tech: NVDA, TSLA)
            regime = "HIGH"
        elif natr >= 1.5:
            stop_mult = 2.5  # Normal Volatility (Standard: GOOGL, AAPL)
            regime = "NORMAL"
        else:
            stop_mult = 2.0  # Low Volatility (Stable: KO, JNJ)
            regime = "STABLE"
            
        # 4. Calculate Targets
        # We aim for 2.0 R:R (Risk/Reward), so Target is 2x the Stop distance
        # BUT for crazy stocks, we might cap the target or scale it differently.
        # Here we stick to 2.0x for consistency.
        target_mult = stop_mult * 2.0 
        
        stop_loss = entry_price - (atr * stop_mult)
        target_price = entry_price + (atr * target_mult)
        
        return stop_loss, target_price, f"{regime} ({stop_mult}x)"
    
    @staticmethod
    def _agent_fundamentals(f, p):
        """
        Agent 8: Fundamental Analysis.
        Uses static class variable 'fundamentals' set by external engine.
        """
        score = 0
        details = {}
        
        fund = StrategyOrchestra.fundamentals
        if not fund:
            return 0, {}
            
        # PE Ratio Logic
        pe = fund.get('trailingPE')
        if pe and pe > 0:
            if pe < 20: 
                score += 10
                details['Fund_LowPE'] = 10
            elif pe > 50:
                score -= 5 # Overvalued penalty
                
        # Revenue Growth
        rev_growth = fund.get('revenueGrowth')
        if rev_growth and rev_growth > 0.20: # >20% growth
            score += 10
            details['Fund_HighGrowth'] = 10
            
        return score, details

    @staticmethod
    def _agent_machine_learning(f, p):
        """Gen-8: ML Confirmation (Lorentzian/KNN) with Veto Power"""
        score = 0
        details = {}
        ml_signal = f.get('ml_signal', 0)
        
        if ml_signal == 1:
            score += 25 # Confirmation
            details["ML_Lorentzian_Bull"] = 25
        else:
            score -= 50 # VETO: If ML says Down, we kill the trade.
            details["ML_Lorentzian_Bear_VETO"] = -50
            
        return score, details

    @staticmethod
    def _agent_deep_learning(f, p):
        """Gen-9: Deep Learning 'Sniper' Agent"""
        score = 0
        details = {}
        
        # We need historical context (DataFrame) for LSTM
        history_window = f.get('history_window')
        
        if history_window is None or history_window.empty:
            return 0, {}
            
        try:
            # Lazy Import to avoid overhead if not used
            from stockwise_ai_core import StockWiseAI
            
            # Singleton storage could be improved, but for now instantiate/load
            # In production, this should be initialized once in the class __init__
            ai = StockWiseAI() 
            
            # Predict
            # AI expects exactly 60 bars.
            if len(history_window) >= 60:
                # Take last 60
                window = history_window.iloc[-60:].copy()
                action, confidence = ai.predict(window)
                
                if action == 2 and confidence > 0.95:
                    score += 100 # INSTANT TRIGGER
                    details["AI_SNIPER_BUY"] = 100
                    details["AI_Confidence"] = float(f"{confidence:.4f}")
        except Exception as e:
            # logger.error(f"AI Agent Error: {e}")
            pass
            
        # return score, details
        # """Gen-8: Fundamental Filter"""
        # score = 0
        # details = {}
        # data = StrategyOrchestra.fundamentals
        
        # if not data: return 0, {}
        
        # # 1. Revenue Growth
        # rev_growth = data.get('revenueGrowth')
        # if rev_growth and rev_growth > 0.15:
        #     score += 15
        #     details["Fund_HighGrowth"] = 15
            
        # # 2. Profit Margins
        # margins = data.get('profitMargins')
        # if margins and margins > 0.20:
        #      score += 10
        #      details["Fund_HighMargin"] = 10

        # return score, details


    # @staticmethod
    # def get_score(features, regime, params=None):
    #     """
    #     Main dispatch method to select the correct agent based on regime.
    #     """
    #     if params is None: params = cfg.STRATEGY_PARAMS
    #
    #     score_details = {
    #         "Trigger_Breakout": 0, "Conf_Volume": 0, "Conf_Slope": 0, "Conf_EMA_Short": 0,
    #         "Conf_PSAR_Bull": 0, "Conf_Kalman_Positive": 0, "Conf_Wavelet_LowNoise": 0,
    #         "Penalty_Wavelet_Noise": 0, "Conf_Sector_Strong": 0, "Penalty_Sector_Weak": 0,
    #         "Penalty_RSI_Overbought": 0, "Feature_SlopeAngle": features.get('slope_angle', 0.0),
    #         "Feature_KalmanSmooth": features.get('kalman_smooth', 0.0),
    #         "Feature_WaveletNoise": features.get('wavelet_std_ratio', 99.0),
    #         "Feature_RSI": features.get('rsi_14', 50),
    #         "Regime": regime
    #     }
    #     score = 0
    #
    #     details = {}  # Initialize details to prevent UnboundLocalError
    #
    #     if regime == "TRENDING_UP":
    #         score, details = StrategyOrchestra._agent_breakout(features, params)
    #     elif regime == "RANGING_SHUFFLE":
    #         score, details = StrategyOrchestra._agent_dip_buyer(features, params)  # <-- USE DIP BUYER
    #     elif regime == "RANGING_BULL":
    #         if features['close'] > features.get('recent_high', 99999):
    #             score, details = StrategyOrchestra._agent_breakout(features, params)
    #         else:
    #             score, details = StrategyOrchestra._agent_dip_buyer(features, params)
    #     elif regime == "BEARISH":
    #         score, details = StrategyOrchestra._agent_bear_defense(features)
    #
    #     # Merge the score breakdown for logging
    #     score_details.update(details)
    #
    #     # --- CAP THE SCORE AT 100 ---
    #     score = min(score, 100)
    #
    #     return score, score_details


    @staticmethod
    def get_score(features, regime, params=None):
        """
        Main dispatch method to select the correct agent based on regime.

        Gen-7 Logic Override: Calculates potential score from all relevant agents
        and selects the highest positive score, allowing high-confidence Dip Buyer
        signals to override conservative regime filters (Falling Knife Paradox fix).
        """
        if params is None: params = cfg.STRATEGY_PARAMS

        score_details = {
            "Trigger_Breakout": 0, "Conf_Volume": 0, "Conf_Slope": 0, "Conf_EMA_Short": 0,
            "Conf_PSAR_Bull": 0, "Conf_Kalman_Positive": 0, "Conf_Wavelet_LowNoise": 0,
            "Penalty_Wavelet_Noise": 0, "Conf_Sector_Strong": 0, "Penalty_Sector_Weak": 0,
            "Penalty_RSI_Overbought": 0, "Feature_SlopeAngle": features.get('slope_angle', 0.0),
            "Feature_KalmanSmooth": features.get('kalman_smooth', 0.0),
            "Feature_WaveletNoise": features.get('wavelet_std_ratio', 99.0),
            "Feature_RSI": features.get('rsi_14', 50),
            "Regime": regime
        }

        # --- Gen-7 Logic Override: Calculate potential score from all relevant agents ---
        # We calculate everything upfront so we can compare them
        breakout_score, breakout_details = StrategyOrchestra._agent_breakout(features, params)
        dip_score, dip_details = StrategyOrchestra._agent_dip_buyer(features, params)
        # bear_score, bear_details = StrategyOrchestra._agent_bear_defense(features)

        # Default initialization
        score = 0
        details = {}
        active_agent = "None"
        
        # --- GEN-8: AGENT CALCULATIONS ---
        ml_score, ml_details = StrategyOrchestra._agent_machine_learning(features, params)
        fund_score, fund_details = StrategyOrchestra._agent_fundamentals(features, params)
        dl_score, dl_details = StrategyOrchestra._agent_deep_learning(features, params)
        
        # BASE SCORE BOOST (Add to whatever agent is selected)
        gen8_boost = ml_score + fund_score + dl_score
        gen8_details = {}
        gen8_details.update(ml_details)
        gen8_details.update(fund_details)
        gen8_details.update(dl_details)

        if regime == "BEARISH":
            # In Bear market, Breakouts are harder (Penalty), Dips are preferred.
            # But if a breakout is HUGE (Score > 80), we take it (Turnaround play).
            breakout_score -= 10
            breakout_details["Penalty_Regime_Bear"] = -10

            if dip_score > breakout_score:
                score = dip_score
                details = dip_details
                active_agent = "DipBuyer_Bear"
            else:
                score = breakout_score
                details = breakout_details
                active_agent = "Breakout_CounterTrend"

        # A. Prioritize Breakout if it has the highest score in Trending/Bullish regimes
        elif regime == "TRENDING_UP":
            # In Bull market, Breakouts are preferred.
            if breakout_score >= dip_score:
                score = breakout_score
                details = breakout_details
                active_agent = "Breakout_Trend"

            else:
                score = dip_score
                details = dip_details
                active_agent = "DipBuyer_Trend"
        else:  # RANGING / SHUFFLE
            # Pure meritocracy: Take the winner
            if breakout_score > dip_score:
                score = breakout_score
                details = breakout_details
                active_agent = "Breakout_Range"
            else:
                score = dip_score
                details = dip_details
                active_agent = "DipBuyer_Range"

        score += gen8_boost
        details.update(gen8_details)

        # Final check: Must be positive or zero
        if score < 0:
            score = 0
            details = {"status": "BLOCKED_BY_PENALTY"}

        # Merge the score breakdown for logging
        score_details.update(details)
        score_details['Active_Agent'] = active_agent

        # --- CAP THE SCORE AT 100 ---
        score = min(score, 100)

        return score, score_details

    # @staticmethod
    # def _agent_breakout(f, p):
    #     """
    #     Gen-7 Breakout Agent
    #
    #     Logic:
    #     - Trigger: Price > Recent High
    #     - Confirmation: EMA Trend, PSAR (Bullish), Keltner Channel Breakout
    #     - Signal Processing: Kalman & Wavelet constraints
    #     """
    #     score = 0
    #     details = {}
    #
    #     # Check if we are running an Ablation Test (disabled features)
    #     # If the list doesn't exist, default to empty (no disables)
    #     disabled_features = getattr(cfg, 'DISABLED_FEATURES', [])
    #
    #     # --- GEN-7 RF PARAMETERS ---
    #     kalman_smooth_threshold = p.get('kalman_smooth_threshold', 0.5)
    #     wavelet_noise_max = p.get('wavelet_noise_max', 1.5)
    #     kalman_dev = f.get('kalman_smooth', 0.0)
    #
    #     # 1. Breakout Trigger
    #     if f['close'] > f.get('recent_high', 99999):
    #         score += cfg.SCORE_TRIGGER_BREAKOUT
    #         details["Trigger_Breakout"] = cfg.SCORE_TRIGGER_BREAKOUT
    #
    #     # 2. EMA Trend Crossover (Wake-Up Signal) [cite: 140, 141, 142]
    #     if "ema_crossover" not in disabled_features:
    #         ema_9 = f.get('ema_9', 0)
    #         ema_21 = f.get('ema_21', 0)
    #         if ema_9 > ema_21:
    #             score += 10
    #             details["Conf_EMA_Crossover"] = 10
    #     # score += ema_score
    #     # details["Conf_EMA_Crossover"] = ema_score
    #
    #     # 3. Parabolic SAR (PSAR) Logic [cite: 155]
    #     if "psar" not in disabled_features:
    #         psar_score = 0
    #         psar = f.get('psar', 0)
    #         ema_9 = f.get('ema_9', 0)
    #         ema_21 = f.get('ema_21', 0)
    #
    #         # # Bullish Regime: PSAR must be below price (PSAR < Price)
    #         # if 0 < psar < f['close']:
    #         #     psar_score = 15
    #         # elif psar > f['close']:
    #         #     # Only penalize if we are NOT in a strong EMA uptrend
    #         #     # If EMA9 > EMA21, we assume trend is strong and PSAR is just lagging.
    #         #     if ema_9 > ema_21:
    #         #         psar_score = 0  # No penalty, ignore lag
    #         #         details["Conf_PSAR_Lag_Ignored"] = 0
    #         #     else:
    #         #         psar_score = -35
    #
    #         score += psar_score
    #         details["Conf_PSAR_Bull"] = psar_score
    #
    #     # 4. Keltner Channel (KC) Breakout Confirmation
    #     if "kc_breakout" not in disabled_features:
    #         kc_upper = f.get('kc_upper', 0)
    #         kc_score = 0
    #         if f['close'] > kc_upper:
    #             kc_score = 10
    #         score += kc_score
    #         details["Conf_KC_Upper"] = kc_score
    #
    #     # 5. Volume Confirmation
    #     if "volume" not in disabled_features:
    #         vol_score = 0
    #         if f['volume'] > f.get('vol_ma', 0) * p.get('vol_multiplier', 1.1):
    #             vol_score = cfg.SCORE_CONFIRM_VOLUME
    #         score += vol_score
    #         details["Conf_Volume"] = vol_score
    #
    #     # 6. Trend Slope/Velocity
    #     if "slope" not in disabled_features:
    #         angle_score = 0
    #         angle = f.get('slope_angle', 0)
    #         if angle > p.get('slope_threshold', 20):
    #             angle_score = cfg.SCORE_CONFIRM_SLOPE
    #         score += angle_score
    #         details["Conf_Slope"] = angle_score
    #
    #     # 7. Kalman Deviation Check (Positive Deviation for trend-following)
    #     if "kalman" not in disabled_features:
    #         kalman_score = 0
    #         if kalman_dev > kalman_smooth_threshold:
    #             kalman_score = cfg.SCORE_CONFIRM_KALMAN
    #         score += kalman_score
    #         details["Conf_Kalman_Positive"] = kalman_score
    #
    #     # 8. Wavelet Noise Check
    #     if "wavelet" not in disabled_features:
    #         wavelet_score = 0
    #         noise_penalty = 0
    #         if f.get('wavelet_std_ratio', 99.0) < wavelet_noise_max:
    #             wavelet_score = cfg.SCORE_CONFIRM_WAVELET
    #         else:
    #             noise_penalty = cfg.SCORE_PENALTY_NOISE  # Negative value
    #         score += wavelet_score + noise_penalty
    #         details["Conf_Wavelet_LowNoise"] = wavelet_score
    #         details["Penalty_Wavelet_Noise"] = noise_penalty
    #
    #     # 9. Safety Checks
    #     if "rsi_safety" not in disabled_features:
    #         rsi_penalty = 0
    #         if f.get('rsi_14', 50) > 82:
    #             rsi_penalty = cfg.SCORE_PENALTY_RSI
    #         score += rsi_penalty
    #         details["Penalty_RSI_Overbought"] = rsi_penalty
    #
    #     return score, details



    #
    # @staticmethod
    # def _agent_dip_buyer(f, p):
    #     """
    #     Gen-7 Dip Buyer Agent
    #
    #     Logic:
    #     - Trigger: WaveTrend Crossover (Oversold) OR RSI Fallback
    #     - Support: Price at Keltner Lower Band (KC)
    #     - Signal Processing: Kalman Negative Deviation (Dip), Low Noise
    #     - CRITICAL: WaveTrend Oversold Crossover is the logic override for 'Falling Knife'.
    #     """
    #     score = 0
    #     details = {}
    #
    #     # --- NEW GEN-7 RF PARAMETERS ---
    #     kalman_smooth_threshold = p.get('kalman_smooth_threshold', 0.5)
    #     wavelet_noise_max = p.get('wavelet_noise_max', 1.5)
    #
    #     wt1 = f.get('wt1', 0)  # Fast Line (Signal) [cite: 165]
    #     wt2 = f.get('wt2', 0)  # Slow Line (Average) [cite: 166]
    #     kc_lower = f.get('kc_lower', 0)  # Keltner Channel Lower Band [cite: 179]
    #
    #     # 1. Primary Trigger: WaveTrend Oversold Crossover [cite: 169, 170]
    #     wt_score = 0
    #     # Condition: Deeply oversold (<-60) AND Fast crosses above Slow
    #
    #     if wt1 < -53 and wt1 > wt2:
    #         wt_score = 45  # High Trigger Bonus (The override logic)
    #     elif f.get('rsi_14', 50) < 40:
    #         wt_score = cfg.SCORE_TRIGGER_DIP  # Fallback Trigger
    #
    #     score += wt_score
    #     details["Trigger_WaveTrend"] = wt_score
    #
    #     # 2. Support Confirmation: Price near Keltner Channel Lower Band [cite: 174]
    #     kc_score = 0
    #     if 0 < kc_lower <= f['close'] * 1.005:  # Price is at or slightly above KC Lower
    #         kc_score = 25
    #     score += kc_score
    #     details["Conf_KC_Lower"] = kc_score
    #
    #     # 3. Kalman Deviation Check (Must be a dip)
    #     kalman_score = 0
    #     # Check for negative deviation (dip below trend)
    #     if f.get('kalman_smooth', 0.0) < 0:
    #         kalman_score = cfg.SCORE_CONFIRM_KALMAN
    #     score += kalman_score
    #     details["Conf_Kalman_Negative"] = kalman_score
    #
    #     # 4. Wavelet Noise Check (Crucial for Dips - must be low noise)
    #     wavelet_score = 0
    #     noise_penalty = 0
    #     wavelet_noise = f.get('wavelet_std_ratio', 99.0)
    #     if wavelet_noise < wavelet_noise_max:
    #         wavelet_score = cfg.SCORE_CONFIRM_WAVELET
    #     else:
    #         noise_penalty = cfg.SCORE_PENALTY_NOISE  # Negative value
    #     score += wavelet_score + noise_penalty
    #     details["Conf_Wavelet_LowNoise"] = wavelet_score
    #     details["Penalty_Wavelet_Noise"] = noise_penalty
    #
    #     # 5. Micro-Structure Confirmation (VWAP Recapture / IBS Reversion)
    #     ibs_score = 0
    #     if f.get('ibs', 0.5) < 0.2:
    #         ibs_score = 15
    #     score += ibs_score
    #     details["Conf_IBS_Reversion"] = ibs_score
    #
    #     vwap_score = 0
    #     if f.get('vwap', 0) > 0 and f['close'] > f['vwap']:
    #         vwap_score = 5
    #     score += vwap_score
    #     details["Conf_VWAP"] = vwap_score
    #
    #     return score, details

        # strategy_engine.py

    @staticmethod
    def _agent_breakout(f, p):
        """
        Gen-7 Breakout Agent (Ablation-Ready)
        """
        # 1. Get the list of disabled features for the current test
        # If running normally, this list is empty.
        disabled_features = getattr(cfg, 'DISABLED_FEATURES', [])

        score = 0
        details = {}

        # --- 1. Breakout Trigger (The Base) ---
        if f.get('recent_high', 0) > 0 and f['high'] > f['recent_high']:
            score += cfg.SCORE_TRIGGER_BREAKOUT
            details["Trigger_Breakout"] = cfg.SCORE_TRIGGER_BREAKOUT
            # print(f"DEBUG: Breakout Triggered! High: {f['high']} > RecentHigh: {f['recent_high']}") # DEBUG
        else:
            # print(f"DEBUG: Breakout Failed. High: {f['high']}, RecentHigh: {f.get('recent_high', 'MISSING')}") # DEBUG
            pass

        # --- 2. EMA Trend Crossover ---
        if "ema_crossover" not in disabled_features:
            if f.get('ema_9', 0) > f.get('ema_21', 0):
                score += 10      # EMA crossover bonus - ENABLED for Gen-7
                details["Conf_EMA_Crossover"] = 10

        # --- 3. MACD Confirmation ---
        if "macd" not in disabled_features:
            # Check if MACD key exists (it might be missing if feature calc failed)
            if f.get('macd', 0) > f.get('macd_signal', 0):
                score += 10
                details["Conf_MACD"] = 10

        # --- 4. ADX Trend Strength ---
        if "adx" not in disabled_features:
            if f.get('ADX_14', 0) > 20:
                score += 5
                details["Conf_ADX_Strong"] = 5

        # --- 5. Keltner Channel ---
        if "kc_breakout" not in disabled_features:
            # Breakout: Close > Keltner Upper Band
            if f['close'] > f.get('kc_upper', 99999):
                score += 10
                details["Conf_KC_Upper"] = 10

        # 5. Candlestick Confirmation (Bullish Continuation / Reversal)
        # Using the new "bullish_continuation" (Rising 3 / 3-Line Strike) for trend confirmation
        if f.get('bullish_continuation', False):
            score += 15
            details["Conf_Candle_Continuation"] = 15
        elif f.get('bullish_reversal', False):  # Fallback to basic patterns (Hammer/Engulfing)
            score += 10
            details["Conf_Candle_Basic"] = 10
            
        # 6. Sector Alignment (if available)
        # If 'rel_strength_sector' > 0, it means we are outperforming the sector
        if f.get('rel_strength_sector', 0) > 0:
            score += cfg.SCORE_CONFIRM_SECTOR
            details["Conf_Sector_Strong"] = cfg.SCORE_CONFIRM_SECTOR
        elif f.get('rel_strength_sector', 0) < -0.02:
            score += cfg.SCORE_PENALTY_SECTOR
            details["Penalty_Sector_Weak"] = cfg.SCORE_PENALTY_SECTOR

        # # --- 7. SNIPER LOCK (The "Perfect Setup" Bonus) ---
        # # User demands 95% Win Rate. We need to identify the "Slam Dunk".
        # # Condition: Strong Trend (ADX>25) + Not Overbought (RSI<70) + High Volume + Trend Align
        # if f.get('adx', 0) > 25 and f.get('rsi_14', 50) < 75 and f['volume'] > f.get('vol_ma', 0):
        #      score += 20
        #      details["Conf_Sniper_Lock"] = 20

        # --- 8. Volume Confirmation ---
        if "volume" not in disabled_features:
            if f['volume'] > f.get('vol_ma', 0) * 1.1:
                score += cfg.SCORE_CONFIRM_VOLUME
                details["Conf_Volume"] = cfg.SCORE_CONFIRM_VOLUME

        # --- 9. Safety Checks (RSI) ---
        if "rsi_safety" not in disabled_features:
            if f.get('rsi_14', 50) > 90:
                score -= 20
                details["Penalty_RSI_Extreme"] = -20

        return score, details

    @staticmethod
    def _agent_dip_buyer(f, p):
        """
        Gen-7 Dip Buyer
        Updates: Added 'Falling Knife' filter (200 EMA).
        """
        # Get disabled features list
        disabled_features = getattr(cfg, 'DISABLED_FEATURES', [])

        score = 0
        details = {}
        
        wt1 = f.get('wt1', 0)
        wt2 = f.get('wt2', 0)

        # 1. Rising Knife Filter (Wait, Falling Knife) with Gen-7 OVERRIDE
        long_trend = f.get('sma_200', f.get('sma_long', 0))
        
        # Calculate Signals FIRST
        wt_score = 0
        is_deep_value = False
        
        # Gen-7 Spec: Trigger if WT1 < -60 (Deep Oversold) AND Crossover
        if wt1 < -60 and wt1 > wt2:
            wt_score = 45  # High Trigger Bonus (The override logic)
            is_deep_value = True
        elif wt1 < -53 and wt1 > wt2:
             wt_score = 30 # Normal trigger
        elif f.get('rsi_14', 50) < 45:
            # Fallback Trigger (Weak)
            wt_score = cfg.SCORE_TRIGGER_DIP 

        # Now Apply Filter with Override Logic
        if long_trend > 0 and f['close'] < long_trend * 0.85:
            if is_deep_value:
                # OVERRIDE: Allowed because it's a "Deep Value" play
                details["Info_Filter_Override"] = "Deep_Value_Catch"
            else:
                # Block trade
                # print(f"DEBUG: DipBuyer Filtered. Close: {f['close']} < LongTrend: {long_trend}") # DEBUG
                return 0, {"Status": "Filter_Bear_Crash"}

        score += wt_score
        details["Trigger_WaveTrend"] = wt_score

        # 2. Support Confirmation: Price near Keltner Channel Lower Band [cite: 174]
        kc_score = 0
        kc_lower = f.get('kc_lower', 0)
        # Logic: If price is dipping into or below the Lower Band
        if 0 < kc_lower and f['close'] <= kc_lower * 1.01: 
            kc_score = 15
            details["Conf_KC_Lower"] = 15
        score += kc_score

        # 3. Trend Support Bonus (+10)
        # If we are dipping BUT still above the 200 EMA, it's a "Bullish Pullback".
        if "trend_support" not in disabled_features:
            if f['close'] > long_trend:
                score += 10
                details["Conf_Trend_Support"] = 10

        # 4. RSI Oversold (+15)
        # Standard Dip logic
        if "rsi_dip" not in disabled_features:
            rsi = f.get('rsi_14', 50)
            if 25 < rsi < 45:
                score += 15
                details["Conf_RSI_Dip"] = 15
            elif rsi <= 25:
                score += 10
                details["Conf_RSI_Extreme"] = 10

        # 5. Candlestick Confirmation (+15)
        if "candles" not in disabled_features:
            if f.get('bullish_pattern', False):
                score += 15
                details["Conf_Candle_Bull"] = 15

        # # --- GEN-8 AGENTS ---
        # # Assuming these agents are meant to be called with 'f' (features) and 'p' (parameters)
        # # and return a score and update details.
        # # The instruction refers to `calculate_score` which uses `self._agent_...` and `row, decision_log, feature_manifest`.
        # # Adapting to `_agent_dip_buyer` context:
        # ml_score, ml_details = StrategyOrchestra._agent_machine_learning(f, p)
        # score += ml_score
        # details.update(ml_details)

        # fund_score, fund_details = StrategyOrchestra._agent_fundamentals(f, p)
        # score += fund_score
        # details.update(fund_details)

        # 6. Volume Spike (+10)
        if "volume_dip" not in disabled_features:
            if f['volume'] > f.get('vol_ma', 0) * 1.2:
                score += 10
                details["Conf_Vol_Capitulation"] = 10
        
        # print(f"DEBUG: DipBuyer Score: {score}, Details: {details}") # DEBUG
        return score, details
    @staticmethod
    def _agent_bear_defense(f):
        # score = 0
        # details = {}
        # defense_score = 0
        #
        # if f.get('rsi_14', 50) < 20:
        #     defense_score = 55
        #
        # score += defense_score
        # details["Defense_RSI_Oversold"] = defense_score
        # return score, details
        score = 0
        details = {}
        defensescore = 0

        # Existing RSI-based defense
        if f.get('rsi14', 50) < 20:
            defensescore = 25
            score += defensescore
            details['DefenseRSIOversold'] = defensescore

        # reward bearish patterns (confirming downside)
        bearish_bonus = 0
        if f.get('bearish_pattern', False):
            bearish_bonus = 10  # tune this
            score += bearish_bonus
        details['ConfBearishPattern'] = bearish_bonus

        # penalize bullish patterns (conflicting with bear regime)
        bullish_penalty = 0
        if f.get('bullish_pattern', False):
            bullish_penalty = 10  # tune this
            score -= bullish_penalty
        details['PenaltyBullishPattern'] = bullish_penalty

        return score, details
