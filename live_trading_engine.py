# live_trading_engine.py

"""
StockWise Gen-12 Live Trading Engine
====================================
The Execution & Defense Matrix.
Houses Agent 4 (The Lifecycle Manager) which monitors active trades, 
ratchets kinetic stops, and executes the Zombie Trade Protocol.
"""

import os
import json
import logging
import requests
import asyncio
import pytz
from datetime import datetime, timedelta
import system_config as cfg
from notification_manager import NotificationManager
import csv
import argparse
import time

# Initialize Logger (Rule 2 Compliance)
logger = logging.getLogger("LiveTradingEngine")

async def scheduled_health_check(ib_client, notifier_instance):
    """
    [Proactive Health Monitoring Routine]
    Continuously monitors the IB Gateway connection.
    Specifically checks status at 07:00, 08:00, 21:00, and 22:00 (Asia/Jerusalem time),
    triggering a Telegram alert if the weekly reauthentication (Soft token=0) disconnected the system.
    """
    israel_tz = pytz.timezone('Asia/Jerusalem')
    target_hours = {7, 8, 21, 22}
    
    logger.info("Proactive Health Monitoring Routine initiated for Israel timezone.")
    
    while True:
        try:
            # Capture exact current time in Israel
            now = datetime.now(israel_tz)
            
            # Perform the critical check exactly when the minute is 00 for target hours
            if now.hour in target_hours and now.minute == 0:
                logger.debug(f"Executing scheduled IB Gateway health check. IST Time: {now.strftime('%H:%M')}")
                
                # Verify physical connection using official IBKR API method
                if not ib_client.isConnected():
                    logger.error("Health Check Failed: ib_client.isConnected() returned False.")
                    # Trigger the Headless Control notification
                    notifier_instance.send_ibkr_disconnect_alert()
                else:
                    logger.debug("Health Check Passed: IB Gateway is actively connected.")
                    
                # Sleep for 60 seconds to avoid multi-triggering within the exact same minute
                await asyncio.sleep(60)
            else:
                # Wake up every 30 seconds to re-evaluate the time
                await asyncio.sleep(30)
                
        except Exception as e:
            logger.error(f"Critical failure in scheduled_health_check loop: {str(e)}")
            await asyncio.sleep(60)

class TradeJournal:
    """
    The Black Box Recorder (Upgraded).
    Now tracks Trend Prediction Accuracy to measure Analysis Quality vs. Execution Quality.
    """
    def __init__(self, filename="StockWise_Trade_Journal.csv"):
        self.filepath = os.path.join(cfg.BASE_DIR, filename)
        self._initialize_csv()

    def _initialize_csv(self):
        # Create file with NEW headers if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp",        # Time of signal
                        "Symbol",           # Ticker
                        "Action",           # BUY/WAIT
                        "Master_Score",     # Final Weighted Score
                        "Tech_Score",       # Technical Pattern Score
                        "AI_Score",         # AI Probability
                        "Setups_Found",     # List of active patterns (e.g., "SQUEEZE|VSA")
                        "Trend_Pre",        # Trend Direction at Entry (UP/DOWN/CHOP)
                        "Trend_Post",       # Trend Direction at Exit (Filled later)
                        "Trend_Success",    # 1 = Correct Prediction, 0 = Failed
                        "Entry_Price",      # Signal Price
                        "Stop_Loss",        # Calculated Stop
                        "Target_Price",     # Calculated Target
                        "Risk_Ratio",       # Reward / Risk
                        "Status",           # SIGNAL_ONLY, EXECUTED, REJECTED
                        "Execution_Price",  # Actual Fill Price
                        "Exit_Price",       # Actual Exit Price
                        "PnL_Percent"       # Final Profit/Loss
                    ])

    def log_signal(self, ticket, df_snapshot=None, status="SIGNAL_ONLY", exec_price=0, exit_price=0, pnl=0):
        """
        Logs a signal event with Trend Verification data.
        """
        try:
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # --- CALCULATION LOGIC ---
                
                # 1. Determine Trend BEFORE (At Entry)
                # We use the slope of the 50SMA or the Regime calculated by StrategyEngine
                trend_pre = "UNKNOWN"
                if df_snapshot is not None and not df_snapshot.empty:
                    last = df_snapshot.iloc[-1]
                    # Logic: If Close > SMA50 -> UP, Else DOWN
                    sma50 = last.get('SMA_50', last['close'])
                    if last['close'] > sma50:
                        trend_pre = "UP"
                    else:
                        trend_pre = "DOWN"
                
                # 2. Determine Trend AFTER (At Exit)
                # Only relevant if we are logging a closed trade
                trend_post = "PENDING"
                trend_success = 0 # Default to 0 until proven correct
                
                if status in ["EXECUTED", "CLOSED"]:
                    # If we closed with profit -> The trend prediction was likely correct
                    if pnl > 0:
                        trend_post = trend_pre # Confirmed
                        trend_success = 1      # SUCCESS (100%)
                    else:
                        # Reversal occurred
                        trend_post = "REVERSAL" if trend_pre == "UP" else "BOUNCE"
                        trend_success = 0      # FAILURE (0%)

                # 3. Risk/Reward Calculation
                entry = ticket.get('limit_price', 0)
                stop = ticket.get('stop_loss', 0)
                target = ticket.get('target_price', 0)
                
                risk = abs(entry - stop)
                reward = abs(target - entry)
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                
                # # --- WRITE ROW ---
                # writer.writerow([
                #     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                #     ticket.get('symbol'),
                #     ticket.get('action'),
                #     f"{ticket.get('master_score', 0):.1f}",
                #     f"{ticket.get('tech_score', 0):.1f}",
                #     f"{ticket.get('ai_score', 0):.1f}",
                #     "|".join(ticket.get('setups_found', [])),
                #     trend_pre,          # Column 1: Trend Before
                #     trend_post,         # Column 2: Trend After
                #     trend_success,      # Column 3: Success Boolean (for averaging in Excel)
                #     f"{entry:.2f}",
                #     f"{stop:.2f}",
                #     f"{target:.2f}",
                #     rr_ratio,
                #     status,
                #     f"{exec_price:.2f}",
                #     f"{exit_price:.2f}",
                #     f"{pnl:.2f}%"
                # ])
                # Extract Nested Scores
                scores = ticket.get('scores', {})
                master = scores.get('master', ticket.get('master_score', 0.0))
                tech = scores.get('tech', ticket.get('tech_score', 0.0))
                ai = scores.get('ai', ticket.get('ai_score', 0.0))

                # --- WRITE ROW ---
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ticket.get('symbol'),
                    ticket.get('action'),
                    f"{master:.1f}",
                    f"{tech:.1f}",
                    f"{ai:.1f}",
                    "|".join(ticket.get('setups_found', [])),
                    trend_pre,          
                    trend_post,         
                    trend_success,      
                    f"{entry:.2f}",
                    f"{stop:.2f}",
                    f"{target:.2f}",
                    rr_ratio,
                    status,
                    f"{exec_price:.2f}",
                    f"{exit_price:.2f}",
                    f"{pnl:.2f}%"
                ])
            
        except Exception as e:
            logger.error(f"Failed to write to Trade Journal: {e}")


class Notifier:
    """
    The Alert Bridge.
    Ensures the human operator knows exactly when to execute a trade.
    """
    @staticmethod
    def trigger_alert(message):
        # 1. Loud Console Output
        print("\n" + "="*60)
        print(f"!!! TRADE ALERT !!!\n{message}")
        print("="*60 + "\n")
        
        # 2. Telegram Output (If configured)
        token = cfg.TELEGRAM_TOKEN
        chat_id = cfg.TELEGRAM_CHAT_ID
        if token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {"chat_id": chat_id, "text": f"🤖 STOCKWISE:\n{message}"}
                requests.post(url, json=payload, timeout=5)
            except Exception as e:
                logger.debug(f"Telegram notification failed: {e}")


class LifecycleManager:
    """
    AGENT 4: The Lifecycle Manager.
    Responsibility: Manage open positions. 
    1. Ratchet Kinetic Trailing Stops based on profit thresholds.
    2. Execute the 'Zombie Protocol' for Orphaned Trades.
    """
    def __init__(self):
        self.stop_cfg = cfg.KINETIC_STOP_CONFIG
        self.defense_cfg = cfg.PORTFOLIO_DEFENSE
        
    def manage_kinetic_stop(self, symbol, position, current_price, current_atr):
        """
        The Asymptotic Acceleration Curve.
        As the trade becomes more profitable, the stop-loss chokes the price 
        tighter to mathematically guarantee we capture the alpha.
        """
        entry_price = position.get("entry_price", current_price)
        current_stop = position.get("stop_loss", current_price - current_atr)
        
        # Track the highest high the stock has reached since we bought it
        highest_high = max(position.get("highest_high", entry_price), current_price)
        
        # Calculate raw percentage profit (ignoring fees for the trigger thresholds)
        profit_pct = (current_price - entry_price) / entry_price
        
        new_stop = current_stop
        phase = "PHASE_1_BREATHING"

        # --- PHASE 4: RUNNER MODE (Let Winners Run) ---
        # When a position hits its original target, we don't sell.
        # Instead we trail ultra-tight with a minimum distance floor.
        if position.get("runner_mode"):
            milestone_cfg = getattr(cfg, 'MILESTONE_ALERT_CONFIG', {})
            runner_atr_mult = milestone_cfg.get('runner_atr_mult', 0.5)
            runner_min_dist = milestone_cfg.get('runner_min_distance_pct', 0.008)

            # ATR-based runner stop
            runner_stop_atr = highest_high - (current_atr * runner_atr_mult)
            # Floor-based runner stop (prevents noise exit when ATR is tiny)
            runner_stop_floor = highest_high * (1 - runner_min_dist)
            # Use the LOWER of the two (= wider stop = more protection from noise)
            runner_stop = min(runner_stop_atr, runner_stop_floor)

            new_stop = max(current_stop, runner_stop)
            phase = "PHASE_4_RUNNER"

        # --- STORY: The Parabolic Choke (Phase 3) ---
        # If the stock has exploded +3.0% into profit, we do not want to give it back.
        # We shrink the stop to just 1.0 ATR from the highest peak.
        # If it blinks, we take the money and run.
        elif profit_pct >= self.stop_cfg["phase3_parabolic_trigger_pct"]:
            choke_stop = highest_high - (current_atr * self.stop_cfg["phase3_atr_mult"])
            new_stop = max(current_stop, choke_stop)
            phase = "PHASE_3_PARABOLIC"
            
        # --- STORY: The Breakeven Ratchet (Phase 2) ---
        # If the stock hits +1.5% profit, we have enough room to cover our taxes and fees.
        # We instantly pull the stop loss up to our entry price. The trade is now "Risk Free".
        elif profit_pct >= self.stop_cfg["phase2_breakeven_trigger_pct"]:
            breakeven_stop = entry_price * (1 + cfg.COSTS_CONFIG["slippage_pct"])
            new_stop = max(current_stop, breakeven_stop)
            phase = "PHASE_2_BREAKEVEN"
            
        # --- STORY: The Breathing Room (Phase 1) ---
        # The trade is new. It needs room to bounce around without getting stopped out by HFT noise.
        # We use a wide 2.0 ATR trailing stop.
        else:
            trail_stop = highest_high - (current_atr * self.stop_cfg["phase1_atr_mult"])
            new_stop = max(current_stop, trail_stop) # 'max' ensures the stop only moves UP, never down.

        logger.debug(f"[{symbol}] Kinetic Stop Math -> Pnl: {profit_pct:.2%}, State: {phase}, Old Stop: {current_stop:.2f}, New Stop: {new_stop:.2f}")
        
        return new_stop, highest_high

    def check_zombie_protocol(self, symbol, position, current_regime):
        """
        The Orphaned Trade Protocol.
        If we bought NVDA because it was in a "TREND", but today Agent 1 says NVDA is "CHOP",
        the fundamental reason we entered the trade is dead. It is now a Zombie.
        """
        entry_regime = position.get("entry_regime", current_regime)
        
        # 1. Identify Mismatch
        if entry_regime != current_regime:
            # The trade is an orphan. Tag it with a death timer.
            if "zombie_timestamp" not in position:
                logger.info(f"[{symbol}] REGIME SHIFT DETECTED ({entry_regime} -> {current_regime}). Trade declared ZOMBIE. TTL initiated.")
                position["zombie_timestamp"] = datetime.now().isoformat()
            
            # 2. Check Time-To-Live (TTL)
            zombie_time = datetime.fromisoformat(position["zombie_timestamp"])
            hours_alive = (datetime.now() - zombie_time).total_seconds() / 3600
            
            if hours_alive >= self.defense_cfg["zombie_trade_ttl_hours"]:
                logger.info(f"[{symbol}] ZOMBIE TTL EXPIRED ({hours_alive:.1f} hours). Initiating Force Liquidation.")
                return True # True = Force Liquidate Now
        
        # Trade is either healthy or still within its 72-hour grace period
        return False


class LiveTradingEngine:
    """
    The Execution Gateway.
    Reads open positions, updates stops via Agent 4, and executes new tickets from the Conductor.
    """
    def __init__(self, broker_api=None):
        self.broker = broker_api # This will connect to IBKR/Alpaca later
        self.lifecycle = LifecycleManager()
        self.notifier = NotificationManager() # Initialize the Voice
        
        # We use a stateful JSON file to track open positions independently of the broker API
        self.positions_file = os.path.join(cfg.DB_DIR, "open_positions.json")
        self.positions = self._load_json(self.positions_file)

    def _process_closed_position(self, ticker, buy_date, buy_price, sell_price):
        """
        [Lifecycle Resolution]
        Calculates net PnL dynamically and triggers the Gen-13 Notification 
        and centralized CSV logging immediately upon position liquidation.
        """
        try:
            if buy_price and float(buy_price) > 0:
                pnl_net = ((float(sell_price) - float(buy_price)) / float(buy_price)) * 100.0
            else:
                pnl_net = 0.0
                
            # Call the updated Notification Manager
            if hasattr(self, 'notifier') and self.notifier:
                self.notifier.send_closed_position_report(
                    ticker=ticker,
                    buy_date=buy_date,
                    buy_price=float(buy_price),
                    sell_price=float(sell_price),
                    pnl_net=pnl_net
                )
                
            logger.info(f"Position Closed Processed: [{ticker}] PnL: {pnl_net:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to process closed position for {ticker}: {str(e)}")

    
    def _load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load positions JSON: {e}")
        return {}

    def _save_json(self, data, path):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save positions JSON: {e}")

    def _write_cooldown(self, symbol, reason="STOP_LOSS_HIT"):
        """
        [Bug 1.4 Fix] Writes a ticker to cooldown_list.json when stop-loss fires.
        The strategy engine reads this file to blacklist the ticker.
        """
        cooldown_path = getattr(cfg, 'COOLDOWN_FILE_PATH', 'data/cooldown_list.json')
        cooldown_hours = getattr(cfg, 'COOLDOWN_PERIOD_HOURS', 24)
        try:
            cooldown_data = {}
            if os.path.exists(cooldown_path):
                with open(cooldown_path, 'r', encoding='utf-8') as f:
                    cooldown_data = json.load(f)

            cooldown_data[symbol] = {
                "timestamp": time.time(),
                "reason": reason,
                "cooldown_hours": cooldown_hours
            }

            os.makedirs(os.path.dirname(cooldown_path) or '.', exist_ok=True)
            with open(cooldown_path, 'w', encoding='utf-8') as f:
                json.dump(cooldown_data, f, indent=4)
            logger.info(f"[{symbol}] Added to cooldown blacklist for {cooldown_hours}h. Reason: {reason}")
        except Exception as e:
            logger.error(f"[{symbol}] Failed to write cooldown: {e}")

    def _calculate_real_breakeven(self, entry_price, qty):
        """
        [Phase 2.5] Calculates the TRUE breakeven price including all round-trip costs.
        The user should not move their stop to breakeven until profit exceeds this.

        Costs: entry commission + exit commission + slippage (both ways) + tax on gain.
        Formula: breakeven = entry_price + total_costs_per_share / (1 - tax_rate)
        The division by (1-tax) accounts for the fact that the gain itself is taxed.
        """
        costs = cfg.COSTS_CONFIG
        milestone_cfg = getattr(cfg, 'MILESTONE_ALERT_CONFIG', {})

        commission_per_share = costs.get('commission_per_share', 0.005)
        min_commission = costs.get('min_commission', 1.00)
        slippage_pct = costs.get('slippage_pct', 0.001)
        tax_rate = costs.get('tax_rate', 0.25)
        buffer_pct = milestone_cfg.get('safe_zone_buffer_pct', 0.002)

        # Round-trip commissions
        entry_commission = max(qty * commission_per_share, min_commission)
        exit_commission = max(qty * commission_per_share, min_commission)
        total_commission = entry_commission + exit_commission

        # Round-trip slippage
        total_slippage = entry_price * slippage_pct * 2  # both entry and exit

        # Total cost per share
        cost_per_share = (total_commission / max(qty, 1)) + total_slippage

        # Breakeven must cover costs AFTER tax: gain / (1 - tax_rate) = cost_per_share
        breakeven_gain_per_share = cost_per_share / (1 - tax_rate) if tax_rate < 1 else cost_per_share

        # Add safety buffer
        buffer_per_share = entry_price * buffer_pct

        breakeven_price = entry_price + breakeven_gain_per_share + buffer_per_share

        logger.debug(f"Real breakeven for {qty} shares @ ${entry_price:.2f}: "
                     f"${breakeven_price:.2f} (costs: ${cost_per_share:.4f}/share, "
                     f"buffer: {buffer_pct:.1%})")

        return round(breakeven_price, 2)

    def _check_and_send_milestone_alert(self, symbol, position, new_stop, current_price):
        """
        [Phase 2.5] Event-driven milestone alerts.
        Sends a Telegram notification when the recommended stop-loss changes significantly.

        Logic:
        1. First alert: only after real breakeven is reached (covers commissions + tax)
        2. Subsequent alerts: when stop changes by > min_stop_change_pct
        3. Cooldown: minimum min_alert_interval_minutes between alerts
        """
        milestone_cfg = getattr(cfg, 'MILESTONE_ALERT_CONFIG', {})
        min_change_pct = milestone_cfg.get('min_stop_change_pct', 0.01)
        min_interval = milestone_cfg.get('min_alert_interval_minutes', 15)

        entry_price = position.get('entry_price', current_price)
        qty = position.get('qty', 10)

        # Calculate real breakeven (first time only, then cache in position)
        if 'real_breakeven' not in position:
            position['real_breakeven'] = self._calculate_real_breakeven(entry_price, qty)

        real_breakeven = position['real_breakeven']

        # Gate 1: Don't send any alerts until price is above real breakeven
        if current_price < real_breakeven and not position.get('breakeven_alerted'):
            return

        # Gate 2: First alert -- breakeven reached for the first time
        if current_price >= real_breakeven and not position.get('breakeven_alerted'):
            position['breakeven_alerted'] = True
            position['last_alerted_stop'] = new_stop
            position['last_alert_time'] = time.time()

            profit_locked_pct = ((new_stop - entry_price) / entry_price) * 100
            msg = (f"**{symbol}: SAFE ZONE REACHED**\n"
                   f"Price: ${current_price:.2f}\n"
                   f"Recommended Stop: ${new_stop:.2f}\n"
                   f"Breakeven: ${real_breakeven:.2f}\n"
                   f"From here you don't lose a cent.\n"
                   f"Locked P&L: {profit_locked_pct:+.1f}%")

            if hasattr(self, 'notifier') and self.notifier:
                self.notifier.send_message(msg)
            logger.info(f"[{symbol}] MILESTONE: Safe Zone reached. Stop: ${new_stop:.2f}")
            return

        # Gate 3: Check cooldown timer
        last_alert_time = position.get('last_alert_time', 0)
        if (time.time() - last_alert_time) < (min_interval * 60):
            return

        # Gate 4: Check if stop change is significant enough
        last_alerted_stop = position.get('last_alerted_stop', entry_price)
        stop_change_pct = abs(new_stop - last_alerted_stop) / current_price

        if stop_change_pct < min_change_pct:
            return

        # All gates passed -- send milestone alert
        position['last_alerted_stop'] = new_stop
        position['last_alert_time'] = time.time()

        profit_pct = ((current_price - entry_price) / entry_price) * 100
        locked_pct = ((new_stop - entry_price) / entry_price) * 100

        # Determine alert level based on profit
        if position.get('runner_mode'):
            phase_label = "RUNNER MODE"
        elif profit_pct >= 10:
            phase_label = "EXTENDED RUN"
        elif profit_pct >= 5:
            phase_label = "STRONG PROFIT"
        else:
            phase_label = "TRAILING UP"

        msg = (f"**{symbol}: {phase_label}**\n"
               f"Current: ${current_price:.2f} ({profit_pct:+.1f}%)\n"
               f"Recommended Stop: ${new_stop:.2f}\n"
               f"Locked Profit: {locked_pct:+.1f}%\n"
               f"Move your stop-loss to ${new_stop:.2f}")

        if hasattr(self, 'notifier') and self.notifier:
            self.notifier.send_message(msg)
        logger.info(f"[{symbol}] MILESTONE: {phase_label}. Stop: ${new_stop:.2f}, Locked: {locked_pct:+.1f}%")

    def execute_ticket(self, ticket, current_regime):
        """
        Receives the "BUY" ticket from the StrategyEngine (Conductor).
        Secures the entry price and hands it to Agent 4 to manage.
        """
        symbol = ticket["symbol"]
        
        if symbol in self.positions:
            logger.debug(f"[{symbol}] Ticket rejected. Position already exists.")
            return False
            
        # --- STORY: The Execution Wrapper ---
        # In production, we fire the API call to Alpaca/IBKR here.
        # For our logic loop, we register the trade in the memory bank so Agent 4 can wake up.
        
        self.positions[symbol] = {
            "entry_price": ticket["limit_price"],
            "qty": ticket["qty"],
            "stop_loss": ticket["stop_loss"],
            "take_profit": ticket["take_profit"],
            "highest_high": ticket["limit_price"],
            "entry_regime": current_regime,
            "entry_time": datetime.now().isoformat()
        }
        
        self._save_json(self.positions, self.positions_file)
        
        # --- NOTIFICATION ---
        # --- NEW: Send Notification ---
        msg = (f"🟢 **BUY SIGNAL DETECTED: {symbol}**\n"
               f"Qty: {ticket['qty']} @ ${ticket['limit_price']:.2f}\n"
               f"Stop: ${ticket['stop_loss']:.2f}\n"
               f"Target: ${ticket['take_profit']:.2f}\n"
               f"Regime: {current_regime}")
        
        if hasattr(self, 'notifier') and self.notifier:
            self.notifier.send_message(msg)

        logger.info(f"[{symbol}] Order Executed. Handed to Agent 4.")
        
        # Return a dictionary mimicking a standard Broker API JSON response
        return {"status": "FILLED", "exec_price": ticket["limit_price"]}

    def manage_open_positions(self, market_data, agent1_router):
        if not self.positions:
            logger.debug("No open positions to manage.")
            return

        liquidated = []

        for symbol, position in self.positions.items():
            try:
                df = market_data.get_stock_data(symbol, days_back=20)
                if df is None or df.empty: continue
                    
                last = df.iloc[-1]
                current_price = last['close']
                current_atr = last.get('atr', current_price * 0.01)
                current_regime = agent1_router.classify_regime(df)

                # --- CHECK STOPS ---
                reason = None
                if current_price <= position["stop_loss"]:
                    reason = "STOP LOSS HIT"
                elif current_price >= position["take_profit"] and not position.get("runner_mode"):
                    # [Phase 2.5b] Don't sell at target -- activate Runner Mode instead
                    position["runner_mode"] = True
                    position["runner_activated_at"] = current_price
                    position["runner_activated_time"] = time.time()
                    logger.info(f"[{symbol}] TARGET REACHED at ${current_price:.2f} -- Runner Mode ACTIVATED")
                    # Send immediate notification
                    if hasattr(self, 'notifier') and self.notifier:
                        entry = position.get('entry_price', 0)
                        gain_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
                        self.notifier.send_message(
                            f"**{symbol}: TARGET REACHED -- RUNNER MODE**\n"
                            f"Entry: ${entry:.2f} | Current: ${current_price:.2f} ({gain_pct:+.1f}%)\n"
                            f"Target was ${position['take_profit']:.2f} -- NOT selling.\n"
                            f"Trailing stop tightened. Let it run!")
                elif self.lifecycle.check_zombie_protocol(symbol, position, current_regime):
                    reason = "ZOMBIE PROTOCOL (Time Expired)"

                if reason:
                    # Execute Sale
                    logger.info(f"[{symbol}] LIQUIDATION: {reason} at {current_price:.2f}")
                    # Bug 1.4 Fix: Blacklist ticker on stop-loss or zombie exit
                    if reason in ("STOP LOSS HIT", "ZOMBIE PROTOCOL (Time Expired)"):
                        self._write_cooldown(symbol, reason=reason)
                    
                    # --- Send Original Execution Reason Notification ---
                    msg = (f"**SELL SIGNAL: {symbol}**\n"
                           f"Reason: {reason}\n"
                           f"Exit Price: ${current_price:.2f}\n"
                           f"PnL: {((current_price - position['entry_price'])/position['entry_price']):.2%}")
                    
                    if hasattr(self, 'notifier') and self.notifier:
                        self.notifier.send_message(msg)
                    
                    # --- GEN-13: Process Position Closure (CSV Logging + Telegram PnL Report) ---
                    # Extract entry time safely, clean ISO format ("2025-02-05T14:30:00" -> "2025-02-05")
                    buy_date_raw = position.get("entry_time", "UNKNOWN")
                    
                    # Architectural Fix: Extract date part from ISO timestamp string
                    buy_date_clean = buy_date_raw.split("T")[0] if "T" in buy_date_raw else buy_date_raw
                    
                    self._process_closed_position(
                        ticker=symbol,
                        buy_date=buy_date_clean,
                        buy_price=position["entry_price"],
                        sell_price=current_price
                    )
                    liquidated.append(symbol)
                    continue

                # 4. Agent 4: Kinetic Trailing Stop
                new_stop, new_high = self.lifecycle.manage_kinetic_stop(symbol, position, current_price, current_atr)
                
                if new_stop > position["stop_loss"]:
                    old_stop = position["stop_loss"]
                    position["stop_loss"] = new_stop
                    position["highest_high"] = new_high
                    logger.info(f"[{symbol}] Kinetic Stop tightened: {old_stop:.2f} -> {new_stop:.2f}")

                    # [Phase 2.5b] Check if this stop change warrants a user alert
                    self._check_and_send_milestone_alert(symbol, position, new_stop, current_price)

            except Exception as e:
                logger.error(f"[{symbol}] Error managing position: {e}", exc_info=True)

        for sym in liquidated:
            del self.positions[sym]
            
        if liquidated:
            self._save_json(self.positions, self.positions_file)


if __name__ == "__main__":
    from data_source_manager import DataSourceManager
    from strategy_engine import StrategyEngine
    from stock_hunter import StockHunter
    from template_matcher import TemplateMatcher
    
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, default="1m", help="Data interval (1m, 5m, 1h)")
    parser.add_argument("--mode", type=str, default="PAPER", help="Trading mode (PAPER/LIVE)") # <--- הוסף את השורה הזו
    args = parser.parse_args()

    logger.info("=== STOCKWISE LIVE ENGINE: INITIALIZING ===")
    
    # 1. Initialize Core Systems
    # market_data: Gateway to Alpaca/Polygon
    # orchestra: The Strategy Brain (Tech + AI)
    # scout: The Nightly Scanner Interface
    market_data = DataSourceManager()
    orchestra = StrategyEngine()
    scout = StockHunter(market_data)
    
    # Initialize Execution Engine (Agent 4)
    live_engine = LiveTradingEngine()
    
    # Initialize Statistics Recorder (The Black Box)
    journal = TradeJournal()

    # Initialize Template Pipeline (Phase 3.8)
    matcher = TemplateMatcher()
    logger.info(f"Template Pipeline loaded: {len(matcher.tm.templates)} templates, "
                f"mode={getattr(cfg, 'SIGNAL_PIPELINE_MODE', 'legacy')}")

    logger.info("ENGINE START: Dynamic Heartbeat Protocol Initiated.")

    # EOD Tracker Initialization
    last_eod_date = None
        
    # Veto Cooldown Cache: Prevents the system from scanning the same rejected stock repeatedly
    cooldown_cache = {}
    COOLDOWN_MINUTES = 30
    
    # --- THE MAIN INFINITE LOOP ---
    while True:
        try:
            # 2. Refresh VIP Target List
            vip_list = scout.get_active_vip_watchlist()
            
            if not vip_list:
                logger.warning("VIP List is empty! Please run stock_hunter.py. Sleeping 60s...")
                time.sleep(60)
                continue
                
            logger.info(f"Loaded {len(vip_list)} VIP targets. Starting Cycle...")

            # State variables for Dynamic Heartbeat Logic
            highest_score_detected = 0.0
            open_positions_count = 0
            
            # --- PHASE 1: DEFENSE (Manage Open Positions) ---
            try:
                # positions = live_engine.api.list_positions()
                # open_positions_count = len(positions)

                # In Shadow Mode, we check the internal dictionary, not an external API
                open_positions_count = len(live_engine.positions)

                if open_positions_count > 0:
                    logger.info(f"Managing {open_positions_count} open positions...")
                    live_engine.manage_open_positions(market_data, orchestra.router)
            except Exception as e:
                logger.error(f"Error checking positions: {e}")

            # --- PHASE 2: OFFENSE (Scan for Opportunities) ---
            for symbol in vip_list:
                # 1. Check Cooldown Cache (Amnesia Loop Prevention)
                if symbol in cooldown_cache:
                    if datetime.now() < cooldown_cache[symbol]:
                        continue # Stock is still in timeout
                    else:
                        del cooldown_cache[symbol] # Cooldown expired, we can checking it again

                try:
                    # Fetch Candle Data (730 days back to guarantee deep history)
                    df = market_data.get_stock_data(symbol, days_back=730, interval=args.interval)

                    if df is not None and not df.empty:
                        # Architectural Fix: ROW GATEKEEPER (Live Engine)
                        if len(df) < 100:
                            logger.warning(f"[{symbol}] Row Gatekeeper Veto: Only {len(df)} rows. Skipping to prevent AI crash.")
                            cooldown_mins = getattr(cfg, 'DATA_STARVATION_COOLDOWN_MINUTES', 120)
                            cooldown_cache[symbol] = datetime.now() + timedelta(minutes=cooldown_mins)
                            continue

                        # --- SIGNAL GENERATION (Pipeline Mode) ---
                        pipeline_mode = getattr(cfg, 'SIGNAL_PIPELINE_MODE', 'legacy')

                        if pipeline_mode in ('templates', 'dual'):
                            # New template-based pipeline
                            # Load stock state from scanner ledger
                            ledger_state = {}
                            try:
                                ledger_path = os.path.join(cfg.DB_DIR, "scan_ledger.json")
                                if os.path.exists(ledger_path):
                                    with open(ledger_path, 'r') as f:
                                        full_ledger = json.load(f)
                                    ledger_state = full_ledger.get(symbol, {}).get('state', {})
                            except Exception as e:
                                logger.debug(f"[{symbol}] Could not load ledger state: {e}")

                            # Calculate features for template evaluation
                            from feature_engine import FeatureEngine
                            fe = FeatureEngine()
                            df_features = fe.calculate_features(df, strategy_config={"active_indicators": ["all"]})

                            # Run template matcher
                            signals = matcher.scan_ticker(symbol, df_features, stock_state=ledger_state)

                            if signals:
                                # Use the best signal (highest confidence)
                                best = signals[0]
                                score = best.get('confidence_score', 0)

                                # Track best score for dynamic sleep
                                if score > highest_score_detected:
                                    highest_score_detected = score

                                # Build ticket compatible with existing execution flow
                                ticket = {
                                    "symbol": symbol,
                                    "action": "BUY",
                                    "master_score": score,
                                    "limit_price": best['entry_price'],
                                    "stop_loss": best['stop_loss'],
                                    "take_profit": best['take_profit'],
                                    "qty": 10,  # Default, will be overridden by RiskActuary
                                    "template_id": best['template_id'],
                                    "template_name": best['template_name'],
                                    "confidence_score": best['confidence_score'],
                                    "risk_reward_ratio": best['risk_reward_ratio'],
                                    "use_runner_mode": best.get('use_runner_mode', False),
                                    "conditions_detail": best.get('conditions_detail', []),
                                    "stock_state": ledger_state,
                                }

                                # Send detailed Telegram alert
                                state_str = " | ".join(f"{k}:{v}" for k, v in ledger_state.items()) if ledger_state else "N/A"
                                blocks_str = ", ".join(d.get('block', '?') for d in best.get('conditions_detail', []))

                                alert_msg = (
                                    f"**BUY SIGNAL: {symbol}**\n"
                                    f"Template: {best['template_name']}\n"
                                    f"Confidence: {best['confidence_score']}%\n"
                                    f"Entry: ${best['entry_price']:.2f}\n"
                                    f"Stop Loss: ${best['stop_loss']:.2f} ({best.get('risk_pct', 0):.1f}%)\n"
                                    f"Take Profit: ${best['take_profit']:.2f} ({best.get('reward_pct', 0):.1f}%)\n"
                                    f"R:R: {best['risk_reward_ratio']:.1f}\n"
                                    f"Runner Mode: {'Yes' if best.get('use_runner_mode') else 'No'}\n"
                                    f"Blocks: [{blocks_str}]\n"
                                    f"State: {state_str}"
                                )

                                if hasattr(live_engine, 'notifier') and live_engine.notifier:
                                    live_engine.notifier.send_message(alert_msg)

                                logger.info(f"[{symbol}] TEMPLATE SIGNAL: {best['template_name']} | "
                                            f"Conf: {score:.0f}% | Entry: ${best['entry_price']:.2f}")

                                # Journal logging
                                journal.log_signal(ticket, df_snapshot=df, status="SIGNAL_DETECTED")

                                # Execute if in auto mode
                                try:
                                    current_regime = orchestra.router.classify_regime(df_features)
                                    result = live_engine.execute_ticket(ticket, current_regime)
                                    if result and result.get('status') == 'FILLED':
                                        journal.log_signal(ticket, df_snapshot=df, status="EXECUTED", exec_price=ticket['limit_price'])
                                    else:
                                        journal.log_signal(ticket, df_snapshot=df, status="REJECTED_BROKER")
                                except Exception as exec_error:
                                    logger.error(f"Execution Failed: {exec_error}")
                                    journal.log_signal(ticket, df_snapshot=df, status="ERROR_EXECUTION")
                            else:
                                # No template signals
                                cooldown_mins = getattr(cfg, 'VETO_COOLDOWN_MINUTES', 30)
                                cooldown_cache[symbol] = datetime.now() + timedelta(minutes=cooldown_mins)

                        if pipeline_mode in ('legacy', 'dual'):
                            # Original pipeline (kept for comparison/fallback)
                            ticket = orchestra.evaluate_ticker(symbol, df)
                            score = ticket.get('master_score', 0.0)

                            if score > highest_score_detected:
                                highest_score_detected = score

                            if pipeline_mode == 'dual':
                                logger.info(f"[{symbol}] DUAL MODE — Legacy score: {score:.1f}")

                            if pipeline_mode == 'legacy':
                                if ticket.get("action") == "BUY":
                                    current_regime = orchestra.router.classify_regime(df)
                                    journal.log_signal(ticket, df_snapshot=df, status="SIGNAL_DETECTED")
                                    try:
                                        result = live_engine.execute_ticket(ticket, current_regime)
                                        if result and result.get('status') == 'FILLED':
                                            journal.log_signal(ticket, df_snapshot=df, status="EXECUTED", exec_price=ticket['limit_price'])
                                    except Exception as exec_error:
                                        logger.error(f"Execution Failed: {exec_error}")
                                else:
                                    cooldown_mins = getattr(cfg, 'VETO_COOLDOWN_MINUTES', 30)
                                    cooldown_cache[symbol] = datetime.now() + timedelta(minutes=cooldown_mins)
                            
                except Exception as inner_e:
                    logger.error(f"Error processing {symbol}: {inner_e}")
                    continue

            # --- PHASE 3: DYNAMIC HEARTBEAT (Smart Sleep) ---
            # Adjust sleep time based on market activity to optimize API usage.
            
            sleep_time = 60 # Default: Patrol Mode
            status_msg = "PATROL MODE"

            if open_positions_count > 0:
                # Money at risk -> Fast Polling (15s)
                sleep_time = 15
                status_msg = f"COMBAT MODE ({open_positions_count} Pos Open)"
                
            elif highest_score_detected >= 50.0:
                # High Potential -> Medium Polling (30s)
                sleep_time = 30
                status_msg = f"STALKING MODE (Best Score: {highest_score_detected:.1f})"
            
            else:
                # Quiet Market -> Slow Polling (60s)
                sleep_time = 60
                status_msg = "PATROL MODE (Scanning...)"

            logger.info(f"{status_msg} | Sleeping {sleep_time}s...")
            time.sleep(sleep_time)

            # --- PHASE 4: END OF DAY (EOD) CRON JOB ---
            current_time = datetime.now()
            
            # Fire at 23:00 IST (US Market Close) or later, ensuring it only fires once per day.
            if current_time.hour >= 23 and current_time.date() != last_eod_date:
                # Lock the trigger IMMEDIATELY before any operations.
                # This guarantees the report never fires twice even if the notification crashes.
                last_eod_date = current_time.date()

                logger.info("Triggering End of Day (EOD) Report...")
                try:
                    # Dynamically read today's setups from the Trade Journal
                    today_signals = 0
                    journal_path = os.path.join(cfg.BASE_DIR, "StockWise_Trade_Journal.csv")
                    if os.path.exists(journal_path):
                        import pandas as pd
                        df_j = pd.read_csv(journal_path)
                        # Check if Timestamp column exists and count today's rows
                        if 'Timestamp' in df_j.columns:
                            df_j['Timestamp'] = pd.to_datetime(df_j['Timestamp'], errors='coerce')
                            
                            # 1. Count Total Suggested Setups (System Signals)
                            today_mask = (df_j['Timestamp'].dt.date == current_time.date()) & (df_j['Status'] == 'SIGNAL_DETECTED')
                            today_signals = len(df_j[today_mask])
                            
                            # 2. Count Actual Executed Trades by the User (Gen-13 Feedback Loop)
                            executed_mask = (df_j['Timestamp'].dt.date == current_time.date()) & (df_j['Status'] == getattr(cfg, 'TRADE_STATUS_EXECUTED', 'CONFIRMED'))
                            user_taken = len(df_j[executed_mask])

                            # Fire the notification with exact suggestion vs execution counts
                            live_engine.notifier.send_eod_summary(
                                date_str=current_time.strftime("%Y-%m-%d"),
                                portfolio_status="Active", 
                                system_pnl="0.0%",  # Update with actual PnL var if available
                                win_rate="0.0%",    # Update with actual Win Rate var if available
                                active_shadow=str(len(live_engine.positions)), 
                                setups_found=today_signals,
                                user_taken=user_taken
                            )
                        else:
                            raise ValueError("Timestamp column missing in journal.")
                    else:
                        # Architectural Fix: Properly indented 'else' block
                        # Fire the notification
                        live_engine.notifier.send_eod_summary(
                            date_str=current_time.strftime("%Y-%m-%d"),
                            portfolio_status="No Active Live Trades.",
                            system_pnl=0.00,  # Will be wired to PnL calculator later
                            win_rate=0.0,
                            active_shadow=0,
                            setups_found=0,
                            user_taken=0
                        )
                    
                    # Lock the trigger so it doesn't fire again until tomorrow
                    last_eod_date = current_time.date()
                    logger.info("EOD Report successfully broadcasted.")
                    
                except Exception as e:
                    logger.error(f"Failed to generate EOD Report: {e}")

        except KeyboardInterrupt:
            logger.info("Engine manually stopped by User.")
            break
        except Exception as e:
            logger.critical(f"CRITICAL ENGINE FAILURE: {e}")
            time.sleep(60) # Fail-safe pause