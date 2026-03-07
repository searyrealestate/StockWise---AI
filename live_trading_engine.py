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

        # --- STORY: The Parabolic Choke (Phase 3) ---
        # If the stock has exploded +3.0% into profit, we do not want to give it back.
        # We shrink the stop to just 1.0 ATR from the highest peak. 
        # If it blinks, we take the money and run.
        if profit_pct >= self.stop_cfg["phase3_parabolic_trigger_pct"]:
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
                elif current_price >= position["take_profit"]:
                    reason = "TAKE PROFIT HIT"
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
                    
                    # Optional: Notify on Stop Adjustment (Reduce noise by only logging)
                    logger.info(f"[{symbol}] Kinetic Stop tightened: {old_stop:.2f} -> {new_stop:.2f}")

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

                        # Full Analysis: Technical Setups + AI
                        ticket = orchestra.evaluate_ticker(symbol, df)
                        score = ticket.get('master_score', 0.0)

                        # Track best score for dynamic sleep
                        if score > highest_score_detected:
                            highest_score_detected = score

                        # --- JOURNALING & EXECUTION LOGIC ---
                        if ticket.get("action") == "BUY":
                            current_regime = orchestra.router.classify_regime(df)
                            logger.info(f"Signal Detected: {symbol} (Score: {score}). Logging to Journal...")
                            
                            journal.log_signal(ticket, df_snapshot=df, status="SIGNAL_DETECTED")
                            try:
                                # Attempt Execution
                                result = live_engine.execute_ticket(ticket, current_regime)
                                if result and result.get('status') == 'FILLED':
                                    journal.log_signal(ticket, df_snapshot=df, status="EXECUTED", exec_price=ticket['limit_price'])
                                else:
                                    journal.log_signal(ticket, df_snapshot=df, status="REJECTED_BROKER")
                            except Exception as exec_error:
                                logger.error(f"Execution Failed: {exec_error}")
                                journal.log_signal(ticket, df_snapshot=df, status="ERROR_EXECUTION")
                        else:
                            # Architectural Fix: Amnesia Loop Prevention (Dynamic Veto Cooldown)
                            cooldown_mins = getattr(cfg, 'VETO_COOLDOWN_MINUTES', 30)
                            logger.debug(f"[{symbol}] Trade Vetoed by Strategy Engine. Initiating {cooldown_mins}m cooldown.")
                            cooldown_cache[symbol] = datetime.now() + timedelta(minutes=cooldown_mins)
                            continue
                            
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