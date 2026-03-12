# portfolio_risk.py

"""
StockWise Gen-13 Portfolio Risk Manager
=======================================
Three pre-entry gates:
1. Correlation Gate: Blocks if new stock correlates > 80% with existing positions
2. Drawdown Gate: Blocks all new entries if portfolio drawdown exceeds limit
3. Weekly Trend Gate: Blocks if weekly trend is bearish

Called by live_trading_engine BEFORE execute_ticket().
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
import system_config as cfg

logger = logging.getLogger("PortfolioRisk")

# Sector mapping for common stocks (expandable)
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer", "META": "Technology", "NVDA": "Technology",
    "TSLA": "Consumer", "AMD": "Technology", "NFLX": "Communication",
    "SPY": "Index", "QQQ": "Index", "DIS": "Communication",
    "JPM": "Financial", "BAC": "Financial", "GS": "Financial",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "WMT": "Consumer", "COST": "Consumer", "HD": "Consumer",
}


class PortfolioRiskManager:
    """
    Pre-entry risk gates for portfolio-level protection.
    """

    def __init__(self):
        self.config = getattr(cfg, 'PORTFOLIO_RISK_CONFIG', {})
        self.portfolio_high_water_mark = 0.0
        self.circuit_breaker_active = False
        self.circuit_breaker_time = None

    def check_all_gates(self, symbol, df, open_positions, market_data=None, portfolio_value=0):
        """
        Run all three risk gates. Returns (approved, reasons).

        Args:
            symbol: New stock to enter
            df: DataFrame for the new stock (daily data)
            open_positions: dict of {symbol: position_dict} currently held
            market_data: DataSourceManager instance (for fetching correlation data)
            portfolio_value: Total portfolio value in dollars

        Returns:
            (True, []) if all gates pass
            (False, ["reason1", "reason2"]) if any gate blocks
        """
        reasons = []

        # Gate 1: Correlation & Sector Check
        corr_ok, corr_reason = self.check_correlation_gate(symbol, open_positions, market_data)
        if not corr_ok:
            reasons.append(corr_reason)

        # Gate 2: Drawdown & Exposure Check
        dd_ok, dd_reason = self.check_drawdown_gate(open_positions, portfolio_value)
        if not dd_ok:
            reasons.append(dd_reason)

        # Gate 3: Weekly Trend Check
        wt_ok, wt_reason = self.check_weekly_trend_gate(symbol, df)
        if not wt_ok:
            reasons.append(wt_reason)

        approved = len(reasons) == 0
        if not approved:
            logger.warning(f"[{symbol}] PORTFOLIO RISK VETO: {reasons}")
        else:
            logger.debug(f"[{symbol}] All portfolio risk gates passed")

        return approved, reasons

    # ========================================
    # GATE 1: Correlation & Sector
    # ========================================
    def check_correlation_gate(self, symbol, open_positions, market_data=None):
        """
        Block if:
        - Same sector already has max_sector_positions open
        - New stock correlates > max_correlation with any open position
        """
        if not open_positions:
            return True, ""

        max_sector = self.config.get('max_sector_positions', 2)
        max_corr = self.config.get('max_correlation', 0.80)

        # Sector check
        new_sector = SECTOR_MAP.get(symbol, "Unknown")
        if new_sector != "Unknown":
            sector_count = sum(
                1 for s in open_positions.keys()
                if SECTOR_MAP.get(s, "Unknown") == new_sector
            )
            if sector_count >= max_sector:
                return False, f"Sector '{new_sector}' already has {sector_count} positions (max {max_sector})"

        # Correlation check (requires market_data to fetch price history)
        if market_data is not None:
            lookback = self.config.get('correlation_lookback_days', 60)
            try:
                df_new = market_data.get_stock_data(symbol, days_back=lookback)
                if df_new is not None and not df_new.empty:
                    new_returns = df_new['close'].pct_change().dropna()

                    for pos_symbol in open_positions.keys():
                        try:
                            df_pos = market_data.get_stock_data(pos_symbol, days_back=lookback)
                            if df_pos is not None and not df_pos.empty:
                                pos_returns = df_pos['close'].pct_change().dropna()

                                # Align by date
                                aligned = pd.concat([new_returns, pos_returns], axis=1, join='inner')
                                if len(aligned) >= 20:
                                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                                    if corr > max_corr:
                                        return False, (f"High correlation with {pos_symbol}: "
                                                       f"{corr:.2f} > {max_corr}")
                        except Exception as e:
                            logger.debug(f"Could not calculate correlation with {pos_symbol}: {e}")
            except Exception as e:
                logger.debug(f"Could not fetch data for correlation check: {e}")

        return True, ""

    # ========================================
    # GATE 2: Drawdown & Exposure
    # ========================================
    def check_drawdown_gate(self, open_positions, portfolio_value):
        """
        Block all new entries if:
        - Portfolio drawdown exceeds max_portfolio_drawdown_pct
        - Total exposure exceeds max_total_exposure_pct
        - Single position would exceed max_single_position_pct
        """
        if portfolio_value <= 0:
            # Can't check without portfolio value — let it through with warning
            logger.debug("Portfolio value unknown, skipping drawdown gate")
            return True, ""

        max_dd = self.config.get('max_portfolio_drawdown_pct', 0.10)
        max_exposure = self.config.get('max_total_exposure_pct', 0.60)
        cooldown_hours = self.config.get('drawdown_cooldown_hours', 24)

        # Update high water mark
        if portfolio_value > self.portfolio_high_water_mark:
            self.portfolio_high_water_mark = portfolio_value

        # Check drawdown
        if self.portfolio_high_water_mark > 0:
            drawdown = (self.portfolio_high_water_mark - portfolio_value) / self.portfolio_high_water_mark
            if drawdown >= max_dd:
                self.circuit_breaker_active = True
                self.circuit_breaker_time = datetime.now()
                return False, f"CIRCUIT BREAKER: Portfolio down {drawdown:.1%} (max {max_dd:.0%})"

        # Check if circuit breaker cooldown has passed
        if self.circuit_breaker_active and self.circuit_breaker_time:
            elapsed = (datetime.now() - self.circuit_breaker_time).total_seconds() / 3600
            if elapsed < cooldown_hours:
                return False, f"Circuit breaker active ({elapsed:.0f}h of {cooldown_hours}h cooldown)"
            else:
                self.circuit_breaker_active = False
                logger.info("Circuit breaker cooldown expired. Resuming new entries.")

        # Check total exposure
        total_invested = sum(
            pos.get('entry_price', 0) * pos.get('qty', 0)
            for pos in open_positions.values()
        )
        exposure_pct = total_invested / portfolio_value
        if exposure_pct >= max_exposure:
            return False, f"Exposure {exposure_pct:.0%} >= max {max_exposure:.0%}"

        return True, ""

    # ========================================
    # GATE 3: Weekly Trend
    # ========================================
    def check_weekly_trend_gate(self, symbol, df):
        """
        Block if weekly trend is bearish.
        Uses daily data and resamples to weekly.
        """
        if not self.config.get('weekly_trend_enabled', True):
            return True, ""

        if not self.config.get('weekly_trend_must_be_bullish', True):
            return True, ""

        if df is None or len(df) < 50:
            return True, ""  # Not enough data for weekly analysis

        try:
            weekly_sma = self.config.get('weekly_sma_period', 40)

            # Resample daily to weekly
            df_weekly = df.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if len(df_weekly) < weekly_sma:
                return True, ""  # Not enough weekly data

            # Calculate weekly SMA
            df_weekly['sma_weekly'] = df_weekly['close'].rolling(window=weekly_sma).mean()

            last_weekly = df_weekly.iloc[-1]
            weekly_close = last_weekly['close']
            weekly_sma_val = last_weekly.get('sma_weekly', 0)

            if weekly_sma_val > 0 and weekly_close < weekly_sma_val:
                return False, (f"Weekly trend BEARISH: close ${weekly_close:.2f} "
                               f"< SMA_{weekly_sma} ${weekly_sma_val:.2f}")

            return True, ""

        except Exception as e:
            logger.debug(f"[{symbol}] Weekly trend check failed: {e}")
            return True, ""  # If we can't check, don't block
