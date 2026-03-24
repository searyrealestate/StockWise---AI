# market_intelligence.py

"""
StockWise Gen-12 Market Intelligence (The Intel)
================================================
The "G-2" of the trading system.
Responsible for gathering "Intel" beyond basic price/volume:
- Macro Health (SPY Analysis) -> Defense Mode
- Sentiment Analysis (News) -> Veto Power
- Fundamental Analysis (Earnings, PE, Graham Number) -> Strategic Validation
- Calendar Events (Earnings Dates) -> Risk Avoidance
"""

import logging
import pandas as pd
from datetime import datetime
import system_config as cfg
import yfinance as yf

# Setup specialized logger for Intelligence gathering events
logger = logging.getLogger("MarketIntel")

class MarketIntelligence:
    def __init__(self, data_manager):
        """
        Initialize with a link to the DataManager for fetching benchmarks (SPY).
        """
        self.dm = data_manager
        self.defense_mode = False # Flag: True if Market is crashing
        self.market_state = "NEUTRAL" # BULL, BEAR, CHOP, NEUTRAL

    def check_macro_health(self):
        """
        Analyzes the S&P 500 (SPY) to determine the overall market state.
        Returns: 'BULL', 'BEAR', 'NEUTRAL'
        Side Effect: Sets self.defense_mode = True if SPY drops > 1.5% in 2 days.
        """
        # Fetch last 2 days of SPY data
        spy = self.dm.fetch_data("SPY", limit=2)
        if spy.empty: return "NEUTRAL" # Fail safe

        if len(spy) < 2: return "NEUTRAL"
        
        # Calculate percentage change from yesterday to today (or last 2 data points)
        change = (spy.iloc[-1]['close'] - spy.iloc[0]['close']) / spy.iloc[0]['close']

        # Defense Trigger: Significant drop
        if change < -0.015: # -1.5% drop
            self.defense_mode = True
            self.market_state = "BEAR"
            logger.warning(f"DEFENSE MODE TRIGGERED: SPY dropped {change:.2%}")
            return "BEAR"
        
        # The following logic is commented out but would broaden state detection
        # self.defense_mode = False
        # self.market_state = "BULL" if change > 0 else "CHOP"
        # return self.market_state
        return "NEUTRAL"
            
        # Legacy/Alternative logic block (Commented out)
        # current_price = spy_df.iloc[-1]['close']
        # prev_close = spy_df.iloc[-1].get('prev_close', spy_df.iloc[0]['close']) # simplified
        
        # # Calculate Intraday Change
        # pct_change = (current_price - prev_close) / prev_close
        
        # crash_trigger = cfg.RISK_CONFIG["spy_crash_trigger_pct"] # -0.015
        
        # if pct_change < crash_trigger:
        #     logger.warning(f"🚨 MACRO DEFENSE ACTIVATED! SPY Down {pct_change:.2%}")
        #     self.defense_mode = True
        #     return False
        # else:
        #     self.defense_mode = False
        #     return True

    def calculate_graham_number(self, fundamentals):
        """
        Calculates the "Graham Number" - Benjamin Graham's formula for fair value.
        Formula: Sqrt(22.5 * Earnings Per Share * Book Value Per Share)
        Used by 'Strategic' and 'Reversion' agents to find deep value.
        """
        try:
            eps = fundamentals.get('trailingEps', 0)
            bv = fundamentals.get('bookValue', 0)
            # Both must be positive for the square root to work (and to be a valid value stock)
            if eps > 0 and bv > 0:
                return (22.5 * eps * bv) ** 0.5
        except:
            pass
        return 0 # Return 0 if data missing or invalid
    
    def analyze_news_sentiment(self, symbol):
        """
        Conducts a quick Sentiment Analysis on recent news titles.
        Veto Role: If aggregate sentiment is highly negative (< -0.2), block BUY orders.
        Uses TextBlob NLP library.
        """
        try:
            from textblob import TextBlob
            import yfinance as yf
            
            # Fetch News via YFinance API
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return True # No news is generally safe (no bad news)
                
            scores = []
            for item in news:
                title = item.get('title', '')
                if title:
                    # Run simple polarity check (-1.0 to +1.0)
                    blob = TextBlob(title)
                    scores.append(blob.sentiment.polarity)
            
            if not scores:
                return True
                
            # Calculate Average Sentiment
            avg_sentiment = sum(scores) / len(scores)
            
            # Veto Logic
            # Threshold: < -0.2 is considered bad enough to pause trading
            if avg_sentiment < -0.2:
                 logger.info(f"News Veto: {symbol} has negative sentiment ({avg_sentiment:.2f})")
                 return False # BLOCK TRADE
                 
            return True # ALLOW TRADE
            
        except ImportError:
            logger.warning("TextBlob or YFinance not found. Skipping News Check.")
            return True
        except Exception as e:
            logger.warning(f"News check denied for {symbol}: {e}")
            return True # Fail open (allow trade) to prevent blocking on API errors

    def check_earnings_safety(self, symbol):
        """
        Risk Management: Checks if an earnings report is imminent (within 7 days).
        Blocks trading if true to avoid volatility gambles.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Get Calendar
            cal = ticker.calendar
            if cal is None or cal.empty:
                return True
                
            # 'Earnings Date' logic is complex due to API variations
            next_earnings = None
            
            # Handling different yfinance versions (structure varies)
            if isinstance(cal, pd.DataFrame):
                # Usually row 0 is next earnings
                pass 
            
            # Robust approach: get_earnings_dates
            dates = ticker.get_earnings_dates(limit=5)
            if dates is None or dates.empty:
                return True
                
            # Filter for future dates only
            now = pd.Timestamp.now().tz_localize(None)
            
            # Clean dataframe index timezone to be compatible with 'now'
            dates.index = dates.index.tz_localize(None)
            
            future_dates = dates.index[dates.index > now]
            
            if not future_dates.empty:
                # Get the nearest future date
                next_earnings = future_dates.sort_values()[0]
                days_to = (next_earnings - now).days
                
                # Veto: If within 7 days, don't trade.
                if days_to is not None and days_to < 7:
                    logger.info(f"Earnings Safety: {symbol} reports in {days_to} days ({next_earnings.date()}).")
                    return False
                    
            return True
            
        except Exception as e:
            # logger.debug(f"Earnings check failed for {symbol}: {e}")
            return True

    def get_fundamentals(self, symbol):
        """
        Aggregates fundamental data for the Strategy Engine decision making.
        Returns a dictionary of key metrics (PE, PEG, Growth).
        """
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info
            
            return {
                "pe_ratio": info.get('trailingPE', 20.0), # Default to 20 if missing
                "sector_pe": 25.0, # Using static constant as sector data is expensive to fetch
                "peg_ratio": info.get('pegRatio', 1.2),
                "earnings_growth_qbq": info.get('earningsGrowth', 0.15),
                "sector": info.get('sector', 'Technology')
            }
        except:
             # Return safe defaults in case of failure
             return {
                "pe_ratio": 20.0,
                "sector_pe": 25.0,
                "peg_ratio": 1.2,
                "earnings_growth_qbq": 0.15,
                "sector": "Unknown"
            }

    def calculate_trin_indicator(self, market_data):
        """
        Calculates TRIN (Arms Index) for Market Internals.
        Formula: (Advancing Issues / Declining Issues) / (Advancing Vol / Declining Vol)
        Currently a Placeholder returning 1.0 (Neutral).
        """
        # Requires broad market breadth data usually fetched separately
        return 1.0 


def get_financial_statements(ticker: yf.Ticker) -> dict:
    """
    Fetches raw financial statements (Income Statement & Balance Sheet).
    Useful for deep dive analysis, but rarely used in high-frequency logic.
    """
    if not ticker:
        return {'income_statement': pd.DataFrame(), 'balance_sheet': pd.DataFrame()}
    try:
        return {
            'income_statement': ticker.financials,
            'balance_sheet': ticker.balance_sheet
        }
    except Exception as e:
        logger.error(f"Error fetching financials for {ticker.ticker}: {e}", exc_info=True)
        return {'income_statement': pd.DataFrame(), 'balance_sheet': pd.DataFrame()}


def get_sector_context(self, symbol):
    """
    Checks if the stock's technical breakout aligns with its Sector.
    Source [14]: Context Analysis (rel_strength_sector)
    BUG: This function is defined *outside* the MarketIntelligence class but uses 'self'.
    It will likely raise a NameError or TypeError if called directly.
    """
    # If NVDA is forming a Bull Flag, is SMH (Semiconductor ETF) also bullish?
    # This reduces false positives from "Fake Breakouts" (Source [13]).
    benchmark = self.dm.get_benchmark_symbol(symbol) # e.g., 'SMH'
    
    df_sector = self.dm.fetch_data(benchmark, limit=50)
    if df_sector.empty: return 0
    
    # Check simple sector trend (Price > SMA50)
    return 1 if df_sector.iloc[-1]['close'] > df_sector.iloc[-1]['sma_50'] else -1


def calculate_key_ratios(ticker: yf.Ticker) -> dict:
    """
    Fetches key, pre-calculated ratios from ticker.info.
    yfinance provides these directly, saving us calculation overhead.
    """
    ratios = {
        'pe_ratio': None,
        'ps_ratio': None,
        'debt_to_equity': None
    }
    if not ticker or not ticker.info:
        return ratios

    try:
        ratios['pe_ratio'] = ticker.info.get('trailingPE')
        ratios['ps_ratio'] = ticker.info.get('priceToSalesTrailing12Months')
        ratios['debt_to_equity'] = ticker.info.get('debtToEquity')

        # Log if data is missing
        for key, val in ratios.items():
            if val is None:
                logger.debug(f"Fundamental data for {ticker.ticker}: '{key}' is 'None'.")

        return ratios
    except Exception as e:
        logger.error(f"Error fetching key ratios for {ticker.ticker}: {e}", exc_info=True)
        return ratios


def check_earnings_anomaly(ticker: yf.Ticker) -> dict:
    """
    Checks the most recent earnings report for an anomaly (actual vs. estimate).
    """
    result = {
        'anomaly_found': False,
        'last_earnings_date': None,
        'surprise_pct': None
    }
    if not ticker:
        return result

    try:
        # Get the quarterly earnings data
        earnings = ticker.earnings_dates
        if earnings is None or earnings.empty:
            logger.info(f"No earnings dates found for {ticker.ticker}.")
            return result

        # Sort by date and get the most recent one
        earnings = earnings.sort_index(ascending=False)
        last_report = earnings.iloc[0]

        actual = last_report.get('Actual')
        estimate = last_report.get('Estimate')

        result['last_earnings_date'] = last_report.name.strftime('%Y-%m-%d')

        if actual is not None and estimate is not None and estimate != 0:
            surprise_pct = ((actual - estimate) / abs(estimate)) * 100
            result['surprise_pct'] = surprise_pct

            # Define an "anomaly" as a miss or beat of more than 5%
            # Logic here specifically checks for Misses (< -5.0) for warning.
            if surprise_pct < -5.0:
                result['anomaly_found'] = True
                logger.info(f"Earnings anomaly for {ticker.ticker}: Missed estimate by {surprise_pct:.2f}%")

        return result

    except Exception as e:
        logger.error(f"Error checking earnings anomaly for {ticker.ticker}: {e}", exc_info=True)
        return result