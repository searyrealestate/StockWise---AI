# fundamental_analyzer.py

"""
Fundamental Analyzer
====================

This module provides functions to fetch and analyze fundamental data
using the yfinance library, avoiding the need for paid API keys.
"""

import yfinance as yf
import pandas as pd
import logging
import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_ticker_object(symbol: str):
    """
    Returns a cached yf.Ticker object.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Check if ticker is valid
        if not ticker.info:
            logger.warning(f"Could not get ticker info for {symbol}. Symbol may be invalid.")
            return None
        return ticker
    except Exception as e:
        logger.error(f"Error creating yfinance Ticker for {symbol}: {e}", exc_info=True)
        return None


def get_financial_statements(ticker: yf.Ticker) -> dict:
    """
    Fetches the raw financial statements (Income Statement & Balance Sheet).
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


def calculate_key_ratios(ticker: yf.Ticker) -> dict:
    """
    Fetches key, pre-calculated ratios from ticker.info.
    yfinance provides these directly, saving us calculation.
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
                logger.info(f"Fundamental data for {ticker.ticker}: '{key}' is 'None'.")

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

            # Define an "anomaly" as a miss of more than 5%
            if surprise_pct < -5.0:
                result['anomaly_found'] = True
                logger.info(f"Earnings anomaly for {ticker.ticker}: Missed estimate by {surprise_pct:.2f}%")

        return result

    except Exception as e:
        logger.error(f"Error checking earnings anomaly for {ticker.ticker}: {e}", exc_info=True)
        return result