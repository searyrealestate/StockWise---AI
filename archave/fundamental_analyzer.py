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
