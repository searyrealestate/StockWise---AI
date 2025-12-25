"""
Enhanced IBKR Manager for Professional StockWise
Professional-grade data integration with Interactive Brokers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import IBKR API, but handle gracefully if not available
try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    import threading
    import queue
    IBKR_API_AVAILABLE = True
except ImportError:
    IBKR_API_AVAILABLE = False
    print("⚠️ IBKR API not installed. Using mock mode.")

class ProfessionalIBKRManager:
    """Professional IBKR data manager with fallback to yfinance"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.connected = False
        self.data_queue = queue.Queue() if IBKR_API_AVAILABLE else None
        self.connection_config = {}
        self.data_cache = {}
        self.log("Professional IBKR Manager initialized", "INFO")
        
    def log(self, message, level="INFO"):
        """Logging method"""
        if self.debug:
            timestamp = datetime.now().strftime('%H:%M:%S')
            color_map = {
                "INFO": "\033[94m",
                "SUCCESS": "\033[92m", 
                "ERROR": "\033[91m",
                "WARNING": "\033[93m"
            }
            reset = "\033[0m"
            print(f"{timestamp} | {color_map.get(level, '')}{message}{reset}")
    
    def connect_with_fallback(self, connection_configs):
        """Try to connect with multiple fallback options"""
        if not IBKR_API_AVAILABLE:
            self.log("IBKR API not available, using fallback mode", "WARNING")
            return False
            
        for config in connection_configs:
            try:
                self.log(f"Attempting connection to {config['name']} on {config['host']}:{config['port']}", "INFO")
                
                # Simulate connection attempt (replace with actual IBKR connection)
                # In production, this would actually connect to IBKR
                time.sleep(0.5)  # Simulate connection delay
                
                # For now, we'll simulate a failed connection to use yfinance
                # Set to True if you have IBKR properly configured
                connection_successful = False
                
                if connection_successful:
                    self.connected = True
                    self.connection_config = config
                    self.log(f"Connected to {config['name']}", "SUCCESS")
                    return True
                    
            except Exception as e:
                self.log(f"Failed to connect to {config['name']}: {e}", "ERROR")
                
        self.log("All IBKR connection attempts failed", "ERROR")
        return False
    
    def get_stock_data(self, symbol, days_back=60):
        """Get historical stock data"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{days_back}"
            if cache_key in self.data_cache:
                cache_time, cached_data = self.data_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 300:  # 5 minute cache
                    self.log(f"Returning cached data for {symbol}", "INFO")
                    return cached_data
            
            # If connected to IBKR, get professional data
            if self.connected and IBKR_API_AVAILABLE:
                # This would be actual IBKR data retrieval
                # For now, return None to trigger yfinance fallback
                return None
            else:
                # Fallback to yfinance
                import yfinance as yf
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back + 30)
                
                df = yf.download(symbol, start=start_date, end=end_date, 
                               progress=False, auto_adjust=True)
                
                if not df.empty:
                    # Cache the data
                    self.data_cache[cache_key] = (datetime.now(), df)
                    return df
                    
        except Exception as e:
            self.log(f"Error getting data for {symbol}: {e}", "ERROR")
            return None
    
    def get_current_price(self, symbol):
        """Get current market price"""
        try:
            if self.connected and IBKR_API_AVAILABLE:
                # Would get real-time price from IBKR
                # For now, use yfinance
                pass
                
            # Fallback to yfinance
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            for field in ['regularMarketPrice', 'currentPrice', 'ask', 'bid']:
                if field in info and info[field]:
                    return float(info[field])
                    
            # If no current price, get last close
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
                
        except Exception as e:
            self.log(f"Error getting current price for {symbol}: {e}", "ERROR")
            
        return None
    
    def validate_symbol(self, symbol):
        """Validate if symbol exists"""
        try:
            if self.connected and IBKR_API_AVAILABLE:
                # Would validate through IBKR
                # For now, use yfinance
                pass
                
            # Fallback validation
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return 'symbol' in info or 'shortName' in info or 'regularMarketPrice' in info
            
        except:
            return False
    
    def get_connection_info(self):
        """Get connection status information"""
        return {
            'connected': self.connected,
            'connection_config': self.connection_config,
            'api_available': IBKR_API_AVAILABLE,
            'data_quality': {
                'successful_requests': 0,
                'failed_requests': 0,
                'timeout_requests': 0,
                'data_latency': []
            }
        }
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.connected = False
            self.log("Disconnected from IBKR", "INFO")


# Additional helper classes for enhanced functionality
class EnhancedSignalDetector:
    """Enhanced signal detection for 95% confidence"""
    
    def __init__(self, debug=False):
        self.debug = debug
        
    def enhanced_signal_decision(self, df, indicators, symbol):
        """Make enhanced signal decision"""
        signals = []
        total_score = 0
        
        # Analyze multiple factors
        trend_strength = self.analyze_trend_strength(df, indicators)
        momentum_quality = self.analyze_momentum_quality(indicators)
        volume_confirmation = self.analyze_volume_pattern(df, indicators)
        
        # Combine scores
        total_score = trend_strength + momentum_quality + volume_confirmation
        
        # Generate confidence
        if abs(total_score) >= 5:
            confidence = 90
            action = "BUY" if total_score > 0 else "SELL"
        elif abs(total_score) >= 3:
            confidence = 75
            action = "BUY" if total_score > 0 else "SELL"
        else:
            confidence = 60
            action = "WAIT"
            
        return {
            'action': action,
            'confidence': confidence,
            'signals': signals,
            'total_score': total_score,
            'score_breakdown': {
                'trend': trend_strength,
                'momentum': momentum_quality,
                'volume': volume_confirmation
            },
            'target_gain_pct': self.calculate_target_gain(confidence)
        }
    
    def analyze_trend_strength(self, df, indicators):
        """Analyze trend strength"""
        score = 0
        
        # Check moving average alignment
        if indicators.get('sma_5', 0) > indicators.get('sma_20', 0):
            score += 1
        if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
            score += 1
            
        # Check price position
        current_price = indicators.get('current_price', 0)
        if current_price > indicators.get('sma_20', 0):
            score += 1
            
        return score
    
    def analyze_momentum_quality(self, indicators):
        """Analyze momentum indicators"""
        score = 0
        
        # RSI analysis
        rsi = indicators.get('rsi_14', 50)
        if 30 < rsi < 70:
            score += 1
        if rsi < 40:
            score += 1  # Oversold bonus
            
        # MACD analysis
        if indicators.get('macd_histogram', 0) > 0:
            score += 1
            
        return score
    
    def analyze_volume_pattern(self, df, indicators):
        """Analyze volume patterns"""
        score = 0
        
        volume_rel = indicators.get('volume_relative', 1.0)
        if volume_rel > 1.5:
            score += 2
        elif volume_rel > 1.2:
            score += 1
            
        return score
    
    def calculate_target_gain(self, confidence):
        """Calculate target gain based on confidence"""
        if confidence >= 90:
            return 10.0
        elif confidence >= 80:
            return 7.0
        elif confidence >= 70:
            return 5.0
        else:
            return 3.7


class ConfidenceBuilder:
    """Build 95% confidence predictions"""
    
    def __init__(self, debug=False):
        self.debug = debug
        
    def calculate_95_percent_confidence(self, symbol, features, signals, market_data):
        """Calculate confidence aiming for 95% accuracy"""
        
        base_confidence = 70
        
        # Feature-based adjustments
        if features[0] > 1.5:  # High volume
            base_confidence += 5
        if features[1] < 0.4:  # Oversold RSI
            base_confidence += 5
        if features[2] > 0:  # Positive MACD
            base_confidence += 5
            
        # Market regime adjustments
        if market_data.get('volatility', 0) < 0.02:
            base_confidence += 3
            
        # Signal agreement bonus
        signal_agreement = len([s for s in signals if 'BUY' in s or 'bullish' in s.lower()])
        if signal_agreement >= 3:
            base_confidence += 10
            
        # Cap at 95%
        final_confidence = min(95, base_confidence)
        
        # Determine recommendation
        if final_confidence >= 85:
            recommendation = "STRONG_BUY"
        elif final_confidence >= 75:
            recommendation = "BUY"
        elif final_confidence >= 65:
            recommendation = "WEAK_BUY"
        else:
            recommendation = "WAIT"
            
        return {
            'confidence': final_confidence,
            'recommendation': recommendation,
            'confidence_factors': {
                'volume': features[0] > 1.5,
                'rsi_oversold': features[1] < 0.4,
                'macd_positive': features[2] > 0,
                'low_volatility': market_data.get('volatility', 0) < 0.02,
                'signal_agreement': signal_agreement >= 3
            }
        }
