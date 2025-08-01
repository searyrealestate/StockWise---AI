CONFIG_CODE = '''
"""
🔧 Configuration file for 95% Confidence Trading System
"""

# 95% Confidence System Settings
CONFIDENCE_SETTINGS = {
    'target_confidence': 95.0,
    'minimum_trading_confidence': 85.0,
    'high_confidence_threshold': 90.0,
    'ultra_confidence_threshold': 95.0,
    'enable_pre_breakout_detection': True,
    'enable_manual_review_flagging': True
}

# Dynamic Profit Targets
PROFIT_TARGETS = {
    95: 0.08,  # 8% for 95%+ confidence
    90: 0.06,  # 6% for 90-94% confidence  
    85: 0.05,  # 5% for 85-89% confidence
    80: 0.04,  # 4% for 80-84% confidence
    75: 0.037, # 3.7% for 75-79% confidence
    'default': 0.037
}

# Risk Management
RISK_SETTINGS = {
    'max_position_risk': 0.02,  # 2% of account per trade
    'stop_loss_pct': 0.06,      # 6% stop loss
    'trailing_stop_activation': 0.5,  # Activate trailing stop at 50% of profit target
    'max_daily_trades': 5,
    'max_weekly_trades': 15
}

# Model Settings
MODEL_SETTINGS = {
    'ensemble_model_dir': 'models/ensemble/',
    'retrain_frequency_days': 30,
    'min_historical_accuracy': 0.70,
    'require_volume_confirmation': True,
    'market_regime_lookback': 20
}
'''