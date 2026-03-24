"""
Bug 1.3 Verification: er_trend replaced with er_slow threshold.
Tests that Setup 1 (DSP_SUPER_TREND) fires when er_slow >= threshold.
"""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_setup1_fires_when_er_slow_above_threshold():
    """Setup 1 should activate when er_slow >= 0.55 AND trend_alignment == 1"""
    from strategy_engine import TacticalSniper
    sniper = TacticalSniper()

    # Build minimal DataFrame with columns the analyze() method needs
    row = {
        'close': 150.0, 'open': 148.0, 'high': 151.0, 'low': 147.0,
        'volume': 1000000, 'sma_50': 145.0, 'sma_200': 140.0,
        'er_slow': 0.65,       # Above threshold (0.55)
        'er_fast': 0.50,
        'trend_alignment': 1,  # Aligned
        'bb_width': 0.20, 'atr': 2.5, 'rvol': 1.5,
        'rsi': 55.0, 'bb_upper': 160.0, 'macd_hist': 0.5,
        'vol_avg_20': 800000
    }
    df = pd.DataFrame([row])

    result = sniper.analyze(df)
    assert 'DSP_SUPER_TREND' in result.get('active_setups', []), \
        f"Setup 1 should fire when er_slow=0.65 >= 0.55. Got: {result}"
    print("PASS: Setup 1 fires correctly with er_slow above threshold")

def test_setup1_blocked_when_er_slow_below_threshold():
    """Setup 1 should NOT activate when er_slow < 0.55"""
    from strategy_engine import TacticalSniper
    sniper = TacticalSniper()

    row = {
        'close': 150.0, 'open': 148.0, 'high': 151.0, 'low': 147.0,
        'volume': 1000000, 'sma_50': 145.0, 'sma_200': 140.0,
        'er_slow': 0.30,       # Below threshold
        'er_fast': 0.20,
        'trend_alignment': 1,
        'bb_width': 0.20, 'atr': 2.5, 'rvol': 1.5,
        'rsi': 55.0, 'bb_upper': 160.0, 'macd_hist': 0.5,
        'vol_avg_20': 800000
    }
    df = pd.DataFrame([row])

    result = sniper.analyze(df)
    setups = result.get('active_setups', [])
    assert 'DSP_SUPER_TREND' not in setups, \
        f"Setup 1 should NOT fire when er_slow=0.30 < 0.55. Got: {setups}"
    print("PASS: Setup 1 correctly blocked when er_slow below threshold")

def test_old_er_trend_column_not_referenced():
    """Verify the string 'er_trend' no longer appears in strategy_engine.py"""
    strategy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'strategy_engine.py')
    with open(strategy_path, 'r') as f:
        content = f.read()
    assert 'er_trend' not in content, \
        "FAIL: 'er_trend' still found in strategy_engine.py!"
    print("PASS: 'er_trend' fully removed from codebase")

if __name__ == '__main__':
    test_setup1_fires_when_er_slow_above_threshold()
    test_setup1_blocked_when_er_slow_below_threshold()
    test_old_er_trend_column_not_referenced()
    print("\n=== All Bug 1.3 tests PASSED ===")
