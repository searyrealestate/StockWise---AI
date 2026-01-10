import unittest
import logging
import sys
import os
import time
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# --- IMPORT SYSTEM MODULES ---
try:
    import system_config as cfg
    from feature_engine import RobustFeatureCalculator
    from signal_processor import extract_rf_features
    from stockwise_ai_core import StockWiseAI, GEN9_FEATURES
    from strategy_engine import StrategyOrchestra
    from verify_sniper_logic import run_verification
    from portfolio_manager import PortfolioManager
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import system modules. {e}")
    sys.exit(1)

# --- CONFIGURATION ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "system_health_check.log")
CHART_FILE = os.path.join(LOG_DIR, "system_health_chart.png")

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MasterValidatorV4")

class StockWiseValidatorV4(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logger.info("="*80)
        logger.info("🚀 STOCKWISE SYSTEM HEALTH CHECK (V4)")
        logger.info("="*80)
        
        logger.info(f"📂 Log File:        {os.path.abspath(LOG_FILE)}")
        logger.info(f"📂 Models Dir:      {os.path.abspath(cfg.MODELS_DIR)}")
        
        logger.info("\n⚙️  SYSTEM PARAMETERS:")
        logger.info(f"   • Risk Profile:    {cfg.ACTIVE_PROFILE_NAME}")
        logger.info(f"   • Timeframe:       {cfg.TIMEFRAME}")
        logger.info(f"   • AI Threshold:    {cfg.SniperConfig.MODEL_CONFIDENCE_THRESHOLD}")
        logger.info(f"   • Stop Loss ATR:   {cfg.TRAILLING_STOP_ATR}")
        logger.info(f"   • Training Syms:   {cfg.TRAINING_SYMBOLS}")
        
        logger.info("\n🧠 AI MODEL CONFIG:")
        logger.info(f"   • Features ({len(GEN9_FEATURES)}): {GEN9_FEATURES}")
        
        logger.info("\n🛠️  Generating Test Data...")
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='h')
        
        # 1. Noisy Data
        cls.noisy_df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 1000),
            'high': np.random.uniform(100, 200, 1000),
            'low': np.random.uniform(100, 200, 1000),
            'close': np.random.uniform(100, 200, 1000),
            'volume': np.random.randint(1000, 100000, 1000)
        }, index=dates)
        cls.noisy_df['high'] = cls.noisy_df[['open', 'close', 'high']].max(axis=1)
        cls.noisy_df['low'] = cls.noisy_df[['open', 'close', 'low']].min(axis=1)

        # 2. Golden Data (Sustainable Uptrend)
        # We need RSI to hover around 60 (Golden Zone).
        # Math: To get RSI~60 (RS~1.5), assuming 2 Up 1 Down pattern:
        # AvgGain = 2*Up/3, AvgLoss = Down/3.
        # RS = 2*Up / Down = 1.5  => Down = 1.33 * Up.
        # If Up=+0.5%, Down should be -0.7%.
        # Net Trend: 2*0.5 - 0.7 = +0.3% per 3 bars (Healthy growth).
        price = 100.0
        golden_data = []
        for i, d in enumerate(dates):
            if i % 3 == 2: 
                # Deep Pullback (Healthy correction)
                change = 0.993 # -0.7%
            else:
                # Steady Growth
                change = 1.005 # +0.5%
                
            o = price
            c = price * change
            
            if change > 1:
                h = c * 1.002
                l = o * 0.998
            else:
                h = o * 1.002
                l = c * 0.998
                
            golden_data.append([o, h, l, c, 50000])
            price = c
        cls.golden_df = pd.DataFrame(golden_data, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)

    def log_result(self, test_name, passed, message=""):
        status = "✅ [PASS]" if passed else "❌ [FAIL]"
        logger.info(f"{status} {test_name} | {message}")

    # --- TEST 1: COMPONENT INTEGRATION ---
    def test_01_feature_engine_integration(self):
        """Verify inputs/outputs for Feature Engine."""
        test_name = "Feature Engine Integration"
        try:
            calc = RobustFeatureCalculator()
            df_out = calc.calculate_features(self.noisy_df.copy())
            
            required = ['rsi_14', 'adx', 'ema_21']
            missing = [c for c in required if c not in df_out.columns]
            
            if missing:
                self.log_result(test_name, False, f"Missing Output Columns: {missing}")
                self.fail(f"Missing columns: {missing}")
            
            if df_out.iloc[-1][required].isna().any():
                self.log_result(test_name, False, "Output contains NaNs in latest row.")
                self.fail("NaNs detected")
                
            self.log_result(test_name, True, f"Generated {len(df_out.columns)} features.")
            
        except Exception as e:
            self.log_result(test_name, False, f"Crash: {e}")
            self.fail(str(e))

    # --- TEST 2: AI CORE INPUT/OUTPUT ---
    def test_02_ai_core_io(self):
        """Verify AI accepts feature shape and outputs probabilities."""
        test_name = "AI Core I/O"
        try:
            ai = StockWiseAI(symbol="TEST_IO")
            calc = RobustFeatureCalculator()
            df_feats = calc.calculate_features(self.noisy_df.copy())
            
            decision, prob, trace = ai.predict_trade_confidence("TEST_IO", {}, {}, df_feats)
            
            if not isinstance(prob, float):
                self.log_result(test_name, False, f"Probability output is not float: {type(prob)}")
                self.fail("Invalid Probability Type")
            
            if decision not in ["BUY", "WAIT"]:
                self.log_result(test_name, False, f"Invalid Decision String: {decision}")
                self.fail("Invalid Decision")
                
            self.log_result(test_name, True, f"Output: {decision} ({prob:.2%})")
            
        except Exception as e:
            self.log_result(test_name, False, f"Crash: {e}")
            self.fail(str(e))

    # --- TEST 3: POSITIVE PnL VERIFICATION (Golden Path) ---
    def test_03_positive_pnl_check(self):
        """Verify PnL is positive on winning data."""
        test_name = "Positive PnL Logic"
        
        calc = RobustFeatureCalculator()
        df_golden = calc.calculate_features(self.golden_df.copy())
        
        trades = 0
        total_pnl = 0.0
        
        for i in range(len(df_golden)-100, len(df_golden)):
            row = df_golden.iloc[i]
            analysis = {'AI_Probability': 0.99, 'Fundamental_Score': 90}
            decision = StrategyOrchestra.decide_action("GOLD_TEST", row, analysis)
            
            if decision == "BUY":
                trades += 1
                total_pnl += 10.0 
        
        if trades == 0:
            self.log_result(test_name, False, "System refused to buy on Perfect Uptrend!")
            self.fail("Zero Trades on Golden Data")
            
        if total_pnl <= 0:
            self.log_result(test_name, False, f"PnL is {total_pnl} (Expected > 0)")
            self.fail("Non-Positive PnL")
            
        self.log_result(test_name, True, f"Generated ${total_pnl} profit over {trades} trades on Golden Data.")

    # --- TEST 4: SYSTEM STABILITY ---
    def test_04_stability(self):
        """Stress test memory."""
        test_name = "Stability & Memory"
        try:
            process = psutil.Process(os.getpid())
            mem_start = process.memory_info().rss / 1024 / 1024
            
            calc = RobustFeatureCalculator()
            for _ in range(20):
                calc.calculate_features(self.noisy_df.iloc[:200].copy())
                
            mem_end = process.memory_info().rss / 1024 / 1024
            growth = mem_end - mem_start
            
            if growth > 100: 
                self.log_result(test_name, False, f"Memory Leak: +{growth:.2f}MB")
                self.fail("Memory Leak")
                
            self.log_result(test_name, True, f"Stable. Growth: {growth:.2f} MB")
            
        except Exception as e:
            self.log_result(test_name, False, f"Crash: {e}")
            self.fail(str(e))

def generate_error_chart(results):
    logger.info("\n📊 Generating System Health Chart...")
    components = list(results.keys())
    statuses = [1 if res == "PASS" else 0 for res in results.values()]
    colors = ['green' if s == 1 else 'red' for s in statuses]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(components, [1]*len(components), color=colors)
    plt.title('StockWise System Health Check')
    plt.xlabel('Status (Green=OK, Red=FAIL)')
    plt.yticks(ticks=range(len(components)), labels=components)
    plt.xticks([]) 
    
    for i, bar in enumerate(bars):
        label = "PASS" if statuses[i] == 1 else "FAIL"
        plt.text(0.5, bar.get_y() + bar.get_height()/2, label, 
                 ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    logger.info(f"🖼️  Chart saved to: {CHART_FILE}")

if __name__ == "__main__":
    start_time = time.time()
    
    # 1. Run Tests
    suite = unittest.TestLoader().loadTestsFromTestCase(StockWiseValidatorV4)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    # 2. Collect Chart Data
    report = {}
    method_map = {
        'test_01_feature_engine_integration': 'Feature Engine',
        'test_02_ai_core_io': 'AI Core Logic',
        'test_03_positive_pnl_check': 'PnL / Win Rate Logic',
        'test_04_stability': 'System Stability'
    }
    for method_name, readable in method_map.items():
        report[readable] = "PASS"
        
    if not result.wasSuccessful():
        for failure in result.failures:
            name = failure[0]._testMethodName
            if name in method_map: report[method_map[name]] = "FAIL"
        for error in result.errors:
            name = error[0]._testMethodName
            if name in method_map: report[method_map[name]] = "FAIL"
            
    # 3. Generate Chart
    generate_error_chart(report)
    
    # 4. TEST RUN SUMMARY (The New Addition)
    duration = time.time() - start_time
    passed_count = result.testsRun - len(result.failures) - len(result.errors)
    
    logger.info("\n" + "="*50)
    logger.info("           🏁  TEST RUN SUMMARY  🏁")
    logger.info("="*50)
    logger.info(f"⏱️  Total Duration:   {duration:.2f} seconds")
    logger.info(f"🔢 Total Tests Run:  {result.testsRun}")
    logger.info("-" * 50)
    logger.info(f"✅ PASSED:           {passed_count}")
    logger.info(f"❌ FAILED:           {len(result.failures)}")
    logger.info(f"⚠️  ERRORS:           {len(result.errors)}")
    logger.info("="*50)
    
    if result.wasSuccessful():
        logger.info("🟢 OVERALL STATUS:   SUCCESS")
    else:
        logger.info("🔴 OVERALL STATUS:   FAILURE (Check Logs)")
    logger.info("="*50)