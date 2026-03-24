import logging
import pandas as pd
import time
from data_source_manager import DataSourceManager
from strategy_engine import StrategyOrchestra

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyFix")

def verify_dsm():
    logger.info("--- Verifying DataSourceManager ID Generation ---")
    dsm = DataSourceManager(use_ibkr=True, allow_fallback=False) # Force IBKR logic path
    
    # We want to check if IDs are increasing.
    # Note: We can't easily check internal state without private access, 
    # but we can observe if it crashes on 'Duplicate ticker ID' when called rapidly.
    
    logger.info("Simulating rapid requests...")
    try:
        # Mocking the _download_from_ibkr to just check ID generation if possible,
        # or just calling get_stock_data if we have a connection.
        # Since we might not have a running TWS, `connect_to_ibkr` might fail gracefuly.
        # But `get_new_req_id` is what we fixed. Let's test checking the method existence.
        
        id1 = dsm.get_new_req_id()
        id2 = dsm.get_new_req_id()
        
        logger.info(f"ID 1: {id1}")
        logger.info(f"ID 2: {id2}")
        
        if id2 > id1:
            logger.info("✅ ID Generation checks out (Incrementing).")
        else:
            logger.error("❌ ID Generation failed.")
            
    except Exception as e:
        logger.error(f"❌ DSM Test Failed: {e}")

def verify_strategy():
    logger.info("--- Verifying Strategy Engine Scope Fix ---")
    engine = StrategyOrchestra()
    
    # Mock Data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=10)
    df = pd.DataFrame({
        'close': [150] * 10,
        'pe_ratio': [10] * 10 
    }, index=dates)
    
    fundamentals = {
        'pe_ratio': 10,
        'sector_pe': 20,
        'peg_ratio': 1.0,
        'news_sentiment': 0.5
    }
    
    # Testing STRATEGIC agent specifically since that's where the UnboundLocalError was
    logger.info("Calling decide_action with STRATEGIC agent...")
    try:
        # We need to make sure config is loaded or mocked. 
        # StrategyOrchestra loads system_config.STRATEGY_CONFIG.
        # Assuming system_config is valid.
        
        decision = engine.decide_action(
            symbol="TEST",
            df=df,
            fundamentals=fundamentals,
            ai_confidence=0.0,
            allowed_agents=["STRATEGIC"]
        )
        
        logger.info(f"Decision: {decision}")
        logger.info("✅ Strategy Logic ran without UnboundLocalError.")
        
    except UnboundLocalError as e:
        logger.error(f"❌ UnboundLocalError Caught: {e}")
    except Exception as e:
        logger.error(f"❌ Other Strategy Error: {e}")

if __name__ == "__main__":
    verify_dsm()
    verify_strategy()
