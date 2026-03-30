"""
Downloads S&P 500 OHLCV data + FeatureEngine features.
Caches to backtest/data/{symbol}.parquet — skips re-download if file covers TEST_END.
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from backtest.config import (
    TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    TEST_START, TEST_END, API_DELAY
)

logger = logging.getLogger("backtest.data_loader")

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DATA_DIR   = _HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── S&P 500 symbol list ────────────────────────────────────────────────────
# ~503 current S&P 500 constituents (as of early 2026)
SP500_SYMBOLS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
    "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN",
    "APH","ADI","ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET",
    "AJG","AIZ","T","ATO","ADSK","AZO","AVB","AVY","AXON","BKR","BALL",
    "BAC","BK","BBWI","BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK",
    "BX","BA","BCR","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","GHC","BG",
    "CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT",
    "CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL",
    "SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C",
    "CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP",
    "ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA","CCI",
    "CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DAL","XRAY","DVN",
    "DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DTE",
    "DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR",
    "ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY",
    "EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO",
    "FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","F","FTNT","FTV",
    "FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEN","GNSS","GILD",
    "GIS","GM","GPC","GOOG","GWW","HAL","HIG","HAS","HCA","DOC","HSIC",
    "HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ",
    "HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD",
    "INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM",
    "JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP",
    "KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX",
    "LW","LVS","LDOS","LEN","LNC","LIN","LYV","LKQ","LMT","L","LOW","LULU",
    "LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH",
    "MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU",
    "MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS",
    "MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE",
    "NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI",
    "ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW",
    "PARA","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX",
    "PNW","PXD","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU",
    "PEG","PSTG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF",
    "RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP",
    "ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG",
    "SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
    "SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP",
    "TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO",
    "TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA",
    "UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VRSN","VRSK","VZ",
    "VRTX","VFC","VTRS","VICI","V","VMC","WRB","GWW","WAB","WBA","WMT",
    "WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WRK","WY","WHR","WMB",
    "WTW","GWW","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"
]
# De-duplicate while preserving order
_seen = set()
SP500_SYMBOLS = [s for s in SP500_SYMBOLS if s not in _seen and not _seen.add(s)]


def _parquet_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}.parquet"


def _is_cached(symbol: str) -> bool:
    """Return True if a valid parquet file exists covering up to TEST_END."""
    p = _parquet_path(symbol)
    if not p.exists():
        return False
    try:
        df = pd.read_parquet(p, columns=["close"])
        if df.empty:
            return False
        last_date = df.index.max()
        if isinstance(last_date, pd.Timestamp):
            return last_date >= pd.Timestamp(TEST_END)
        return False
    except Exception:
        return False


def download_all(symbols: list, max_symbols: int = None) -> dict:
    """
    Download and cache OHLCV + features for each symbol.
    Returns {symbol: DataFrame} for successfully downloaded symbols.
    """
    # Lazy imports (avoid loading heavy libs at module level)
    from data_source_manager import DataSourceManager
    from feature_engine import FeatureEngine

    if max_symbols:
        symbols = symbols[:max_symbols]

    dsm = DataSourceManager(use_ibkr=False, allow_fallback=True)
    fe  = FeatureEngine()

    results = {}
    ok = 0
    failed = 0
    total = len(symbols)
    t0 = time.time()

    for i, symbol in enumerate(symbols, 1):
        if _is_cached(symbol):
            try:
                df = pd.read_parquet(_parquet_path(symbol))
                results[symbol] = df
                ok += 1
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
                failed += 1
        else:
            try:
                raw = dsm.get_stock_data(symbol, days_back=730, interval='1d')
                if raw is None or raw.empty or len(raw) < 100:
                    logger.warning(f"{symbol}: insufficient data ({len(raw) if raw is not None else 0} rows)")
                    failed += 1
                    time.sleep(API_DELAY)
                    continue

                df = fe.calculate_features(raw, strategy_config={"active_indicators": ["all"]})
                if df is None or df.empty:
                    logger.warning(f"{symbol}: feature engine returned empty DataFrame")
                    failed += 1
                    time.sleep(API_DELAY)
                    continue

                df.to_parquet(_parquet_path(symbol))
                results[symbol] = df
                ok += 1
                time.sleep(API_DELAY)

            except Exception as e:
                logger.warning(f"{symbol}: download failed — {e}")
                failed += 1
                time.sleep(API_DELAY)

        if i % 25 == 0 or i == total:
            elapsed = (time.time() - t0) / 60
            pct = i / total * 100
            print(f"{i}/{total} ({pct:.0f}%) | {ok} OK, {failed} failed | elapsed: {elapsed:.1f}m")

    print(f"Download complete: {ok}/{total} symbols loaded")
    return results


def load_and_split(symbols: list) -> tuple:
    """
    Load parquet files and split into (train, val, test) dicts.
    Each dict: {symbol: DataFrame}.  Zero date overlap guaranteed.
    Symbols with < 200 rows in train are dropped.
    """
    train_data, val_data, test_data = {}, {}, {}

    train_s = pd.Timestamp(TRAIN_START)
    train_e = pd.Timestamp(TRAIN_END)
    val_s   = pd.Timestamp(VAL_START)
    val_e   = pd.Timestamp(VAL_END)
    test_s  = pd.Timestamp(TEST_START)
    test_e  = pd.Timestamp(TEST_END)

    dropped = 0
    for symbol in symbols:
        p = _parquet_path(symbol)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            tr = df[(df.index >= train_s) & (df.index <= train_e)].copy()
            va = df[(df.index >= val_s)   & (df.index <= val_e)].copy()
            te = df[(df.index >= test_s)  & (df.index <= test_e)].copy()

            if len(tr) < 200:
                dropped += 1
                continue

            train_data[symbol] = tr
            if not va.empty:
                val_data[symbol] = va
            if not te.empty:
                test_data[symbol] = te

        except Exception as e:
            logger.warning(f"load_and_split failed for {symbol}: {e}")

    logger.info(
        f"Split complete: {len(train_data)} train / {len(val_data)} val / "
        f"{len(test_data)} test symbols. Dropped (insufficient train rows): {dropped}"
    )
    return train_data, val_data, test_data
