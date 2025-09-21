# analyze_trades.py
import pandas as pd
import glob
import os
from data_manager import DataManager


def analyze_trade_entries(backtest_log_path: str, data_manager: DataManager):
    """
    Analyzes trade entries to determine if they were 'dip buys' or 'rising buys'.
    """
    print(f"🔎 Analyzing trade log: {backtest_log_path}")
    trades_df = pd.read_csv(backtest_log_path, parse_dates=['entry_date'])

    buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
    if buy_trades.empty:
        print("No 'BUY' trades found in the log to analyze.")
        return

    results = []
    look_forward_days = 5  # How many days to check for a dip after buying

    for _, trade in buy_trades.iterrows():
        symbol = trade['symbol']
        entry_date = trade['entry_date']
        entry_price = trade['entry_price']

        # Load the historical data for this stock
        stock_df = data_manager.load_feature_file(symbol)
        if stock_df is None:
            continue

        # Find the data for the period immediately following the trade
        trade_window = stock_df[stock_df.index > entry_date].head(look_forward_days)
        if trade_window.empty:
            continue

        lowest_price_after_buy = trade_window['Low'].min()
        drawdown = (entry_price - lowest_price_after_buy) / entry_price * 100

        # Categorize the trade
        trade_type = "Dip Buy" if drawdown > 1.0 else "Rising Buy"  # If price dropped more than 1%

        results.append({
            'symbol': symbol,
            'entry_date': entry_date.date(),
            'entry_price': entry_price,
            'lowest_price_in_5_days': lowest_price_after_buy,
            'max_drawdown_pct': drawdown,
            'trade_type': trade_type
        })

    if not results:
        print("Could not generate any analysis results.")
        return

    analysis_df = pd.DataFrame(results)

    # --- Display Summary ---
    print("\n--- Trade Entry Analysis Summary ---")
    print(analysis_df.round(2))

    dip_buy_pct = (analysis_df['trade_type'] == 'Dip Buy').mean() * 100
    print(f"\n📈 Percentage of trades that were 'Dip Buys': {dip_buy_pct:.2f}%")

    output_path = "backtest_results/trade_entry_analysis.csv"
    analysis_df.to_csv(output_path, index=False)
    print(f"✅ Full report saved to: {output_path}")


if __name__ == "__main__":
    # Assumes your backtester saves a trade log named 'trade_log.csv'
    # and that you have test data available for the 2% agent
    log_file = "backtest_results/trade_log.csv"
    data_dir = "models/NASDAQ-testing set/features/2per_profit"

    if os.path.exists(log_file) and os.path.exists(data_dir):
        # We use a DataManager to easily access the historical price data
        dm = DataManager(feature_dir=data_dir)
        analyze_trade_entries(log_file, dm)
    else:
        print(f"Error: Make sure a trade log exists at '{log_file}' and data exists at '{data_dir}'.")
        print("You may need to run the backtester first to generate the log.")