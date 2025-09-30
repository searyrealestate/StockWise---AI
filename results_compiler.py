"""
Master Results Compiler
=======================

This script serves as a centralized compiler for all artifacts and logs generated
during a full version test of the trading system. It automatically finds the
latest logs and result files from each stage of the pipeline (data generation,
tuning, evaluation, and backtesting), processes them, and aggregates them into a
single, master Excel workbook for easy comparison and record-keeping.

The script is version-aware; it associates all collected data with a specific
'Version ID' provided by the user, allowing for a historical log of performance
and changes over time.

Functionality:
--------------
1.  Parses Logs: Extracts key metrics from plain-text log files for stages like
    data generation and hyperparameter tuning.
2.  Processes CSVs: Reads and summarizes structured data from CSV files generated
    during model evaluation, backtesting (equity curves), and trade analysis.
3.  Calculates KPIs: Computes high-level financial metrics like Total Return,
    Max Drawdown, and Sharpe Ratio from raw equity curve data.
4.  Updates Master Workbook: Creates or updates a master Excel file
    ('Master_Test_Results.xlsx') with separate sheets for each stage of the
    pipeline. It intelligently appends the new version's results without
    duplicating or removing old data.

Usage:
------
The script is run from the command line after one or more stages of the main
test pipeline have been completed.

    python results_compiler.py --version-id <version_id> --change-description "<description>"

Example:
    python results_compiler.py --version-id "v3.2.0" --change-description "Added VIX as a new feature."


how to run the script?
Generate Visuals: Run the visualization script to create all charts.
python results_compiler.py --version-id "v3.1.2" --change-description "Added VIX as new feature."

Generate the Final Report: Run the new report generator.
python generate_report.py --version-id "v3.1.2"

After these steps, you will have a file named reports/StockWise_Report_v3.1.2.html
that contains all of your tables and charts in a single, professional report.

"""


# results_compiler.py
import pandas as pd
import numpy as np
import os
import glob
import argparse
import re
from datetime import datetime

# --- Configuration ---
REPORTS_DIR = "reports"
BACKTEST_RESULTS_DIR = os.path.join(REPORTS_DIR, "backtest_results")
MODELS_DIR = "models"
LOGS_DIR = "logs"
EXCEL_FILE_PATH = os.path.join(REPORTS_DIR, "Master_Test_Results.xlsx")
SHEET_NAMES = {
    "versions": "Version Log",
    "datagen": "Data Generation Summary",
    "tuning": "Hyperparameter Tuning Summary",
    "evaluation": "Model Evaluation Summary",
    "backtest": "Backtest Summary",
    "trade_analysis": "Trade Analysis Summary"
}


def find_latest_log_file(log_directory: str) -> str or None:
    """Finds the most recently modified file in a directory."""
    if not os.path.isdir(log_directory):
        return None
    log_files = glob.glob(os.path.join(log_directory, '*.log'))
    if not log_files:
        return None
    return max(log_files, key=os.path.getmtime)


def process_data_gen_summary() -> pd.DataFrame:
    """Parses the latest data generation log to extract key metrics."""
    print("🔎 Processing Data Generation log...")
    log_file = find_latest_log_file(os.path.join(LOGS_DIR, "Create_parquet_file_log"))
    if not log_file:
        print("⚠️ Data Generation log not found.")
        return pd.DataFrame()

    with open(log_file, 'r') as f:
        content = f.read()

    processed = re.search(r"Processed (\d+) stocks successfully", content)
    skipped = re.search(r"Skipped (\d+) stocks", content)
    thresholds = re.search(r"Global Volatility Thresholds Calculated: Low < (\d+\.\d+), High > (\d+\.\d+)", content)

    data = {
        "Stocks Processed": int(processed.group(1)) if processed else 0,
        "Stocks Skipped": int(skipped.group(1)) if skipped else 0,
        "Low Vol Threshold": float(thresholds.group(1)) if thresholds else 0,
        "High Vol Threshold": float(thresholds.group(2)) if thresholds else 0,
    }
    return pd.DataFrame([data])


def process_tuning_summary() -> pd.DataFrame:
    """Parses the latest hyperparameter tuning log."""
    print("🔎 Processing Hyperparameter Tuning log...")
    log_file = find_latest_log_file(os.path.join(LOGS_DIR, "hyperparameter_tuner_log"))
    if not log_file:
        print("⚠️ Hyperparameter Tuning log not found.")
        return pd.DataFrame()

    with open(log_file, 'r') as f:
        lines = f.readlines()

    results = []
    pattern = re.compile(r"Tuning finished for (.+?) for (.+?) Agent. Best F1-Score: (\d+\.\d+)")
    for line in lines:
        match = pattern.search(line)
        if match:
            results.append({
                "Agent": match.group(2).strip().replace("Pct", "% Profit"),
                "Model Name": match.group(1).strip(),
                "Best F1-Score": float(match.group(3))
            })
    return pd.DataFrame(results)


def calculate_metrics_from_equity_curve(equity_df: pd.DataFrame) -> dict:
    """Calculates financial KPIs from a portfolio equity curve."""
    if len(equity_df) < 2:
        return {'Total Return (%)': 0, 'Max Drawdown (%)': 0, 'Sharpe Ratio': 0}

    total_return = (equity_df.iloc[-1] / equity_df.iloc[0] - 1) * 100
    drawdown = (equity_df - equity_df.cummax()) / equity_df.cummax()
    max_drawdown = abs(drawdown.min()) * 100
    returns = equity_df.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0

    return {
        'Total Return (%)': total_return,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }


def process_backtest_results() -> pd.DataFrame:
    """Finds and processes all equity curve CSVs to summarize financial performance."""
    print("🔎 Processing backtest financial results...")
    equity_files = glob.glob(os.path.join(BACKTEST_RESULTS_DIR, 'equity_curve_*_Backtesting.csv'))

    all_metrics = []
    for file in equity_files:
        try:
            agent_name = os.path.basename(file).split('_')[2]
            equity_df = pd.read_csv(file, index_col=0, header=0, names=['Date', 'Portfolio_Value'])
            equity_df = equity_df['Portfolio_Value']

            metrics = calculate_metrics_from_equity_curve(equity_df)
            metrics['Agent'] = agent_name.replace('pct', '% Profit')
            all_metrics.append(metrics)
        except Exception as e:
            print(f"⚠️ Could not process file {file}: {e}")

    return pd.DataFrame(all_metrics)


def process_model_evaluation_results() -> pd.DataFrame:
    """Finds and processes all model evaluation summaries."""
    print("🔎 Processing model evaluation (F1-score) results...")
    eval_files = glob.glob(os.path.join(MODELS_DIR, '**', 'gen3_model_performance_summary.csv'), recursive=True)

    all_evals = []
    for file in eval_files:
        try:
            agent_name = file.split(os.sep)[-2].replace('NASDAQ-gen3-', '').replace('pct', '% Profit')
            eval_df = pd.read_csv(file)
            eval_df['Agent'] = agent_name
            all_evals.append(eval_df)
        except Exception as e:
            print(f"⚠️ Could not process file {file}: {e}")

    return pd.concat(all_evals, ignore_index=True) if all_evals else pd.DataFrame()


def process_trade_analysis_summary() -> pd.DataFrame:
    """Processes the trade entry analysis CSV files."""
    print("🔎 Processing Trade Analysis results...")
    analysis_files = glob.glob(os.path.join(BACKTEST_RESULTS_DIR, 'trade_entry_analysis_*.csv'))

    all_analyses = []
    for file in analysis_files:
        try:
            agent_name = os.path.basename(file).split('_')[3].replace('.csv', '')
            df = pd.read_csv(file)
            dip_buy_pct = (df['trade_type'] == 'Dip Buy').mean() * 100
            all_analyses.append({
                "Agent": agent_name.replace('pct', '% Profit'),
                "Total Trades": len(df),
                "Percent Dip Buys": dip_buy_pct
            })
        except Exception as e:
            print(f"⚠️ Could not process file {file}: {e}")

    return pd.DataFrame(all_analyses)


def update_workbook(version_id: str, change_desc: str, dfs: dict):
    """Creates or updates the Master Test Results Excel workbook with all data."""
    print(f"📝 Updating workbook for Version ID: {version_id}")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    sheets = {}
    if os.path.exists(EXCEL_FILE_PATH):
        print(f"   - Found existing workbook at '{EXCEL_FILE_PATH}'. Appending new results.")
        with pd.ExcelFile(EXCEL_FILE_PATH) as xls:
            for key, name in SHEET_NAMES.items():
                if name in xls.sheet_names:
                    sheets[key] = pd.read_excel(xls, sheet_name=name)
    else:
        print(f"   - No existing workbook found. Creating '{EXCEL_FILE_PATH}'.")

    # Ensure all sheets exist, even if empty
    if 'versions' not in sheets:
        sheets['versions'] = pd.DataFrame(columns=['Version ID', 'Date', 'Key Change / Hypothesis'])

    # Create and append the new version log entry
    new_version_entry = pd.DataFrame([{'Version ID': version_id, 'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                       'Key Change / Hypothesis': change_desc}])
    sheets['versions'] = pd.concat([sheets['versions'], new_version_entry], ignore_index=True).drop_duplicates(
        subset=['Version ID'], keep='last')

    # Append new results to each corresponding sheet
    for key, df_new in dfs.items():
        if not df_new.empty:
            df_new['Version ID'] = version_id
            df_old = sheets.get(key, pd.DataFrame())

            # CORRECTED: Only filter if the old dataframe is not empty
            if not df_old.empty:
                df_old = df_old[df_old['Version ID'] != version_id]

            sheets[key] = pd.concat([df_old, df_new], ignore_index=True)

    with pd.ExcelWriter(EXCEL_FILE_PATH, engine='openpyxl') as writer:
        for key, df_to_write in sheets.items():
            if not df_to_write.empty:
                df_to_write.to_excel(writer, sheet_name=SHEET_NAMES[key], index=False)

    print(f"✅ Workbook updated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile all test results into the Master Test Results Workbook.")
    parser.add_argument('--version-id', required=True, type=str, help='A unique ID for this test run (e.g., "v3.1.2").')
    parser.add_argument('--change-description', required=True, type=str,
                        help='A description of the change being tested.')
    args = parser.parse_args()

    all_dfs = {
        "datagen": process_data_gen_summary(),
        "tuning": process_tuning_summary(),
        "evaluation": process_model_evaluation_results(),
        "backtest": process_backtest_results(),
        "trade_analysis": process_trade_analysis_summary()
    }

    update_workbook(
        version_id=args.version_id,
        change_desc=args.change_description,
        dfs=all_dfs
    )