# utils.py
"""
Comprehensive HTML Report Generator
===================================

This script serves as the final step in the version testing pipeline. It
gathers all the compiled results and visual artifacts for a specific version
test and consolidates them into a single, professional, self-contained HTML report.

The resulting report provides a complete overview of a version's performance,
from data generation metrics to final backtest results, making it easy to
analyze, share, and archive.

How it Works:
-------------
1.  **Generates Visuals**: It first runs the `visualize_results.py` script to
    ensure all comparison charts (Sharpe Ratio, Max Drawdown, Equity Curves)
    are created and up-to-date.
2.  **Loads Master Data**: It reads all the data sheets from the master Excel
    workbook (`Master_Test_Results.xlsx`).
3.  **Filters by Version**: It filters the data from each sheet to isolate the
    results corresponding to the `--version-id` provided by the user.
4.  **Creates HTML Tables**: The filtered data for each stage (Data Generation,
    Tuning, Evaluation, Backtesting) is converted into formatted HTML tables.
5.  **Embeds Charts**: It reads the generated PNG chart images, encodes them
    into base64 strings, and embeds them directly into the HTML file. This
    makes the final report fully self-contained.
6.  **Populates Template**: It injects the version information, change description,
    HTML tables, and embedded charts into a master HTML template.

Output:
-------
A single HTML file named `reports/StockWise_Report_<version_id>.html` that
can be viewed in any web browser without needing any external files.

Usage:
------
The script is run from the command line after the `results_compiler.py` has
been run for all stages of a version test.

    python generate_report.py --version-id <version_id>

Example:
    python generate_report.py --version-id "v3.2.0"
"""

import pandas as pd
import numpy as np


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates a DataFrame from any source.
    - Handles empty or None DataFrames.
    - Flattens multi-level column indexes (from yfinance).
    - Converts all column names to lowercase.
    - Standardizes 'adj close' to 'close'.
    - FIX: Removes all timezone information to prevent comparison errors.
    - Validates that essential columns are present.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # This prevents the "ambiguous value" error during backtesting.
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns = [col.lower() for col in df.columns]

    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # --- Handle Adjusted Close ---
    # Use 'adj close' as the new 'close' if it exists
    if 'adj close' in df.columns:
        df['close'] = df['adj close']
        # We drop all columns that might be 'adj close' in case of duplicates
        df = df.drop(columns=[col for col in df.columns if col == 'adj close'])

    # --- THIS IS THE FIX for the TypeError ---
    # 4. Remove all timezone info from the index
    if pd.api.types.is_datetime64_any_dtype(df.index) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # --- END FIX ---

    standard_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in standard_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    existing_cols = [col for col in standard_cols if col in df.columns]

    # Critical check: Ensure we have the required columns *after* cleaning
    if not all(col in df.columns for col in standard_cols):
        # This can happen if yfinance returns a weird format
        return pd.DataFrame()  # Return empty, not a subset

    if existing_cols:
        df.dropna(subset=existing_cols, inplace=True)

    # Return only the standard columns
    return df[existing_cols]