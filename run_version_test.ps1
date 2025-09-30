# run_version_test.ps1 (V2 - With Pre-Flight Checks)
"""
.SYNOPSIS
    A master script to run an end-to-end machine learning experiment pipeline for a new version of the trading system.

.DESCRIPTION
    This PowerShell script orchestrates the entire workflow for testing a new version of the AI trading model. It guides the user through a series of distinct stages, from initial unit tests to final financial backtesting, with manual checkpoints at each step.

    The script is designed to be configured for each new experiment by simply editing the variables at the top. It runs a series of Python scripts and pytest commands in the correct sequence. After each major stage, it calls a 'results_compiler.py' script to aggregate and log the outputs of that stage, associating them with the specified version ID.

.STAGES
    The pipeline is divided into the following sequential stages:
    1.  Pre-Flight Checks: Runs all unit tests to ensure the codebase is stable before starting the time-consuming data processes.
    2.  Data Generation: Executes the main data processing pipeline to create the feature and label sets.
    3.  Hyperparameter Tuning: Runs the Optuna-based tuner to find the best hyperparameters for the specified agent.
    4.  Model Training & Evaluation: Trains the full suite of specialist models and then evaluates their performance on the test set.
    5.  Financial Backtesting: Runs the final, comprehensive backtest to measure the financial performance (e.g., Sharpe Ratio) of the newly trained agent.

.CONFIGURATION
    To run a new version test, edit the following variables at the top of the script:
    - $ versionId: A unique identifier for this test run (e.g., 'v3.2.0').
    - $ changeDescription: A detailed description of the changes being tested and the hypothesis.
    - $ agentToFocusOn: The specific agent (e.g., 'dynamic') that will be tuned, trained, and evaluated.

.USAGE
    .\run_version_test.ps1
"""


# --- CONFIGURE YOUR TEST RUN HERE ---
$versionId = "v3.2.0"
$changeDescription = """Added VIX as a new feature.
Key Change: Added three new market context features: the VIX Index (vix_close), Chaikin Money Flow (cmf),
and Correlation to Bonds (corr_tlt).

Hypothesis: Providing the models with a better understanding of overall market sentiment (VIX),institutional money flow (CMF),
and risk-on/risk-off appetite (TLT correlation) will improve their predictive accuracy and increase the final Sharpe Ratio of the Dynamic Agent.
"""
$agentToFocusOn = "dynamic"
# --- END CONFIGURATION ---

# A helper function to print headers
function Write-StageHeader {
    param($Message)
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host "    $Message" -ForegroundColor Green
    Write-Host "================================================================================"
}

# A helper function to check for errors and exit
function Check-CommandSuccess {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ SCRIPT FAILED. Halting execution." -ForegroundColor Red
        exit 1
    }
}

# --- STAGE 0: PRE-FLIGHT CHECKS ---
Write-StageHeader "STAGE 0: RUNNING ALL UNIT TESTS (PRE-FLIGHT CHECK)"

pytest test_create_parquet_NASDAQ_file.py
Check-CommandSuccess

pytest unit_test_model_trainer.py
Check-CommandSuccess

pytest unit_test_model_evaluator.py
Check-CommandSuccess

# Add any other unit test files here
Write-Host "✅ All unit tests passed."
Read-Host -Prompt "Press Enter to begin the full data pipeline, or Ctrl+C to stop"

# --- STAGE 1: DATA GENERATION ---
Write-StageHeader "STAGE 1: RUNNING DATA GENERATION"
python Create_parquet_file_NASDAQ.py
Check-CommandSuccess
python results_compiler.py --version-id $versionId --change-description $changeDescription --stage datagen
Check-CommandSuccess
Read-Host -Prompt "Press Enter to continue to Stage 2 (Tuning), or Ctrl+C to stop"

# --- STAGE 2: HYPERPARAMETER TUNING ---
Write-StageHeader "STAGE 2: RUNNING HYPERPARAMETER TUNING"
python hyperparameter_tuner.py --agent $agentToFocusOn
Check-CommandSuccess
python results_compiler.py --version-id $versionId --stage tuning --agent $agentToFocusOn
Check-CommandSuccess
Read-Host -Prompt "Press Enter to continue to Stage 3 (Training), or Ctrl+C to stop"

# --- STAGE 3: MODEL TRAINING & EVALUATION ---
Write-StageHeader "STAGE 3: RUNNING MODEL TRAINER & EVALUATOR"
python model_trainer.py --agent $agentToFocusOn
Check-CommandSuccess
python model_evaluator.py --agent $agentToFocusOn
Check-CommandSuccess
python results_compiler.py --version-id $versionId --stage evaluation --agent $agentToFocusOn
Check-CommandSuccess
Read-Host -Prompt "Press Enter to continue to Stage 4 (Backtesting), or Ctrl+C to stop"

# --- STAGE 4: FINANCIAL BACKTESTING ---
Write-StageHeader "STAGE 4: RUNNING FINANCIAL BACKTEST"
pytest -s test_system_performance.py --mode=long
Check-CommandSuccess
python results_compiler.py --version-id $versionId --stage backtest --agent $agentToFocusOn
Check-CommandSuccess

Write-StageHeader "✅ FULL PIPELINE COMPLETE"