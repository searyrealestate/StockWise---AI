"""
Pytest Configuration for System Performance Tests
===============================================

This file (`conftest.py`) is a special pytest plugin that allows for the customization
of the test framework's behavior. It is used to add custom command-line options
for controlling the scope and execution of the system performance backtests.

This setup provides flexibility for both quick development checks and comprehensive
validation runs.

Command-Line Options Added:
---------------------------
--mode : str
    Determines the duration and scope of the backtests.
    - 'short' (default): Runs a quick backtest on a smaller, recent date range.
                         Ideal for rapid checks during development.
    - 'long': Runs a full, comprehensive backtest over the entire available
              historical dataset.

--debug-agent : str
    Specifies a single trading agent to run the backtests for. If provided,
    pytest will skip tests for all other agents.
    - Example: 'dynamic', '2pct', '3pct'
    - If not provided, tests will run for all agents.

Usage Examples:
---------------
# Run a quick, short backtest for all agents
pytest -s --mode=short

# Run a full, comprehensive backtest for all agents
pytest -s --mode=long

# Run a full backtest in debug mode for only the 'dynamic' agent
pytest -s --mode=long --debug-agent=dynamic
"""


import pytest

def pytest_addoption(parser):
    """Adds global command-line options to pytest."""
    parser.addoption("--mode", action="store", default="short",
                     help="Backtest mode: 'short' or 'long'")
    parser.addoption("--debug-agent", action="store", default=None,
                     help="Run in debug mode for a single agent (e.g., 'dynamic').")


@pytest.fixture(scope="session")
def backtest_mode(request):
    """Fixture that returns the value of the --mode option."""
    return request.config.getoption("--mode")