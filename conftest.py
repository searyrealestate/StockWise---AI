import pytest

def pytest_addoption(parser):
    """Adds a custom command-line option to specify the backtest mode."""
    parser.addoption("--mode", action="store", default="long", help="Backtest mode: 'short' or 'long'")

@pytest.fixture(scope="session")
def backtest_mode(request):
    """Fixture that returns the value of the --mode option."""
    return request.config.getoption("--mode")