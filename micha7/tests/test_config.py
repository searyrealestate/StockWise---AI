"""TDD test suite for micha7.config (ConfigLoader + get_logger).

All tests written BEFORE implementation — expect RED on first run.
"""

import io
import json
import logging
import pathlib
import re

import pytest

from micha7.config import ConfigError, ConfigLoader, get_logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_config(tmp_path):
    """Write a minimal config.json into tmp_path and return its path."""
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "meta": {"name": "micha7_analyzer", "config_version": "1.0.0"},
                "logging": {
                    "level": "INFO",
                    "format": "json",
                    "directory": str(tmp_path / "logs"),
                    "console": True,
                },
                "numbers": {"score": 5, "ratio": 1.5},
            }
        ),
        encoding="utf-8",
    )
    return cfg


@pytest.fixture()
def loader(base_config, tmp_path):
    """Return a loaded ConfigLoader backed by tmp_path config."""
    cl = ConfigLoader(
        config_path=base_config,
        local_path=tmp_path / "config.local.json",
    )
    cl.load()
    return cl


# ---------------------------------------------------------------------------
# ConfigLoader — load()
# ---------------------------------------------------------------------------


def test_load_reads_config_json(base_config, tmp_path):
    cl = ConfigLoader(
        config_path=base_config,
        local_path=tmp_path / "config.local.json",
    )
    data = cl.load()
    assert data["meta"]["name"] == "micha7_analyzer"


def test_load_missing_file_raises_configerror(tmp_path):
    cl = ConfigLoader(
        config_path=tmp_path / "nonexistent.json",
        local_path=tmp_path / "config.local.json",
    )
    with pytest.raises(ConfigError):
        cl.load()


def test_load_invalid_json_raises_configerror(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text("{invalid json", encoding="utf-8")
    cl = ConfigLoader(
        config_path=bad,
        local_path=tmp_path / "config.local.json",
    )
    with pytest.raises(ConfigError):
        cl.load()


def test_local_overrides_merge_over_base(base_config, tmp_path):
    local = tmp_path / "config.local.json"
    local.write_text(
        json.dumps({"logging": {"level": "DEBUG"}}),
        encoding="utf-8",
    )
    cl = ConfigLoader(config_path=base_config, local_path=local)
    cl.load()
    assert cl.get("logging.level") == "DEBUG"
    # base key not in local must still be present
    assert cl.get("meta.name") == "micha7_analyzer"


def test_local_missing_is_ok(base_config, tmp_path):
    cl = ConfigLoader(
        config_path=base_config,
        local_path=tmp_path / "config.local.json",  # does not exist
    )
    data = cl.load()
    assert data is not None


# ---------------------------------------------------------------------------
# ConfigLoader — get()
# ---------------------------------------------------------------------------


def test_get_returns_value_for_existing_path(loader):
    assert loader.get("logging.level") == "INFO"


def test_get_returns_default_for_missing_path(loader):
    assert loader.get("nope.x", default=5) == 5


def test_get_type_mismatch_raises(loader):
    with pytest.raises(ConfigError):
        loader.get("logging.level", expected_type=int)


def test_get_below_min_raises(loader):
    # numbers.score == 5; set min_val=10 → should raise
    with pytest.raises(ConfigError):
        loader.get("numbers.score", min_val=10)


def test_get_above_max_raises(loader):
    # numbers.score == 5; set max_val=3 → should raise
    with pytest.raises(ConfigError):
        loader.get("numbers.score", max_val=3)


def test_get_bool_not_treated_as_int(base_config, tmp_path):
    """A bool value with expected_type=int must raise (bool is subclass of int
    in Python, but ConfigLoader must reject the implicit coercion)."""
    cl = ConfigLoader(config_path=base_config, local_path=tmp_path / "config.local.json")
    cl.load()
    with pytest.raises(ConfigError):
        cl.get("logging.console", expected_type=int)


# ---------------------------------------------------------------------------
# ConfigLoader — require()
# ---------------------------------------------------------------------------


def test_require_missing_raises(loader):
    with pytest.raises(ConfigError):
        loader.require("nope.missing")


def test_require_present_returns_value(loader):
    assert loader.require("meta.name") == "micha7_analyzer"


# ---------------------------------------------------------------------------
# Logger — get_logger()
# ---------------------------------------------------------------------------


@pytest.fixture()
def json_log_line(loader, tmp_path):
    """Capture one JSON log line emitted by get_logger."""
    logger = get_logger("test.component", loader=loader, correlation_id="test-corr-id")
    # Attach an in-memory handler to capture output
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    # Copy the JSON formatter from the first handler already on the logger
    if logger.handlers:
        handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)
    logger.info(
        "test message",
        extra={"event": "TEST_EVENT", "context": {"key": "val"}, "correlation_id": "test-corr-id"},
    )
    handler.flush()
    stream.seek(0)
    line = stream.getvalue().strip().split("\n")[0]
    logger.removeHandler(handler)
    return line


def test_get_logger_emits_valid_json(json_log_line):
    data = json.loads(json_log_line)
    for key in ("timestamp", "component", "level", "event", "message", "correlation_id", "context"):
        assert key in data, f"Missing key: {key}"


def test_log_timestamp_is_utc_iso8601_z(json_log_line):
    data = json.loads(json_log_line)
    ts = data["timestamp"]
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z"
    assert re.fullmatch(pattern, ts), f"Timestamp does not match UTC ISO 8601 Z format: {ts!r}"


def test_get_logger_idempotent_no_duplicate_handlers(loader):
    logger1 = get_logger("idempotent.test", loader=loader)
    count_after_first = len(logger1.handlers)
    logger2 = get_logger("idempotent.test", loader=loader)
    assert logger1 is logger2
    assert len(logger2.handlers) == count_after_first


def test_correlation_id_propagates(loader, tmp_path):
    stream = io.StringIO()
    logger = get_logger("corr.test", loader=loader, correlation_id="abc-123")
    handler = logging.StreamHandler(stream)
    if logger.handlers:
        handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)
    logger.info(
        "msg",
        extra={"event": "E", "context": {}, "correlation_id": "abc-123"},
    )
    handler.flush()
    stream.seek(0)
    data = json.loads(stream.getvalue().strip().split("\n")[0])
    assert data["correlation_id"] == "abc-123"
    logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Feature config — defaults and range validation (D-23, Q1-Q4)
# ---------------------------------------------------------------------------

_REAL_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.json"


@pytest.fixture()
def features_loader(tmp_path):
    """ConfigLoader backed by the real micha7/config.json (no local override)."""
    cl = ConfigLoader(
        config_path=_REAL_CONFIG_PATH,
        local_path=tmp_path / "config.local.json",
    )
    cl.load()
    return cl


def test_features_candle_hammer_wick_ratio(features_loader):
    val = features_loader.get(
        "features.candle.hammer_wick_ratio",
        expected_type=float,
        min_val=1.5,
        max_val=4.0,
    )
    assert val == 2.0


def test_features_cci_overbought(features_loader):
    val = features_loader.get(
        "features.cci.overbought",
        expected_type=int,
        min_val=50,
        max_val=200,
    )
    assert val == 100


def test_features_cci_oversold(features_loader):
    val = features_loader.get(
        "features.cci.oversold",
        expected_type=int,
        min_val=-200,
        max_val=-50,
    )
    assert val == -100


def test_features_sr_lookback_n(features_loader):
    val = features_loader.get(
        "features.sr.lookback_n",
        expected_type=int,
        min_val=2,
        max_val=20,
    )
    assert val == 5


def test_features_sr_cluster_atr(features_loader):
    val = features_loader.get(
        "features.sr.cluster_atr",
        expected_type=float,
        min_val=0.1,
        max_val=2.0,
    )
    assert val == 0.5


def test_features_gap_max_age_bars(features_loader):
    val = features_loader.get(
        "features.gap.max_age_bars",
        expected_type=int,
        min_val=5,
        max_val=250,
    )
    assert val == 60


def test_features_param_out_of_range_raises(features_loader):
    # max_age_bars=60; min_val=100 → 60 < 100 → ConfigError
    with pytest.raises(ConfigError):
        features_loader.get("features.gap.max_age_bars", min_val=100)
