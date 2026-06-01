"""Configuration loading and structured logging for micha7.

ConfigLoader: loads config.json, merges optional config.local.json overrides,
and validates values against type/range with defaults (Fail-Loud on invalid).
Logger: structured single-line JSON logging (simulator-readable).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ConfigError
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict that is *override* deep-merged on top of *base*.

    Nested dicts are merged recursively; all other types are replaced.
    Neither input is mutated.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class ConfigLoader:
    """Load config.json and optionally merge config.local.json on top.

    Usage::

        loader = ConfigLoader()
        loader.load()
        level = loader.get("logging.level", default="INFO")
    """

    def __init__(
        self,
        config_path: str | Path = "config.json",
        local_path: str | Path = "config.local.json",
    ) -> None:
        self._config_path = Path(config_path)
        self._local_path = Path(local_path)
        self._data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """Read config_path (required) and deep-merge local_path if present.

        Returns the merged config dict.
        Raises ConfigError on missing file or invalid JSON.
        """
        base = self._read_json(self._config_path, required=True)
        if self._local_path.exists():
            local = self._read_json(self._local_path, required=False)
            base = _deep_merge(base, local)
        self._data = base
        return self._data

    def get(
        self,
        path: str,
        default: Any = None,
        expected_type: type | None = None,
        min_val: Any = None,
        max_val: Any = None,
    ) -> Any:
        """Look up a dotted *path* in the loaded config.

        - Missing path → return *default* (no error).
        - Present but *expected_type* mismatch → raise ConfigError.
          Bool values are never accepted as int even though ``bool`` is a
          subclass of ``int`` in Python.
        - Numeric value outside [*min_val*, *max_val*] → raise ConfigError.
        """
        value = self._lookup(path)
        if value is _MISSING:
            return default

        if expected_type is not None:
            # Reject bool masquerading as int
            if expected_type is int and isinstance(value, bool):
                raise ConfigError(
                    f"Config key '{path}': expected int but got bool."
                )
            if not isinstance(value, expected_type):
                raise ConfigError(
                    f"Config key '{path}': expected {expected_type.__name__} "
                    f"but got {type(value).__name__}."
                )

        if min_val is not None and value < min_val:
            raise ConfigError(
                f"Config key '{path}': value {value!r} is below minimum {min_val!r}."
            )
        if max_val is not None and value > max_val:
            raise ConfigError(
                f"Config key '{path}': value {value!r} is above maximum {max_val!r}."
            )

        return value

    def require(self, path: str) -> Any:
        """Return value at *path*; raise ConfigError if missing."""
        value = self._lookup(path)
        if value is _MISSING:
            raise ConfigError(f"Required config key '{path}' is missing.")
        return value

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_json(self, path: Path, *, required: bool) -> dict:
        if not path.exists():
            if required:
                raise ConfigError(f"Config file not found: {path}")
            return {}
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc

    def _lookup(self, path: str) -> Any:
        """Traverse dotted *path* in self._data; return _MISSING if absent."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return _MISSING
            node = node[part]
        return node


# Sentinel for "key not present" — avoids None ambiguity
class _MissingSentinel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record (simulator-readable)."""

    def __init__(self, correlation_id: str | None = None) -> None:
        super().__init__()
        self._correlation_id = correlation_id

    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S")
            + "Z"
        )
        payload = {
            "timestamp": ts,
            "component": record.name,
            "level": record.levelname,
            "event": getattr(record, "event", None),
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", self._correlation_id),
            "context": getattr(record, "context", {}),
        }
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_DIR = "outputs/logs"
_DEFAULT_CONSOLE = True


def get_logger(
    name: str,
    loader: ConfigLoader | None = None,
    correlation_id: str | None = None,
) -> logging.Logger:
    """Return a stdlib Logger configured with the JSON formatter.

    Idempotent: calling twice with the same *name* returns the same Logger
    and does not add duplicate handlers.

    Log directory is created if absent. Level/directory/console are read
    from *loader* when provided; otherwise defaults are used.
    """
    logger = logging.getLogger(name)

    # Idempotency: if handlers already attached, return as-is
    if logger.handlers:
        return logger

    # --- resolve settings from loader or defaults ---
    if loader is not None:
        level_str = loader.get("logging.level", default=_DEFAULT_LOG_LEVEL)
        log_dir = loader.get("logging.directory", default=_DEFAULT_LOG_DIR)
        console = loader.get("logging.console", default=_DEFAULT_CONSOLE)
    else:
        level_str = _DEFAULT_LOG_LEVEL
        log_dir = _DEFAULT_LOG_DIR
        console = _DEFAULT_CONSOLE

    level = getattr(logging, str(level_str).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = _JsonFormatter(correlation_id=correlation_id)

    # --- file handler ---
    log_path = Path(log_dir)
    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler(log_path / f"{name.replace('.', '_')}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- optional console handler ---
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
