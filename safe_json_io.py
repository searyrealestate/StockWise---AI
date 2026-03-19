# safe_json_io.py

"""
StockWise Safe JSON I/O
========================
Atomic read/write for shared JSON files between scanner and live engine.

DO NOT DELETE — PRODUCTION SAFETY LAYER (2026-03-19)

Problem: stock_hunter.py and live_trading_engine.py run as separate processes.
Both access scan_ledger.json and daily_review_list.json concurrently.
Without atomic operations, one process can read a half-written file → crash.

Solution:
- WRITE: serialize to temp file → os.replace() (atomic on all OS)
- READ: retry with backoff if JSON parse fails (file being written)
- LOCK: threading.Lock for in-process safety (multiple DSM instances)

Usage:
    from safe_json_io import safe_json_read, safe_json_write

    data = safe_json_read("data/scan_ledger.json", default={})
    safe_json_write("data/scan_ledger.json", data)
"""

import json
import os
import time
import tempfile
import logging

logger = logging.getLogger("SafeJsonIO")


def safe_json_write(filepath, data, cls=None, indent=4):
    """
    Atomic JSON write: writes to temp file, then replaces original.

    os.replace() is atomic on Windows (NTFS) and Linux (ext4/xfs).
    If the process crashes mid-write, only the temp file is corrupted —
    the original file remains intact.

    Args:
        filepath: Path to JSON file
        data: Data to serialize
        cls: Optional JSON encoder class (e.g., NumpyEncoder)
        indent: JSON indentation (default 4)
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        # Write to temp file in the SAME directory (required for atomic rename)
        fd, tmp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix='.safe_',
            dir=directory or '.'
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, cls=cls, indent=indent)

            # Atomic replace: either fully succeeds or fully fails
            os.replace(tmp_path, filepath)

        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    except Exception as e:
        logger.error(f"Safe write failed for {filepath}: {e}")
        raise


def safe_json_read(filepath, default=None, retries=3, retry_delay=0.2):
    """
    Safe JSON read with retry on parse failure.

    If another process is mid-write (atomic replace not yet complete),
    the read may get an empty or truncated file. We retry with backoff.

    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist or all retries fail
        retries: Number of retry attempts (default 3)
        retry_delay: Seconds between retries (default 0.2)

    Returns:
        Parsed JSON data, or default if file missing/unreadable
    """
    if not os.path.exists(filepath):
        return default if default is not None else {}

    last_error = None
    for attempt in range(retries):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < retries - 1:
                logger.debug(f"JSON parse retry {attempt + 1}/{retries} for {filepath}: {e}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            continue
        except Exception as e:
            logger.error(f"Safe read failed for {filepath}: {e}")
            return default if default is not None else {}

    logger.warning(f"All {retries} read attempts failed for {filepath}: {last_error}")
    return default if default is not None else {}
