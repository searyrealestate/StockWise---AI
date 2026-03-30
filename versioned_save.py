"""
StockWise — Versioned Output Saver
===================================
Saves a timestamped copy of output files to a history folder.
Each copy includes the git commit hash for traceability.

Usage:
    from versioned_save import save_versioned_copy
    save_versioned_copy("data/backtest_results.json", "backtest_history")

Result:
    data/backtest_results.json              <- latest (unchanged)
    data/backtest_history/
      backtest_results_20260329_143022_eb3677a.json  <- versioned copy
"""

import logging
import os
import shutil
import subprocess
from datetime import datetime

logger = logging.getLogger("VersionedSave")


def _get_git_short_hash():
    """Get current git commit short hash. Returns 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def save_versioned_copy(source_path, history_folder_name, label=None):
    """
    Copy a file to a history folder with timestamp + git hash.

    Args:
        source_path: Path to the file to copy (e.g. "data/backtest_results.json")
        history_folder_name: Name of subfolder inside the source's parent dir
                            (e.g. "backtest_history")
        label: Optional label to append (e.g. "pre_fix" or "post_fix")

    Returns:
        str: Path to the versioned copy, or None if failed.

    Example:
        save_versioned_copy("data/backtest_results.json", "backtest_history")
        -> data/backtest_history/backtest_results_20260329_143022_eb3677a.json

        save_versioned_copy("data/report.docx", "report_history", label="post_squeeze_fix")
        -> data/report_history/report_20260329_143022_eb3677a_post_squeeze_fix.docx
    """
    if not os.path.exists(source_path):
        logger.warning(f"Versioned save: source not found: {source_path}")
        return None

    try:
        # Build history directory path
        source_dir = os.path.dirname(source_path) or "."
        history_dir = os.path.join(source_dir, history_folder_name)
        os.makedirs(history_dir, exist_ok=True)

        # Build versioned filename
        basename = os.path.basename(source_path)
        name, ext = os.path.splitext(basename)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = _get_git_short_hash()

        if label:
            versioned_name = f"{name}_{timestamp}_{git_hash}_{label}{ext}"
        else:
            versioned_name = f"{name}_{timestamp}_{git_hash}{ext}"

        dest_path = os.path.join(history_dir, versioned_name)

        # Copy file
        shutil.copy2(source_path, dest_path)

        logger.info(f"Versioned copy saved: {dest_path}")
        return dest_path

    except Exception as e:
        logger.warning(f"Versioned save failed for {source_path}: {e}")
        return None


def list_history(history_dir, limit=10):
    """
    List recent versioned files in a history folder.

    Args:
        history_dir: Path to history folder
        limit: Max files to return (most recent first)

    Returns:
        list of (filename, size_bytes, modified_time) tuples
    """
    if not os.path.exists(history_dir):
        return []

    files = []
    for f in os.listdir(history_dir):
        fp = os.path.join(history_dir, f)
        if os.path.isfile(fp):
            stat = os.stat(fp)
            files.append((f, stat.st_size, stat.st_mtime))

    # Sort by modification time, most recent first
    files.sort(key=lambda x: x[2], reverse=True)
    return files[:limit]
