"""
StockWise — Versioned Save Tests
Validates timestamped copies are created correctly.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from versioned_save import _get_git_short_hash, list_history, save_versioned_copy


# ═══════════════════════════════════════════════════════════
# Unit tests
# ═══════════════════════════════════════════════════════════

class TestVersionedSave:
    """Tests for versioned output saving."""

    def test_creates_versioned_copy(self, tmp_path):
        """T1: Creates a timestamped copy in history folder."""
        source = tmp_path / "results.json"
        source.write_text('{"test": true}')

        dest = save_versioned_copy(str(source), "history")

        assert dest is not None
        assert os.path.exists(dest)
        assert "history" in dest
        assert "results_" in os.path.basename(dest)

    def test_preserves_original(self, tmp_path):
        """T2: Original file is NOT modified or deleted."""
        source = tmp_path / "results.json"
        source.write_text('{"original": true}')

        save_versioned_copy(str(source), "history")

        assert source.exists()
        assert json.loads(source.read_text()) == {"original": True}

    def test_includes_git_hash(self, tmp_path):
        """T3: Versioned filename includes git hash."""
        source = tmp_path / "data.json"
        source.write_text("{}")

        dest = save_versioned_copy(str(source), "hist")
        basename = os.path.basename(dest)

        # Should have format: data_YYYYMMDD_HHMMSS_<hash>.json
        parts = basename.replace(".json", "").split("_")
        assert len(parts) >= 4, f"Expected timestamp+hash, got: {basename}"

    def test_includes_label(self, tmp_path):
        """T4: Optional label is appended to filename."""
        source = tmp_path / "report.docx"
        source.write_text("fake docx")

        dest = save_versioned_copy(str(source), "history", label="post_fix")
        basename = os.path.basename(dest)

        assert "post_fix" in basename

    def test_multiple_runs_no_overwrite(self, tmp_path):
        """T5: Multiple runs create separate files (different timestamps)."""
        source = tmp_path / "results.json"
        source.write_text('{"run": 1}')

        dest1 = save_versioned_copy(str(source), "history")
        time.sleep(1.1)  # ensure different second
        source.write_text('{"run": 2}')
        dest2 = save_versioned_copy(str(source), "history")

        assert dest1 != dest2
        assert os.path.exists(dest1)
        assert os.path.exists(dest2)

    def test_missing_source_returns_none(self, tmp_path):
        """T6: Non-existent source → returns None, no crash."""
        dest = save_versioned_copy(str(tmp_path / "nope.json"), "history")
        assert dest is None

    def test_creates_history_dir(self, tmp_path):
        """T7: History folder is created if it doesn't exist."""
        source = tmp_path / "data.json"
        source.write_text("{}")

        history_dir = tmp_path / "new_history"
        assert not history_dir.exists()

        save_versioned_copy(str(source), "new_history")
        assert history_dir.exists()

    def test_git_hash_returns_string(self):
        """T8: _get_git_short_hash returns a non-empty string."""
        h = _get_git_short_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_list_history_returns_recent(self, tmp_path):
        """T9: list_history returns files sorted most recent first."""
        hist_dir = tmp_path / "history"
        hist_dir.mkdir()

        for i in range(3):
            (hist_dir / f"file_{i}.json").write_text(f'{{"i": {i}}}')
            time.sleep(0.1)

        files = list_history(str(hist_dir), limit=2)
        assert len(files) == 2
        # Most recent should be first
        assert "file_2" in files[0][0]

    def test_list_history_empty_dir(self, tmp_path):
        """T10: list_history on empty dir → empty list."""
        hist_dir = tmp_path / "empty"
        hist_dir.mkdir()
        assert list_history(str(hist_dir)) == []

    def test_list_history_missing_dir(self):
        """T11: list_history on non-existent dir → empty list."""
        assert list_history("/nonexistent/path") == []


# ═══════════════════════════════════════════════════════════
# Regression guards
# ═══════════════════════════════════════════════════════════

class TestVersionedSaveRegression:
    """Regression guards."""

    def test_docx_extension_preserved(self, tmp_path):
        """R1: DOCX files keep .docx extension in versioned copy."""
        source = tmp_path / "report.docx"
        source.write_text("fake")
        dest = save_versioned_copy(str(source), "history")
        assert dest.endswith(".docx")

    def test_json_extension_preserved(self, tmp_path):
        """R2: JSON files keep .json extension in versioned copy."""
        source = tmp_path / "data.json"
        source.write_text("{}")
        dest = save_versioned_copy(str(source), "history")
        assert dest.endswith(".json")
