"""Tests for file cleanup utilities (app/utils/files.py)."""

from pathlib import Path

from app.utils.files import cleanup_files


class TestCleanupFiles:
    """cleanup_files deletion behavior tests."""

    def test_deletes_existing_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("data")
        f2.write_text("data")
        cleanup_files(f1, f2)
        assert not f1.exists()
        assert not f2.exists()

    def test_skips_none_paths(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("data")
        # Should not raise
        cleanup_files(None, f1, None)
        assert not f1.exists()

    def test_ignores_already_missing_files(self, tmp_path):
        missing = tmp_path / "gone.txt"
        # Should not raise FileNotFoundError
        cleanup_files(missing)

    def test_no_args_is_noop(self):
        cleanup_files()

    def test_mixed_existing_and_missing(self, tmp_path):
        exists = tmp_path / "exists.txt"
        exists.write_text("data")
        missing = tmp_path / "missing.txt"
        cleanup_files(exists, missing)
        assert not exists.exists()
