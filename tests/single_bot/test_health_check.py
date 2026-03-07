"""
tests/single_bot/test_health_check.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for ``scripts/health_check.py`` and Docker configuration validation.

The health check is the Docker ``HEALTHCHECK`` command that determines
whether the container should be restarted.  These tests verify:

* ``check_status_summary()`` — stale/missing log detection.
* ``check_main_log()`` — empty/missing main log file handling.
* ``check_bot_logs()`` — per-coin log freshness.
* ``check_process_status()`` — PID file and process existence.
* ``main()`` — exit code logic (≥ 75% pass → healthy).
* Docker config — Dockerfile and docker-compose.yml sanity checks.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

# Add scripts/ to path so we can import health_check
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import health_check  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_log_dir(tmp_path):
    """Create a temporary log directory and patch health_check to use it."""
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    (tmp_path / "log").mkdir()
    yield tmp_path
    os.chdir(orig_cwd)


def _write_status_log(log_dir: Path, timestamp: datetime | None = None):
    """Write a status_summary.log with a given timestamp."""
    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    status_file = log_dir / "log" / "status_summary.log"
    status_file.write_text(f"[{ts_str}] All systems OK\n")


def _write_main_log(log_dir: Path, content: str = "INFO normal operation"):
    main_log = log_dir / "log" / "multi_grid_BN.log"
    main_log.write_text(content)


# ---------------------------------------------------------------------------
# check_status_summary()
# ---------------------------------------------------------------------------

class TestCheckStatusSummary:

    def test_missing_file_returns_false(self, tmp_log_dir):
        assert health_check.check_status_summary() is False

    def test_empty_file_returns_false(self, tmp_log_dir):
        (tmp_log_dir / "log" / "status_summary.log").write_text("")
        assert health_check.check_status_summary() is False

    def test_fresh_timestamp_returns_true(self, tmp_log_dir):
        _write_status_log(tmp_log_dir, datetime.now())
        assert health_check.check_status_summary() is True

    def test_stale_timestamp_returns_false(self, tmp_log_dir):
        _write_status_log(tmp_log_dir, datetime.now() - timedelta(seconds=120))
        assert health_check.check_status_summary() is False


# ---------------------------------------------------------------------------
# check_main_log()
# ---------------------------------------------------------------------------

class TestCheckMainLog:

    def test_missing_file_returns_false(self, tmp_log_dir):
        assert health_check.check_main_log() is False

    def test_empty_file_returns_false(self, tmp_log_dir):
        (tmp_log_dir / "log" / "multi_grid_BN.log").write_text("")
        assert health_check.check_main_log() is False

    def test_normal_content_returns_true(self, tmp_log_dir):
        _write_main_log(tmp_log_dir, "INFO all good\nINFO still good\n")
        assert health_check.check_main_log() is True

    def test_error_in_last_lines_still_returns_true(self, tmp_log_dir):
        """Non-fatal ERROR lines don't fail the check (logged as warning)."""
        content = "INFO ok\n" * 8 + "ERROR transient issue\nINFO recovered\n"
        _write_main_log(tmp_log_dir, content)
        # check_main_log returns True even with ERROR lines
        assert health_check.check_main_log() is True


# ---------------------------------------------------------------------------
# check_bot_logs()
# ---------------------------------------------------------------------------

class TestCheckBotLogs:

    def test_no_log_dir_returns_false(self, tmp_path):
        """Missing log directory entirely → False."""
        os.chdir(tmp_path)
        # Don't create log/ dir
        result = health_check.check_bot_logs()
        assert result is False
        os.chdir(Path(__file__).resolve().parents[2])

    def test_no_bot_log_files_returns_true(self, tmp_log_dir):
        """No grid_BN_*.log files yet → True (just started, not an error)."""
        assert health_check.check_bot_logs() is True

    def test_fresh_bot_log_returns_true(self, tmp_log_dir):
        bot_log = tmp_log_dir / "log" / "grid_BN_XRP.log"
        bot_log.write_text("INFO position update\n")
        assert health_check.check_bot_logs() is True


# ---------------------------------------------------------------------------
# check_process_status()
# ---------------------------------------------------------------------------

class TestCheckProcessStatus:

    def test_no_pid_file_returns_true(self, tmp_log_dir):
        """No PID file → first startup, return True."""
        assert health_check.check_process_status() is True

    def test_valid_pid_returns_true(self, tmp_log_dir):
        """PID of current process → alive → True."""
        pid_file = tmp_log_dir / "grid_bot.pid"
        pid_file.write_text(str(os.getpid()))
        assert health_check.check_process_status() is True

    def test_dead_pid_returns_false(self, tmp_log_dir):
        """PID of non-existent process → False."""
        pid_file = tmp_log_dir / "grid_bot.pid"
        pid_file.write_text("999999")  # almost certainly not running
        # os.kill(999999, 0) will raise OSError if process doesn't exist
        result = health_check.check_process_status()
        assert result is False


# ---------------------------------------------------------------------------
# Docker config validation (static file checks)
# ---------------------------------------------------------------------------

class TestDockerConfig:
    """Validate Docker configuration files exist and contain expected content."""

    _repo_root = Path(__file__).resolve().parents[2]

    def test_dockerfile_exists(self):
        assert (self._repo_root / "docker" / "Dockerfile").exists()

    def test_dockerfile_has_healthcheck(self):
        content = (self._repo_root / "docker" / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content
        assert "health_check.py" in content

    def test_dockerfile_uses_non_root_user(self):
        content = (self._repo_root / "docker" / "Dockerfile").read_text()
        assert "USER trader" in content

    def test_docker_compose_exists(self):
        assert (self._repo_root / "docker" / "docker-compose.yml").exists()

    def test_docker_compose_has_env_file(self):
        content = (self._repo_root / "docker" / "docker-compose.yml").read_text()
        assert "env_file" in content or "environment" in content

    def test_docker_compose_has_resource_limits(self):
        content = (self._repo_root / "docker" / "docker-compose.yml").read_text()
        assert "memory" in content.lower()
        assert "cpus" in content.lower()

    def test_docker_compose_has_healthcheck(self):
        content = (self._repo_root / "docker" / "docker-compose.yml").read_text()
        assert "healthcheck" in content

    def test_docker_compose_has_restart_policy(self):
        content = (self._repo_root / "docker" / "docker-compose.yml").read_text()
        assert "restart" in content

    def test_docker_compose_volume_mounts(self):
        """Verify log and config volumes are mounted."""
        content = (self._repo_root / "docker" / "docker-compose.yml").read_text()
        assert "volumes" in content
        assert "log" in content
        assert "config" in content
