"""
tests/single_bot/test_config_validation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for environment-variable parsing and ``validate_config()`` logic.

These tests verify that:
* Required env vars (``API_KEY``, ``API_SECRET``) are enforced.
* Numeric env vars parse correctly and reject garbage input.
* Boolean env vars accept case-insensitive "true"/"false".
* ``validate_config()`` raises ``ValueError`` for invalid configs.
* Layer-gate env vars default to OFF (0 / False) when unset.
* The new ``VEL_DIR_EMA_PERIOD`` env var defaults to 36 and parses ints.
"""

from __future__ import annotations

import importlib
import os
from unittest import mock

import pytest

# We import these AFTER patching env in each test to exercise the parsing.
# For tests that call validate_config() we dynamically reload the module.
MODULE_PATH = "src.single_bot.bitunix_bot"


def _reload_bot_module(env_overrides: dict | None = None):
    """Reload ``bitunix_bot`` with patched env vars and return the module."""
    env = {
        # Minimum viable env for module load (prevents ValueErrors on parse)
        "API_KEY": "test-key",
        "API_SECRET": "test-secret",
        "COIN_NAME": "XRP",
        "GRID_SPACING": "0.015",
        "INITIAL_QUANTITY": "1",
        "LEVERAGE": "2",
        "ENABLE_NOTIFICATIONS": "false",
    }
    if env_overrides:
        env.update(env_overrides)

    with mock.patch.dict(os.environ, env, clear=False):
        mod = importlib.import_module(MODULE_PATH)
        mod = importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# validate_config() — required env vars
# ---------------------------------------------------------------------------

class TestValidateConfigRequired:
    """Ensure validate_config rejects missing/invalid required vars."""

    def test_missing_api_key_raises(self):
        mod = _reload_bot_module({"API_KEY": "", "API_SECRET": "secret"})
        with pytest.raises(ValueError, match="API_KEY"):
            mod.validate_config()

    def test_missing_api_secret_raises(self):
        mod = _reload_bot_module({"API_KEY": "key", "API_SECRET": ""})
        with pytest.raises(ValueError, match="API_SECRET"):
            mod.validate_config()

    def test_grid_spacing_zero_raises(self):
        """GRID_SPACING=0 causes ZeroDivisionError at module load (POSITION_THRESHOLD)."""
        with pytest.raises(ZeroDivisionError):
            _reload_bot_module({"GRID_SPACING": "0.0"})

    def test_grid_spacing_one_raises(self):
        mod = _reload_bot_module({"GRID_SPACING": "1.0"})
        with pytest.raises(ValueError, match="GRID_SPACING"):
            mod.validate_config()

    def test_grid_spacing_negative_raises(self):
        mod = _reload_bot_module({"GRID_SPACING": "-0.01"})
        with pytest.raises(ValueError, match="GRID_SPACING"):
            mod.validate_config()

    def test_initial_quantity_zero_raises(self):
        mod = _reload_bot_module({"INITIAL_QUANTITY": "0"})
        with pytest.raises(ValueError, match="INITIAL_QUANTITY"):
            mod.validate_config()

    def test_leverage_zero_raises(self):
        mod = _reload_bot_module({"LEVERAGE": "0"})
        with pytest.raises(ValueError, match="LEVERAGE"):
            mod.validate_config()

    def test_leverage_over_100_raises(self):
        mod = _reload_bot_module({"LEVERAGE": "101"})
        with pytest.raises(ValueError, match="LEVERAGE"):
            mod.validate_config()

    def test_valid_config_passes(self):
        """No exception for a valid minimal config."""
        mod = _reload_bot_module()
        mod.validate_config()  # should not raise


# ---------------------------------------------------------------------------
# Numeric env var parsing
# ---------------------------------------------------------------------------

class TestNumericEnvParsing:
    """Verify that numeric env vars parse to correct types and defaults."""

    def test_grid_spacing_default(self):
        mod = _reload_bot_module()
        assert mod.GRID_SPACING == 0.015

    def test_grid_spacing_custom(self):
        mod = _reload_bot_module({"GRID_SPACING": "0.025"})
        assert mod.GRID_SPACING == 0.025

    def test_atr_parabolic_mult_default_zero(self):
        mod = _reload_bot_module()
        assert mod.ATR_PARABOLIC_MULT == 0.0

    def test_atr_parabolic_mult_custom(self):
        mod = _reload_bot_module({"ATR_PARABOLIC_MULT": "2.5"})
        assert mod.ATR_PARABOLIC_MULT == 2.5

    def test_vel_atr_mult_default_zero(self):
        mod = _reload_bot_module()
        assert mod.VEL_ATR_MULT == 0.0

    def test_vel_atr_mult_custom(self):
        mod = _reload_bot_module({"VEL_ATR_MULT": "1.6"})
        assert mod.VEL_ATR_MULT == 1.6

    def test_vel_dir_ema_period_default_36(self):
        mod = _reload_bot_module()
        assert mod.VEL_DIR_EMA_PERIOD == 36

    def test_vel_dir_ema_period_custom_120(self):
        mod = _reload_bot_module({"VEL_DIR_EMA_PERIOD": "120"})
        assert mod.VEL_DIR_EMA_PERIOD == 120

    def test_atr_cooldown_default_zero(self):
        mod = _reload_bot_module()
        assert mod.ATR_COOLDOWN == 0

    def test_atr_cooldown_custom(self):
        mod = _reload_bot_module({"ATR_COOLDOWN": "4"})
        assert mod.ATR_COOLDOWN == 4

    def test_regime_ema_period_default_175(self):
        mod = _reload_bot_module()
        assert mod.REGIME_EMA_PERIOD == 175


# ---------------------------------------------------------------------------
# Boolean env var parsing
# ---------------------------------------------------------------------------

class TestBooleanEnvParsing:
    """Boolean env vars should accept case-insensitive true/false."""

    def test_vel_dir_only_default_false(self):
        mod = _reload_bot_module()
        assert mod.VEL_DIR_ONLY is False

    def test_vel_dir_only_true_lowercase(self):
        mod = _reload_bot_module({"VEL_DIR_ONLY": "true"})
        assert mod.VEL_DIR_ONLY is True

    def test_vel_dir_only_true_uppercase(self):
        mod = _reload_bot_module({"VEL_DIR_ONLY": "TRUE"})
        assert mod.VEL_DIR_ONLY is True

    def test_vel_dir_only_true_mixedcase(self):
        mod = _reload_bot_module({"VEL_DIR_ONLY": "True"})
        assert mod.VEL_DIR_ONLY is True

    def test_vel_dir_only_false_explicit(self):
        mod = _reload_bot_module({"VEL_DIR_ONLY": "false"})
        assert mod.VEL_DIR_ONLY is False

    def test_vel_dir_only_garbage_is_false(self):
        """Non-'true' value treated as False (no crash)."""
        mod = _reload_bot_module({"VEL_DIR_ONLY": "yes"})
        assert mod.VEL_DIR_ONLY is False

    def test_htf_ema_align_default_false(self):
        mod = _reload_bot_module()
        assert mod.HTF_EMA_ALIGN is False

    def test_regime_vote_mode_default_false(self):
        mod = _reload_bot_module()
        assert mod.REGIME_VOTE_MODE is False

    def test_atr_regime_adaptive_default_false(self):
        mod = _reload_bot_module()
        assert mod.ATR_REGIME_ADAPTIVE is False


# ---------------------------------------------------------------------------
# Telegram notification config
# ---------------------------------------------------------------------------

class TestTelegramConfig:
    """Validate Telegram fallback logic in validate_config()."""

    def test_notifications_disabled_when_tokens_missing(self):
        """ENABLE_NOTIFICATIONS=true but no tokens → auto-disabled."""
        mod = _reload_bot_module({
            "ENABLE_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "",
        })
        mod.validate_config()
        assert mod.ENABLE_NOTIFICATIONS is False

    def test_notifications_stay_enabled_with_tokens(self):
        """ENABLE_NOTIFICATIONS=true with tokens → stays enabled."""
        mod = _reload_bot_module({
            "ENABLE_NOTIFICATIONS": "true",
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "TELEGRAM_CHAT_ID": "-100123456",
        })
        mod.validate_config()
        assert mod.ENABLE_NOTIFICATIONS is True


# ---------------------------------------------------------------------------
# Layer gate defaults (all off when not set)
# ---------------------------------------------------------------------------

class TestLayerGateDefaults:
    """All layer gates should default to OFF to avoid surprises."""

    def test_all_layers_off_by_default(self):
        mod = _reload_bot_module()
        assert mod.ATR_PARABOLIC_MULT == 0.0   # L1: off
        assert mod.ATR_REGIME_ADAPTIVE is False  # L1b: off
        assert mod.ATR_COOLDOWN == 0             # L1c: off
        assert mod.ATR_ACCEL_LOOKBACK == 0       # L1d: off
        assert mod.HTF_EMA_ALIGN is False        # L2: off
        assert mod.REGIME_VOTE_MODE is False     # L3: off
        assert mod.GRID_SLEEP_ATR_THRESH == 0.0  # L4: off
        assert mod.VEL_ATR_MULT == 0.0           # L5: off
        assert mod.VEL_DIR_ONLY is False         # L5b: off
        assert mod.VEL_DIR_EMA_PERIOD == 36      # L5c: default
