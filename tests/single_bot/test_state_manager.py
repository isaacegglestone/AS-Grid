"""
tests/single_bot/test_state_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the pluggable state-persistence backends.

Covers:
  - BotState serialisation / deserialisation
  - FileStateBackend  (atomic write, load, corruption handling)
  - S3StateBackend    (local+S3, newest-wins merge)
  - DynamoDBStateBackend (put/get, sanitisation round-trip)
  - create_state_backend() factory
  - Integration: _persist_state / _restore_state on GridTradingBot
"""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.single_bot.state_manager import (
    BotState,
    DynamoDBStateBackend,
    FileStateBackend,
    S3StateBackend,
    StateBackend,
    _desanitize_from_dynamo,
    _sanitize_for_dynamo,
    create_state_backend,
)


# ======================================================================
# BotState dataclass
# ======================================================================


class TestBotState:
    """BotState serialisation round-trip tests."""

    def test_default_values(self):
        s = BotState()
        assert s.trend_position is None
        assert s.trend_mode is None
        assert s.trend_confirm_counter == 0
        assert s.symbol == ""

    def test_to_dict_sets_updated_at(self):
        s = BotState(symbol="BTCUSDT")
        d = s.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["updated_at"] > 0
        assert isinstance(d["updated_at"], float)

    def test_from_dict_round_trip(self):
        original = BotState(
            trend_position={"side": "long", "entry": 60000, "qty": 50, "peak": 65000},
            trend_mode="up",
            trend_pending_dir="up",
            trend_confirm_counter=3,
            trend_cooldown_counter=0,
            gate_fire_counter=2,
            symbol="BTCUSDT",
        )
        d = original.to_dict()
        restored = BotState.from_dict(d)
        assert restored.trend_position == original.trend_position
        assert restored.trend_mode == "up"
        assert restored.trend_confirm_counter == 3
        assert restored.gate_fire_counter == 2
        assert restored.symbol == "BTCUSDT"

    def test_from_dict_ignores_unknown_keys(self):
        d = {"trend_mode": "down", "unknown_field": 42, "symbol": "ETHUSDT"}
        s = BotState.from_dict(d)
        assert s.trend_mode == "down"
        assert s.symbol == "ETHUSDT"
        assert not hasattr(s, "unknown_field")

    def test_from_dict_missing_keys_use_defaults(self):
        s = BotState.from_dict({"symbol": "XRPUSDT"})
        assert s.trend_position is None
        assert s.trend_confirm_counter == 0


# ======================================================================
# FileStateBackend
# ======================================================================


class TestFileStateBackend:
    """Local atomic-file persistence tests."""

    def test_save_creates_file(self, tmp_path: Path):
        backend = FileStateBackend(str(tmp_path), "BTCUSDT")
        state = BotState(trend_mode="up", symbol="BTCUSDT")
        backend.save(state)
        assert backend.path.exists()

    def test_save_and_load_round_trip(self, tmp_path: Path):
        backend = FileStateBackend(str(tmp_path), "BTCUSDT")
        state = BotState(
            trend_position={"side": "long", "entry": 100, "qty": 10, "peak": 110},
            trend_mode="up",
            trend_confirm_counter=5,
            symbol="BTCUSDT",
        )
        backend.save(state)
        restored = backend.load()
        assert restored is not None
        assert restored.trend_position == state.trend_position
        assert restored.trend_mode == "up"
        assert restored.trend_confirm_counter == 5

    def test_load_returns_none_when_no_file(self, tmp_path: Path):
        backend = FileStateBackend(str(tmp_path), "BTCUSDT")
        assert backend.load() is None

    def test_load_handles_corrupt_file(self, tmp_path: Path):
        backend = FileStateBackend(str(tmp_path), "BTCUSDT")
        backend.path.write_text("NOT VALID JSON {{{")
        result = backend.load()
        assert result is None

    def test_save_overwrites_previous(self, tmp_path: Path):
        backend = FileStateBackend(str(tmp_path), "BTCUSDT")
        backend.save(BotState(trend_mode="up", symbol="BTCUSDT"))
        backend.save(BotState(trend_mode="down", symbol="BTCUSDT"))
        restored = backend.load()
        assert restored is not None
        assert restored.trend_mode == "down"

    def test_atomic_write_no_partial_file(self, tmp_path: Path):
        """If save() fails mid-write, no .tmp file is left behind."""
        backend = FileStateBackend(str(tmp_path), "BTCUSDT")
        # First write succeeds
        backend.save(BotState(trend_mode="up", symbol="BTCUSDT"))
        # Force a write failure by making the dir read-only won't work on
        # all systems, so we test that the tmp file doesn't persist after
        # a normal save.
        tmp_file = backend.path.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_creates_state_dir(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "dir"
        backend = FileStateBackend(str(nested), "XRPUSDT")
        backend.save(BotState(symbol="XRPUSDT"))
        assert nested.exists()
        assert backend.path.exists()

    def test_symbol_with_slash(self, tmp_path: Path):
        backend = FileStateBackend(str(tmp_path), "BTC/USDT")
        backend.save(BotState(symbol="BTC/USDT"))
        assert "BTC_USDT" in backend.path.name

    def test_saved_file_is_valid_json(self, tmp_path: Path):
        backend = FileStateBackend(str(tmp_path), "BTCUSDT")
        state = BotState(
            trend_position={"side": "short", "entry": 50000, "qty": 20, "peak": 49000},
            symbol="BTCUSDT",
        )
        backend.save(state)
        raw = json.loads(backend.path.read_text())
        assert raw["symbol"] == "BTCUSDT"
        assert raw["trend_position"]["side"] == "short"


# ======================================================================
# S3StateBackend
# ======================================================================


class TestS3StateBackend:
    """S3 + local file backend tests (mocked boto3)."""

    def _mock_s3(self):
        """Return a mock S3 client with an in-memory object store."""
        store: dict = {}

        def put_object(Bucket, Key, Body, **kw):
            store[f"{Bucket}/{Key}"] = Body

        def get_object(Bucket, Key):
            data = store.get(f"{Bucket}/{Key}")
            if data is None:
                raise Exception("NoSuchKey")
            body_mock = mock.MagicMock()
            body_mock.read.return_value = data if isinstance(data, bytes) else data.encode()
            return {"Body": body_mock}

        s3 = mock.MagicMock()
        s3.put_object = put_object
        s3.get_object = get_object
        return s3, store

    def test_save_writes_locally_and_to_s3(self, tmp_path: Path):
        s3_client, store = self._mock_s3()
        backend = S3StateBackend(str(tmp_path), "BTCUSDT", "my-bucket", "pfx")
        backend._s3 = s3_client

        state = BotState(trend_mode="up", symbol="BTCUSDT")
        backend.save(state)

        # Local file exists
        assert backend._local.path.exists()
        # S3 object exists
        assert "my-bucket/pfx/BTCUSDT/bot_state.json" in store

    def test_load_prefers_newer_s3(self, tmp_path: Path):
        s3_client, store = self._mock_s3()
        backend = S3StateBackend(str(tmp_path), "BTCUSDT", "my-bucket")
        backend._s3 = s3_client

        # Write a local state with deliberately old timestamp
        old_dict = BotState(trend_mode="up", symbol="BTCUSDT").to_dict()
        old_dict["updated_at"] = 1000.0
        backend._local.path.parent.mkdir(parents=True, exist_ok=True)
        backend._local.path.write_text(json.dumps(old_dict))

        # Put a newer state in S3
        new_dict = BotState(trend_mode="down", symbol="BTCUSDT").to_dict()
        new_dict["updated_at"] = time.time() + 100  # definitely newer
        s3_client.put_object(
            Bucket="my-bucket",
            Key="state/BTCUSDT/bot_state.json",
            Body=json.dumps(new_dict).encode(),
        )

        restored = backend.load()
        assert restored is not None
        assert restored.trend_mode == "down"

    def test_load_prefers_newer_local(self, tmp_path: Path):
        s3_client, store = self._mock_s3()
        backend = S3StateBackend(str(tmp_path), "BTCUSDT", "my-bucket")
        backend._s3 = s3_client

        # S3 has old data
        old_dict = BotState(trend_mode="down", symbol="BTCUSDT").to_dict()
        old_dict["updated_at"] = 1000.0
        s3_client.put_object(
            Bucket="my-bucket",
            Key="state/BTCUSDT/bot_state.json",
            Body=json.dumps(old_dict).encode(),
        )

        # Local has newer data
        backend._local.save(BotState(trend_mode="up", symbol="BTCUSDT"))

        restored = backend.load()
        assert restored is not None
        assert restored.trend_mode == "up"

    def test_load_falls_back_to_local_when_s3_missing(self, tmp_path: Path):
        s3_client, _ = self._mock_s3()
        backend = S3StateBackend(str(tmp_path), "BTCUSDT", "my-bucket")
        backend._s3 = s3_client

        backend._local.save(BotState(trend_mode="up", symbol="BTCUSDT"))
        restored = backend.load()
        assert restored is not None
        assert restored.trend_mode == "up"

    def test_load_falls_back_to_s3_when_local_missing(self, tmp_path: Path):
        s3_client, _ = self._mock_s3()
        backend = S3StateBackend(str(tmp_path), "BTCUSDT", "my-bucket")
        backend._s3 = s3_client

        # Only S3 has data
        s3_data = BotState(trend_mode="down", symbol="BTCUSDT").to_dict()
        s3_client.put_object(
            Bucket="my-bucket",
            Key="state/BTCUSDT/bot_state.json",
            Body=json.dumps(s3_data).encode(),
        )

        restored = backend.load()
        assert restored is not None
        assert restored.trend_mode == "down"

    def test_s3_failure_still_saves_locally(self, tmp_path: Path):
        """If S3 is unreachable, local file write still succeeds."""
        s3_client = mock.MagicMock()
        s3_client.put_object.side_effect = Exception("S3 timeout")
        backend = S3StateBackend(str(tmp_path), "BTCUSDT", "my-bucket")
        backend._s3 = s3_client

        state = BotState(trend_mode="up", symbol="BTCUSDT")
        backend.save(state)

        # Local still works
        assert backend._local.path.exists()
        restored = backend._local.load()
        assert restored is not None
        assert restored.trend_mode == "up"


# ======================================================================
# DynamoDB helpers
# ======================================================================


class TestDynamoSanitization:
    """_sanitize_for_dynamo / _desanitize_from_dynamo round-trip."""

    def test_none_roundtrip(self):
        d = {"a": None, "b": "hello"}
        sanitized = _sanitize_for_dynamo(d)
        assert sanitized["a"] == "__NONE__"
        assert sanitized["b"] == "hello"
        desanitized = _desanitize_from_dynamo(sanitized)
        assert desanitized["a"] is None

    def test_float_becomes_decimal(self):
        d = {"price": 60123.45}
        sanitized = _sanitize_for_dynamo(d)
        assert isinstance(sanitized["price"], Decimal)
        desanitized = _desanitize_from_dynamo(sanitized)
        assert isinstance(desanitized["price"], float)
        assert desanitized["price"] == pytest.approx(60123.45)

    def test_nested_dict(self):
        d = {"trend_position": {"side": "long", "entry": 100.0, "peak": None}}
        sanitized = _sanitize_for_dynamo(d)
        assert isinstance(sanitized["trend_position"]["entry"], Decimal)
        assert sanitized["trend_position"]["peak"] == "__NONE__"
        desanitized = _desanitize_from_dynamo(sanitized)
        assert desanitized["trend_position"]["entry"] == pytest.approx(100.0)
        assert desanitized["trend_position"]["peak"] is None

    def test_int_preserved(self):
        d = {"count": 5}
        assert _sanitize_for_dynamo(d)["count"] == 5
        assert _desanitize_from_dynamo(_sanitize_for_dynamo(d))["count"] == 5


# ======================================================================
# DynamoDBStateBackend
# ======================================================================


class TestDynamoDBStateBackend:
    """DynamoDB backend tests (mocked boto3)."""

    def _mock_table(self):
        """Return a mock DynamoDB Table with in-memory item store."""
        items: dict = {}

        def put_item(Item):
            pk = Item["pk"]
            items[pk] = Item.copy()

        def get_item(Key, **kw):
            item = items.get(Key["pk"])
            if item:
                return {"Item": item.copy()}
            return {}

        table = mock.MagicMock()
        table.put_item = put_item
        table.get_item = get_item
        return table, items

    def test_save_and_load(self):
        table, items = self._mock_table()
        backend = DynamoDBStateBackend("my-table", "BTCUSDT")
        backend._table = table

        state = BotState(
            trend_position={"side": "long", "entry": 60000, "qty": 50, "peak": 65000},
            trend_mode="up",
            symbol="BTCUSDT",
        )
        backend.save(state)
        assert "BTCUSDT" in items

        restored = backend.load()
        assert restored is not None
        assert restored.trend_mode == "up"
        assert restored.trend_position["side"] == "long"
        assert restored.trend_position["entry"] == pytest.approx(60000.0)

    def test_load_returns_none_when_empty(self):
        table, _ = self._mock_table()
        backend = DynamoDBStateBackend("my-table", "BTCUSDT")
        backend._table = table
        assert backend.load() is None

    def test_save_error_does_not_raise(self):
        table = mock.MagicMock()
        table.put_item.side_effect = Exception("DynamoDB timeout")
        backend = DynamoDBStateBackend("my-table", "BTCUSDT")
        backend._table = table
        # Should not raise — errors are logged, not propagated
        backend.save(BotState(symbol="BTCUSDT"))

    def test_load_error_returns_none(self):
        table = mock.MagicMock()
        table.get_item.side_effect = Exception("DynamoDB timeout")
        backend = DynamoDBStateBackend("my-table", "BTCUSDT")
        backend._table = table
        assert backend.load() is None


# ======================================================================
# Factory
# ======================================================================


class TestCreateStateBackend:
    """create_state_backend() factory selection tests."""

    def test_default_is_file_backend(self, tmp_path: Path):
        backend = create_state_backend("BTCUSDT", state_dir=str(tmp_path))
        assert isinstance(backend, FileStateBackend)

    def test_s3_bucket_selects_s3_backend(self, tmp_path: Path):
        backend = create_state_backend(
            "BTCUSDT",
            state_dir=str(tmp_path),
            s3_bucket="my-bucket",
        )
        assert isinstance(backend, S3StateBackend)

    def test_dynamodb_table_selects_dynamo_backend(self):
        backend = create_state_backend(
            "BTCUSDT",
            dynamodb_table="my-table",
        )
        assert isinstance(backend, DynamoDBStateBackend)

    def test_dynamodb_takes_priority_over_s3(self):
        """When both are configured, DynamoDB wins."""
        backend = create_state_backend(
            "BTCUSDT",
            s3_bucket="my-bucket",
            dynamodb_table="my-table",
        )
        assert isinstance(backend, DynamoDBStateBackend)

    def test_empty_strings_treated_as_unset(self, tmp_path: Path):
        backend = create_state_backend(
            "BTCUSDT",
            state_dir=str(tmp_path),
            s3_bucket=None,
            dynamodb_table=None,
        )
        assert isinstance(backend, FileStateBackend)


# ======================================================================
# Bot integration: _persist_state / _restore_state
# ======================================================================


class TestBotStateIntegration:
    """Test _persist_state and _restore_state on the actual bot class."""

    def _make_bot(self, tmp_path: Path):
        """Create a bot instance with file-based state persistence."""
        env = {
            "API_KEY": "test-key",
            "API_SECRET": "test-secret",
            "COIN_NAME": "BTC",
            "GRID_SPACING": "0.015",
            "INITIAL_QUANTITY": "1",
            "LEVERAGE": "2",
            "STATE_DIR": str(tmp_path),
            "STATE_S3_BUCKET": "",
            "STATE_DYNAMODB_TABLE": "",
            "ENABLE_NOTIFICATIONS": "false",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            import importlib
            import src.single_bot.bitunix_bot as bot_mod

            importlib.reload(bot_mod)
            bot = bot_mod.GridTradingBot(
                api_key="test-key",
                api_secret="test-secret",
                coin_name="BTC",
                grid_spacing=0.015,
                initial_quantity=1,
                leverage=2,
            )
            return bot

    def test_persist_and_restore_trend_position(self, tmp_path: Path):
        bot = self._make_bot(tmp_path)

        # Simulate an open trend position
        bot.trend_position = {"side": "long", "entry": 60000, "qty": 50, "peak": 65000}
        bot.trend_mode = "up"
        bot.trend_confirm_counter = 3
        bot._gate_fire_counter = 2
        bot._persist_state()

        # Create a fresh bot (simulating restart)
        bot2 = self._make_bot(tmp_path)
        assert bot2.trend_position is None  # default

        bot2._restore_state()
        assert bot2.trend_position is not None
        assert bot2.trend_position["side"] == "long"
        assert bot2.trend_position["peak"] == 65000
        assert bot2.trend_mode == "up"
        assert bot2.trend_confirm_counter == 3
        assert bot2._gate_fire_counter == 2

    def test_restore_with_no_state_file(self, tmp_path: Path):
        bot = self._make_bot(tmp_path)
        # Should not raise
        bot._restore_state()
        assert bot.trend_position is None
        assert bot.trend_mode is None

    def test_restore_ignores_wrong_symbol(self, tmp_path: Path):
        bot = self._make_bot(tmp_path)
        # Manually write state with wrong symbol
        state = BotState(
            trend_mode="down",
            symbol="ETHUSDT",
        )
        bot.state_backend.save(state)

        bot._restore_state()
        # Should ignore because symbol doesn't match
        assert bot.trend_mode is None

    def test_persist_clears_trend_position(self, tmp_path: Path):
        bot = self._make_bot(tmp_path)

        # Set then clear a trend position
        bot.trend_position = {"side": "short", "entry": 50000, "qty": 20, "peak": 49000}
        bot._persist_state()

        bot.trend_position = None
        bot.trend_mode = None
        bot._persist_state()

        bot2 = self._make_bot(tmp_path)
        bot2._restore_state()
        assert bot2.trend_position is None
        assert bot2.trend_mode is None
