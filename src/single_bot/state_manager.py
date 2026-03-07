"""
src/single_bot/state_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pluggable state-persistence backends for the Bitunix grid-trading bot.

The bot's critical runtime state (trend-capture position, trend mode,
confirmation counters) is held in-memory.  Without persistence a process
crash loses that state — most dangerously the ``trend_position`` dict
that drives the trailing-stop logic.

Three backends are provided, selected automatically by
``create_state_backend()`` based on environment variables:

    +-----------+-----------------------------+-------------------+
    | Priority  | Backend                     | Env var trigger    |
    +-----------+-----------------------------+-------------------+
    | 1 (best)  | DynamoDB                    | STATE_DYNAMODB_TABLE |
    | 2         | Local file **+ S3 sync**    | STATE_S3_BUCKET    |
    | 3 (default)| Atomic local JSON file     | *(always)*         |
    +-----------+-----------------------------+-------------------+

Usage in bitunix_bot.py::

    from state_manager import BotState, create_state_backend

    backend = create_state_backend(symbol="BTCUSDT")
    backend.save(BotState(trend_position={...}, ...))
    restored = backend.load()
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# State data-class
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BotState:
    """Serialisable snapshot of the bot's critical in-memory state."""

    # Trend-capture runtime fields (mirrors bitunix_bot.py attributes)
    trend_position: Optional[Dict[str, Any]] = None   # {side, entry, qty, peak}
    trend_mode: Optional[str] = None                   # "up" | "down" | None
    trend_pending_dir: Optional[str] = None
    trend_confirm_counter: int = 0
    trend_cooldown_counter: int = 0
    gate_fire_counter: int = 0

    # Metadata
    symbol: str = ""
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["updated_at"] = time.time()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BotState":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ──────────────────────────────────────────────────────────────────────
# Abstract backend
# ──────────────────────────────────────────────────────────────────────

class StateBackend(ABC):
    """Interface every persistence backend must implement."""

    @abstractmethod
    def save(self, state: BotState) -> None:
        ...

    @abstractmethod
    def load(self) -> Optional[BotState]:
        ...


# ──────────────────────────────────────────────────────────────────────
# 1. Local file backend (default)
# ──────────────────────────────────────────────────────────────────────

class FileStateBackend(StateBackend):
    """
    Atomic local-file persistence using the same ``.tmp`` → ``os.replace()``
    → ``fsync`` pattern proven in ``binance_multi_bot.py``.
    """

    def __init__(self, state_dir: str, symbol: str) -> None:
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        safe_sym = symbol.replace("/", "_")
        self._path = self._dir / f"bot_state_{safe_sym}.json"

    @property
    def path(self) -> Path:
        return self._path

    def save(self, state: BotState) -> None:
        data = state.to_dict()
        tmp = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(self._path))
            logger.debug("State persisted to %s", self._path)
        except Exception as exc:
            logger.error("Failed to persist state to %s: %s", self._path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def load(self) -> Optional[BotState]:
        if not self._path.exists():
            logger.info("No state file found at %s", self._path)
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = BotState.from_dict(data)
            age = time.time() - state.updated_at
            logger.info(
                "Restored state from %s (age=%.0fs)",
                self._path, age,
            )
            return state
        except Exception as exc:
            logger.error("Failed to load state from %s: %s", self._path, exc)
            return None


# ──────────────────────────────────────────────────────────────────────
# 2. S3 backend (local file + S3 sync)
# ──────────────────────────────────────────────────────────────────────

class S3StateBackend(StateBackend):
    """
    Writes locally **first** (fast, crash-safe), then syncs to S3.

    On load, reads both local and S3, and uses whichever is newer.  This
    ensures recovery even when the local volume is lost (e.g. EC2 instance
    replacement).
    """

    def __init__(
        self,
        state_dir: str,
        symbol: str,
        bucket: str,
        prefix: str = "state",
    ) -> None:
        self._local = FileStateBackend(state_dir, symbol)
        self._bucket = bucket
        safe_sym = symbol.replace("/", "_")
        self._key = f"{prefix}/{safe_sym}/bot_state.json"
        self._s3 = None  # lazy

    def _get_s3(self):  # noqa: ANN202
        if self._s3 is None:
            import boto3  # noqa: F811 – optional dependency
            self._s3 = boto3.client("s3")
        return self._s3

    def save(self, state: BotState) -> None:
        # Always write locally first (fast, crash-safe)
        self._local.save(state)
        # Then sync to S3
        try:
            s3 = self._get_s3()
            body = json.dumps(state.to_dict(), separators=(",", ":"))
            s3.put_object(
                Bucket=self._bucket,
                Key=self._key,
                Body=body.encode("utf-8"),
                ContentType="application/json",
            )
            logger.debug("State synced to s3://%s/%s", self._bucket, self._key)
        except Exception as exc:
            logger.warning("S3 state sync failed (local file OK): %s", exc)

    def load(self) -> Optional[BotState]:
        local_state = self._local.load()

        s3_state: Optional[BotState] = None
        try:
            s3 = self._get_s3()
            resp = s3.get_object(Bucket=self._bucket, Key=self._key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            s3_state = BotState.from_dict(data)
            logger.info(
                "S3 state loaded from s3://%s/%s (age=%.0fs)",
                self._bucket, self._key,
                time.time() - s3_state.updated_at,
            )
        except Exception as exc:
            logger.info("No S3 state available: %s", exc)

        # Use whichever is newer
        if local_state and s3_state:
            if s3_state.updated_at > local_state.updated_at:
                logger.info("Using S3 state (newer than local)")
                return s3_state
            logger.info("Using local state (newer than S3)")
            return local_state
        return local_state or s3_state


# ──────────────────────────────────────────────────────────────────────
# 3. DynamoDB backend (HA deployments)
# ──────────────────────────────────────────────────────────────────────

class DynamoDBStateBackend(StateBackend):
    """
    DynamoDB persistence for high-availability multi-instance deployments.

    Table schema required::

        Partition key: ``pk`` (String) — set to the trading symbol
        No sort key needed (single item per symbol)

    Strongly-consistent reads ensure the latest state is always returned.
    """

    def __init__(self, table_name: str, symbol: str) -> None:
        self._table_name = table_name
        self._symbol = symbol
        self._table = None  # lazy

    def _get_table(self):  # noqa: ANN202
        if self._table is None:
            import boto3  # noqa: F811 – optional dependency
            dynamodb = boto3.resource("dynamodb")
            self._table = dynamodb.Table(self._table_name)
        return self._table

    def save(self, state: BotState) -> None:
        try:
            table = self._get_table()
            item = state.to_dict()
            item["pk"] = self._symbol
            item = _sanitize_for_dynamo(item)
            table.put_item(Item=item)
            logger.debug(
                "State persisted to DynamoDB table=%s pk=%s",
                self._table_name, self._symbol,
            )
        except Exception as exc:
            logger.error("DynamoDB state save failed: %s", exc)

    def load(self) -> Optional[BotState]:
        try:
            table = self._get_table()
            resp = table.get_item(
                Key={"pk": self._symbol},
                ConsistentRead=True,
            )
            item = resp.get("Item")
            if not item:
                logger.info("No DynamoDB state for pk=%s", self._symbol)
                return None
            item.pop("pk", None)
            item = _desanitize_from_dynamo(item)
            state = BotState.from_dict(item)
            logger.info(
                "Restored state from DynamoDB (age=%.0fs)",
                time.time() - state.updated_at,
            )
            return state
        except Exception as exc:
            logger.error("DynamoDB state load failed: %s", exc)
            return None


# ──────────────────────────────────────────────────────────────────────
# DynamoDB helpers
# ──────────────────────────────────────────────────────────────────────

def _sanitize_for_dynamo(d: dict) -> dict:
    """Replace ``None`` values and ``float`` with DynamoDB-safe types."""
    from decimal import Decimal

    result: dict = {}
    for k, v in d.items():
        if v is None:
            result[k] = "__NONE__"
        elif isinstance(v, dict):
            result[k] = _sanitize_for_dynamo(v)
        elif isinstance(v, float):
            result[k] = Decimal(str(v))
        else:
            result[k] = v
    return result


def _desanitize_from_dynamo(d: dict) -> dict:
    """Reverse DynamoDB sanitisation (``__NONE__`` → ``None``, Decimal → float)."""
    result: dict = {}
    for k, v in d.items():
        if v == "__NONE__":
            result[k] = None
        elif isinstance(v, dict):
            result[k] = _desanitize_from_dynamo(v)
        elif hasattr(v, "__float__"):  # Decimal
            result[k] = float(v)
        else:
            result[k] = v
    return result


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────

def create_state_backend(
    symbol: str,
    state_dir: str = "state",
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "state",
    dynamodb_table: Optional[str] = None,
) -> StateBackend:
    """
    Pick the best available backend based on configuration.

    Priority: DynamoDB > S3 + local > local-only.
    """
    if dynamodb_table:
        logger.info(
            "State backend: DynamoDB (table=%s, symbol=%s)",
            dynamodb_table, symbol,
        )
        return DynamoDBStateBackend(table_name=dynamodb_table, symbol=symbol)
    if s3_bucket:
        logger.info(
            "State backend: S3 (bucket=%s, prefix=%s) + local file (%s)",
            s3_bucket, s3_prefix, state_dir,
        )
        return S3StateBackend(
            state_dir=state_dir,
            symbol=symbol,
            bucket=s3_bucket,
            prefix=s3_prefix,
        )
    logger.info("State backend: local file (%s)", state_dir)
    return FileStateBackend(state_dir=state_dir, symbol=symbol)
