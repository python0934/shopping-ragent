"""
Snowflake ID generator — mirrors Java Hutool IdUtil.getSnowflakeNextIdStr().

Produces 19-digit numeric strings compatible with existing database rows.
Format: 41-bit timestamp + 10-bit worker-id + 12-bit sequence = 63-bit int.
"""

from __future__ import annotations

import threading
import time


class SnowflakeGenerator:
    """Thread-safe Snowflake ID generator."""

    # Epoch: 1970-01-01 00:00:00 UTC (Unix epoch, matches Java Hutool default)
    # Using Unix epoch ensures IDs are ~19 digits, compatible with existing DB rows.
    EPOCH = 0

    # Bit allocation
    WORKER_ID_BITS = 10
    SEQUENCE_BITS = 12
    MAX_WORKER_ID = (1 << WORKER_ID_BITS) - 1  # 1023
    MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1      # 4095

    # Shifts
    WORKER_ID_SHIFT = SEQUENCE_BITS              # 12
    TIMESTAMP_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS  # 22

    def __init__(self, worker_id: int = 1):
        if worker_id < 0 or worker_id > self.MAX_WORKER_ID:
            raise ValueError(f"worker_id must be in [0, {self.MAX_WORKER_ID}]")
        self._worker_id = worker_id
        self._sequence = 0
        self._last_timestamp = -1
        self._lock = threading.Lock()

    def _current_millis(self) -> int:
        return int(time.time() * 1000)

    def _wait_next_millis(self, last_ts: int) -> int:
        ts = self._current_millis()
        while ts <= last_ts:
            ts = self._current_millis()
        return ts

    def next_id(self) -> int:
        """Generate the next snowflake ID as an integer."""
        with self._lock:
            timestamp = self._current_millis()

            if timestamp < self._last_timestamp:
                # Clock moved backwards — wait until caught up
                timestamp = self._wait_next_millis(self._last_timestamp)

            if timestamp == self._last_timestamp:
                self._sequence = (self._sequence + 1) & self.MAX_SEQUENCE
                if self._sequence == 0:
                    timestamp = self._wait_next_millis(self._last_timestamp)
            else:
                self._sequence = 0

            self._last_timestamp = timestamp

            return (
                ((timestamp - self.EPOCH) << self.TIMESTAMP_SHIFT)
                | (self._worker_id << self.WORKER_ID_SHIFT)
                | self._sequence
            )

    def next_id_str(self) -> str:
        """Generate the next snowflake ID as a 19-digit string."""
        return str(self.next_id())


# Module-level singleton (worker_id=1, sufficient for single-process deployment)
_generator = SnowflakeGenerator(worker_id=1)


def get_snowflake_id() -> int:
    """Return next snowflake ID as int."""
    return _generator.next_id()


def get_snowflake_id_str() -> str:
    """Return next snowflake ID as string (19 digits)."""
    return _generator.next_id_str()
