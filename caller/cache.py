"""
SQLite-based response caching.
"""

import json
import time
import hashlib
import asyncio
from loguru import logger
from pathlib import Path
from cachetools import LRUCache
from typing import Optional, Type, Generic, TypeVar
from pydantic import BaseModel, ValidationError
import aiosqlite

from .types import (
    ChatHistory,
    InferenceConfig,
)

APIResponse = TypeVar("APIResponse", bound=BaseModel)


def deterministic_hash(input: str) -> str:
    return hashlib.sha1(input.encode()).hexdigest()


def file_cache_key(
    messages: ChatHistory,
    config: InferenceConfig,
) -> tuple[str, str]:
    """Returns: str_key, cache_key"""
    to_dump = {
        "messages": messages.model_dump(exclude_none=True),
        "config": config.model_dump(exclude_none=True),
    }
    str_key = json.dumps(to_dump, sort_keys=True)
    cache_key = deterministic_hash(str_key)

    return str_key, cache_key


class CacheConfig(BaseModel):
    """
    Configuration for cache behavior.
    """

    base_path: str | None = ".cache/caller"  # None = disable caching
    no_cache_models: set[str] = set()

    cache_chunk_size: int = 128
    max_entries_in_memory: int = 8192
    max_entries_in_disk: int | None = 131072


class Backend:
    """SQLite cache backend."""

    def __init__(self, safe_model_name: str, cache_config: CacheConfig):
        if cache_config.base_path is None:
            raise ValueError("base_path cannot be None when trying to enable caching")

        self.db_path = Path(cache_config.base_path) / f"{safe_model_name}.db"
        self.cache_config = cache_config
        self.in_memory_cache = LRUCache(maxsize=cache_config.max_entries_in_memory)

        self._initialized = False

        # Eviction
        self._evict_counter = 0
        self._eviction_check_interval = 1000  # check every 1000 inserts

        # Persistent connections
        self._read_db: aiosqlite.Connection | None = None
        self._write_db: aiosqlite.Connection | None = None

        # Write-behind queue
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._writer_task: asyncio.Task | None = None

        # Locks
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Open persistent write connection and set up schema
            self._write_db = aiosqlite.connect(self.db_path)
            self._write_db._thread.daemon = True  # Don't block interpreter shutdown
            await self._write_db
            await self._write_db.execute("PRAGMA journal_mode = WAL")
            await self._write_db.execute("PRAGMA synchronous = NORMAL")

            await self._write_db.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    str_key TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_used INTEGER NOT NULL
                )
            """
            )

            await self._write_db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_used
                ON cache_entries(last_used)
            """
            )

            await self._write_db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON cache_entries(created_at)
            """
            )

            await self._write_db.commit()

            # Open persistent read connection
            self._read_db = aiosqlite.connect(self.db_path)
            self._read_db._thread.daemon = True  # Don't block interpreter shutdown
            await self._read_db
            await self._read_db.execute("PRAGMA journal_mode = WAL")

            # Start background writer
            self._writer_task = asyncio.create_task(self._background_writer())

            self._initialized = True

    async def _background_writer(self):
        """Background task that drains the write queue in batches."""
        while True:
            batch: list[tuple] = []

            # Block until at least one item arrives
            item = await self._write_queue.get()
            if item is None:
                # Sentinel: shut down
                break
            batch.append(item)

            # Drain up to 255 more items (total 256) with a short timeout
            deadline = asyncio.get_event_loop().time() + 0.5
            while len(batch) < 256:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._write_queue.get(), timeout=remaining
                    )
                    if item is None:
                        # Sentinel arrived mid-batch; process batch then exit
                        await self._flush_batch(batch)
                        return
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            await self._flush_batch(batch)

    async def _flush_batch(self, batch: list[tuple]):
        """Coalesce and execute a batch of writes."""
        assert self._write_db is not None

        # Coalesce: deduplicate by cache_key, "put" supersedes "touch"
        puts: dict[str, tuple] = {}  # cache_key -> entry values
        touches: dict[str, int] = {}  # cache_key -> timestamp

        for item in batch:
            kind = item[0]
            cache_key = item[1]
            if kind == "put":
                # put supersedes any pending touch
                touches.pop(cache_key, None)
                puts[cache_key] = item[2:]  # (str_key, response, created_at, last_used)
            elif kind == "touch":
                # Only keep touch if no put pending for this key
                if cache_key not in puts:
                    touches[cache_key] = item[2]  # timestamp

        try:
            if puts:
                await self._write_db.executemany(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (cache_key, str_key, response, created_at, last_used)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (cache_key, vals[0], vals[1], vals[2], vals[3])
                        for cache_key, vals in puts.items()
                    ],
                )

            if touches:
                await self._write_db.executemany(
                    "UPDATE cache_entries SET last_used = ? WHERE cache_key = ?",
                    [(ts, key) for key, ts in touches.items()],
                )

            await self._write_db.commit()

            # Eviction check (only count puts toward counter)
            self._evict_counter += len(puts)
            if (
                self.cache_config.max_entries_in_disk is not None
                and puts
                and self._evict_counter >= self._eviction_check_interval
            ):
                self._evict_counter = 0
                async with self._write_db.execute(
                    "SELECT COUNT(*) FROM cache_entries"
                ) as cursor:
                    row = await cursor.fetchone()
                    assert row is not None
                    count = row[0]

                logger.debug(
                    f"Evicting old cache entries, current count: {count}/{self.cache_config.max_entries_in_disk}"
                )

                if count > self.cache_config.max_entries_in_disk:
                    to_delete = count - int(
                        self.cache_config.max_entries_in_disk * 0.9
                    )
                    await self._write_db.execute(
                        """
                        DELETE FROM cache_entries
                        WHERE cache_key IN (
                            SELECT cache_key FROM cache_entries
                            ORDER BY last_used ASC
                            LIMIT ?
                        )
                        """,
                        (to_delete,),
                    )
                    await self._write_db.commit()

        except Exception:
            logger.warning("Failed to flush write batch", exc_info=True)

    async def get_entry(self, cache_key: str) -> dict | None:
        await self.initialize()
        assert self._read_db is not None
        current_time = int(time.time())

        if cache_key in self.in_memory_cache:
            entry = self.in_memory_cache[cache_key]
            entry["last_used"] = current_time
            self._write_queue.put_nowait(("touch", cache_key, current_time))
            return entry

        self._read_db.row_factory = aiosqlite.Row
        async with self._read_db.execute(
            "SELECT * FROM cache_entries WHERE cache_key = ?", (cache_key,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

            target_entry = dict(row)
            target_created_at = target_entry["created_at"]

        target_entry["last_used"] = current_time
        self.in_memory_cache[cache_key] = target_entry
        self._write_queue.put_nowait(("touch", cache_key, current_time))

        await self._prefetch_chunk(target_created_at, cache_key)
        return target_entry

    async def _prefetch_chunk(self, created_at: int, cache_key: str) -> None:
        """Prefetch a chunk of entries with similar creation time."""
        try:
            assert self._read_db is not None
            self._read_db.row_factory = aiosqlite.Row

            async with self._read_db.execute(
                """
                SELECT * FROM cache_entries
                WHERE created_at >= ? AND cache_key != ?
                ORDER BY created_at ASC
                LIMIT ?
            """,
                (created_at, cache_key, self.cache_config.cache_chunk_size),
            ) as cursor:
                rows = await cursor.fetchall()

                for row in rows:
                    entry = dict(row)
                    entry_key = entry["cache_key"]

                    if entry_key not in self.in_memory_cache:
                        self.in_memory_cache[entry_key] = entry

        except Exception:
            logger.warning("Failed to prefetch chunk", exc_info=True)

    async def put_entry(self, entry: dict):
        await self.initialize()
        cache_key = entry["cache_key"]

        # Add to in-memory cache immediately
        self.in_memory_cache[cache_key] = entry

        # Enqueue write — returns immediately
        self._write_queue.put_nowait(
            (
                "put",
                cache_key,
                entry["str_key"],
                entry["response"],
                entry["created_at"],
                entry["last_used"],
            )
        )

    async def shutdown(self):
        """Flush remaining writes and close connections."""
        if self._writer_task is not None:
            # Send sentinel to stop background writer
            await self._write_queue.put(None)
            await self._writer_task
            self._writer_task = None

        if self._read_db is not None:
            await self._read_db.close()
            self._read_db = None

        if self._write_db is not None:
            await self._write_db.close()
            self._write_db = None

        self._initialized = False


class Cache(Generic[APIResponse]):
    """
    Interface for cache operations.
    """
    def __init__(
        self,
        safe_model_name: str,
        response_type: Type[APIResponse],
        cache_config: CacheConfig,
    ):
        self.safe_model_name = safe_model_name
        self.response_type = response_type
        self.cache_config = cache_config
        self.backend = Backend(safe_model_name, cache_config)

        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return
            await self.backend.initialize()
            self._initialized = True

    async def put_entry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
        response: APIResponse,
    ) -> None:
        await self.initialize()

        str_key, cache_key = file_cache_key(messages=messages, config=config)

        await self.backend.put_entry(
            entry={
                "cache_key": cache_key,
                "str_key": str_key,
                "response": response.model_dump_json(exclude_none=True),
                "created_at": int(time.time()),
                "last_used": int(time.time()),
            }
        )

    async def get_entry(
        self,
        messages: ChatHistory,
        config: InferenceConfig,
    ) -> Optional[APIResponse]:
        await self.initialize()

        _, cache_key = file_cache_key(messages=messages, config=config)
        entry = await self.backend.get_entry(cache_key)

        if entry:
            try:
                response = self.response_type.model_validate_json(entry["response"])
                return response
            except ValidationError:
                logger.warning(f"Failed to validate cache entry for key {cache_key}.\nResponse: {entry['response']}", exc_info=True)
                return None
        return None

    async def shutdown(self):
        """Flush remaining writes and close connections."""
        await self.backend.shutdown()
