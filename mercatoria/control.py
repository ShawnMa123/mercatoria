"""Publication authority: immutable control states behind one control head.

The database is the *only* record of what is live. ``current.txt`` is an
export for legacy tooling and the migration input; nothing reads it to decide
which index serves requests.

A control state is written exactly once and never updated. Publishing appends a
generation and moves the single head row inside one transaction, so a crash
before commit leaves the previous state in force and a crash after commit is
fully recovered by reading the head again.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS control_states (
    generation INTEGER PRIMARY KEY,
    manifest_id TEXT NOT NULL,
    index_version TEXT NOT NULL,
    build_profile_id TEXT NOT NULL,
    runtime_config_id TEXT NOT NULL,
    created_by_job_id TEXT,
    created_at TEXT NOT NULL,
    manifest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS control_head (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    generation INTEGER NOT NULL
);
"""

# Schema 2 adds the non-sensitive configuration a generation was published
# with, plus the operator's desired (possibly not yet built) configuration.
# Both are additive: an existing deployment keeps its rows and gets an empty
# column, which is exactly the "cannot reconstruct the config" case.
SCHEMA_V2_COLUMNS = (
    ("control_states", "config", "TEXT"),
)
SCHEMA_V2_TABLES = (
    """CREATE TABLE IF NOT EXISTS desired_config (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    config TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""",
)


class ControlError(RuntimeError):
    pass


class ControlConflict(ControlError):
    """The head moved since the caller read its base generation."""


class ControlSchemaTooNew(ControlError):
    """The control database was written by a newer mercatoria build."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _loads(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def manifest_id(manifest: dict[str, Any]) -> str:
    """Stable id of a document set: identities plus the fields that matter."""
    docs = [
        {
            "doc_id": doc.get("doc_id"),
            "path": doc.get("path"),
            "jurisdiction": doc.get("jurisdiction"),
            "effective_from": doc.get("effective_from"),
            "effective_to": doc.get("effective_to"),
        }
        for doc in manifest.get("docs") or []
    ]
    docs.sort(key=lambda doc: str(doc.get("doc_id")))
    raw = json.dumps(docs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ControlState:
    generation: int
    index_version: str
    manifest: dict[str, Any] = field(repr=False, compare=False)
    manifest_id: str = ""
    build_profile_id: str = ""
    runtime_config_id: str = ""
    created_by_job_id: str | None = None
    created_at: str = ""
    # Non-sensitive configuration this generation serves with. ``None`` means
    # the row predates config persistence: its runtime config can only be
    # reconstructed if the deployment config still has the same fingerprint.
    config: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def snapshot_version(self) -> str:
        """Opaque, outward-facing identifier of this control generation."""
        return str(self.generation)

    def describe(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "generation": self.generation,
            "index_version": self.index_version,
            "manifest_id": self.manifest_id,
            "build_profile_id": self.build_profile_id,
            "runtime_config_id": self.runtime_config_id,
            "documents": len(self.manifest.get("docs") or []),
            "created_by_job_id": self.created_by_job_id,
            "created_at": self.created_at,
            "config_persisted": self.config is not None,
        }


class ControlStore:
    """The authority, with one writer connection and one committed-view reader.

    ``self.conn`` is the writer: every state change happens inside its
    ``BEGIN IMMEDIATE`` ... ``COMMIT`` block, serialised by ``_write_lock``.
    ``self._read_conn`` is a second, ``query_only`` connection used by every
    *public* read method, so a request thread can never observe rows the writer
    has not committed: SQLite only shows a second connection committed state,
    and this one never opens a transaction of its own.

    Lock order (never take them the other way around):

    1. ``PublicationCoordinator.gate`` — the process-wide publish/lease
       boundary; it covers the control commit *and* the head-check+pin.
    2. ``ControlStore._write_lock`` — one control transaction at a time.
    3. ``ControlStore._read_lock`` — one committed-view read at a time.

    A read that happens inside the publication gate takes 1 then 3, and the
    publish path takes 1 then 2. No path holds a control lock and then waits
    for the publication gate, so the two chains cannot deadlock. Transaction
    internals that must see their own uncommitted writes (``publish`` checking
    the base generation) use :meth:`_head_on` on the write connection instead of
    the public readers, which would self-lock.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self.conn = self._connect(writer=True)
        self._read_conn: sqlite3.Connection | None = None
        self._migrate()
        self._read_conn = self._connect(writer=False)

    def _connect(self, *, writer: bool) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        if writer:
            conn.execute("PRAGMA journal_mode=WAL")
        else:
            # Read-only view: a public read can never leave a transaction (or a
            # pending write) behind on the handle other threads read from.
            conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _migrate(self) -> None:
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            self.conn.close()
            raise ControlSchemaTooNew(
                f"control schema {version} is newer than this build ({SCHEMA_VERSION})"
            )
        if version < 1:
            # Additive only, and idempotent: a half-created schema is completed
            # on the next start, an existing one keeps its rows.
            self.conn.executescript(SCHEMA)
            self.conn.execute("PRAGMA user_version=1")
        if version < 2:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                existing = {
                    row[1]
                    for row in self.conn.execute("PRAGMA table_info(control_states)")
                }
                for table, column, decl in SCHEMA_V2_COLUMNS:
                    if column not in existing:
                        self.conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
                        )
                for statement in SCHEMA_V2_TABLES:
                    self.conn.execute(statement)
                self.conn.execute("PRAGMA user_version=2")
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        reader, self._read_conn = self._read_conn, None
        if reader is not None:
            reader.close()
        self.conn.close()

    # -- committed-view reads ---------------------------------------------

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Run one read on the committed-view connection."""
        reader = self._read_conn
        if reader is None:
            raise ControlError("control store is closed")
        with self._read_lock:
            return reader.execute(sql, params).fetchall()

    @staticmethod
    def _head_on(conn: sqlite3.Connection) -> int | None:
        """Head as seen by ``conn`` (unlocked: only for a held transaction)."""
        row = conn.execute(
            "SELECT generation FROM control_head WHERE singleton=1"
        ).fetchone()
        return int(row[0]) if row else None

    def _row(self, row: sqlite3.Row | None) -> ControlState | None:
        if row is None:
            return None
        try:
            manifest = json.loads(row["manifest"])
        except json.JSONDecodeError:
            manifest = {"docs": []}
        return ControlState(
            generation=int(row["generation"]),
            index_version=str(row["index_version"]),
            manifest=manifest if isinstance(manifest, dict) else {"docs": []},
            manifest_id=str(row["manifest_id"]),
            build_profile_id=str(row["build_profile_id"]),
            runtime_config_id=str(row["runtime_config_id"]),
            created_by_job_id=row["created_by_job_id"],
            created_at=str(row["created_at"]),
            config=_loads(row["config"]) if "config" in row.keys() else None,
        )

    def head_generation(self) -> int | None:
        rows = self._read("SELECT generation FROM control_head WHERE singleton=1")
        return int(rows[0][0]) if rows else None

    def head(self) -> ControlState | None:
        # One statement, one snapshot: the head row and the state it names can
        # never come from two different commits.
        rows = self._read(
            """SELECT states.* FROM control_states AS states
               JOIN control_head AS head ON states.generation = head.generation
               WHERE head.singleton = 1"""
        )
        return self._row(rows[0]) if rows else None

    def get(self, generation: int) -> ControlState | None:
        rows = self._read(
            "SELECT * FROM control_states WHERE generation=?", (int(generation),)
        )
        return self._row(rows[0]) if rows else None

    def states(self, limit: int = 50) -> list[ControlState]:
        rows = self._read(
            "SELECT * FROM control_states ORDER BY generation DESC LIMIT ?",
            (int(limit),),
        )
        return [state for state in (self._row(row) for row in rows) if state is not None]

    # -- desired (not yet built) configuration ----------------------------

    def desired_config(self) -> dict[str, Any] | None:
        """Operator intent that may still need an explicit rebuild."""
        rows = self._read("SELECT config FROM desired_config WHERE singleton=1")
        return _loads(rows[0][0]) if rows else None

    @staticmethod
    def _write_desired(
        conn: sqlite3.Connection, config: dict[str, Any], at: str | None = None
    ) -> None:
        """Upsert the desired row on ``conn`` (inside the caller's transaction)."""
        conn.execute(
            """INSERT INTO desired_config(singleton, config, updated_at)
               VALUES (1, ?, ?)
               ON CONFLICT(singleton) DO UPDATE SET
                   config=excluded.config, updated_at=excluded.updated_at""",
            (json.dumps(config, ensure_ascii=False), at or _now()),
        )

    def set_desired_config(self, config: dict[str, Any]) -> None:
        with self._write_lock:
            self._write_desired(self.conn, config)

    def publish(
        self,
        *,
        index_version: str,
        manifest: dict[str, Any],
        build_profile_id: str,
        runtime_config_id: str,
        base_generation: int | None = None,
        created_by_job_id: str | None = None,
        config: dict[str, Any] | None = None,
        expect_empty: bool = False,
        desired_config: dict[str, Any] | None = None,
    ) -> ControlState:
        """Append a generation and move the head, or raise ControlConflict.

        ``base_generation=None`` means "whatever the head is now" and is only
        safe while the caller holds the data-directory write lock.
        ``expect_empty=True`` is the explicit first-publication CAS: the head
        must not exist yet, so two candidates built from nothing cannot both
        publish.

        ``desired_config`` is written in the *same* transaction: a crash can
        never leave the active state advanced while the accepted operator intent
        is the previous value, and a failed desired write rolls the head back.
        """
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                # The head of *this* transaction: the committed-view reader
                # would not see the row this transaction just wrote, and taking
                # its lock here would nest control locks.
                current = self._head_on(conn)
                if expect_empty and current is not None:
                    raise ControlConflict(
                        f"control head is not empty: generation {current} is already published"
                    )
                if base_generation is not None and current != base_generation:
                    raise ControlConflict(
                        f"control head moved: expected generation {base_generation}, "
                        f"found {current}"
                    )
                generation = (current or 0) + 1
                state = ControlState(
                    generation=generation,
                    index_version=str(index_version),
                    manifest=manifest,
                    manifest_id=manifest_id(manifest),
                    build_profile_id=build_profile_id,
                    runtime_config_id=runtime_config_id,
                    created_by_job_id=created_by_job_id,
                    created_at=_now(),
                    config=config,
                )
                conn.execute(
                    """INSERT INTO control_states
                       (generation, manifest_id, index_version, build_profile_id,
                        runtime_config_id, created_by_job_id, created_at, manifest,
                        config)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        state.generation,
                        state.manifest_id,
                        state.index_version,
                        state.build_profile_id,
                        state.runtime_config_id,
                        state.created_by_job_id,
                        state.created_at,
                        json.dumps(manifest, ensure_ascii=False),
                        json.dumps(config, ensure_ascii=False) if config is not None else None,
                    ),
                )
                conn.execute(
                    """INSERT INTO control_head(singleton, generation) VALUES (1, ?)
                       ON CONFLICT(singleton) DO UPDATE SET generation=excluded.generation""",
                    (state.generation,),
                )
                if desired_config is not None:
                    self._write_desired(conn, desired_config, state.created_at)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return state


__all__ = [
    "ControlConflict",
    "ControlError",
    "ControlSchemaTooNew",
    "ControlState",
    "ControlStore",
    "manifest_id",
]
