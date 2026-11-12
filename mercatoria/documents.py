"""SQLite registry for uploaded documents.

The registry is deliberately separate from the retrieval index: uploads need
status and error reporting even when no index exists, and an index rebuild must
not lose the document list.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    parsed_path TEXT,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    jurisdiction TEXT NOT NULL,
    language TEXT NOT NULL,
    precision TEXT NOT NULL,
    applies_to TEXT NOT NULL,
    effective_from TEXT,
    effective_to TEXT,
    source_doc TEXT NOT NULL,
    size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""

STATUSES = {"queued", "parsing", "parsed", "indexing", "ready", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _loads(value: str | None) -> list[str]:
    try:
        data = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    return [str(v) for v in data] if isinstance(data, list) else []


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        data["applies_to"] = _loads(data.get("applies_to"))
        return data

    def get(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._row(row)

    def get_by_sha(self, sha256: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return self._row(row)

    def list(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self._row(row) for row in rows if row is not None]

    def insert(self, record: dict[str, Any]) -> None:
        now = _now()
        values = {
            "doc_id": record["doc_id"],
            "filename": record["filename"],
            "original_path": record["original_path"],
            "parsed_path": record.get("parsed_path"),
            "sha256": record["sha256"],
            "status": record.get("status", "queued"),
            "error": record.get("error"),
            "jurisdiction": record["jurisdiction"],
            "language": record.get("language", "zh"),
            "precision": record.get("precision", "exact"),
            "applies_to": json.dumps(record.get("applies_to") or [], ensure_ascii=False),
            "effective_from": record.get("effective_from"),
            "effective_to": record.get("effective_to"),
            "source_doc": record["source_doc"],
            "size": int(record.get("size", 0)),
            "chunk_count": int(record.get("chunk_count", 0)),
            "created_at": record.get("created_at", now),
            "updated_at": now,
        }
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        self.conn.execute(
            f"INSERT OR REPLACE INTO documents ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        self.conn.commit()

    def update(self, doc_id: str, **fields: Any) -> None:
        allowed = {
            "parsed_path",
            "status",
            "error",
            "chunk_count",
            "source_doc",
            "jurisdiction",
            "language",
            "precision",
            "applies_to",
            "effective_from",
            "effective_to",
        }
        values = {k: v for k, v in fields.items() if k in allowed}
        if not values:
            return
        if "applies_to" in values and not isinstance(values["applies_to"], str):
            values["applies_to"] = json.dumps(values["applies_to"], ensure_ascii=False)
        values["updated_at"] = _now()
        assignments = ", ".join(f"{key}=?" for key in values)
        self.conn.execute(
            f"UPDATE documents SET {assignments} WHERE doc_id=?",
            (*values.values(), doc_id),
        )
        self.conn.commit()

    def delete(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        self.conn.commit()

    def manifest_docs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM documents WHERE status IN ('parsed','indexing','ready')"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            record = self._row(row)
            if not record or not record.get("parsed_path"):
                continue
            path = Path(record["parsed_path"])
            if not path.is_file():
                continue
            out.append(
                {
                    "doc_id": record["doc_id"],
                    "path": str(path),
                    "source_doc": record["source_doc"],
                    "jurisdiction": record["jurisdiction"],
                    "language": record["language"],
                    "precision": record["precision"],
                    "applies_to": record["applies_to"],
                    "effective_from": record["effective_from"],
                    "effective_to": record["effective_to"],
                    "snapshot_date": None,
                }
            )
        return out


def combined_manifest(base: dict[str, Any], registry: DocumentRegistry | None) -> dict[str, Any]:
    docs: dict[str, dict[str, Any]] = {}
    for doc in base.get("docs") or []:
        docs[str(doc["doc_id"])] = doc
    if registry is not None:
        for doc in registry.manifest_docs():
            docs[str(doc["doc_id"])] = doc
    ordered = [docs[key] for key in sorted(docs)]
    return {
        "manifest_version": base.get("manifest_version", "upload"),
        "corpus_dir": base.get("corpus_dir", "."),
        "documents": len(ordered),
        "docs": ordered,
    }
