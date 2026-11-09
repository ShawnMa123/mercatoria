"""SQLite-backed index: BM25 postings + a dense vector matrix.

For the POC corpus (~10k chunks, 1024 dimensions) dense search is a single
NumPy matmul over a ~40 MB float32 matrix. That is faster and far simpler than
running Elasticsearch/FAISS. If the corpus grows past ~100k chunks, replace
``Index.vector_scores`` behind this class; the API does not change.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .chunker import ChunkRecord, chunk_document
from .config import AppConfig
from .tokenizer import term_counts, tokenize

SCHEMA = """
CREATE TABLE docs (
    doc_id TEXT PRIMARY KEY,
    source_doc TEXT NOT NULL,
    jurisdiction TEXT NOT NULL,
    language TEXT NOT NULL,
    precision TEXT NOT NULL,
    applies_to TEXT NOT NULL,
    effective_from TEXT,
    effective_to TEXT,
    snapshot_date TEXT,
    path TEXT NOT NULL
);
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    article_label TEXT NOT NULL,
    text TEXT NOT NULL,
    section TEXT NOT NULL,
    part INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    src_start INTEGER NOT NULL,
    src_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    jurisdiction TEXT NOT NULL,
    precision TEXT NOT NULL,
    applies_to TEXT NOT NULL,
    effective_from TEXT,
    effective_to TEXT
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE INDEX idx_chunks_jur ON chunks(jurisdiction);
CREATE INDEX idx_chunks_article ON chunks(article_id);
CREATE TABLE postings (
    term TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    tf INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
CREATE TABLE terms (
    term TEXT PRIMARY KEY,
    df INTEGER NOT NULL
);
CREATE TABLE vectors (
    chunk_id TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class IndexNotFound(RuntimeError):
    pass


class CorpusUnavailable(RuntimeError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version(cfg: AppConfig) -> str:
    # Microseconds avoid colliding with an index that is still open during a
    # same-second rebuild (Windows refuses to replace an open file).
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{ts}-{cfg.chunking_hash()}"


def _meta_set(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (key, value if isinstance(value, str) else _json(value)),
    )


def _meta_get(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def _row_to_chunk(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "article_id": row["article_id"],
        "article_label": row["article_label"],
        "text": row["text"],
        "section": row["section"],
        "part": row["part"],
        "ordinal": row["ordinal"],
        "src_start": row["src_start"],
        "src_end": row["src_end"],
        "token_count": row["token_count"],
        "jurisdiction": row["jurisdiction"],
        "precision": row["precision"],
        "applies_to": _loads(row["applies_to"], []),
        "effective_from": row["effective_from"],
        "effective_to": row["effective_to"],
    }


class Index:
    def __init__(self, path: Path, conn: sqlite3.Connection):
        self.path = path
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.version = str(_meta_get(conn, "version", ""))
        self.built_at = str(_meta_get(conn, "built_at", ""))
        self.embedding_model = str(_meta_get(conn, "embedding_model", ""))
        self.chunking_hash = str(_meta_get(conn, "chunking_hash", ""))
        self.snapshot_date = _meta_get(conn, "snapshot_date", None)
        self.avgdl = float(_meta_get(conn, "avgdl", 0.0) or 0.0)
        self.docs = {
            row["doc_id"]: dict(row)
            for row in conn.execute("SELECT * FROM docs ORDER BY doc_id")
        }
        self.chunks = [
            _row_to_chunk(row)
            for row in conn.execute("SELECT * FROM chunks ORDER BY chunk_id")
        ]
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.chunk_pos = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.vectors = self._load_vectors()
        self.jurisdictions = sorted({c["jurisdiction"] for c in self.chunks})
        self.precision: dict[str, str] = {}
        for c in self.chunks:
            self.precision.setdefault(c["jurisdiction"], c["precision"])
        self.article_count = len({c["article_id"] for c in self.chunks})
        self.known_applies_to = sorted(
            {value for c in self.chunks for value in c["applies_to"]}
        )
        self.config = _loads(_meta_get(self.conn, "config", "{}"), {})

    def _load_vectors(self) -> np.ndarray:
        rows = list(
            self.conn.execute("SELECT vec FROM vectors ORDER BY chunk_id")
        )
        # chunks were loaded ORDER BY chunk_id, so the two orders align.
        if rows and len(rows) != len(self.chunks):
            raise ValueError(
                f"index is inconsistent: {len(self.chunks)} chunks, {len(rows)} vectors"
            )
        if not rows:
            dim = int(_meta_get(self.conn, "embedding_dim", 0) or 0)
            return np.zeros((0, dim), dtype=np.float32)
        dim = int(_meta_get(self.conn, "embedding_dim", 0) or 0)
        matrix = np.vstack([np.frombuffer(row["vec"], dtype=np.float32) for row in rows])
        if dim and matrix.shape[1] != dim:
            raise ValueError(f"vector dim mismatch: meta={dim}, data={matrix.shape[1]}")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def close(self) -> None:
        self.conn.close()

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def documents(self) -> int:
        return len(self.docs)

    def corpus_block(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.total_chunks else "empty",
            "version": self.version,
            "snapshot_date": self.snapshot_date,
            "documents": self.documents,
            "article_count": self.article_count,
            "chunk_count": self.total_chunks,
            "jurisdictions": self.jurisdictions,
            "precision": self.precision,
            "built_at": self.built_at,
            "embedding_model": self.embedding_model,
        }

    def bm25_scores(
        self, candidate_ids: set[str], terms: Iterable[str], k1: float, b: float
    ) -> dict[str, float]:
        terms = [t for t in terms if t]
        if not terms or not candidate_ids:
            return {}
        placeholders = ",".join("?" for _ in terms)
        rows = self.conn.execute(
            f"""SELECT p.term, p.chunk_id, p.tf, t.df
                FROM postings p JOIN terms t ON t.term = p.term
                WHERE p.term IN ({placeholders})""",
            terms,
        )
        n = max(1, self.total_chunks)
        avgdl = self.avgdl or 1.0
        scores: dict[str, float] = {}
        for term, chunk_id, tf, df in rows:
            if chunk_id not in candidate_ids:
                continue
            chunk = self.chunk_by_id[chunk_id]
            dl = chunk["token_count"] or 1
            idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + float(idf * tf * (k1 + 1.0) / denom)
        return scores

    def vector_scores(self, candidate_indices: list[int], query_vector: np.ndarray) -> np.ndarray:
        if not candidate_indices:
            return np.zeros((0,), dtype=np.float32)
        return self.vectors[candidate_indices] @ query_vector.astype(np.float32)


def build_index(
    manifest: dict[str, Any],
    cfg: AppConfig,
    output_path: str | Path,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    activate: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Build a complete index file and optionally activate it.

    ``embed_fn`` is injectable so tests and offline experiments can avoid the
    network. In production it defaults to the configured external gateway.
    """
    from .embed import EmbeddingClient

    version = _version(cfg)
    out = Path(output_path)
    if out.suffix != ".db" or out.is_dir():
        out = out / f"index-{version}.db"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    conn = sqlite3.connect(tmp)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")

    embed = embed_fn or EmbeddingClient(cfg.embedding).embed
    corpus_dir = cfg.resolved_corpus_dir()

    all_chunks: list[tuple[dict[str, Any], ChunkRecord]] = []
    for i, doc in enumerate(manifest["docs"], 1):
        path = corpus_dir / doc["path"]
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise FileNotFoundError(f"corpus file missing: {path}") from exc
        conn.execute(
            """INSERT OR REPLACE INTO docs
               (doc_id, source_doc, jurisdiction, language, precision,
                applies_to, effective_from, effective_to, snapshot_date, path)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                doc["doc_id"],
                doc["source_doc"],
                doc["jurisdiction"],
                doc["language"],
                doc["precision"],
                _json(doc.get("applies_to") or []),
                doc.get("effective_from"),
                doc.get("effective_to"),
                doc.get("snapshot_date"),
                doc["path"],
            ),
        )
        records = chunk_document(doc, raw, cfg.chunking)
        for rec in records:
            all_chunks.append((doc, rec))
        if progress:
            progress("chunk", i, len(manifest["docs"]))

    chunk_rows = []
    posting_rows: list[tuple[str, str, int]] = []
    texts: list[str] = []
    chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for doc, rec in all_chunks:
        if rec.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(rec.chunk_id)
        token_counts = term_counts(rec.text)
        chunk_rows.append(
            (
                rec.chunk_id,
                rec.doc_id,
                rec.article_id,
                rec.article_label,
                rec.text,
                rec.section,
                rec.part,
                rec.ordinal,
                rec.start,
                rec.end,
                sum(token_counts.values()),
                doc["jurisdiction"],
                doc["precision"],
                _json(doc.get("applies_to") or []),
                doc.get("effective_from"),
                doc.get("effective_to"),
            )
        )
        for term, tf in token_counts.items():
            posting_rows.append((term, rec.chunk_id, int(tf)))
        texts.append(rec.text)
        chunk_ids.append(rec.chunk_id)

    conn.executemany(
        """INSERT OR REPLACE INTO chunks
           (chunk_id, doc_id, article_id, article_label, text, section, part,
            ordinal, src_start, src_end, token_count, jurisdiction, precision,
            applies_to, effective_from, effective_to)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        chunk_rows,
    )
    # Keep SQLite parameter/transaction sizes reasonable.
    for i in range(0, len(posting_rows), 50_000):
        conn.executemany(
            "INSERT OR REPLACE INTO postings(term, chunk_id, tf) VALUES (?,?,?)",
            posting_rows[i : i + 50_000],
        )
    conn.execute(
        "INSERT OR REPLACE INTO terms(term, df) SELECT term, COUNT(*) FROM postings GROUP BY term"
    )

    if progress:
        progress("embed", 0, len(texts))
    try:
        matrix = embed(texts) if texts else np.zeros((0, 0), dtype=np.float32)
        if matrix.shape[0] != len(texts):
            raise ValueError("embedding count mismatch")
    except Exception:
        conn.close()
        tmp.unlink(missing_ok=True)
        raise
    if progress:
        progress("embed", len(texts), len(texts))
    dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    conn.executemany(
        "INSERT OR REPLACE INTO vectors(chunk_id, dim, vec) VALUES (?,?,?)",
        [
            (cid, dim, sqlite3.Binary(matrix[i].astype(np.float32).tobytes()))
            for i, cid in enumerate(chunk_ids)
        ],
    )

    avgdl = (
        conn.execute("SELECT AVG(token_count) FROM chunks").fetchone()[0] or 0.0
    )
    article_count = conn.execute("SELECT COUNT(DISTINCT article_id) FROM chunks").fetchone()[0]
    jurisdictions = [
        row[0] for row in conn.execute("SELECT DISTINCT jurisdiction FROM docs ORDER BY jurisdiction")
    ]
    precision = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT jurisdiction, MAX(precision) FROM docs GROUP BY jurisdiction"
        )
    }
    snapshot_date = conn.execute(
        "SELECT MAX(snapshot_date) FROM docs WHERE snapshot_date IS NOT NULL"
    ).fetchone()[0]
    _meta_set(conn, "version", version)
    _meta_set(conn, "built_at", _now())
    _meta_set(conn, "embedding_model", cfg.embedding.model)
    _meta_set(conn, "embedding_dim", dim)
    _meta_set(conn, "chunking_hash", cfg.chunking_hash())
    _meta_set(conn, "avgdl", float(avgdl))
    _meta_set(conn, "chunk_count", len(chunk_rows))
    _meta_set(conn, "doc_count", len(manifest["docs"]))
    _meta_set(conn, "article_count", int(article_count))
    _meta_set(conn, "jurisdictions", jurisdictions)
    _meta_set(conn, "precision", precision)
    _meta_set(conn, "snapshot_date", snapshot_date)
    _meta_set(conn, "config", cfg.public_dict())
    conn.commit()
    conn.close()

    os.replace(tmp, out)
    if activate:
        activate_index(out.parent, version)
    if progress:
        progress("done", len(chunk_rows), len(chunk_rows))
    return {
        "version": version,
        "path": str(out),
        "documents": len(manifest["docs"]),
        "chunks": len(chunk_rows),
        "articles": int(article_count),
        "embedding_dim": dim,
        "avgdl": float(avgdl),
    }


def _current_file(index_dir: Path) -> Path:
    return index_dir / "current.txt"


def activate_index(index_dir: str | Path, version: str) -> None:
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    matches = sorted(d.glob(f"index-{version}*.db"))
    if not matches:
        raise FileNotFoundError(f"index version not found: {version}")
    _current_file(d).write_text(matches[-1].name, encoding="utf-8")


def open_index(path: str | Path) -> Index:
    p = Path(path)
    if not p.is_file():
        raise IndexNotFound(f"index not found: {p}")
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return Index(p, conn)


def open_current(index_dir: str | Path) -> Index:
    d = Path(index_dir)
    cur = _current_file(d)
    if cur.is_file():
        name = cur.read_text(encoding="utf-8").strip()
        if name:
            return open_index(d / name)
    candidates = sorted(d.glob("index-*.db"))
    if not candidates:
        raise IndexNotFound(f"no index in {d}")
    return open_index(candidates[-1])
