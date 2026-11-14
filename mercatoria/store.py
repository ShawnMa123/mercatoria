"""SQLite-backed index: BM25 postings + a dense vector matrix.

For the POC corpus (~10k chunks, 1024 dimensions) dense search is a single
NumPy matmul over a ~40 MB float32 matrix. That is faster and far simpler than
running Elasticsearch/FAISS. If the corpus grows past ~100k chunks, replace
``Index.vector_scores`` behind this class; the API does not change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .chunker import ChunkRecord, chunk_document, cn_to_int
from .config import AppConfig, embedding_space_id, snapshot_config_payload
from .tokenizer import term_counts, tokenize

# Bump when a table/column the reader relies on changes. Opening an older
# index is supported (missing columns fall back), a newer one is refused.
SCHEMA_VERSION = 2

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
    path TEXT NOT NULL,
    superseded_by TEXT NOT NULL DEFAULT ''
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
    index_token_count INTEGER,
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


class SchemaTooNew(RuntimeError):
    """The index was written by a newer mercatoria build."""


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


SYNTHETIC_ARTICLE_MARKERS = (":doc", ":preamble", ":section-", ":chunk-")


class _Unset:
    """Distinguish "caller did not bind a base" from an explicit ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "BASE_UNSET"


BASE_UNSET = _Unset()

# Cap on articles returned by follow_refs so one densely referenced article
# cannot blow up a response.
ARTICLE_FOLLOW_LIMIT = 20

_CN_NUM = r"[零〇一二三四五六七八九十百千0-9]+"
# "本法第五十六条、第五十七条第一款": one prefix, several article numbers.
RE_CN_SELF_GROUP = re.compile(
    rf"(?:本法|本条例|本办法|本规定|本章)\s*"
    rf"((?:第{_CN_NUM}条(?:之{_CN_NUM})?)"
    rf"(?:\s*[、和及与]\s*第{_CN_NUM}条(?:之{_CN_NUM})?)*)"
)
RE_CN_ONE = re.compile(rf"第({_CN_NUM})条(?:之({_CN_NUM}))?")
RE_CN_RANGE = re.compile(rf"第({_CN_NUM})条\s*(?:至|－|-|~|～)\s*第({_CN_NUM})条")
RE_CN_RELATIVE = re.compile(r"(前条|上条|下条|前款)")
RE_CN_TITLED = re.compile(rf"《([^》]{{2,60}})》\s*(第({_CN_NUM})条(?:之({_CN_NUM}))?)")
RE_EN_SECTION = re.compile(r"(?:^|[^A-Za-z])[Ss]ection\s+(\d+(?:\.\d+)*[A-Za-z]?)")
RE_EN_SECTION_MARK = re.compile(r"§\s*(\d+(?:\.\d+)*[A-Za-z]?)")
RE_EN_ARTICLE = re.compile(r"\bArticle\s+(\d+[A-Za-z]?)\b")


def is_synthetic_article(article_id: str) -> bool:
    """True for fallback chunks that are not legal articles."""
    return any(marker in article_id for marker in SYNTHETIC_ARTICLE_MARKERS)


def _row_to_chunk(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    token_count = int(row["token_count"] or 0)
    # Older indexes (schema < 2) indexed the raw text, so one count means both.
    index_token_count = (
        int(row["index_token_count"] or 0) if "index_token_count" in keys else token_count
    )
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
        "token_count": token_count,
        "index_token_count": index_token_count,
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
        self.embedding_space_id = str(_meta_get(conn, "embedding_space_id", ""))
        self.chunking_hash = str(_meta_get(conn, "chunking_hash", ""))
        self.snapshot_date = _meta_get(conn, "snapshot_date", None)
        self.schema_version = int(_meta_get(conn, "schema_version", 1) or 1)
        if self.schema_version > SCHEMA_VERSION:
            conn.close()
            raise SchemaTooNew(
                f"index schema {self.schema_version} is newer than this build "
                f"({SCHEMA_VERSION}); upgrade mercatoria or rebuild the index"
            )
        # schema < 2 stored the indexed token count (header included) in
        # token_count and had no index_token_count column.
        self.schema_outdated = self.schema_version < SCHEMA_VERSION
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
        # The inverted index is a few MB at POC scale. Keeping it in memory
        # removes every SQLite call from the query path, which is also the
        # only safe answer to concurrent requests on a shared connection.
        self.terms: dict[str, int] = {
            str(row["term"]): int(row["df"])
            for row in conn.execute("SELECT term, df FROM terms")
        }
        self.postings: dict[str, list[tuple[str, int]]] = {}
        for row in conn.execute("SELECT term, chunk_id, tf FROM postings"):
            self.postings.setdefault(str(row["term"]), []).append(
                (str(row["chunk_id"]), int(row["tf"]))
            )
        n = max(1, len(self.chunks))
        self.idf: dict[str, float] = {
            term: float(np.log(1.0 + (n - df + 0.5) / (df + 0.5)))
            for term, df in self.terms.items()
        }
        # Article-level aggregation: one long article is several chunks, and
        # external agents need to read it whole and reach its neighbours.
        self.articles: dict[str, list[dict[str, Any]]] = {}
        for chunk in self.chunks:
            self.articles.setdefault(chunk["article_id"], []).append(chunk)
        for article_chunks in self.articles.values():
            article_chunks.sort(key=lambda c: c["ordinal"])
        self.doc_articles: dict[str, list[str]] = {}
        for article_id, article_chunks in sorted(
            self.articles.items(), key=lambda item: item[1][0]["ordinal"]
        ):
            self.doc_articles.setdefault(article_chunks[0]["doc_id"], []).append(article_id)
        self.jurisdictions = sorted({c["jurisdiction"] for c in self.chunks})
        self.precision: dict[str, str] = {}
        for c in self.chunks:
            self.precision.setdefault(c["jurisdiction"], c["precision"])
        self.article_count = len({c["article_id"] for c in self.chunks})
        self.known_applies_to = sorted(
            {value for c in self.chunks for value in c["applies_to"]}
        )
        self._references: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._doc_id_by_source_doc: dict[str, str] = {
            str(doc.get("source_doc") or "").strip(): doc_id
            for doc_id, doc in self.docs.items()
            if str(doc.get("source_doc") or "").strip()
        }
        self.config = _loads(_meta_get(self.conn, "config", "{}"), {})
        # What the build was made from: its own manifest, build config and base
        # generation. Activation reads this record instead of borrowing the
        # current state's manifest, which would publish a different document set
        # than the one that was built.
        self.candidate = _loads(_meta_get(self.conn, "candidate", "null"), None)
        # Everything is in memory now; a closed connection cannot be raced by
        # concurrent request threads.
        self.conn.close()

    def _load_vectors(self) -> np.ndarray:
        """Decode vectors and put row *i* exactly at ``chunk_ids[i]``.

        The old implementation selected ``vec`` ordered by chunk_id and assumed
        that order matched the chunk list. Equal lengths therefore proved
        nothing: a wrong or missing id still produced a matrix of the right
        shape, and a tampered candidate could be published. The id set is now
        matched explicitly, and any missing, extra, duplicate or differently
        sized vector refuses the index.
        """
        rows = list(self.conn.execute("SELECT chunk_id, vec FROM vectors"))
        dim = int(_meta_get(self.conn, "embedding_dim", 0) or 0)
        if not rows:
            if self.chunks:
                raise ValueError(
                    f"index is inconsistent: {len(self.chunks)} chunks but no vectors"
                )
            self.vector_chunk_ids = []
            return np.zeros((0, dim), dtype=np.float32)
        by_id: dict[str, bytes] = {}
        for chunk_id, blob in rows:
            key = str(chunk_id)
            if key in by_id:
                raise ValueError(f"index is inconsistent: duplicate vector for {key}")
            by_id[key] = blob
        missing = [cid for cid in self.chunk_ids if cid not in by_id]
        extra = [cid for cid in by_id if cid not in self.chunk_pos]
        if missing or extra:
            raise ValueError(
                "index is inconsistent: vector ids do not match chunks "
                f"(missing={missing[:5]}, extra={extra[:5]})"
            )
        vectors: list[np.ndarray] = []
        width = 0
        for chunk_id in self.chunk_ids:
            vector = np.frombuffer(by_id[chunk_id], dtype=np.float32)
            if vector.ndim != 1 or vector.shape[0] == 0:
                raise ValueError(f"index is inconsistent: empty vector for {chunk_id}")
            if not width:
                width = int(vector.shape[0])
            elif int(vector.shape[0]) != width:
                raise ValueError(
                    f"index is inconsistent: vector width {vector.shape[0]} for "
                    f"{chunk_id} differs from {width}"
                )
            vectors.append(vector)
        if dim and width != dim:
            raise ValueError(f"vector dim mismatch: meta={dim}, data={width}")
        self.vector_chunk_ids = list(self.chunk_ids)
        return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)

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
        avgdl = self.avgdl or 1.0
        scores: dict[str, float] = {}
        for term in set(terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for chunk_id, tf in postings:
                if chunk_id not in candidate_ids:
                    continue
                dl = self.chunk_by_id[chunk_id]["index_token_count"] or 1
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf * (k1 + 1.0) / denom
        return scores

    def article_chunks(self, article_id: str) -> list[dict[str, Any]]:
        return self.articles.get(article_id, [])

    def article_text(self, article_chunks: list[dict[str, Any]]) -> str:
        """Reassemble the raw article text from its chunks.

        Chunks of a split article overlap (``chunking.overlap``), and every
        chunk keeps its absolute ``src_start``/``src_end``, so the overlap is
        exactly the prefix already emitted.
        """
        pieces: list[str] = []
        cursor = 0
        for chunk in article_chunks:
            text = chunk["text"]
            start = int(chunk["src_start"])
            if not pieces:
                pieces.append(text)
            else:
                overlap = max(0, min(len(text), cursor - start))
                pieces.append(text[overlap:])
            cursor = max(cursor, start + len(text))
        return "".join(pieces)

    def article_payload(
        self, article_id: str, neighbors: int = 0, follow_refs: int = 0
    ) -> dict[str, Any] | None:
        article_chunks = self.article_chunks(article_id)
        if not article_chunks:
            return None
        first = article_chunks[0]
        doc = self.docs.get(first["doc_id"], {})
        followed = self.article_followed(article_id, follow_refs)
        return {
            "article_id": article_id,
            "doc_id": first["doc_id"],
            "source_doc": doc.get("source_doc", first["doc_id"]),
            "article_label": first["article_label"],
            "section": first["section"],
            "jurisdiction": first["jurisdiction"],
            "applies_to": first["applies_to"],
            "effective_from": first["effective_from"],
            "effective_to": first["effective_to"],
            "superseded_by": doc.get("superseded_by") or None,
            "text": self.article_text(article_chunks),
            "chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "part": chunk["part"],
                    "ordinal": chunk["ordinal"],
                    "token_count": chunk["token_count"],
                    "text": chunk["text"],
                }
                for chunk in article_chunks
            ],
            "neighbors": self.article_neighbors(article_id, neighbors),
            "references": self.article_references(article_id),
            "followed": followed,
            "truncated": len(followed) >= ARTICLE_FOLLOW_LIMIT,
        }

    def article_summary(self, article_id: str) -> dict[str, Any] | None:
        chunks = self.article_chunks(article_id)
        if not chunks:
            return None
        return {
            "article_id": article_id,
            "doc_id": chunks[0]["doc_id"],
            "article_label": chunks[0]["article_label"],
            "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            "superseded_by": self.docs.get(chunks[0]["doc_id"], {}).get("superseded_by") or None,
            "text": self.article_text(chunks),
        }

    def article_followed(self, article_id: str, depth: int) -> list[dict[str, Any]]:
        """Summaries of the articles this one references (1 hop per depth).

        The result is capped so a densely cross-referenced article cannot
        explode the response; ``truncated`` on the payload reports the cap.
        """
        if depth <= 0:
            return []
        seen: set[str] = {article_id}
        frontier = [article_id]
        out: list[dict[str, Any]] = []
        for _ in range(min(2, depth)):
            next_frontier: list[str] = []
            for aid in frontier:
                refs = self.article_references(aid)["resolved"]
                for ref in refs:
                    target = ref["article_id"]
                    if target in seen:
                        continue
                    seen.add(target)
                    summary = self.article_summary(target)
                    if summary is not None:
                        out.append(summary)
                    next_frontier.append(target)
                    if len(out) >= ARTICLE_FOLLOW_LIMIT:
                        return out
            frontier = next_frontier
        return out

    def article_references(self, article_id: str) -> dict[str, list[dict[str, Any]]]:
        """Same-document (and unambiguous cross-document) article references."""
        cached = self._references.get(article_id)
        if cached is not None:
            return cached
        result: dict[str, list[dict[str, Any]]] = {"resolved": [], "unresolved": []}
        chunks = self.article_chunks(article_id)
        if chunks:
            text = self.article_text(chunks)
            doc_id = chunks[0]["doc_id"]
            seen: set[tuple[str, str]] = set()

            def add(target: str | None, snippet: str) -> None:
                if target is None:
                    if (snippet, "") not in seen:
                        seen.add((snippet, ""))
                        result["unresolved"].append({"text": snippet})
                    return
                if target == article_id and not snippet.endswith("前款"):
                    return
                if (snippet, target) in seen:
                    return
                seen.add((snippet, target))
                if any(item["article_id"] == target for item in result["resolved"]):
                    return
                label_chunks = self.article_chunks(target)
                if not label_chunks:
                    result["unresolved"].append({"text": snippet})
                    return
                result["resolved"].append(
                    {
                        "text": snippet,
                        "article_id": target,
                        "article_label": label_chunks[0]["article_label"],
                    }
                )

            for group in RE_CN_SELF_GROUP.finditer(text):
                for match in RE_CN_ONE.finditer(group.group(1)):
                    add(
                        self._resolve_number(doc_id, match.group(1), match.group(2)),
                        match.group(0),
                    )
            for match in RE_CN_RANGE.finditer(text):
                start = cn_to_int(match.group(1))
                end = cn_to_int(match.group(2))
                if start is None or end is None or end < start or end - start > 30:
                    result["unresolved"].append({"text": match.group(0)})
                    continue
                for number in range(start, end + 1):
                    add(self._resolve_number(doc_id, str(number), None), match.group(0))
            for match in RE_CN_TITLED.finditer(text):
                title, snippet, number, suffix = match.group(1), match.group(2), match.group(3), match.group(4)
                target_doc = self._doc_id_by_source_doc.get(title.strip())
                add(
                    self._resolve_number(target_doc, number, suffix) if target_doc else None,
                    snippet,
                )
            for match in RE_CN_RELATIVE.finditer(text):
                snippet = match.group(1)
                if snippet == "前款":
                    add(article_id, snippet)
                else:
                    add(self._relative_article(article_id, snippet), snippet)
            for match in RE_EN_SECTION.finditer(text):
                add(self._resolve_number(doc_id, match.group(1), None), match.group(0).strip())
            for match in RE_EN_SECTION_MARK.finditer(text):
                add(self._resolve_number(doc_id, match.group(1), None), match.group(0))
            for match in RE_EN_ARTICLE.finditer(text):
                add(self._resolve_number(doc_id, match.group(1), None), match.group(0))

        self._references[article_id] = result
        return result

    def _resolve_number(
        self, doc_id: str | None, number: str, suffix: str | None
    ) -> str | None:
        if not doc_id:
            return None
        number = number.strip()
        raw = number
        if number.isdigit() and suffix:
            suffix_int = cn_to_int(suffix)
            candidate = f"{doc_id}:{int(number)}.{suffix_int if suffix_int is not None else suffix}"
            if candidate in self.articles:
                return candidate
        if re.fullmatch(r"[零〇一二三四五六七八九十百千]+", number):
            value = cn_to_int(number)
            if value is None:
                return None
            number = str(value)
        for candidate in (
            f"{doc_id}:{number}",
            f"{doc_id}:sec-{number}",
            f"{doc_id}:art-{number}",
        ):
            if candidate in self.articles:
                return candidate
        return None

    def _relative_article(self, article_id: str, keyword: str) -> str | None:
        chunks = self.article_chunks(article_id)
        if not chunks:
            return None
        sequence = [
            aid
            for aid in self.doc_articles.get(chunks[0]["doc_id"], [])
            if not is_synthetic_article(aid)
        ]
        if article_id not in sequence:
            return None
        position = sequence.index(article_id)
        index = position - 1 if keyword in {"前条", "上条"} else position + 1
        if 0 <= index < len(sequence):
            return sequence[index]
        return None

    def article_neighbors(self, article_id: str, count: int) -> dict[str, list[dict[str, Any]]]:
        """Previous/next legal articles in the same document, in source order."""
        empty: dict[str, list[dict[str, Any]]] = {"previous": [], "next": []}
        if count <= 0:
            return empty
        article_chunks = self.article_chunks(article_id)
        if not article_chunks:
            return empty
        sequence = [
            aid
            for aid in self.doc_articles.get(article_chunks[0]["doc_id"], [])
            if not is_synthetic_article(aid)
        ]
        if article_id not in sequence:
            return empty
        position = sequence.index(article_id)

        def summary(aid: str) -> dict[str, Any]:
            chunks = self.article_chunks(aid)
            return {
                "article_id": aid,
                "article_label": chunks[0]["article_label"],
                "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
                "text": self.article_text(chunks),
            }

        return {
            "previous": [
                summary(aid) for aid in sequence[max(0, position - count) : position]
            ],
            "next": [summary(aid) for aid in sequence[position + 1 : position + 1 + count]],
        }

    def vector_scores(self, candidate_indices: list[int], query_vector: np.ndarray) -> np.ndarray:
        if not candidate_indices:
            return np.zeros((0,), dtype=np.float32)
        return self.vectors[candidate_indices] @ query_vector.astype(np.float32)


def _reusable_vectors(
    path: Path | None,
    space_id: str,
    wanted: set[str],
    hashes: dict[str, str],
) -> dict[str, np.ndarray]:
    """Vectors from the *published* index for unchanged chunk IDs.

    Rebuilding after an upload should only embed the new chunks. Reading the
    published SQLite file directly keeps this independent of the in-memory
    Index held by the serving process.

    Reuse is gated on the index's recorded ``embedding_space_id``: the same
    model name on another endpoint (or another explicit space version) is a
    different space, and injecting its vectors would relabel old data as the
    new space. An index written before space ids existed has no trustworthy
    identity and is not reused (it is rebuilt instead).
    """
    if path is None or not path.is_file():
        return {}
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='embedding_space_id'").fetchone()
        if not row or str(row[0]) != space_id:
            return {}
        # The header text comes from document metadata (source_doc/section),
        # which the space id does not cover, so compare the per-chunk hash of
        # the exact indexed text before reusing a vector.
        row = conn.execute("SELECT value FROM meta WHERE key='index_text_hash'").fetchone()
        old_hashes = _loads(row[0], {}) if row else {}
        if not isinstance(old_hashes, dict) or not old_hashes:
            return {}
        out: dict[str, np.ndarray] = {}
        for chunk_id, blob in conn.execute("SELECT chunk_id, vec FROM vectors"):
            if chunk_id not in wanted:
                continue
            if old_hashes.get(str(chunk_id)) != hashes.get(str(chunk_id)):
                continue
            out[str(chunk_id)] = np.frombuffer(blob, dtype=np.float32)
        return out
    finally:
        conn.close()


def build_index(
    manifest: dict[str, Any],
    cfg: AppConfig,
    output_path: str | Path,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    activate: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
    publish: Any = None,
    base_generation: Any = BASE_UNSET,
) -> dict[str, Any]:
    """Build a complete index file and optionally publish it.

    ``embed_fn`` is injectable so tests and offline experiments can avoid the
    network. In production it defaults to the configured external gateway.

    ``activate=False`` only leaves a candidate file behind: the control head,
    the served snapshot and ``current.txt`` are untouched, so a staged build
    cannot become live through a restart either.

    ``publish`` is the owning :class:`PublicationCoordinator`. Callers that do
    not pass one get a coordinator without a snapshot holder (offline CLI):
    the commit happens, the process-local install does not.

    ``base_generation`` lets the caller bind the base to the same snapshot it
    took the build config from (``None`` = the head must be empty). Left unset,
    the base is read here, which is only correct for offline callers that own
    the directory exclusively.
    """
    from .embed import EmbeddingClient
    from .control import ControlStore

    version = _version(cfg)
    probe = ControlStore(cfg.resolved_control_path())
    try:
        if base_generation is BASE_UNSET:
            # Captured before building: publication fails if the head moved
            # meanwhile instead of silently overwriting a newer state.
            base_generation = probe.head_generation()
        source_index = _published_index_path(probe, cfg.resolved_index_dir())
    finally:
        probe.close()
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
                applies_to, effective_from, effective_to, snapshot_date, path,
                superseded_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
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
                doc.get("superseded_by") or "",
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
    index_hashes: dict[str, str] = {}
    seen_chunk_ids: set[str] = set()
    for doc, rec in all_chunks:
        if rec.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(rec.chunk_id)
        # Only the retrieval signal is enriched: rec.text (and therefore the
        # chunk_id and the chunk text served to callers) stays the raw article.
        index_text = rec.text
        if cfg.chunking.index_context_header:
            header = " | ".join(
                part for part in (doc.get("source_doc"), rec.section, rec.article_label) if part
            )
            if header:
                index_text = f"[{header}]\n{rec.text}"
        index_hashes[rec.chunk_id] = hashlib.sha1(
            index_text.encode("utf-8")
        ).hexdigest()[:16]
        token_counts = term_counts(index_text)
        raw_token_count = sum(term_counts(rec.text).values())
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
                raw_token_count,
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
        texts.append(index_text)
        chunk_ids.append(rec.chunk_id)

    conn.executemany(
        """INSERT OR REPLACE INTO chunks
           (chunk_id, doc_id, article_id, article_label, text, section, part,
            ordinal, src_start, src_end, token_count, index_token_count,
            jurisdiction, precision, applies_to, effective_from, effective_to)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
        if not texts:
            matrix = np.zeros((0, 0), dtype=np.float32)
        else:
            reuse = _reusable_vectors(
                source_index,
                embedding_space_id(cfg.embedding),
                set(chunk_ids),
                index_hashes,
            )
            missing = [i for i, cid in enumerate(chunk_ids) if cid not in reuse]
            fresh = embed([texts[i] for i in missing]) if missing else None
            if fresh is not None:
                if fresh.shape[0] != len(missing):
                    raise ValueError("embedding count mismatch")
                dim = int(fresh.shape[1])
            else:
                dim = int(next(iter(reuse.values())).shape[0])
            matrix = np.zeros((len(texts), dim), dtype=np.float32)
            for i, cid in enumerate(chunk_ids):
                cached = reuse.get(cid)
                if cached is not None and cached.shape[0] == dim:
                    matrix[i] = cached
            if fresh is not None:
                for row, index in enumerate(missing):
                    matrix[index] = fresh[row]
            if progress and missing:
                progress("embed", len(texts) - len(missing), len(texts))
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
        conn.execute("SELECT AVG(index_token_count) FROM chunks").fetchone()[0] or 0.0
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
    _meta_set(conn, "schema_version", SCHEMA_VERSION)
    _meta_set(conn, "index_text_hash", json.dumps(index_hashes, sort_keys=True))
    _meta_set(conn, "built_at", _now())
    _meta_set(conn, "embedding_model", cfg.embedding.model)
    _meta_set(conn, "embedding_dim", dim)
    _meta_set(conn, "embedding_space_id", embedding_space_id(cfg.embedding))
    _meta_set(conn, "chunking_hash", cfg.chunking_hash())
    _meta_set(conn, "avgdl", float(avgdl))
    _meta_set(conn, "chunk_count", len(chunk_rows))
    _meta_set(conn, "doc_count", len(manifest["docs"]))
    _meta_set(conn, "article_count", int(article_count))
    _meta_set(conn, "jurisdictions", jurisdictions)
    _meta_set(conn, "precision", precision)
    _meta_set(conn, "snapshot_date", snapshot_date)
    _meta_set(conn, "config", snapshot_config_payload(cfg))
    # Everything activation needs, recorded by the build that produced it.
    _meta_set(
        conn,
        "candidate",
        {"base_generation": base_generation, "manifest": manifest},
    )
    conn.commit()
    conn.close()

    os.replace(tmp, out)
    result = {
        "version": version,
        "path": str(out),
        "documents": len(manifest["docs"]),
        "chunks": len(chunk_rows),
        "articles": int(article_count),
        "embedding_dim": dim,
        "avgdl": float(avgdl),
        "activated": False,
        "committed": False,
        "installed": False,
        "generation": None,
        "snapshot_version": None,
        "warnings": [],
    }
    if activate:
        from .publication import PublicationCoordinator

        owned = None
        publisher = publish
        if publisher is None:
            owned = ControlStore(cfg.resolved_control_path())
            publisher = PublicationCoordinator(owned, cfg, cfg.resolved_index_dir())
        try:
            publication = publisher.publish_built(
                version, manifest, cfg, base_generation=base_generation
            )
        finally:
            if owned is not None:
                owned.close()
        result["activated"] = publication.committed
        result["committed"] = publication.committed
        result["installed"] = publication.installed
        result["generation"] = publication.generation
        result["snapshot_version"] = publication.snapshot_version
        result["warnings"] = list(publication.warnings)
    if progress:
        # This runs after the commit. A broken progress sink must not turn a
        # published state into a "failed, uncommitted" job.
        try:
            progress("done", len(chunk_rows), len(chunk_rows))
        except Exception as exc:
            result["warnings"].append(f"final progress report failed: {exc}")
    return result


def _current_file(index_dir: Path) -> Path:
    return index_dir / "current.txt"


def index_path(index_dir: str | Path, version: str) -> Path:
    """File name of one index version. The version is the whole name."""
    return Path(index_dir) / f"index-{version}.db"


def _published_index_path(control: Any, index_dir: Path) -> Path | None:
    """Index file the control head points at, else the legacy export.

    Once a head exists, ``current.txt`` is only an export and is never consulted:
    a stale pointer must not make a rebuild reuse vectors from another state.
    """
    head = control.head()
    if head is not None:
        path = index_path(index_dir, head.index_version)
        return path if path.is_file() else None
    current = _current_file(index_dir)
    if current.is_file():
        try:
            name = current.read_text(encoding="utf-8").strip()
        except OSError:
            name = ""
        if name:
            path = index_dir / name
            if path.is_file():
                return path
    return None


def export_current(index_dir: str | Path, version: str) -> None:
    """Write ``current.txt`` atomically.

    This file is an export for old tooling and the migration input; the control
    head decides what serves requests. Replacement is atomic so a reader never
    sees a truncated name.
    """
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = index_path(d, version).name
    if not (d / name).is_file():
        raise FileNotFoundError(f"index version not found: {version}")
    target = _current_file(d)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(name, encoding="utf-8")
    os.replace(tmp, target)


def open_index(path: str | Path) -> Index:
    p = Path(path)
    if not p.is_file():
        raise IndexNotFound(f"index not found: {p}")
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        return Index(p, conn)
    except BaseException as exc:
        # A truncated or foreign file must not leak an open handle: on Windows
        # that keeps the data directory undeletable for the rest of the process.
        conn.close()
        if isinstance(exc, sqlite3.Error):
            raise IndexNotFound(f"{p} is not a usable index: {exc}") from exc
        raise


def open_current(index_dir: str | Path) -> Index:
    """Open whatever ``current.txt`` names.

    Migration and test helper only: serving code reads the control head and
    opens that exact version, so an unpublished candidate is never picked up.
    """
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
