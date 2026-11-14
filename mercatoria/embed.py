"""OpenAI-compatible embedding client.

The service deliberately calls an external gateway instead of loading a model:
one HTTP dependency, no CUDA, no model cache. The current gateway exposes
``POST /v1/embeddings`` and returns 1024-dimensional vectors for
``my-team-2k/qwen3.7-text-embedding``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import EmbeddingConfig
from .http_client import HttpError, JsonHttp


class EmbeddingError(RuntimeError):
    pass


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class EmbeddingClient:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        # Cache identity is the whole space, not just the model name: two
        # endpoints can serve the same model name and return different vectors.
        from .config import embedding_space_id

        self.space_id = embedding_space_id(cfg)
        self._http = JsonHttp(cfg.base, cfg.timeout)
        self._cache: sqlite3.Connection | None = None
        self._cache_lock = threading.Lock()
        if cfg.cache_path:
            path = Path(cfg.cache_path)
            if not path.is_absolute():
                from .config import REPO_ROOT

                path = REPO_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(
                path, check_same_thread=False, isolation_level=None
            )
            self._cache.execute("PRAGMA journal_mode=WAL")
            self._cache.execute("PRAGMA busy_timeout=5000")
            # A new table rather than a column: rows written before spaces
            # existed have no trustworthy identity, so they are never read. The
            # old ``embeddings`` table is left untouched (deleting the cache
            # file reclaims it).
            self._cache.execute(
                """CREATE TABLE IF NOT EXISTS embeddings_by_space (
                       space_id TEXT NOT NULL,
                       text_hash TEXT NOT NULL,
                       dim INTEGER NOT NULL,
                       vec BLOB NOT NULL,
                       PRIMARY KEY (space_id, text_hash)
                   )"""
            )

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_get(self, text: str) -> np.ndarray | None:
        if self._cache is None:
            return None
        with self._cache_lock:
            row = self._cache.execute(
                "SELECT dim, vec FROM embeddings_by_space WHERE space_id=? AND text_hash=?",
                (self.space_id, self._key(text)),
            ).fetchone()
        if not row:
            return None
        return np.frombuffer(row[1], dtype=np.float32)

    def _cache_put(self, text: str, vector: np.ndarray) -> None:
        if self._cache is None:
            return
        with self._cache_lock:
            self._cache.execute(
                """INSERT OR REPLACE INTO embeddings_by_space
                   (space_id, text_hash, dim, vec) VALUES (?,?,?,?)""",
                (
                    self.space_id,
                    self._key(text),
                    int(vector.shape[0]),
                    sqlite3.Binary(vector.astype(np.float32).tobytes()),
                ),
            )

    def _post(self, texts: list[str]) -> np.ndarray:
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = "Bearer " + self.cfg.api_key
        body = {"model": self.cfg.model, "input": texts}
        last: Exception | None = None
        for attempt in range(max(1, self.cfg.retries)):
            try:
                payload = self._http.post("embeddings", body, headers)
                data = payload.get("data")
                if not isinstance(data, list) or len(data) != len(texts):
                    raise EmbeddingError(
                        f"embedding response mismatch: expected {len(texts)}, got {type(data).__name__}"
                    )
                vectors = [item["embedding"] for item in data]
                matrix = np.asarray(vectors, dtype=np.float32)
                if matrix.ndim != 2:
                    raise EmbeddingError("embedding response is not a matrix")
                if self.cfg.dim and matrix.shape[1] != self.cfg.dim:
                    raise EmbeddingError(
                        f"embedding dim mismatch: configured {self.cfg.dim}, got {matrix.shape[1]}"
                    )
                return _normalize(matrix)
            except HttpError as exc:
                last = EmbeddingError(str(exc))
                if 400 <= exc.status < 500 and exc.status != 429:
                    break
                if attempt + 1 < max(1, self.cfg.retries):
                    time.sleep(2**attempt)
            except (RuntimeError, TimeoutError, KeyError, TypeError) as exc:
                last = exc
                if attempt + 1 < max(1, self.cfg.retries):
                    time.sleep(2**attempt)
        raise EmbeddingError(f"embedding request failed: {last}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.cfg.dim or 0), dtype=np.float32)

        results: list[np.ndarray | None] = [self._cache_get(text) for text in texts]
        missing = [i for i, vector in enumerate(results) if vector is None]
        if missing:
            missing_texts = [texts[i] for i in missing]
            size = max(1, min(self.cfg.batch_size, self.cfg.max_batch))
            batches = [missing_texts[i : i + size] for i in range(0, len(missing_texts), size)]
            workers = max(1, min(self.cfg.concurrency, len(batches)))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    chunks = list(pool.map(self._post, batches))
            else:
                chunks = [self._post(batch) for batch in batches]
            fresh = np.vstack(chunks)
            for index, vector in zip(missing, fresh):
                results[index] = vector
                self._cache_put(texts[index], vector)

        final = np.vstack([vector for vector in results if vector is not None])
        if self.cfg.dim and final.shape[1] != self.cfg.dim:
            raise EmbeddingError(
                f"embedding dim mismatch: configured {self.cfg.dim}, got {final.shape[1]}"
            )
        return final

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
