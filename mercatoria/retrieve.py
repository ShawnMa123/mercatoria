"""Filter-first hybrid retrieval over the local SQLite index."""

from __future__ import annotations

import copy
import datetime as dt
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from .config import AppConfig
from .embed import EmbeddingClient
from .glossary import expand_query
from .rerank import RerankClient, RerankError
from .store import CorpusUnavailable, Index
from .tokenizer import extract_matches, query_terms

MAX_QUERIES = 16


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if str(v).strip()]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _precision_note(mode: str) -> str | None:
    if mode == "degraded":
        return "无修订历史，按快照日推定；不按日期过滤"
    if mode == "snapshot":
        return "条约/单版本文本；不按日期过滤"
    return None


def _map_fields(filters: dict[str, Any], fields: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(filters or {})
    fields = fields or {}
    if "business_type" in fields and not out.get("applies_to"):
        out["applies_to"] = _as_list(fields["business_type"])
    if "jurisdiction" in fields and not out.get("jurisdictions"):
        out["jurisdictions"] = _as_list(fields["jurisdiction"])
    if "country" in fields and not out.get("jurisdictions"):
        out["jurisdictions"] = _as_list(fields["country"])
    if "as_of" in fields and not out.get("as_of"):
        out["as_of"] = fields["as_of"]
    if "date" in fields and not out.get("as_of"):
        out["as_of"] = fields["date"]
    if "doc_ids" in fields and not out.get("doc_ids"):
        out["doc_ids"] = _as_list(fields["doc_ids"])
    if "article_ids" in fields and not out.get("article_ids"):
        out["article_ids"] = _as_list(fields["article_ids"])
    return out


def _quota_for(
    requested_j: list[str], top_k: int, per_jurisdiction_k: int | None
) -> int:
    """Slots reserved for each requested jurisdiction in the final top_k.

    ``per_jurisdiction_k`` wins when given. Otherwise a cross-jurisdiction
    comparison query (2+ jurisdictions) reserves min(2, top_k // n) slots per
    jurisdiction so one high-scoring law cannot take every slot.
    """
    if per_jurisdiction_k is not None:
        return max(0, int(per_jurisdiction_k))
    if len(requested_j) < 2:
        return 0
    return max(1, min(2, top_k // len(requested_j)))


def _apply_jurisdiction_quota(
    ordered: list[tuple[str, dict[str, Any]]],
    top_k: int,
    requested_j: list[str],
    quota: int,
    index: "Index",
    min_ratio: float = 0.0,
) -> list[tuple[str, dict[str, Any]]]:
    """Keep up to ``quota`` slots per requested jurisdiction, then fill.

    ``min_ratio`` softens the reservation: a jurisdiction only keeps its slot
    when its best candidate scores at least ``min_ratio`` times the global
    top-1. 0 keeps the hard quota.
    """
    if quota <= 0 or len(ordered) <= top_k:
        return ordered[:top_k]
    wanted = set(requested_j)
    best: dict[str, float] = {}
    for cid, hit in ordered:
        jurisdiction = index.chunk_by_id[cid]["jurisdiction"]
        if jurisdiction in wanted and jurisdiction not in best:
            best[jurisdiction] = float(hit.get("score", 0.0))
    top_score = float(ordered[0][1].get("score", 0.0)) if ordered else 0.0
    if min_ratio > 0:
        wanted = {
            jurisdiction
            for jurisdiction in wanted
            if best.get(jurisdiction, 0.0) >= min_ratio * top_score and top_score > 0
        }
    picked: list[tuple[str, dict[str, Any]]] = []
    taken: set[str] = set()
    counts: dict[str, int] = {j: 0 for j in requested_j}
    for cid, hit in ordered:
        jurisdiction = index.chunk_by_id[cid]["jurisdiction"]
        if jurisdiction in wanted and counts[jurisdiction] < quota:
            picked.append((cid, hit))
            taken.add(cid)
            counts[jurisdiction] += 1
            if len(picked) >= top_k:
                return picked
    for cid, hit in ordered:
        if cid in taken:
            continue
        picked.append((cid, hit))
        if len(picked) >= top_k:
            break
    return picked


class Retriever:
    def __init__(
        self,
        index: Index,
        cfg: AppConfig,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        rerank_fn: Callable[[str, list[str], int], tuple[list[tuple[int, float]], float]]
        | None = None,
    ):
        self.index = index
        self.cfg = cfg
        self.embed = embed_fn or EmbeddingClient(cfg.embedding).embed
        # Keep the client even when disabled: a request can opt in with
        # "rerank": true without rebuilding the retriever.
        self._rerank = rerank_fn if rerank_fn is not None else RerankClient(cfg.rerank).rerank
        self._rerank_forced = rerank_fn is not None
        self.rerank_enabled = bool(self._rerank_forced or cfg.rerank.enabled)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = 256
        # Query-vector LRU. /v1/retrieve is a sync route, so concurrent
        # requests share one Retriever: guard every mutation.
        # Response+vector caches are touched from request threads.
        self._cache_lock = threading.Lock()
        self._embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._embed_cache_size = max(0, int(cfg.embedding.query_cache_size))
        self._embed_lock = threading.Lock()

    def _cache_key(
        self,
        queries: list[str],
        top_k: int,
        filters: dict[str, Any],
        fields: dict[str, Any],
        rerank: bool,
        per_jurisdiction_k: int | None = None,
    ) -> str:
        # Every knob that can change the response must be part of the key:
        # /v1/config rewrites these in place, and a stale hit would then serve
        # results computed under the previous settings.
        payload = {
            "queries": queries,
            "top_k": top_k,
            "filters": filters,
            "fields": fields,
            "vector_weight": self.cfg.retrieval.vector_weight,
            "threshold": self.cfg.retrieval.similarity_threshold,
            "bm25_k1": self.cfg.retrieval.bm25_k1,
            "bm25_b": self.cfg.retrieval.bm25_b,
            "expand_cross_language": self.cfg.retrieval.expand_cross_language,
            "query_stopwords": self.cfg.retrieval.query_stopwords,
            "bm25_saturation": self.cfg.retrieval.bm25_saturation,
            "quota_min_ratio": self.cfg.retrieval.quota_min_ratio,
            "confidence": {
                "high": self.cfg.retrieval.confidence.high,
                "medium": self.cfg.retrieval.confidence.medium,
                "min_gap": self.cfg.retrieval.confidence.min_gap,
            },
            "multi_query_bonus": self.cfg.retrieval.multi_query_bonus,
            "rerank": rerank,
            "candidates": self.cfg.rerank.candidates,
            "fusion": self.cfg.rerank.fusion,
            "rerank_weight": self.cfg.rerank.weight,
            "rerank_top_n": self.cfg.rerank.top_n,
            "rerank_min_score": self.cfg.rerank.min_score,
            "rerank_model": self.cfg.rerank.model,
            "per_jurisdiction_k": per_jurisdiction_k,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
        with self._embed_lock:
            self._embed_cache.clear()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(cached)

    def _cache_put(self, key: str, response: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = copy.deepcopy(response)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _select(
        self, filters: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, int], str | None, dict[str, Any]]:
        rows = self.index.chunks
        total = len(rows)

        requested_j = [j.upper() for j in _as_list(filters.get("jurisdictions"))]
        requested_a = [a.lower() for a in _as_list(filters.get("applies_to"))]
        requested_doc = set(_as_list(filters.get("doc_ids")))
        requested_art = set(_as_list(filters.get("article_ids")))
        as_of = filters.get("as_of")
        if as_of:
            try:
                as_of_date = dt.date.fromisoformat(str(as_of))
            except ValueError as exc:
                raise ValueError(f"as_of must be YYYY-MM-DD: {as_of}") from exc
        else:
            as_of_date = None

        uncovered_j = [j for j in requested_j if j not in self.index.jurisdictions]
        uncovered_a = [a for a in requested_a if a not in self.index.known_applies_to]

        # 1) jurisdiction: requested set intersection with covered corpus.
        if requested_j:
            after_j = [r for r in rows if r["jurisdiction"] in requested_j]
        else:
            after_j = rows
        # 2) business type: empty applies_to means universal.
        if requested_a:
            after_a = [
                r
                for r in after_j
                if not r["applies_to"] or any(a in r["applies_to"] for a in requested_a)
            ]
        else:
            after_a = after_j
        # 3) temporal: exact jurisdictions filter; degraded/snapshot do not.
        if as_of_date is None:
            after_t = [r for r in after_a if not r["effective_to"]]
        else:
            after_t = []
            for r in after_a:
                if r["precision"] in {"degraded", "snapshot"}:
                    after_t.append(r)
                    continue
                start = r["effective_from"]
                end = r["effective_to"]
                if start and dt.date.fromisoformat(str(start)[:10]) > as_of_date:
                    continue
                if end and dt.date.fromisoformat(str(end)[:10]) < as_of_date:
                    continue
                after_t.append(r)

        # 4) explicit document/article scope (cross-reference lookups).
        if requested_doc:
            after_doc = [r for r in after_t if r["doc_id"] in requested_doc]
        else:
            after_doc = after_t
        if requested_art:
            after_art = [r for r in after_doc if r["article_id"] in requested_art]
        else:
            after_art = after_doc

        funnel = {
            "total": total,
            "after_jurisdiction": len(after_j),
            "after_applies_to": len(after_a),
            "after_as_of": len(after_t),
            "after_doc": len(after_doc),
            "after_article": len(after_art),
            "after_retrieval": 0,
        }
        zero_reason = None
        if total == 0:
            raise CorpusUnavailable("index is empty")
        if not after_j:
            zero_reason = "no_jurisdiction"
        elif not after_a:
            zero_reason = "no_business_type"
        elif not after_t:
            zero_reason = "no_temporal"
        elif not after_doc:
            zero_reason = "no_doc"
        elif not after_art:
            zero_reason = "no_article"

        uncovered = {
            "jurisdictions": uncovered_j,
            "applies_to": uncovered_a,
        }
        return after_art, funnel, zero_reason, uncovered

    def _temporal_block(self, as_of: str | None) -> dict[str, Any]:
        return {
            "as_of": as_of,
            "by_jurisdiction": {
                j: {
                    "mode": self.index.precision.get(j, "exact"),
                    "note": _precision_note(self.index.precision.get(j, "exact")),
                }
                for j in self.index.jurisdictions
            },
        }

    def _field_matches(
        self,
        requested_j: list[str],
        requested_a: list[str],
        rows: list[dict[str, Any]],
        as_of: str | None,
    ) -> dict[str, Any]:
        matched_j = sorted({r["jurisdiction"] for r in rows})
        matched_a = sorted({a for r in rows for a in r["applies_to"]})
        precision = {
            j: self.index.precision.get(j, "exact")
            for j in (requested_j or self.index.jurisdictions)
        }
        return {
            "jurisdictions": {
                "requested": requested_j,
                "matched": matched_j,
                "uncovered": [j for j in requested_j if j not in matched_j],
            },
            "applies_to": {
                "requested": requested_a,
                "matched": matched_a,
                "uncovered": [a for a in requested_a if a not in matched_a],
            },
            "as_of": {"requested": as_of, "precision": precision},
        }

    def _embed_queries(self, queries: list[str]) -> tuple[np.ndarray, int, bool]:
        """Embed queries through the per-Retriever LRU.

        Returns (matrix aligned with ``queries``, cache_hits, degraded).
        ``degraded`` is True only when the gateway failed and the caller
        opted into the BM25-only fallback.
        """
        model = self.cfg.embedding.model
        keys = [f"{model}\x1f{query}" for query in queries]
        vectors: dict[str, np.ndarray] = {}
        missing: list[str] = []
        hits = 0
        with self._embed_lock:
            for key in keys:
                if key in vectors or key in missing:
                    continue
                cached = self._embed_cache.get(key)
                if cached is None:
                    missing.append(key)
                else:
                    self._embed_cache.move_to_end(key)
                    vectors[key] = cached
                    hits += 1
        degraded = False
        if missing:
            texts = [key.split("\x1f", 1)[1] for key in missing]
            try:
                fresh = self.embed(texts)
                if len(fresh) != len(texts):
                    raise ValueError("embedding count does not match query count")
            except Exception:
                if not self.cfg.retrieval.fallback_bm25_on_embed_error:
                    raise
                degraded = True
                dim = int(self.index.vectors.shape[1]) if self.index.vectors.ndim == 2 else 0
                fresh = np.zeros((len(texts), dim), dtype=np.float32)
            else:
                if self._embed_cache_size:
                    with self._embed_lock:
                        for key, vector in zip(missing, fresh):
                            self._embed_cache[key] = vector
                            self._embed_cache.move_to_end(key)
                        while len(self._embed_cache) > self._embed_cache_size:
                            self._embed_cache.popitem(last=False)
            for key, vector in zip(missing, fresh):
                vectors[key] = vector
        matrix = np.vstack([vectors[key] for key in keys])
        return matrix, hits, degraded

    def _term_coverage(self, text: str, queries: list[str]) -> float:
        """IDF-weighted share of query terms that occur in ``text``.

        A question whose rare terms (CBAM, ICSID, a fabricated article number)
        never appear is not answerable from this corpus, even when the generic
        words produce a decent cosine.
        """
        lowered = text.lower()
        terms: list[str] = []
        seen: set[str] = set()
        for query in queries:
            for term in query_terms(query, use_stopwords=self.cfg.retrieval.query_stopwords):
                if term not in seen:
                    seen.add(term)
                    terms.append(term)
        if not terms:
            return 0.0
        numerator = denominator = 0.0
        for term in terms:
            weight = self.index.idf.get(term, 1.0)
            denominator += weight
            if term in lowered:
                numerator += weight
        return numerator / denominator if denominator else 0.0

    def _confidence(self, chunks: list[dict[str, Any]], queries: list[str]) -> str:
        """Coarse reliability label for the returned set (diagnostics only)."""
        conf = self.cfg.retrieval.confidence
        if not chunks:
            return "low"
        top = chunks[0]
        best = float(top["score_parts"].get("first_stage") or 0.0)
        second = (
            float(chunks[1]["score_parts"].get("first_stage") or 0.0)
            if len(chunks) > 1
            else 0.0
        )
        gap = best - second
        coverage = self._term_coverage(top.get("text") or "", queries)
        weight = max(0.0, min(1.0, float(conf.vector_weight)))
        score = (1.0 - weight) * coverage + weight * float(top["score_parts"].get("vector") or 0.0)
        if score >= conf.high and gap >= conf.min_gap:
            return "high"
        if score >= conf.medium:
            return "medium"
        return "low"

    def retrieve(
        self,
        queries: list[str] | str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        fields: dict[str, Any] | None = None,
        use_rerank: bool | None = None,
        per_jurisdiction_k: int | None = None,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in (queries or []) if q and q.strip()]
        if not queries:
            raise ValueError("queries must contain at least one non-empty string")
        if len(queries) > MAX_QUERIES:
            raise ValueError(f"queries must contain at most {MAX_QUERIES} items")
        if self.cfg.retrieval.expand_cross_language:
            queries = [expand_query(query) for query in queries]
        top_k = int(top_k or self.cfg.retrieval.top_k)
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be in [1, 100]")

        filters = _map_fields(filters or {}, fields)
        if per_jurisdiction_k is None and filters.get("per_jurisdiction_k") is not None:
            try:
                per_jurisdiction_k = int(filters["per_jurisdiction_k"])
            except (TypeError, ValueError) as exc:
                raise ValueError("per_jurisdiction_k must be an integer") from exc
        if per_jurisdiction_k is not None and not 0 <= per_jurisdiction_k <= 100:
            raise ValueError("per_jurisdiction_k must be in [0, 100]")
        rerank_on = self.rerank_enabled if use_rerank is None else bool(use_rerank)
        rerank = self._rerank if rerank_on else None
        cache_key = self._cache_key(
            queries, top_k, filters, fields or {}, rerank_on, per_jurisdiction_k
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            cached.setdefault("diagnostics", {})["cache_hit"] = True
            return cached
        requested_j = [j.upper() for j in _as_list(filters.get("jurisdictions"))]
        requested_a = [a.lower() for a in _as_list(filters.get("applies_to"))]
        as_of = filters.get("as_of")
        t_filter = time.perf_counter()
        rows, funnel, zero_reason, uncovered = self._select(filters)
        timings = {"filter": round((time.perf_counter() - t_filter) * 1000, 2)}

        corpus = self.index.corpus_block()
        temporal = self._temporal_block(as_of)
        diagnostics: dict[str, Any] = {
            "funnel": funnel,
            "zero_reason": zero_reason,
            "requested_uncovered": uncovered,
        }
        response: dict[str, Any] = {
            "schema_version": "2",
            "chunks": [],
            "corpus": corpus,
            "temporal": temporal,
            "diagnostics": diagnostics,
            "index": {
                "version": self.index.version,
                "built_at": self.index.built_at,
                "chunk_count": self.index.total_chunks,
                "embedding_model": self.index.embedding_model,
                "chunking_hash": self.index.chunking_hash,
            },
            "field_matches": self._field_matches(requested_j, requested_a, rows, as_of),
        }
        if zero_reason:
            diagnostics["confidence"] = "low"
            timings["total"] = round((time.perf_counter() - t_start) * 1000, 2)
            diagnostics["timings_ms"] = timings
            self._cache_put(cache_key, response)
            return response

        candidate_ids = [r["chunk_id"] for r in rows]
        candidate_set = set(candidate_ids)
        candidate_indices = [self.index.chunk_pos[cid] for cid in candidate_ids]
        t_embed = time.perf_counter()
        query_vectors, embed_hits, embed_degraded = self._embed_queries(queries)
        timings["embed"] = round((time.perf_counter() - t_embed) * 1000, 2)
        diagnostics["embedding_cache_hits"] = embed_hits
        if embed_degraded:
            diagnostics["degraded"] = ["embedding"]

        merged: dict[str, dict[str, Any]] = {}
        cfg = self.cfg.retrieval
        rerank_cfg = self.cfg.rerank
        rerank_meta: dict[str, Any] = {
            "enabled": rerank is not None,
            "applied": False,
            "model": rerank_cfg.model if rerank is not None else None,
            "candidates": 0,
            "blend_weight": rerank_cfg.weight if rerank is not None else None,
            "fusion": rerank_cfg.fusion if rerank is not None else None,
            "latency_ms": 0.0,
            "error": None,
        }

        def absorb(
            qi: int,
            cid: str,
            score: float,
            first_stage: float,
            rerank_score: float | None,
            b_raw: float,
            v_raw: float,
            b_norm: float,
            v_norm: float,
        ) -> None:
            current = merged.get(cid)
            if current is None:
                merged[cid] = {
                    "score": score,
                    "first_stage": first_stage,
                    "rerank": rerank_score,
                    "query_indices": {qi},
                    "bm25": b_raw,
                    "vector": v_raw,
                    "bm25_norm": b_norm,
                    "vector_norm": v_norm,
                }
                return
            current["query_indices"].add(qi)
            if score > current["score"]:
                current.update(
                    score=score,
                    first_stage=first_stage,
                    rerank=rerank_score,
                    bm25=b_raw,
                    vector=v_raw,
                    bm25_norm=b_norm,
                    vector_norm=v_norm,
                )

        quota = _quota_for(requested_j, top_k, per_jurisdiction_k)
        # A per-jurisdiction quota can only promote what the first stage
        # offers. With a bare top_k pool every candidate comes from the
        # dominant jurisdiction, so widen the pool whenever slots are reserved.
        pool_depth = top_k
        if quota:
            pool_depth = min(200, max(top_k * 3, quota * max(1, len(requested_j)) * 5))

        for qi, (query, qvec) in enumerate(zip(queries, query_vectors)):
            terms = query_terms(query, use_stopwords=cfg.query_stopwords)
            t_bm25 = time.perf_counter()
            bm25 = self.index.bm25_scores(candidate_set, terms, cfg.bm25_k1, cfg.bm25_b)
            raw_bm25 = [float(bm25.get(cid, 0.0)) for cid in candidate_ids]
            timings["bm25"] = round(timings.get("bm25", 0.0) + (time.perf_counter() - t_bm25) * 1000, 2)
            t_vec = time.perf_counter()
            raw_vec = (
                self.index.vector_scores(candidate_indices, qvec)
                if not embed_degraded
                else np.zeros((len(candidate_indices),), dtype=np.float32)
            )
            timings["vector"] = round(timings.get("vector", 0.0) + (time.perf_counter() - t_vec) * 1000, 2)
            vec_list = [float(x) for x in raw_vec]
            bm25_norm = _minmax(raw_bm25)
            # A batch whose best BM25 is weak (one low-IDF hit) must not be
            # rescaled to 1.0: that lets a literal coincidence outrank a
            # strong vector match. Scale by magnitude, saturating at 8
            # (bm25_saturation=0 restores plain _minmax for ablation).
            saturation = float(cfg.bm25_saturation or 0.0)
            if bm25_norm and saturation > 0:
                scale = min(1.0, max(raw_bm25) / saturation)
                if scale < 1.0:
                    bm25_norm = [value * scale for value in bm25_norm]
            vec_norm = _minmax(vec_list)
            vector_weight = 0.0 if embed_degraded else cfg.vector_weight
            scored: list[tuple[float, str, float, float, float, float]] = []
            for i, cid in enumerate(candidate_ids):
                score = (1.0 - vector_weight) * bm25_norm[i] + vector_weight * vec_norm[i]
                if cfg.similarity_threshold and score < cfg.similarity_threshold:
                    continue
                scored.append((score, cid, raw_bm25[i], vec_list[i], bm25_norm[i], vec_norm[i]))
            scored.sort(key=lambda item: (-item[0], item[1]))
            if not scored:
                continue

            if rerank is None:
                for score, cid, b_raw, v_raw, b_norm, v_norm in scored[:pool_depth]:
                    absorb(qi, cid, score, score, None, b_raw, v_raw, b_norm, v_norm)
                continue

            # A cross-encoder only sees what the first stage offers, so widen
            # the pool beyond top_k and let the reranker choose the final N.
            depth = max(pool_depth, int(rerank_cfg.candidates or 0), 1)
            pool = scored[:depth]
            texts = [self.index.chunk_by_id[item[1]]["text"] for item in pool]
            rerank_meta["candidates"] += len(pool)
            pairs: list[tuple[int, float]] = []
            try:
                pairs, latency = rerank(query, texts, len(texts))
                rerank_meta["latency_ms"] = max(rerank_meta["latency_ms"], latency)
            except Exception as exc:  # reranking must never fail the request
                rerank_meta["error"] = f"{type(exc).__name__}: {exc}"[:300]

            if pairs:
                rerank_meta["applied"] = True
                pairs = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
                rerank_map = {
                    idx: score for idx, score in pairs if 0 <= idx < len(pool)
                }
                first_norm = _minmax([item[0] for item in pool])
                rerank_values = list(rerank_map.values())
                rerank_norm = _minmax(rerank_values)
                norm_by_idx = {
                    idx: rerank_norm[i] for i, idx in enumerate(rerank_map)
                }
                ranked: list[tuple[float, float | None, float, str, float, float, float, float]] = []
                for idx, (fscore, cid, b_raw, v_raw, b_norm, v_norm) in enumerate(pool):
                    rscore = rerank_map.get(idx)
                    rnorm = norm_by_idx.get(idx, 0.0) if rscore is not None else 0.0
                    if rerank_cfg.fusion == "rerank":
                        final = rnorm
                    elif rerank_cfg.fusion == "weighted":
                        final = rerank_cfg.weight * rnorm + (1.0 - rerank_cfg.weight) * first_norm[idx]
                    else:
                        # Max fusion keeps a strong lexical (exact legal term)
                        # match visible even when the cross-encoder dislikes it.
                        final = max(first_norm[idx], rerank_cfg.weight * rnorm)
                    ranked.append((final, rscore, fscore, cid, b_raw, v_raw, b_norm, v_norm))
                ranked.sort(
                    key=lambda item: (
                        -item[0],
                        -(item[1] if item[1] is not None else -1.0),
                        item[3],
                    )
                )
                for final, rscore, fscore, cid, b_raw, v_raw, b_norm, v_norm in ranked[:top_k]:
                    absorb(qi, cid, final, fscore, rscore, b_raw, v_raw, b_norm, v_norm)
            else:
                for score, cid, b_raw, v_raw, b_norm, v_norm in pool[:top_k]:
                    absorb(qi, cid, score, score, None, b_raw, v_raw, b_norm, v_norm)

        diagnostics["rerank"] = rerank_meta
        if not merged:
            diagnostics["zero_reason"] = "no_match"
            diagnostics["confidence"] = "low"
            timings["total"] = round((time.perf_counter() - t_start) * 1000, 2)
            diagnostics["timings_ms"] = timings
            self._cache_put(cache_key, response)
            return response

        ordered = sorted(
            merged.items(), key=lambda item: (-item[1]["score"], -item[1]["first_stage"], item[0])
        )
        # Multi-query fusion: a chunk that answers several sub-queries is more
        # likely to be the intended article than one that scores highly on a
        # single phrasing. Single-query requests are unaffected.
        if len(queries) > 1 and cfg.multi_query_bonus:
            ordered = sorted(
                ordered,
                key=lambda item: (
                    -(
                        item[1]["score"]
                        + cfg.multi_query_bonus
                        * min(len(item[1]["query_indices"]) - 1, 2)
                    ),
                    -item[1]["first_stage"],
                    item[0],
                ),
            )
        t_rank = time.perf_counter()
        ordered = _apply_jurisdiction_quota(
            ordered,
            top_k,
            requested_j,
            quota,
            self.index,
            float(cfg.quota_min_ratio or 0.0),
        )
        diagnostics["per_jurisdiction_k"] = quota or None
        if quota:
            slots: dict[str, int] = {}
            for cid, _hit in ordered:
                jurisdiction = self.index.chunk_by_id[cid]["jurisdiction"]
                slots[jurisdiction] = slots.get(jurisdiction, 0) + 1
            diagnostics["quota_slots"] = slots
        funnel["after_retrieval"] = len(ordered)
        chunks: list[dict[str, Any]] = []
        for rank, (cid, hit) in enumerate(ordered, 1):
            row = self.index.chunk_by_id[cid]
            doc = self.index.docs.get(row["doc_id"], {})
            q_indices = sorted(hit["query_indices"])
            matched_terms: list[dict[str, Any]] = []
            seen_terms: set[str] = set()
            for qi in q_indices:
                for match in extract_matches(queries[qi], row["text"]):
                    term = str(match["term"])
                    if term in seen_terms:
                        continue
                    seen_terms.add(term)
                    matched_terms.append({**match, "query_index": qi})
            matched_terms.sort(key=lambda m: (int(m["start"]), -int(m["end"])))
            matched_by = []
            if hit["vector"] != 0.0:
                matched_by.append("vec")
            if hit["bm25"] != 0.0:
                matched_by.append("kw")
            superseded_by = doc.get("superseded_by") or ""
            chunks.append(
                {
                    "chunk_id": cid,
                    "doc_id": row["doc_id"],
                    "source_doc": doc.get("source_doc", row["doc_id"]),
                    "text": row["text"],
                    "score": round(float(hit["score"]), 6),
                    "score_parts": {
                        "vector": round(float(hit["vector"]), 6),
                        "keyword": round(float(hit["bm25"]), 6),
                        "vector_norm": round(float(hit["vector_norm"]), 6),
                        "keyword_norm": round(float(hit["bm25_norm"]), 6),
                        # first_stage is the BM25+cosine hybrid; hybrid is
                        # kept as an alias for older consumers.
                        "first_stage": round(float(hit["first_stage"]), 6),
                        "hybrid": round(float(hit["first_stage"]), 6),
                        "rerank": None if hit["rerank"] is None else round(float(hit["rerank"]), 6),
                        "fusion_bonus": round(
                            float(cfg.multi_query_bonus) * min(len(q_indices) - 1, 2), 6
                        )
                        if len(queries) > 1
                        else 0.0,
                    },
                    "articles": [row["article_label"]],
                    "article_ids": [row["article_id"]],
                    "article_offsets": [
                        {"article_id": row["article_id"], "start": 0, "end": len(row["text"])}
                    ],
                    "effective_from": row["effective_from"],
                    "effective_to": row["effective_to"],
                    "jurisdictions": [row["jurisdiction"]],
                    "applies_to": row["applies_to"],
                    "superseded_by": superseded_by or None,
                    "matched_by": matched_by,
                    "matched_queries": q_indices,
                    "matched_terms": matched_terms,
                    "matched_fields": {
                        "doc_id": row["doc_id"],
                        "article_id": row["article_id"],
                        "jurisdiction": row["jurisdiction"],
                        "applies_to": row["applies_to"],
                        "effective_from": row["effective_from"],
                        "effective_to": row["effective_to"],
                        "requested_jurisdictions": requested_j,
                        "requested_applies_to": requested_a,
                        "as_of": as_of,
                    },
                    "rank": rank,
                }
            )
        response["chunks"] = chunks
        superseded = {
            chunk["doc_id"]: chunk["superseded_by"]
            for chunk in chunks
            if chunk.get("superseded_by")
        }
        if superseded:
            diagnostics["warnings"] = [
                f"{doc_id} has been superseded by {replacement}"
                for doc_id, replacement in sorted(superseded.items())
            ]
        diagnostics["confidence"] = self._confidence(chunks, queries)
        timings["rank"] = round((time.perf_counter() - t_rank) * 1000, 2)
        timings["total"] = round((time.perf_counter() - t_start) * 1000, 2)
        diagnostics["timings_ms"] = timings
        self._cache_put(cache_key, response)
        return response
