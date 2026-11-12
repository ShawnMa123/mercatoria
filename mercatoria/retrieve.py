"""Filter-first hybrid retrieval over the local SQLite index."""

from __future__ import annotations

import copy
import datetime as dt
import json
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from .config import AppConfig
from .embed import EmbeddingClient
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
    return out


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
        if rerank_fn is not None:
            self.rerank = rerank_fn
        elif cfg.rerank.enabled:
            self.rerank = RerankClient(cfg.rerank).rerank
        else:
            self.rerank = None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = 256

    def _cache_key(
        self,
        queries: list[str],
        top_k: int,
        filters: dict[str, Any],
        fields: dict[str, Any],
    ) -> str:
        payload = {
            "queries": queries,
            "top_k": top_k,
            "filters": filters,
            "fields": fields,
            "vector_weight": self.cfg.retrieval.vector_weight,
            "threshold": self.cfg.retrieval.similarity_threshold,
            "rerank": self.rerank is not None,
            "candidates": self.cfg.rerank.candidates,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        cached = self._cache.get(key)
        if cached is None:
            return None
        self._cache.move_to_end(key)
        return copy.deepcopy(cached)

    def _cache_put(self, key: str, response: dict[str, Any]) -> None:
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

        funnel = {
            "total": total,
            "after_jurisdiction": len(after_j),
            "after_applies_to": len(after_a),
            "after_as_of": len(after_t),
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

        uncovered = {
            "jurisdictions": uncovered_j,
            "applies_to": uncovered_a,
        }
        return after_t, funnel, zero_reason, uncovered

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

    def retrieve(
        self,
        queries: list[str] | str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(queries, str):
            queries = [queries]
        queries = [q.strip() for q in (queries or []) if q and q.strip()]
        if not queries:
            raise ValueError("queries must contain at least one non-empty string")
        if len(queries) > MAX_QUERIES:
            raise ValueError(f"queries must contain at most {MAX_QUERIES} items")
        top_k = int(top_k or self.cfg.retrieval.top_k)
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be in [1, 100]")

        filters = _map_fields(filters or {}, fields)
        cache_key = self._cache_key(queries, top_k, filters, fields or {})
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        requested_j = [j.upper() for j in _as_list(filters.get("jurisdictions"))]
        requested_a = [a.lower() for a in _as_list(filters.get("applies_to"))]
        as_of = filters.get("as_of")
        rows, funnel, zero_reason, uncovered = self._select(filters)

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
            self._cache_put(cache_key, response)
            return response

        candidate_ids = [r["chunk_id"] for r in rows]
        candidate_set = set(candidate_ids)
        candidate_indices = [self.index.chunk_pos[cid] for cid in candidate_ids]
        query_vectors = self.embed(queries)
        if len(query_vectors) != len(queries):
            raise ValueError("embedding count does not match query count")

        merged: dict[str, dict[str, Any]] = {}
        cfg = self.cfg.retrieval
        rerank_cfg = self.cfg.rerank
        rerank_meta: dict[str, Any] = {
            "enabled": self.rerank is not None,
            "applied": False,
            "model": rerank_cfg.model if self.rerank is not None else None,
            "candidates": 0,
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

        for qi, (query, qvec) in enumerate(zip(queries, query_vectors)):
            terms = query_terms(query)
            bm25 = self.index.bm25_scores(candidate_set, terms, cfg.bm25_k1, cfg.bm25_b)
            raw_bm25 = [float(bm25.get(cid, 0.0)) for cid in candidate_ids]
            raw_vec = self.index.vector_scores(candidate_indices, qvec)
            vec_list = [float(x) for x in raw_vec]
            bm25_norm = _minmax(raw_bm25)
            vec_norm = _minmax(vec_list)
            scored: list[tuple[float, str, float, float, float, float]] = []
            for i, cid in enumerate(candidate_ids):
                score = (1.0 - cfg.vector_weight) * bm25_norm[i] + cfg.vector_weight * vec_norm[i]
                if cfg.similarity_threshold and score < cfg.similarity_threshold:
                    continue
                scored.append((score, cid, raw_bm25[i], vec_list[i], bm25_norm[i], vec_norm[i]))
            scored.sort(key=lambda item: (-item[0], item[1]))
            if not scored:
                continue

            if self.rerank is None:
                for score, cid, b_raw, v_raw, b_norm, v_norm in scored[:top_k]:
                    absorb(qi, cid, score, score, None, b_raw, v_raw, b_norm, v_norm)
                continue

            # A cross-encoder only sees what the first stage offers, so widen
            # the pool beyond top_k and let the reranker choose the final N.
            depth = max(top_k, int(rerank_cfg.candidates or 0), 1)
            pool = scored[:depth]
            texts = [self.index.chunk_by_id[item[1]]["text"] for item in pool]
            rerank_meta["candidates"] += len(pool)
            pairs: list[tuple[int, float]] = []
            try:
                pairs, latency = self.rerank(query, texts, len(texts))
                rerank_meta["latency_ms"] = max(rerank_meta["latency_ms"], latency)
            except Exception as exc:  # reranking must never fail the request
                rerank_meta["error"] = f"{type(exc).__name__}: {exc}"[:300]

            if pairs:
                rerank_meta["applied"] = True
                pairs = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
                pairs = [(idx, score) for idx, score in pairs if 0 <= idx < len(pool)]
                for idx, rscore in pairs[:top_k]:
                    score, cid, b_raw, v_raw, b_norm, v_norm = pool[idx]
                    absorb(qi, cid, rscore, score, rscore, b_raw, v_raw, b_norm, v_norm)
                # A reranker that returns fewer rows than top_k must still
                # yield top_k chunks: fill from first-stage order.
                if len(pairs) < top_k:
                    seen = {pool[idx][1] for idx, _ in pairs}
                    for score, cid, b_raw, v_raw, b_norm, v_norm in pool:
                        if cid in seen:
                            continue
                        absorb(qi, cid, score, score, None, b_raw, v_raw, b_norm, v_norm)
                        seen.add(cid)
                        if len(seen) >= top_k:
                            break
            else:
                for score, cid, b_raw, v_raw, b_norm, v_norm in pool[:top_k]:
                    absorb(qi, cid, score, score, None, b_raw, v_raw, b_norm, v_norm)

        diagnostics["rerank"] = rerank_meta
        if not merged:
            diagnostics["zero_reason"] = "no_match"
            self._cache_put(cache_key, response)
            return response

        ordered = sorted(
            merged.items(), key=lambda item: (-item[1]["score"], -item[1]["first_stage"], item[0])
        )[:top_k]
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
        self._cache_put(cache_key, response)
        return response
