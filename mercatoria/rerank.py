"""Cross-encoder reranking over an external HTTP API.

Two request shapes are supported:

* ``dashscope`` -- the native contract of ``gte-rerank-v2`` on Alibaba Cloud
  Model Studio: ``{"model", "input": {"query", "documents"}, "parameters"}``
  answered with ``{"output": {"results": [...]}}``.
* ``cohere`` -- the ``POST /v1/rerank`` contract shared by Cohere, Jina,
  Voyage, TEI and magpie: ``{"model", "query", "documents", "top_n"}``
  answered with ``{"results": [...]}``.

Reranking is best-effort by design. A network failure must not turn a
successful retrieval into an error, so the caller falls back to first-stage
order and reports the failure in ``diagnostics.rerank``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import RerankConfig


class RerankError(RuntimeError):
    pass


class RerankClient:
    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg

    @property
    def url(self) -> str:
        base = self.cfg.base.rstrip("/")
        path = self.cfg.path.strip()
        if not path:
            return base
        return base + "/" + path.lstrip("/")

    def _payload(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        if self.cfg.style == "dashscope":
            return {
                "model": self.cfg.model,
                "input": {"query": query, "documents": documents},
                "parameters": {"return_documents": False, "top_n": top_n},
            }
        return {
            "model": self.cfg.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }

    @staticmethod
    def _parse(payload: dict[str, Any]) -> list[tuple[int, float]]:
        results = payload.get("results")
        if results is None and isinstance(payload.get("output"), dict):
            results = payload["output"].get("results")
        if results is None:
            results = payload.get("data")
        if not isinstance(results, list):
            raise RerankError("rerank response has no results list")

        parsed: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            raw = item.get("relevance_score", item.get("score"))
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            parsed.append((index, score))
        parsed.sort(key=lambda pair: (-pair[1], pair[0]))
        return parsed

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            raw = resp.read()
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise RerankError("rerank response is not JSON") from exc
        if not isinstance(payload, dict):
            raise RerankError("rerank response is not a JSON object")
        return payload

    def rerank(
        self, query: str, documents: list[str], top_n: int = 0
    ) -> tuple[list[tuple[int, float]], float]:
        """Return ``([(document_index, score)], latency_ms)`` sorted by score."""
        if not documents:
            return [], 0.0

        wanted = int(top_n or self.cfg.top_n or 0)
        if wanted <= 0 or wanted > len(documents):
            wanted = len(documents)
        body = self._payload(query, documents, wanted)
        attempts = max(1, int(self.cfg.retries) + 1)
        last: Exception | None = None
        started = time.perf_counter()

        for attempt in range(attempts):
            try:
                payload = self._post(body)
                results = [
                    (index, score)
                    for index, score in self._parse(payload)
                    if 0 <= index < len(documents)
                ]
                if self.cfg.min_score > 0:
                    results = [pair for pair in results if pair[1] >= self.cfg.min_score]
                return results, (time.perf_counter() - started) * 1000.0
            except urllib.error.HTTPError as exc:
                detail = exc.read(300).decode("utf-8", "replace")
                last = RerankError(f"HTTP {exc.code}: {detail}")
                # 4xx other than 429 will not succeed on retry.
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
            except RerankError as exc:
                last = exc
                break
            if attempt + 1 < attempts:
                time.sleep(0.2 * (attempt + 1))

        raise RerankError(f"rerank request failed: {last}") from last
