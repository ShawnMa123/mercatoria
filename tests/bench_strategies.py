"""Benchmark chunking and rerank strategies over the 1000-question set.

Run in the test container against the running service:
  python tests/bench_strategies.py --base http://host.docker.internal:8080 \
      --out /eval/strategies.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

CHUNKING = [
    ("article-2048-ov120", {"unit": "article", "max_chars": 2048, "overlap": 120}),
    ("article-1024-ov120", {"unit": "article", "max_chars": 1024, "overlap": 120}),
    ("article-4096-ov120", {"unit": "article", "max_chars": 4096, "overlap": 120}),
    ("article-2048-ov0", {"unit": "article", "max_chars": 2048, "overlap": 0}),
    ("section-2048-ov120", {"unit": "section", "max_chars": 2048, "overlap": 120}),
]

RERANK = [
    ("off", {"enabled": False}, False),
    ("max-w0.5-c10", {"enabled": True, "fusion": "max", "weight": 0.5, "candidates": 10}, True),
    ("max-w0.8-c10", {"enabled": True, "fusion": "max", "weight": 0.8, "candidates": 10}, True),
    ("max-w1.0-c10", {"enabled": True, "fusion": "max", "weight": 1.0, "candidates": 10}, True),
    ("weighted-w0.5-c10", {"enabled": True, "fusion": "weighted", "weight": 0.5, "candidates": 10}, True),
    ("rerank-pure-c10", {"enabled": True, "fusion": "rerank", "weight": 1.0, "candidates": 10}, True),
    ("max-w0.8-c20", {"enabled": True, "fusion": "max", "weight": 0.8, "candidates": 20}, True),
]


_LOCAL = threading.local()


def _connection(url: str) -> http.client.HTTPConnection:
    parts = urlsplit(url)
    key = (parts.scheme, parts.hostname, parts.port)
    if getattr(_LOCAL, "key", None) != key:
        old = getattr(_LOCAL, "conn", None)
        if old is not None:
            old.close()
        cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        _LOCAL.conn = cls(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), timeout=300)
        _LOCAL.key = key
    return _LOCAL.conn


def _request(method: str, url: str, body: dict | None = None, timeout: float = 300) -> dict:
    parts = urlsplit(url)
    path = parts.path + (("?" + parts.query) if parts.query else "")
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
    last: Exception | None = None
    for _attempt in range(2):
        conn = _connection(url)
        try:
            conn.request(method, path, body=data, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {raw.decode('utf-8', 'replace')[:300]}")
            return json.loads(raw.decode("utf-8"))
        except (http.client.HTTPException, OSError) as exc:
            last = exc
            try:
                conn.close()
            finally:
                _LOCAL.conn = None
    raise RuntimeError(f"request failed: {last}")

def _wait(base: str, job_id: str, timeout: float = 1800) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = _request("GET", f"{base}/v1/jobs/{job_id}", timeout=30)
        if job.get("status") != "running":
            return job
        time.sleep(1.0)
    raise TimeoutError(job_id)


def _hit(case: dict, chunks: list[dict]) -> tuple[int, bool, bool, bool]:
    docs = {str(x) for x in case.get("expected_doc_ids", [])}
    articles = {str(x) for x in case.get("expected_article_ids", [])}
    keywords = [str(k).lower() for k in case.get("expected_keywords", [])]
    best_rank = 0
    doc_hit = keyword_hit = article_hit = False
    for rank, chunk in enumerate(chunks, 1):
        chunk_docs = {str(chunk.get("doc_id"))}
        chunk_articles = {str(x) for x in (chunk.get("article_ids") or [])}
        text = str(chunk.get("text", "")).lower()
        d = bool(docs & chunk_docs)
        a = bool(articles & chunk_articles)
        k = any(word in text for word in keywords)
        doc_hit |= d
        article_hit |= a
        keyword_hit |= k
        if case.get("synthetic"):
            matched = a and (k or not keywords)
        elif articles:
            matched = a and (k or not keywords)
        else:
            matched = d and k
        if matched and not best_rank:
            best_rank = rank
    return best_rank, doc_hit, keyword_hit, article_hit


def _query(base: str, case: dict, top_k: int, rerank: bool, timeout: float) -> dict:
    started = time.perf_counter()
    body = {"queries": [case["question"]], "top_k": top_k, "rerank": rerank}
    payload = _request("POST", f"{base}/v1/retrieve", body, timeout=timeout)
    return {
        "id": case["id"],
        "synthetic": bool(case.get("synthetic")),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "rank": 0,
        "doc_hit": False,
        "keyword_hit": False,
        "article_hit": False,
        "chunks": payload.get("chunks") or [],
        "rerank": payload.get("diagnostics", {}).get("rerank", {}),
    }


def run_queries(
    base: str, cases: list[dict], top_k: int, rerank: bool, concurrency: int
) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda c: _query(base, c, top_k, rerank, 120), cases))
    for result, case in zip(results, cases):
        rank, doc_hit, keyword_hit, article_hit = _hit(case, result["chunks"])
        result.update(
            rank=rank, doc_hit=doc_hit, keyword_hit=keyword_hit, article_hit=article_hit
        )
        result["hit"] = rank > 0
        del result["chunks"]
    return results


def summarize(rows: list[dict]) -> dict:
    def bucket(items: list[dict]) -> dict:
        if not items:
            return {"n": 0, "hit5": 0.0, "article_hit5": 0.0, "mrr": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        n = len(items)
        return {
            "n": len(items),
            "hit5": round(sum(1 for r in items if r["hit"]) / n, 4),
            "article_hit5": round(sum(1 for r in items if r["article_hit"]) / n, 4),
            "mrr": round(
                sum((1.0 / r["rank"] if r["rank"] else 0.0) for r in items) / n, 4
            ),
            "p50_ms": round(statistics.median([r["latency_ms"] for r in items]), 1),
            "p95_ms": round(
                sorted(r["latency_ms"] for r in items)[max(0, int(len(items) * 0.95) - 1)],
                1,
            ),
        }

    return {
        "all": bucket(rows),
        "real": bucket([r for r in rows if not r["synthetic"]]),
        "synthetic": bucket([r for r in rows if r["synthetic"]]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--questions", default="tests/eval/questions_1000.json")
    ap.add_argument("--out", default="/eval/strategies.json")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-chunking", action="store_true")
    ap.add_argument("--skip-rerank", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    cases = data["questions"]
    real = [q for q in cases if not q.get("synthetic")]
    synthetic = [q for q in cases if q.get("synthetic")]
    subset = real + synthetic[::5][: max(0, 200 - len(real))]

    report: dict = {"chunking": [], "rerank": []}

    for name, chunking in ([] if args.skip_chunking else CHUNKING):
        print(f"[chunk] {name}", flush=True)
        _request("PUT", f"{args.base}/v1/config", {"chunking": chunking})
        started = time.perf_counter()
        job = _request("POST", f"{args.base}/v1/admin/rebuild", {"background": True})
        done = _wait(args.base, job["job_id"])
        rebuild_s = time.perf_counter() - started
        if done.get("status") != "done":
            print(f"  rebuild failed: {done.get('error')}", flush=True)
            continue
        result = done.get("result") or {}
        rows = run_queries(args.base, cases, args.top_k, False, args.concurrency)
        entry = {
            "name": name,
            "chunking": chunking,
            "rebuild_seconds": round(rebuild_s, 1),
            "version": result.get("version"),
            "chunks": result.get("chunks"),
            "summary": summarize(rows),
        }
        report["chunking"].append(entry)
        print(json.dumps(entry["summary"], ensure_ascii=False), flush=True)

    if not args.skip_chunking:
        baseline = {"unit": "article", "max_chars": 2048, "overlap": 120}
        _request("PUT", f"{args.base}/v1/config", {"chunking": baseline})
        job = _request("POST", f"{args.base}/v1/admin/rebuild", {"background": True})
        _wait(args.base, job["job_id"])

    for name, rerank_cfg, use_rerank in ([] if args.skip_rerank else RERANK):
        print(f"[rerank] {name}", flush=True)
        _request("PUT", f"{args.base}/v1/config", {"rerank": rerank_cfg})
        rows = run_queries(args.base, subset, args.top_k, use_rerank, args.concurrency)
        entry = {"name": name, "config": rerank_cfg, "summary": summarize(rows)}
        report["rerank"].append(entry)
        print(json.dumps(entry["summary"], ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
