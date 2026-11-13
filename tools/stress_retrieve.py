"""Concurrency stress for the retrieval surface (P2-5).

Runs N worker threads issuing random dev questions for ``--seconds`` and
reports throughput, p50/p99 latency, error rate, and RSS growth. Exercises the
shared Retriever (response cache + query-vector cache + index).

Usage:
    python tools/stress_retrieve.py [--threads 8] [--seconds 60] [--endpoint http://127.0.0.1:8080]
    python tools/stress_retrieve.py --local --threads 32
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import tracemalloc
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CASES = ROOT / "corpus" / "eval" / "test_cases.json"


def _p(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--local", action="store_true", help="call the package, not HTTP")
    ap.add_argument("--config", default="config.example.json")
    args = ap.parse_args()

    prompts = [case["q_zh"] for case in json.loads(CASES.read_text(encoding="utf-8"))]

    if args.local:
        from mercatoria.config import load_config
        from mercatoria.retrieve import Retriever
        from mercatoria.store import open_current

        cfg = load_config(args.config)
        retriever = Retriever(open_current(cfg.resolved_index_dir()), cfg)

        def call(prompt: str) -> None:
            retriever.retrieve([prompt], top_k=5)
    else:
        def call(prompt: str) -> None:
            body = json.dumps({"queries": [prompt], "top_k": 5}).encode("utf-8")
            request = urllib.request.Request(
                args.endpoint.rstrip("/") + "/v1/retrieve",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()

    lock = threading.Lock()
    latencies: list[float] = []
    errors: list[str] = []
    stop = time.time() + args.seconds
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()

    def worker(seed: int) -> None:
        rng = random.Random(seed)
        while time.time() < stop:
            prompt = rng.choice(prompts)
            started = time.perf_counter()
            try:
                call(prompt)
                elapsed = (time.perf_counter() - started) * 1000
                with lock:
                    latencies.append(elapsed)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        list(pool.map(worker, range(args.threads)))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(latencies)
    print(
        json.dumps(
            {
                "threads": args.threads,
                "seconds": args.seconds,
                "requests": len(latencies),
                "rps": round(len(latencies) / args.seconds, 2),
                "p50_ms": _p(ordered, 0.5),
                "p99_ms": _p(ordered, 0.99),
                "errors": len(errors),
                "error_samples": errors[:3],
                "rss_growth_mb": round((current - baseline) / 1024 / 1024, 2),
                "peak_mb": round(peak / 1024 / 1024, 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
