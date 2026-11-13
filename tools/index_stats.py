"""Index memory/startup statistics (P2-6).

Reports load time, resident growth attributed to the in-memory index, and how
``postings`` scales, then extrapolates to 5x/10x the corpus. Read-only; the
conclusion (dict vs CSR) is written to stdout for the report.

Usage:
    python tools/index_stats.py [--index ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mercatoria.config import load_config  # noqa: E402
from mercatoria.store import open_current, open_index  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.example.json")
    ap.add_argument("--index", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    path = args.index or cfg.resolved_index_dir()
    tracemalloc.start()
    started = time.perf_counter()
    index = open_index(path) if str(path).endswith(".db") else open_current(path)
    load_s = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    postings = sum(len(items) for items in index.postings.values())
    bytes_per_posting = current / max(1, postings)
    report = {
        "index": str(index.path),
        "chunks": index.total_chunks,
        "postings": postings,
        "distinct_terms": len(index.postings),
        "vectors_mb": round(index.vectors.nbytes / 1024 / 1024, 1),
        "traced_index_mb": round(current / 1024 / 1024, 1),
        "traced_peak_mb": round(peak / 1024 / 1024, 1),
        "load_seconds": round(load_s, 2),
        "bytes_per_posting": round(bytes_per_posting, 1),
        "extrapolated_5x": {
            "chunks": index.total_chunks * 5,
            "postings": postings * 5,
            "vectors_mb": round(index.vectors.nbytes * 5 / 1024 / 1024, 1),
            "index_mb": round(current * 5 / 1024 / 1024, 1),
            "load_seconds": round(load_s * 5, 1),
        },
        "extrapolated_10x": {
            "chunks": index.total_chunks * 10,
            "postings": postings * 10,
            "vectors_mb": round(index.vectors.nbytes * 10 / 1024 / 1024, 1),
            "index_mb": round(current * 10 / 1024 / 1024, 1),
            "load_seconds": round(load_s * 10, 1),
        },
        "recommendation": (
            "≤50k chunks the dict postings stay well under 1 GB and load in seconds; "
            "keep the current structure. Past ~100k chunks the vector matrix dominates "
            "(1024 dims * 4 B/chunk); switch postings to a CSR (term -> (indptr, ids, tf)) "
            "and consider float16 or an ANN index, not an ORM rewrite."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
