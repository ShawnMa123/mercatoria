"""Evaluate the local SQLite/NumPy retriever against corpus/eval/test_cases.json.

This is the local counterpart of eval_retrieval.py. It calls the package
directly (no HTTP) and reuses the same strict/loose keyword scoring so the
numbers are comparable to the RAGFlow baseline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))

from eval_retrieval import score_case  # noqa: E402

from mercatoria.config import load_config  # noqa: E402
from mercatoria.retrieve import Retriever  # noqa: E402
from mercatoria.store import open_current, open_index  # noqa: E402

CASES_FILE = TOOLS.parent / "corpus" / "eval" / "test_cases.json"
OUT_DIR = CASES_FILE.parent


def _pick(cases: list[dict], limit: int) -> list[dict]:
    if not limit:
        return cases
    by_cat: dict[str, list[dict]] = {}
    for case in cases:
        by_cat.setdefault(case["category"], []).append(case)
    per = max(1, limit // max(1, len(by_cat)))
    picked: list[dict] = []
    for items in by_cat.values():
        picked.extend(items[:per])
    return picked[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.example.json")
    ap.add_argument("--index", default=None, help="index .db path; default current")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", choices=["zh", "en", "both"], default="zh")
    ap.add_argument("--top-k", type=int, default=5, help="scoring cutoff for Hit@k")
    ap.add_argument("--retrieval-k", type=int, default=10, help="chunks fetched (>= top-k for Hit@10)")
    ap.add_argument("--vec-weight", type=float, default=None)
    ap.add_argument("--sim-threshold", type=float, default=None)
    ap.add_argument("--bm25-k1", type=float, default=None)
    ap.add_argument("--bm25-b", type=float, default=None)
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.vec_weight is not None:
        cfg.retrieval.vector_weight = args.vec_weight
    if args.sim_threshold is not None:
        cfg.retrieval.similarity_threshold = args.sim_threshold
    if args.bm25_k1 is not None:
        cfg.retrieval.bm25_k1 = args.bm25_k1
    if args.bm25_b is not None:
        cfg.retrieval.bm25_b = args.bm25_b
    index = open_index(args.index) if args.index else open_current(cfg.resolved_index_dir())
    retriever = Retriever(index, cfg)

    src = Path(args.cases) if args.cases else CASES_FILE
    cases = json.loads(src.read_text(encoding="utf-8"))
    cases = [c for c in cases if c.get("expect") != "refuse"]
    cases = _pick(cases, args.limit)

    rows = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        questions = []
        if args.lang in {"zh", "both"}:
            questions.append(("zh", case["q_zh"]))
        if args.lang in {"en", "both"}:
            questions.append(("en", case["q_en"]))
        for lang, question in questions:
            started = time.time()
            try:
                result = retriever.retrieve([question], top_k=max(args.top_k, args.retrieval_k))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] ERR {exc}", file=sys.stderr)
                continue
            elapsed = round(time.time() - started, 3)
            chunks = [
                {"content": c["text"], "document_keyword": c["source_doc"]}
                for c in result["chunks"]
            ]
            sc = score_case(case, chunks)
            rows.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "lang": lang,
                    "question": question,
                    "elapsed": elapsed,
                    "zero_reason": result["diagnostics"].get("zero_reason"),
                    **sc,
                }
            )
        if i % 25 == 0:
            print(f"  {i}/{len(cases)}  已用 {time.time() - t0:.0f}s", flush=True)

    if not rows:
        print("无结果", file=sys.stderr)
        return 1

    n = len(rows)
    summary = {
        "tag": args.tag,
        "lang": args.lang,
        "top_k": args.top_k,
        "retrieval_k": max(args.top_k, args.retrieval_k),
        "vector_weight": cfg.retrieval.vector_weight,
        "similarity_threshold": cfg.retrieval.similarity_threshold,
        "bm25_k1": cfg.retrieval.bm25_k1,
        "bm25_b": cfg.retrieval.bm25_b,
        "n": n,
        "hit5": round(sum(1 for r in rows if r["hit5"]) / n, 4),
        "hit10": round(sum(1 for r in rows if r["hit10"]) / n, 4),
        "hit5_loose": round(sum(1 for r in rows if r["hit5_loose"]) / n, 4),
        "mrr": round(statistics.mean(r["rr"] for r in rows), 4),
        "has_any": round(sum(1 for r in rows if r["has_any"]) / n, 4),
        "avg_chunks": round(statistics.mean(r["n_chunks"] for r in rows), 2),
    }
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)
    summary["by_category"] = {
        cat: {
            "n": len(items),
            "hit5": round(sum(1 for r in items if r["hit5"]) / len(items), 4),
        }
        for cat, items in by_cat.items()
    }

    ts = datetime.now().strftime("%H%M%S")
    out = OUT_DIR / f"local_{args.tag}_{ts}.json"
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print("\n===== 本地检索评测 =====")
    print(f"用例数             : {n}")
    print(f"Hit@5 (strict)     : {summary['hit5']:.4f}")
    print(f"Hit@10 (strict)    : {summary['hit10']:.4f}")
    print(f"Hit@5 (loose)      : {summary['hit5_loose']:.4f}")
    print(f"MRR (strict)       : {summary['mrr']:.4f}")
    print(f"有返回比例         : {summary['has_any']:.4f}")
    print(f"平均 chunk 数      : {summary['avg_chunks']}")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
