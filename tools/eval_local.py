"""Evaluate the local SQLite/NumPy retriever against corpus/eval/test_cases.json.

This is the local counterpart of eval_retrieval.py. It calls the package
directly (no HTTP) and reuses the same strict/loose keyword scoring so the
numbers are comparable to the RAGFlow baseline.

Scoring口径 (report which one you used!):
  default          v2 rule: a chunk is an article head when its article_id is a
                   real article (parts of long articles included).
  --legacy-scoring pre-v2 rule: the chunk text must start with an article label.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))

from eval_retrieval import score_case, score_jurisdiction_coverage  # noqa: E402

from mercatoria.config import load_config  # noqa: E402
from mercatoria.retrieve import Retriever  # noqa: E402
from mercatoria.store import open_current, open_index  # noqa: E402

CASES_FILE = TOOLS.parent / "corpus" / "eval" / "test_cases.json"
HOLDOUT_FILE = TOOLS.parent / "tests" / "eval" / "holdout_cases.json"
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


def _holdout_ids() -> set[str]:
    if not HOLDOUT_FILE.is_file():
        return set()
    data = json.loads(HOLDOUT_FILE.read_text(encoding="utf-8"))
    cases = data.get("cases") or data
    return {str(case.get("id")) for case in cases if isinstance(case, dict)}


def _chunk_payload(chunk: dict) -> dict:
    return {
        "content": chunk["text"],
        "document_keyword": chunk["source_doc"],
        "doc_id": chunk.get("doc_id"),
        "article_ids": chunk.get("article_ids") or [],
        "jurisdictions": chunk.get("jurisdictions") or [],
    }


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
    ap.add_argument("--workers", type=int, default=1, help="parallel retrieval threads")
    ap.add_argument(
        "--fusion",
        action="store_true",
        help="send q_zh + q_en as one multi-query request (A6: multi-query fusion)",
    )
    ap.add_argument("--multi-query-bonus", type=float, default=None)
    ap.add_argument("--legacy-scoring", action="store_true", help="pre-v2 article-head rule")
    ap.add_argument("--holdout", action="store_true", help="score the frozen hold-out set")
    ap.add_argument(
        "--no-quota",
        action="store_true",
        help="disable auto per-jurisdiction quota (per_jurisdiction_k=0)",
    )
    ap.add_argument(
        "--jurisdiction-filter",
        action="store_true",
        help="pass expected_jurisdictions as the jurisdictions filter (coverage evaluation)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.multi_query_bonus is not None:
        cfg.retrieval.multi_query_bonus = args.multi_query_bonus
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
    all_cases = json.loads(src.read_text(encoding="utf-8"))
    if args.holdout:
        wanted = _holdout_ids()
        cases = [c for c in all_cases if str(c.get("id")) in wanted]
        if not cases:
            print("hold-out 文件为空或未生成：python tools/make_holdout.py", file=sys.stderr)
            return 1
    else:
        excluded = _holdout_ids()
        cases = [c for c in all_cases if str(c.get("id")) not in excluded]
    cases = [c for c in cases if c.get("expect") != "refuse"]
    cases = _pick(cases, args.limit)

    retrieval_k = max(args.top_k, args.retrieval_k)
    tasks: list[tuple[int, dict, str, list[str]]] = []
    for position, case in enumerate(cases):
        if args.fusion:
            tasks.append((position, case, "both", [case["q_zh"], case["q_en"]]))
            continue
        if args.lang in {"zh", "both"}:
            tasks.append((position, case, "zh", [case["q_zh"]]))
        if args.lang in {"en", "both"}:
            tasks.append((position, case, "en", [case["q_en"]]))

    def run(task: tuple[int, dict, str, list[str]]) -> dict | None:
        position, case, lang, questions = task
        filters: dict = {}
        if args.jurisdiction_filter and case.get("expected_jurisdictions"):
            filters["jurisdictions"] = case["expected_jurisdictions"]
        started = time.time()
        try:
            result = retriever.retrieve(
                questions,
                top_k=retrieval_k,
                filters=filters or None,
                per_jurisdiction_k=0 if args.no_quota else None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{case['id']}] ERR {exc}", file=sys.stderr)
            return None
        elapsed = round(time.time() - started, 3)
        chunks = [_chunk_payload(chunk) for chunk in result["chunks"]]
        scored = score_case(case, chunks, legacy_article_head=args.legacy_scoring)
        coverage = score_jurisdiction_coverage(case, chunks, k=args.top_k)
        return {
            "case_id": case["id"],
            "category": case["category"],
            "lang": lang,
            "question": " | ".join(questions),
            "elapsed": elapsed,
            "zero_reason": result["diagnostics"].get("zero_reason"),
            "position": position,
            **scored,
            "coverage": coverage,
        }

    rows: list[dict] = []
    t0 = time.time()
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index_done, row in enumerate(pool.map(run, tasks), 1):
                if row is not None:
                    rows.append(row)
                if index_done % 50 == 0:
                    print(f"  {index_done}/{len(tasks)}  已用 {time.time() - t0:.0f}s", flush=True)
    else:
        for index_done, task in enumerate(tasks, 1):
            row = run(task)
            if row is not None:
                rows.append(row)
            if index_done % 25 == 0:
                print(f"  {index_done}/{len(tasks)}  已用 {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda row: (row["position"], row["lang"]))

    if not rows:
        print("无结果", file=sys.stderr)
        return 1

    n = len(rows)
    coverage_rows = [row for row in rows if row["coverage"]]
    summary = {
        "tag": args.tag,
        "lang": args.lang,
        "top_k": args.top_k,
        "retrieval_k": retrieval_k,
        "vector_weight": cfg.retrieval.vector_weight,
        "similarity_threshold": cfg.retrieval.similarity_threshold,
        "bm25_k1": cfg.retrieval.bm25_k1,
        "bm25_b": cfg.retrieval.bm25_b,
        "legacy_scoring": bool(args.legacy_scoring),
        "fusion": bool(args.fusion),
        "multi_query_bonus": cfg.retrieval.multi_query_bonus,
        "holdout": bool(args.holdout),
        "quota": not args.no_quota,
        "jurisdiction_filter": bool(args.jurisdiction_filter),
        "n": n,
        "hit5": round(sum(1 for r in rows if r["hit5"]) / n, 4),
        "hit10": round(sum(1 for r in rows if r["hit10"]) / n, 4),
        "hit5_loose": round(sum(1 for r in rows if r["hit5_loose"]) / n, 4),
        "mrr": round(statistics.mean(r["rr"] for r in rows), 4),
        "has_any": round(sum(1 for r in rows if r["has_any"]) / n, 4),
        "avg_chunks": round(statistics.mean(r["n_chunks"] for r in rows), 2),
        "coverage_n": len(coverage_rows),
        "coverage5": round(
            sum(1 for r in coverage_rows if r["coverage"]["covered"]) / len(coverage_rows), 4
        )
        if coverage_rows
        else None,
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
    if summary["coverage5"] is not None:
        print(f"跨法域覆盖率@5     : {summary['coverage5']:.4f} (n={summary['coverage_n']})")
    print(f"口径               : legacy_scoring={summary['legacy_scoring']} quota={summary['quota']}")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
