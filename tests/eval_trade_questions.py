"""Run the sourced trade-law question set against a running mercatoria service.

Questions are paraphrases of public customs-exam questions, MOFCOM/Customs FAQ
answers and CISG/WTO materials; every row keeps its source URL.

Usage:
  python tests/eval_trade_questions.py --base http://localhost:8080
  python tests/eval_trade_questions.py --base http://localhost:8080 --answer-limit 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SET = HERE / "eval" / "trade_law_questions.json"


def _post(base: str, path: str, body: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload, (time.perf_counter() - started) * 1000.0


def _text(value: object) -> str:
    return str(value or "").lower()


def _rank(case: dict, chunks: list[dict]) -> tuple[int, bool, bool]:
    docs = {str(d) for d in case.get("expected_doc_ids", [])}
    keywords = [_text(k) for k in case.get("expected_keywords", []) if str(k).strip()]
    for index, chunk in enumerate(chunks, 1):
        doc_hit = str(chunk.get("doc_id")) in docs
        body = _text(chunk.get("text"))
        keyword_hit = any(k in body for k in keywords)
        if doc_hit and keyword_hit:
            return index, True, True
    # Keep diagnostic detail for a near miss.
    doc_hit = any(str(c.get("doc_id")) in docs for c in chunks)
    keyword_hit = any(any(k in _text(c.get("text")) for k in keywords) for c in chunks)
    return 0, doc_hit, keyword_hit


def run(
    base: str,
    top_k: int,
    answer_limit: int,
    out_path: Path | None,
    answer_ids: str = "",
) -> dict:
    data = json.loads(DEFAULT_SET.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for case in data["questions"]:
        payload, latency = _post(
            base,
            "/v1/retrieve",
            {"queries": [case["question"]], "top_k": top_k},
            timeout=120,
        )
        chunks = payload.get("chunks") or []
        rank, doc_hit, keyword_hit = _rank(case, chunks)
        row = {
            "id": case["id"],
            "category": case["category"],
            "language": case.get("language", "zh"),
            "question": case["question"],
            "source": case.get("source"),
            "source_url": case.get("source_url"),
            "corpus_gap": bool(case.get("corpus_gap")),
            "coverage_note": case.get("coverage_note"),
            "latency_ms": round(latency, 1),
            "rank": rank,
            "hit": rank > 0,
            "doc_hit": doc_hit,
            "keyword_hit": keyword_hit,
            "rerank": payload.get("diagnostics", {}).get("rerank", {}),
            "top": [
                {
                    "doc_id": c.get("doc_id"),
                    "article_id": (c.get("article_ids") or [None])[0],
                    "score": c.get("score"),
                    "text": str(c.get("text", ""))[:120],
                }
                for c in chunks[:top_k]
            ],
        }
        rows.append(row)
        print(
            f"{row['id']} {row['category']:<6} hit={int(row['hit'])} "
            f"rank={row['rank']} {row['latency_ms']:.0f}ms"
        )

    wanted = [x.strip() for x in answer_ids.split(",") if x.strip()]
    answered: list[dict] = []
    for row in rows:
        if wanted and row["id"] not in wanted:
            continue
        if not wanted and len(answered) >= answer_limit:
            break
        case = next(q for q in data["questions"] if q["id"] == row["id"])
        payload, latency = _post(
            base,
            "/v1/chat",
            {
                "messages": [{"role": "user", "content": case["question"]}],
                "top_k": top_k,
            },
            timeout=180,
        )
        references = payload.get("references") or []
        answered.append(
            {
                "id": row["id"],
                "answer_status": payload.get("answer_status"),
                "latency_ms": round(latency, 1),
                "answer": str(payload.get("answer") or "")[:800],
                "citations": len(payload.get("citations") or []),
                "references": len(references),
                "full_reference_text": all(bool(r.get("text")) for r in references),
            }
        )
        print(
            f"  answer {row['id']} status={answered[-1]['answer_status']} "
            f"citations={answered[-1]['citations']} {latency:.0f}ms"
        )

    by_category: dict[str, dict] = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for category, items in sorted(groups.items()):
        hits = [r for r in items if r["hit"]]
        reciprocal = [1.0 / r["rank"] if r["rank"] else 0.0 for r in items]
        by_category[category] = {
            "n": len(items),
            "hit5": round(len(hits) / len(items), 4),
            "mrr": round(sum(reciprocal) / len(items), 4),
            "p50_ms": round(statistics.median([r["latency_ms"] for r in items]), 1),
        }

    covered = [r for r in rows if not r.get("corpus_gap")]
    gaps = [r for r in rows if r.get("corpus_gap")]
    summary = {
        "n": len(rows),
        "hit5": round(sum(1 for r in rows if r["hit"]) / len(rows), 4),
        "hit5_covered": round(
            sum(1 for r in covered if r["hit"]) / len(covered), 4
        )
        if covered
        else 0.0,
        "covered_n": len(covered),
        "corpus_gaps": [
            {"id": r["id"], "question": r["question"], "note": r.get("coverage_note")}
            for r in gaps
        ],
        "mrr": round(
            sum((1.0 / r["rank"] if r["rank"] else 0.0) for r in rows) / len(rows), 4
        ),
        "doc_hit": round(sum(1 for r in rows if r["doc_hit"]) / len(rows), 4),
        "keyword_hit": round(sum(1 for r in rows if r["keyword_hit"]) / len(rows), 4),
        "p50_ms": round(statistics.median([r["latency_ms"] for r in rows]), 1),
        "p95_ms": round(
            sorted(r["latency_ms"] for r in rows)[max(0, int(len(rows) * 0.95) - 1)], 1
        ),
        "by_category": by_category,
        "by_language": {
            lang: {
                "n": len(items),
                "hit5": round(sum(1 for r in items if r["hit"]) / len(items), 4),
            }
            for lang, items in {
                "zh": [r for r in rows if r["language"] == "zh"],
                "en": [r for r in rows if r["language"] == "en"],
            }.items()
            if items
        },
    }
    result = {"summary": summary, "rows": rows, "answered": answered}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--answer-limit", type=int, default=0)
    ap.add_argument("--answer-ids", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out) if args.out else None
    run(args.base, args.top_k, args.answer_limit, out, args.answer_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
