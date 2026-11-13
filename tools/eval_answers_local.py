"""Answer-quality metrics for the local /v1/chat surface.

Unlike retrieval-only evals this calls the configured LLM, so it is slower and
depends on the gateway being up. Metrics are objective (no LLM judge):

  * answered / skipped / error rate
  * refusal correctness on `no_answer` questions (marker based, same list as
    tools/run_eval.py)
  * citation presence and citation validity: every ``[chunk_id]`` in the answer
    must exist in the retrieved chunks (fabricated ids are counted)
  * grounding: cited chunks whose text contains an expected keyword
  * expected-keyword coverage of the answer text
  * latency p50/p95

Usage:
    python tools/eval_answers_local.py --base http://127.0.0.1:8099 [--limit 20]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from run_eval import REFUSE_MARKERS  # noqa: E402

FIXTURE = ROOT / "tests" / "eval" / "trade_law_questions.json"
OUT_DIR = ROOT / "data" / "eval"
CITATION = re.compile(r"\[([0-9a-f]{8,32})\]")


def post(base: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def refuses(answer: str) -> bool:
    text = (answer or "").lower()
    return any(marker.lower() in text for marker in REFUSE_MARKERS)


def evaluate(base: str, cases: list[dict], top_k: int) -> dict:
    rows: list[dict] = []
    for case in cases:
        started = time.perf_counter()
        try:
            payload = post(
                base,
                "/v1/chat",
                {"messages": [{"role": "user", "content": case["question"]}], "top_k": top_k},
            )
            elapsed = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "id": case["id"],
                    "type": case.get("type", "answer"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
            continue

        answer = str(payload.get("answer") or "")
        chunks = payload.get("chunks") or []
        chunk_ids = {str(chunk.get("chunk_id")) for chunk in chunks}
        cited = CITATION.findall(answer)
        valid = [cid for cid in cited if cid in chunk_ids]
        keywords = [str(k) for k in case.get("expected_keywords") or []]
        lowered = answer.lower()
        keyword_hits = [kw for kw in keywords if kw.lower() in lowered]
        cited_text = " ".join(
            str(chunk.get("text") or "") for chunk in chunks if chunk.get("chunk_id") in set(valid)
        ).lower()
        grounded = [kw for kw in keywords if kw.lower() in cited_text]
        rows.append(
            {
                "id": case["id"],
                "type": case.get("type", "answer"),
                "status": payload.get("answer_status"),
                "latency_ms": round(elapsed, 1),
                "answer_chars": len(answer),
                "citations": len(cited),
                "valid_citations": len(valid),
                "invalid_citations": [cid for cid in cited if cid not in chunk_ids][:5],
                "citation_present": bool(cited),
                "keywords": keywords,
                "keyword_hits": keyword_hits,
                "grounded_keywords": grounded,
                "refused": refuses(answer),
                "retrieved": len(chunks),
                "confidence": (payload.get("diagnostics") or {}).get("confidence"),
                "answer": answer[:600],
            }
        )
    return {"rows": rows}


def summarise(rows: list[dict]) -> dict:
    answered = [r for r in rows if r["type"] != "no_answer" and r["status"] != "error"]
    no_answer = [r for r in rows if r["type"] == "no_answer" and r["status"] != "error"]
    live = [r for r in rows if r["status"] not in {"error", "skipped"}]

    def rate(items: list[dict], predicate) -> float | None:
        return round(sum(1 for item in items if predicate(item)) / len(items), 4) if items else None

    with_keywords = [r for r in answered if r["keywords"]]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    ordered = sorted(latencies)
    return {
        "n": len(rows),
        "answered_n": len(answered),
        "no_answer_n": len(no_answer),
        "error_n": sum(1 for r in rows if r["status"] == "error"),
        "skipped_n": sum(1 for r in rows if r["status"] == "skipped"),
        "refusal_rate_on_no_answer": rate(no_answer, lambda r: r["refused"]),
        "false_refusal_rate_on_answered": rate(answered, lambda r: r["refused"]),
        "citation_presence_rate": rate(live, lambda r: r["citation_present"]),
        "citation_validity_rate": rate(
            [r for r in live if r["citations"]],
            lambda r: r["valid_citations"] == r["citations"],
        ),
        "fabricated_citation_answers": sum(1 for r in live if r["invalid_citations"]),
        "keyword_all_hit_rate": rate(
            with_keywords, lambda r: len(r["keyword_hits"]) == len(r["keywords"])
        )
        if with_keywords
        else None,
        "keyword_any_hit_rate": rate(with_keywords, lambda r: bool(r["keyword_hits"]))
        if with_keywords
        else None,
        "grounded_all_keywords_rate": rate(
            with_keywords, lambda r: len(r["grounded_keywords"]) == len(r["keywords"])
        )
        if with_keywords
        else None,
        "avg_answer_chars": round(statistics.mean([r["answer_chars"] for r in live]), 1)
        if live
        else None,
        "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "p95_ms": (ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8099")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma separated question ids")
    ap.add_argument("--tag", default="local")
    args = ap.parse_args()

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = data["questions"]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    result = evaluate(args.base, cases, args.top_k)
    summary = summarise(result["rows"])
    result["summary"] = summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"answers_{args.tag}_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
