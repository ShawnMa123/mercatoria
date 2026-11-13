"""Compare jurisdiction-quota strategies on precision-sensitive questions.

Cases (``tests/eval/quota_cases.json``) are real questions about ONE legal
system, retrieved with a filter that also requests a distractor jurisdiction:
the setting where a hard quota can spend a slot on a law nobody asked about.

Strategies:
    off    per_jurisdiction_k = 0
    hard   auto quota (min(2, top_k // n))
    soft   auto quota + retrieval.quota_min_ratio (default 0.6)

Also reports how often a naive "jurisdiction keywords in the question" trigger
would fire, since P1-7 asks for that conclusion only (not an implementation).

Usage:
    python tools/eval_quota.py [--index ...] [--out docs/quota-eval-<date>.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))

from eval_retrieval import score_case  # noqa: E402

from mercatoria.config import load_config  # noqa: E402
from mercatoria.retrieve import Retriever  # noqa: E402
from mercatoria.store import open_current, open_index  # noqa: E402

CASES = ROOT / "tests" / "eval" / "quota_cases.json"

JURISDICTION_WORDS = {
    "CN": ("中国", "我国", "中华人民共和国", "海关总署"),
    "US": ("美国", "CBP", "eCFR", "CFR"),
    "WTO": ("WTO", "世贸", "GATT"),
    "JP": ("日本", "関税"),
    "AU": ("澳洲", "澳大利亚"),
    "EU": ("欧盟", "EU"),
    "INT": ("CISG", "联合国"),
}


def _mentions(question: str) -> list[str]:
    lowered = question.lower()
    return [
        jurisdiction
        for jurisdiction, words in JURISDICTION_WORDS.items()
        if any(word.lower() in lowered for word in words)
    ]


def run_strategy(retriever: Retriever, cases: list[dict], strategy: str) -> dict:
    hits: list[int] = []
    rr: list[float] = []
    slots: dict[str, int] = {}
    for case in cases:
        filters = {"jurisdictions": case["requested_jurisdictions"]}
        quota = 0 if strategy == "off" else None
        result = retriever.retrieve(
            [case["question"]], top_k=5, filters=filters, per_jurisdiction_k=quota
        )
        chunks = [
            {
                "content": chunk["text"],
                "document_keyword": chunk["source_doc"],
                "doc_id": chunk.get("doc_id"),
                "article_ids": chunk.get("article_ids") or [],
                "jurisdictions": chunk.get("jurisdictions") or [],
            }
            for chunk in result["chunks"]
        ]
        scored = score_case(case, chunks)
        hits.append(1 if scored["hit5"] else 0)
        rr.append(scored["rr"])
        for jurisdiction, count in (result["diagnostics"].get("quota_slots") or {}).items():
            slots[jurisdiction] = slots.get(jurisdiction, 0) + count
    n = len(cases)
    return {
        "strategy": strategy,
        "n": n,
        "hit5": round(sum(hits) / n, 4),
        "mrr": round(statistics.mean(rr), 4),
        "slots": slots,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.example.json")
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ratio", type=float, default=0.6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    index = open_index(args.index) if args.index else open_current(cfg.resolved_index_dir())
    retriever = Retriever(index, cfg)
    data = json.loads(CASES.read_text(encoding="utf-8"))
    cases = data["cases"]

    results = [run_strategy(retriever, cases, "off"), run_strategy(retriever, cases, "hard")]
    cfg.retrieval.quota_min_ratio = args.ratio
    retriever.clear_cache()
    results.append(run_strategy(retriever, cases, "soft"))

    mentions = [_mentions(case["question"]) for case in cases]
    would_fire = sum(1 for m in mentions if len(m) > 1)
    wrong_trigger = sum(
        1
        for case, m in zip(cases, mentions)
        if len(m) > 1 and not set(case["relevant_jurisdictions"]) <= set(m)
    )

    lines = [
        "# 配额精度代价（P1-7）",
        "",
        f"- 生成日期：{date.today().isoformat()}",
        f"- 用例：{len(cases)}（每题只有一个法域相关，但过滤器请求了 2 个法域）",
        f"- 软配额阈值 `quota_min_ratio`：{args.ratio}",
        "",
        "| 策略 | Hit@5 | MRR | 各法域占用槽位 |",
        "|---|---|---|---|",
    ]
    for item in results:
        slots = ", ".join(f"{key}:{value}" for key, value in sorted(item["slots"].items())) or "-"
        lines.append(
            f"| {item['strategy']} | {item['hit5']:.4f} | {item['mrr']:.4f} | {slots} |"
        )
    lines += [
        "",
        "## 关键词触发配额的可行性",
        "",
        f"- 题目同时提到 2 个以上法域名的：{would_fire}/{len(cases)}",
        f"- 其中会误判（真实相关法域不在提到的法域里）的：{wrong_trigger}/{would_fire or 1}",
        "",
        "结论：不启用基于问题文本的关键词触发；多法域过滤已经由调用方显式给出，",
        "用文本猜测只会引入误判。",
    ]
    out = Path(args.out) if args.out else ROOT / "docs" / f"quota-eval-{date.today().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
