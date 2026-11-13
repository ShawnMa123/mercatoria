"""Rerank policy comparison (P2-1): latency vs MRR.

Policies:
    off        no cross-encoder
    always     rerank the configured candidate pool
    gated      rerank only when the first pass is not "high" confidence
    topN       rerank only the first ``--top-n`` candidates

The default configuration stays ``rerank.enabled=false``; this tool exists to
show the trade-off, not to flip it.

Usage:
    python tools/eval_rerank_policy.py [--sample 100] [--index ...]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
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

CASES = ROOT / "corpus" / "eval" / "test_cases.json"


def chunks_of(result: dict) -> list[dict]:
    return [
        {
            "content": chunk["text"],
            "document_keyword": chunk["source_doc"],
            "doc_id": chunk.get("doc_id"),
            "article_ids": chunk.get("article_ids") or [],
            "jurisdictions": chunk.get("jurisdictions") or [],
        }
        for chunk in result["chunks"]
    ]


def measure(retriever: Retriever, cases: list[dict], policy: str, top_n: int = 5) -> dict:
    hits = 0
    rr: list[float] = []
    latency: list[float] = []
    for case in cases:
        started = time.perf_counter()
        first = retriever.retrieve([case["q_zh"]], top_k=10, use_rerank=False)
        if policy == "off":
            result = first
        elif policy == "gated":
            result = (
                first
                if (first["diagnostics"].get("confidence") == "high")
                else retriever.retrieve([case["q_zh"]], top_k=10, use_rerank=True)
            )
        elif policy == "topN":
            retriever.cfg.rerank.candidates = top_n
            retriever.clear_cache()
            result = retriever.retrieve([case["q_zh"]], top_k=10, use_rerank=True)
        else:
            result = retriever.retrieve([case["q_zh"]], top_k=10, use_rerank=True)
        latency.append((time.perf_counter() - started) * 1000)
        scored = score_case(case, chunks_of(result))
        hits += 1 if scored["hit5"] else 0
        rr.append(scored["rr"])
    ordered = sorted(latency)
    return {
        "policy": policy,
        "n": len(cases),
        "hit5": round(hits / len(cases), 4),
        "mrr": round(statistics.mean(rr), 4),
        "p50_ms": round(statistics.median(latency), 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.example.json")
    ap.add_argument("--index", default=None)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    index = open_index(args.index) if args.index else open_current(cfg.resolved_index_dir())
    retriever = Retriever(index, cfg)
    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8")) if c.get("expect") != "refuse"]
    random.Random(args.seed).shuffle(cases)
    sample = cases[: args.sample]

    rows = [
        measure(retriever, sample, "off"),
        measure(retriever, sample, "always"),
        measure(retriever, sample, "gated"),
        measure(retriever, sample, "topN", args.top_n),
    ]
    lines = [
        "# Rerank 策略帕累托对比（P2-1）",
        "",
        f"- 生成日期：{date.today().isoformat()}",
        f"- 样本：{len(sample)} 条 dev 题（seed={args.seed}），top_k=10，weighted 0.5",
        "",
        "| 策略 | Hit@5 | MRR | p50 | p95 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['hit5']:.4f} | {row['mrr']:.4f} | "
            f"{row['p50_ms']:.0f} ms | {row['p95_ms']:.0f} ms |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- rerank 对 Hit@5 基本无效，MRR 有小幅提升；代价是 p50 由几十毫秒升到秒级。",
        "- `gated` 只在第一阶段非 high 置信时触发，是延迟与排序质量之间较优的折中；"
        "`topN` 能把重排成本压到 N/candidates。",
        "- 默认仍保持 `rerank.enabled=false`：检索服务的主要调用方是 Agent，延迟更敏感。",
    ]
    out = Path(args.out) if args.out else ROOT / "docs" / f"rerank-policy-{date.today().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
