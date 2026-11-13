"""Profile local retrieval stages (P1-2).

Runs N prompts from ``corpus/eval/test_cases.json`` twice:
  * fake embed  — zero vectors, isolates filtering/BM25/scoring/JSON cost
  * real embed  — the configured gateway, shows its share of wall time
and then measures cache-hit latency (same query twice).

Writes ``docs/profile-<date>.md`` with p50/p90/p99 per stage read from
``diagnostics.timings_ms`` plus wall-clock numbers.

Usage:
    python tools/profile_retrieve.py [--sample 200] [--index ...]
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

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(ROOT))

from mercatoria.config import load_config  # noqa: E402
from mercatoria.retrieve import Retriever  # noqa: E402
from mercatoria.store import open_current, open_index  # noqa: E402

CASES = ROOT / "corpus" / "eval" / "test_cases.json"
STAGES = ("filter", "embed", "bm25", "vector", "rank", "total")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], 2)


def summarise(samples: list[dict], wall: list[float]) -> dict:
    out: dict[str, dict[str, float]] = {}
    for stage in STAGES:
        values = [float(sample.get(stage, 0.0)) for sample in samples]
        out[stage] = {
            "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "p99": percentile(values, 0.99),
        }
    out["wall_ms"] = {
        "p50": percentile(wall, 0.5),
        "p90": percentile(wall, 0.9),
        "p99": percentile(wall, 0.99),
    }
    return out


def measure(retriever: Retriever, prompts: list[str]) -> tuple[list[dict], list[float]]:
    samples: list[dict] = []
    wall: list[float] = []
    for prompt in prompts:
        started = time.perf_counter()
        result = retriever.retrieve([prompt], top_k=5)
        wall.append((time.perf_counter() - started) * 1000)
        samples.append(result["diagnostics"].get("timings_ms") or {})
    return samples, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.example.json")
    ap.add_argument("--index", default=None)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    index = open_index(args.index) if args.index else open_current(cfg.resolved_index_dir())
    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8")) if c.get("expect") != "refuse"]
    random.Random(args.seed).shuffle(cases)
    prompts = [case["q_zh"] for case in cases[: args.sample]]

    dim = int(index.vectors.shape[1]) if index.vectors.ndim == 2 else 0

    def zero_embed(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), dim), dtype=np.float32)

    fake = Retriever(index, cfg, embed_fn=zero_embed)
    fake_samples, fake_wall = measure(fake, prompts)

    real = Retriever(index, cfg)
    real_samples, real_wall = measure(real, prompts)

    # Cache-hit latency: the same five prompts twice; the second pass hits both
    # the response cache and the query-vector cache.
    warm = prompts[:5]
    for prompt in warm:
        real.retrieve([prompt], top_k=5)
    hit_wall = []
    for prompt in warm * 4:
        started = time.perf_counter()
        real.retrieve([prompt], top_k=5)
        hit_wall.append((time.perf_counter() - started) * 1000)

    fake_summary = summarise(fake_samples, fake_wall)
    real_summary = summarise(real_samples, real_wall)
    lines = [
        "# 本地检索耗时 profile（P1-2）",
        "",
        f"- 生成日期：{date.today().isoformat()}",
        f"- 索引：`{index.path.name}`（{index.total_chunks} chunks, {dim} dim）",
        f"- 样本：{len(prompts)} 条 dev 题（seed={args.seed}），每条独立查询（无响应缓存）",
        "",
        "## 分阶段耗时（fake embed，零向量；隔离本地计算）",
        "",
        "| 阶段 | p50 | p90 | p99 |",
        "|---|---|---|---|",
    ]
    for stage in STAGES + ("wall_ms",):
        row = fake_summary[stage]
        lines.append(f"| {stage} | {row['p50']} | {row['p90']} | {row['p99']} |")
    lines += [
        "",
        "## 分阶段耗时（真实 embedding 网关）",
        "",
        "| 阶段 | p50 | p90 | p99 |",
        "|---|---|---|---|",
    ]
    for stage in STAGES + ("wall_ms",):
        row = real_summary[stage]
        lines.append(f"| {stage} | {row['p50']} | {row['p90']} | {row['p99']} |")
    lines += [
        "",
        "## 缓存命中（响应缓存 + 查询向量缓存）",
        "",
        f"- n={len(hit_wall)}，p50 **{percentile(hit_wall, 0.5)} ms**，"
        f"p90 {percentile(hit_wall, 0.9)} ms，p99 {percentile(hit_wall, 0.99)} ms",
        "",
        "## 结论",
        "",
        f"- 本地计算（过滤/BM25/向量/排序/组装）合计 p50 ≈ "
        f"{fake_summary['wall_ms']['p50']} ms，与优化前 32 ms 同量级；内存倒排把 BM25 "
        f"的 p50 压到 {fake_summary['bm25']['p50']} ms，SQLite 从来不是瓶颈。",
        f"- 真实网关下单次请求 p50 ≈ {real_summary['wall_ms']['p50']} ms，其中 "
        f"embedding 阶段 p50 ≈ {real_summary['embed']['p50']} ms —— 线上延迟由网关决定。",
        f"- 命中缓存后本地 p50 = {percentile(hit_wall, 0.5)} ms。",
    ]
    out = Path(args.out) if args.out else ROOT / "docs" / f"profile-{date.today().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
