"""把评测结果汇总成 Markdown 报告。

用法：
    python make_report.py --results corpus/eval/results_<ts>.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "eval-report.md"


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100*a/b:.1f}%)" if b else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    rows = load(Path(args.results))
    modes = sorted({r["thinking"] for r in rows}, reverse=True)

    lines: list[str] = []
    w = lines.append

    w("# 法规检索评测报告")
    w("")
    w(f"> 结果文件：`{args.results}`")
    w(f"> 用例总数：{len(rows) // len(modes)} × {len(modes)} 种模式 = {len(rows)} 次调用")
    w("")

    # ── 总览 ──
    w("## 1. 总览")
    w("")
    w("| 指标 | " + " | ".join(f"thinking={m}" for m in modes) + " |")
    w("|---|" + "---|" * len(modes))

    def agg(mode: str) -> dict:
        rs = [r for r in rows if r["thinking"] == mode]
        n = len(rs)
        kw = [r["score"]["keyword_hit"] for r in rs if r["score"].get("keyword_hit") is not None]
        el = [r.get("elapsed") or r["score"].get("elapsed") or 0 for r in rs]
        refuse = [r for r in rs if r["score"].get("refuse_correct") is not None]
        return {
            "n": n,
            "kw_mean": statistics.mean(kw) if kw else 0,
            "kw_full": sum(1 for x in kw if x >= 1.0),
            "kw_zero": sum(1 for x in kw if x == 0),
            "kw_n": len(kw),
            "refuse_ok": sum(1 for r in refuse if r["score"]["refuse_correct"]),
            "refuse_n": len(refuse),
            "with_refs": sum(1 for r in rs if r["score"].get("has_refs")),
            "partial": sum(1 for r in rs if r["score"].get("partial_with_gap")),
            "hard_refuse": sum(1 for r in rs if r["score"].get("full_refuse_with_refs")),
            "empty": sum(1 for r in rs if r["score"].get("empty_answer")),
            "el_mean": statistics.mean(el) if el else 0,
            "el_med": statistics.median(el) if el else 0,
        }

    stats = {m: agg(m) for m in modes}
    s0 = stats[modes[0]]
    w(f"| 调用次数 | " + " | ".join(str(stats[m]["n"]) for m in modes) + " |")
    w(f"| 关键词覆盖率（均值） | " + " | ".join(f"{stats[m]['kw_mean']:.3f}" for m in modes) + " |")
    w(f"| 关键词全中 | " + " | ".join(pct(stats[m]["kw_full"], stats[m]["kw_n"]) for m in modes) + " |")
    w(f"| 关键词全失 | " + " | ".join(pct(stats[m]["kw_zero"], stats[m]["kw_n"]) for m in modes) + " |")
    w(f"| 有引用比例 | " + " | ".join(pct(stats[m]["with_refs"], stats[m]["n"]) for m in modes) + " |")
    w(f"| 部分作答+声明缺口 | " + " | ".join(str(stats[m]["partial"]) for m in modes) + " |")
    w(f"| 有引用却完全拒答 | " + " | ".join(str(stats[m]["hard_refuse"]) for m in modes) + " |")
    w(f"| 空答案 | " + " | ".join(str(stats[m]["empty"]) for m in modes) + " |")
    w(f"| 拒答正确率 | " + " | ".join(pct(stats[m]["refuse_ok"], stats[m]["refuse_n"]) for m in modes) + " |")
    w(f"| 平均耗时 | " + " | ".join(f"{stats[m]['el_mean']:.1f}s" for m in modes) + " |")
    w(f"| 中位耗时 | " + " | ".join(f"{stats[m]['el_med']:.1f}s" for m in modes) + " |")
    w("")

    if len(modes) == 2:
        a, b = stats[modes[0]], stats[modes[1]]
        ratio = a["el_mean"] / b["el_mean"] if b["el_mean"] else 0
        w(f"**思考开关对比**：开思考耗时是老模式的 **{ratio:.1f}×**，"
          f"关键词覆盖率差异 **{a['kw_mean']-b['kw_mean']:+.3f}**。")
        w("")

    # ── 按场景 ──
    w("## 2. 按业务场景")
    w("")
    by_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_cat[r["category"]][r["thinking"]].append(r["score"])

    w("| 场景 | 用例 | " + " | ".join(f"覆盖率 think={m}" for m in modes) + " |")
    w("|---|---|" + "---|" * len(modes))
    for cat in sorted(by_cat, key=lambda c: -sum(len(v) for v in by_cat[c].values())):
        cells = []
        for m in modes:
            scs = by_cat[cat].get(m, [])
            kws = [s["keyword_hit"] for s in scs if s.get("keyword_hit") is not None]
            cells.append(f"{statistics.mean(kws):.3f}" if kws else "—")
        n = max(len(v) for v in by_cat[cat].values())
        w(f"| {cat} | {n} | " + " | ".join(cells) + " |")
    w("")

    # ── 失败案例 ──
    # 分三类，因为处置完全不同：
    #   A. 空答案        → 生成失败（如 max_tokens 不够）
    #   B. 有引用却硬拒答 → 检索/生成链路有问题，需要排查
    #   C. 部分作答+声明缺口 → 语料确实缺，属正常，不算失败
    w("## 3. 失败案例分类")
    w("")
    w("按处置方式分三类，**C 类不算故障**：")
    w("")
    w("| 类型 | 含义 | 处置 |")
    w("|---|---|---|")
    w("| A 空答案 | 生成完全失败 | 检查 max_tokens、模型可用性 |")
    w("| B 有引用却硬拒答 | 检索到了但没用上 | 排查检索/生成链路 |")
    w("| C 部分作答+声明缺口 | 语料确实没有该内容 | 正常，模型行为正确 |")
    w("")
    for mode in modes:
        rs = [r for r in rows if r["thinking"] == mode]
        a = [r for r in rs if r["score"].get("empty_answer")]
        b = [r for r in rs if r["score"].get("full_refuse_with_refs")]
        c = [r for r in rs if r["score"].get("partial_with_gap")]
        w(f"### thinking={mode}")
        w("")
        w(f"- A 空答案：{len(a)}")
        w(f"- B 有引用却硬拒答：{len(b)}")
        w(f"- C 部分作答+声明缺口：{len(c)}")
        w("")
        if b:
            w("**B 类明细**（需要排查）：")
            w("")
            w("| # | 问题 | 引用数 | 首个引用摘要 |")
            w("|---|---|---|---|")
            for r in b[:15]:
                q = r["question"][:36].replace("|", "/")
                first = ""
                if r.get("refs"):
                    first = (r["refs"][0].get("content") or "")[:55].replace("\n", " ").replace("|", "/")
                w(f"| {r['case_id']} | {q} | {r['score'].get('n_refs',0)} | {first} |")
            w("")
        # 关键词低分但答案实质正确的，单独列出供人工复核
        low_kw = [
            r for r in rs
            if r["score"].get("keyword_hit") is not None
            and r["score"]["keyword_hit"] < 0.5
            and not r["score"].get("empty_answer")
        ]
        if low_kw:
            w(f"**关键词低分（{len(low_kw)} 条，需人工复核是否真的答错）**：")
            w("")
            w("| # | 问题 | 关键词命中 | 答案摘要 |")
            w("|---|---|---|---|")
            for r in low_kw[:15]:
                q = r["question"][:36].replace("|", "/")
                ans = (r["answer"] or "")[:60].replace("\n", " ").replace("|", "/")
                w(f"| {r['case_id']} | {q} | {r['score']['keyword_hit']} | {ans} |")
            w("")

    # ── 法域 ──
    w("## 4. 按法域（有引用的文档归属）")
    w("")
    w("> 说明：`ref_docs` 取自 RAGFlow 返回的 `document_keyword`，部分为空。")
    w("")
    for mode in modes:
        rs = [r for r in rows if r["thinking"] == mode]
        empties = sum(1 for r in rs if r["score"].get("has_refs") and not any(r["score"].get("ref_docs") or []))
        w(f"- thinking={mode}：{empties} 条有引用但文档名为空（{100*empties/max(1,len(rs)):.0f}%）")
    w("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
