"""纯检索评测（不调 LLM），可算 Recall@k / MRR@k。

为什么单独做：
- 调 LLM 慢（5-20s/条），纯检索只要 0.3-1s/条
- 生成质量会掩盖检索质量。要分开评估。
- 能算出与 Legal-DC 可比的 Recall@5 / MRR@5

真值来源：
    每条用例人工标注 expected_articles（可选）。
    未标注时，用「关键词出现在哪些 chunk」自动构建弱真值 —— 只能算相对指标。

用法：
    python eval_retrieval.py --kb <id>                    # 全量
    python eval_retrieval.py --kb <id> --limit 100
    python eval_retrieval.py --kb <id> --vec-weight 0.7 --tag vecw70
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from ragflow_config import api_key  # noqa: E402

BASE = "http://localhost:9380"
CASES_FILE = Path(__file__).resolve().parent.parent / "corpus" / "eval" / "test_cases.json"
OUT_DIR = CASES_FILE.parent

# 归一化条号：把各法域的标识统一成可比较的 key
PAT_CN = re.compile(r"^第([零一二三四五六七八九十百千0-9]+)条")
PAT_CN_REST = re.compile(r"第([零一二三四五六七八九十百千0-9]+)条")
PAT_US = re.compile(r"^§?\s*(\d+\.\d+[a-z]?)")
PAT_ART = re.compile(r"^Article\s+(\d+[A-Za-z]?)", re.I)
PAT_SEC = re.compile(r"^Section\s+(\d+[A-Za-z]?)", re.I)

CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}


def cn_to_int(s: str) -> int | None:
    """中文数字转整数，支持到千位。"""
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch == "十":
            section += (number or 1) * 10
            number = 0
        elif ch == "百":
            section += (number or 1) * 100
            number = 0
        elif ch == "千":
            section += (number or 1) * 1000
            number = 0
        else:
            return None
    return total + section + number


def article_key(doc: str, content: str) -> str | None:
    """从 chunk 内容首行抽条文标识，与文档名拼成唯一 key。"""
    first = content.strip().split("\n", 1)[0].strip()
    first = first.lstrip("#").strip()
    base = re.sub(r"\(\d{4}-\d{2}-\d{2}\)\.md$", "", doc).strip()
    base = re.sub(r"\.md$", "", base)

    m = PAT_CN.match(first)
    if m:
        n = cn_to_int(m.group(1))
        if n is not None:
            return f"{base}:第{n}条"
    m = PAT_US.match(first)
    if m:
        return f"{base}:{m.group(1)}"
    m = PAT_ART.match(first)
    if m:
        return f"{base}:art-{m.group(1)}"
    m = PAT_SEC.match(first)
    if m:
        return f"{base}:sec-{m.group(1)}"
    return None


def retrieve(q: str, kb: str, key: str, p: dict, timeout: int = 180) -> dict:
    body = {
        "question": q,
        "dataset_ids": [kb],
        "top_k": p["top_k"],
        "top_n": p["top_n"],
        "similarity_threshold": p["sim_threshold"],
        "vector_similarity_weight": p["vec_weight"],
        "keyword": True,
    }
    req = urllib.request.Request(
        BASE + "/api/v1/retrieval",
        json.dumps(body, ensure_ascii=False).encode("utf-8"),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return {
        "chunks": (d.get("data") or {}).get("chunks") or [],
        "elapsed": round(time.time() - t0, 3),
    }


def score_case(case: dict, chunks: list[dict]) -> dict:
    """计算命中情况。

    判据设计（按严格程度分三级，分别统计）：
      strict  : chunk 以条文标识开头，且全部关键词命中
                → 最接近「找对了那一条法条」
      loose   : chunk 以条文标识开头，且至少一个关键词命中
      weak    : 只要出现关键词（几乎必然命中，仅作对照）

    只报 strict 与 loose。weak 已知无区分度，保留用于说明问题。
    """
    kws = [k.lower() for k in (case.get("must_mention") or [])]

    keys: list[str] = []
    strict_idx: list[int] = []
    loose_idx: list[int] = []

    for i, c in enumerate(chunks):
        content = c.get("content") or ""
        doc = c.get("document_keyword") or c.get("docnm_kwd") or ""
        k = article_key(doc, content)
        is_article_head = k is not None
        if k and k not in keys:
            keys.append(k)

        text = content.lower()
        present = [kw for kw in kws if kw in text]
        if is_article_head and kws and len(present) == len(kws):
            strict_idx.append(i)
        if is_article_head and present:
            loose_idx.append(i)

    hit5 = any(i < 5 for i in strict_idx)
    hit10 = any(i < 10 for i in strict_idx)
    hit5_loose = any(i < 5 for i in loose_idx)
    rr = 0.0
    for i in strict_idx:
        rr = 1.0 / (i + 1)
        break

    return {
        "n_chunks": len(chunks),
        "n_articles": len(keys),
        "top_articles": keys[:5],
        "strict_idx": strict_idx[:5],
        "loose_idx": loose_idx[:5],
        "hit5": hit5,
        "hit10": hit10,
        "hit5_loose": hit5_loose,
        "rr": round(rr, 4),
        "has_any": len(chunks) > 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default=None, help="指定用例文件，默认 test_cases.json")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--vec-weight", type=float, default=0.3)
    ap.add_argument("--sim-threshold", type=float, default=0.2)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=1024)
    args = ap.parse_args()

    key = api_key()
    src = Path(args.cases) if args.cases else CASES_FILE
    cases = json.loads(src.read_text(encoding="utf-8"))
    cases = [c for c in cases if c.get("expect") != "refuse"]
    if args.limit:
        by_cat: dict[str, list[dict]] = {}
        for c in cases:
            by_cat.setdefault(c["category"], []).append(c)
        per = max(1, args.limit // len(by_cat))
        picked = []
        for items in by_cat.values():
            picked.extend(items[:per])
        cases = picked[: args.limit]

    p = {
        "vec_weight": args.vec_weight,
        "sim_threshold": args.sim_threshold,
        "top_n": args.top_n,
        "top_k": args.top_k,
    }
    print(f"配置 {p}，用例 {len(cases)} 条")

    rows = []
    t_start = time.time()
    for i, case in enumerate(cases, 1):
        try:
            r = retrieve(case["q_zh"], args.kb, key, p)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] ERR {e}", file=sys.stderr)
            continue
        sc = score_case(case, r["chunks"])
        rows.append({
            "case_id": case["id"], "category": case["category"],
            "question": case["q_zh"], "elapsed": r["elapsed"], **sc,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(cases)}  已用 {time.time()-t_start:.0f}s", flush=True)

    if not rows:
        print("无结果", file=sys.stderr)
        return 1

    n = len(rows)
    summary = {
        "tag": args.tag,
        "params": p,
        "n": n,
        "hit5": round(sum(1 for r in rows if r["hit5"]) / n, 4),
        "hit10": round(sum(1 for r in rows if r["hit10"]) / n, 4),
        "hit5_loose": round(sum(1 for r in rows if r["hit5_loose"]) / n, 4),
        "mrr": round(statistics.mean(r["rr"] for r in rows), 4),
        "has_any": round(sum(1 for r in rows if r["has_any"]) / n, 4),
        "avg_chunks": round(statistics.mean(r["n_chunks"] for r in rows), 2),
        "avg_elapsed": round(statistics.mean(r["elapsed"] for r in rows), 3),
    }
    # 分场景
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    summary["by_category"] = {
        cat: {
            "n": len(rs),
            "hit5": round(sum(1 for r in rs if r["hit5"]) / len(rs), 4),
        }
        for cat, rs in by_cat.items()
    }

    ts = datetime.now().strftime("%H%M%S")
    out = OUT_DIR / f"retrieval_{args.tag}_{ts}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print("\n===== 检索评测 =====")
    print(f"用例数             : {n}")
    print(f"Hit@5 (strict)     : {summary['hit5']:.4f}   ← 条文开头 + 全部关键词命中")
    print(f"Hit@10 (strict)    : {summary['hit10']:.4f}")
    print(f"Hit@5 (loose)      : {summary['hit5_loose']:.4f}   ← 条文开头 + 至少一个关键词")
    print(f"MRR (strict)       : {summary['mrr']:.4f}")
    print(f"有返回比例         : {summary['has_any']:.4f}")
    print(f"平均 chunk 数      : {summary['avg_chunks']}")
    print(f"平均耗时           : {summary['avg_elapsed']}s")
    print("\n按场景 Hit@5：")
    for cat, s in sorted(summary["by_category"].items(), key=lambda x: x[1]["hit5"]):
        print(f"  {cat:14s} n={s['n']:3d}  {s['hit5']:.3f}")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
