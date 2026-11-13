"""双路检索评测：原文 + 译文，合并后对比单路效果。

验证假设：跨语言场景下，先把查询译成英文再检索，能显著提升召回。

用法：
    python eval_dualpath.py --kb <id>
    python eval_dualpath.py --kb <id> --cases corpus/eval/cases_crosslang.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from ragflow_config import api_key, base_url  # noqa: E402

RAGFLOW = base_url()
GATEWAY = "http://localhost:3425/v1/chat/completions"
CASES_FILE = Path(__file__).resolve().parent.parent / "corpus" / "eval" / "test_cases.json"
OUT_DIR = CASES_FILE.parent

from eval_retrieval import article_key, score_case  # noqa: E402


def has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def translate(q: str, timeout: int = 60) -> str:
    body = {
        "model": "local-wb/cn:deepseek-v4.1-flash",
        "messages": [{
            "role": "user",
            "content": "Translate the following into precise English legal terminology. "
                       "Output only the translation, no explanation.\n\n" + q,
        }],
        "max_tokens": 300,
        "temperature": 0,
    }
    req = urllib.request.Request(
        GATEWAY, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d["choices"][0]["message"]["content"].strip()


def retrieve(q: str, kb: str, key: str, p: dict, timeout: int = 180) -> list[dict]:
    body = {
        "question": q, "dataset_ids": [kb],
        "top_k": p["top_k"], "top_n": p["top_n"],
        "similarity_threshold": p["sim_threshold"],
        "vector_similarity_weight": p["vec_weight"],
        "keyword": True,
    }
    req = urllib.request.Request(
        RAGFLOW + "/api/v1/retrieval",
        json.dumps(body, ensure_ascii=False).encode("utf-8"),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return (d.get("data") or {}).get("chunks") or []


def merge(a: list[dict], b: list[dict], top_n: int) -> list[dict]:
    """按内容去重合并两路结果，保留相似度高者。"""
    seen: dict[str, dict] = {}
    for c in a + b:
        k = c.get("content") or c.get("id") or ""
        prev = seen.get(k)
        if prev is None or (c.get("similarity") or 0) > (prev.get("similarity") or 0):
            seen[k] = c
    return sorted(seen.values(), key=lambda x: -(x.get("similarity") or 0))[:top_n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--cases", default=str(CASES_FILE))
    ap.add_argument("--tag", default="dual")
    ap.add_argument("--vec-weight", type=float, default=0.5)
    ap.add_argument("--sim-threshold", type=float, default=0.2)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=1024)
    args = ap.parse_args()

    key = api_key()
    p = {"vec_weight": args.vec_weight, "sim_threshold": args.sim_threshold,
         "top_n": args.top_n, "top_k": args.top_k}

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = [c for c in cases if c.get("expect") != "refuse"]
    # 只测需要翻译的（中文问句）
    cases = [c for c in cases if has_cjk(c["q_zh"])]
    print(f"用例 {len(cases)} 条（中文提问）")

    rows = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        q = case["q_zh"]
        try:
            orig = retrieve(q, args.kb, key, p)
            en = translate(q)
            trans = retrieve(en, args.kb, key, p)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] ERR {e}", file=sys.stderr)
            continue

        merged = merge(orig, trans, p["top_n"])
        so = score_case(case, orig)
        st = score_case(case, trans)
        sm = score_case(case, merged)

        rows.append({
            "case_id": case["id"], "category": case["category"],
            "question": q, "translated": en,
            "orig_hit5": so["hit5"], "trans_hit5": st["hit5"], "merged_hit5": sm["hit5"],
            "orig_rr": so["rr"], "trans_rr": st["rr"], "merged_rr": sm["rr"],
            "n_orig": len(orig), "n_trans": len(trans), "n_merged": len(merged),
            "top_articles_orig": so["top_articles"][:3],
            "top_articles_trans": st["top_articles"][:3],
            "top_articles_merged": sm["top_articles"][:3],
        })
        if i % 10 == 0:
            print(f"  {i}/{len(cases)}  {time.time()-t0:.0f}s", flush=True)

    n = len(rows)
    if not n:
        print("无结果", file=sys.stderr)
        return 1

    summary = {
        "tag": args.tag, "n": n, "params": p,
        "orig_hit5": round(sum(r["orig_hit5"] for r in rows) / n, 4),
        "trans_hit5": round(sum(r["trans_hit5"] for r in rows) / n, 4),
        "merged_hit5": round(sum(r["merged_hit5"] for r in rows) / n, 4),
        "orig_mrr": round(statistics.mean(r["orig_rr"] for r in rows), 4),
        "trans_mrr": round(statistics.mean(r["trans_rr"] for r in rows), 4),
        "merged_mrr": round(statistics.mean(r["merged_rr"] for r in rows), 4),
    }
    out = OUT_DIR / f"dualpath_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n===== 双路检索对比 =====")
    print(f"用例数        : {n}")
    print(f"{'':16s} {'Hit@5':>8s} {'MRR':>8s}")
    print(f"{'原文检索':16s} {summary['orig_hit5']:8.4f} {summary['orig_mrr']:8.4f}")
    print(f"{'译文检索':16s} {summary['trans_hit5']:8.4f} {summary['trans_mrr']:8.4f}")
    print(f"{'合并(双路)':14s} {summary['merged_hit5']:8.4f} {summary['merged_mrr']:8.4f}")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
