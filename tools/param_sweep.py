"""RAGFlow 检索参数对比实验。

在固定用例集上切换检索参数，观察召回质量变化。

用法：
    # 默认参数基线
    python param_sweep.py --kb <id> --limit 60 --tag baseline

    # 切换向量权重
    python param_sweep.py --kb <id> --limit 60 --vec-weight 0.7 --tag vecw70

    # 批量扫（会自动串行跑完所有配置）
    python param_sweep.py --kb <id> --limit 60 --sweep

可扫的参数：
- --vec-weight      向量相似度权重（默认 0.3，即关键词占 0.7）
- --sim-threshold   相似度阈值（默认 0.2）
- --top-n           送入 LLM 的 chunk 数（默认 8）
- --top-k           候选池大小（默认 1024）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from ragflow_config import api_key, base_url  # noqa: E402

BASE = base_url()
CASES_FILE = Path(__file__).resolve().parent.parent / "corpus" / "eval" / "test_cases.json"
OUT_DIR = CASES_FILE.parent


def call(method: str, path: str, key: str, body: dict | None = None, timeout: int = 300) -> dict:
    data = None
    headers = {"Authorization": "Bearer " + key}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data, headers, method=method)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def make_chat(kb: str, key: str, p: dict, name: str) -> str:
    body = {
        "name": name,
        "dataset_ids": [kb],
        "llm_id": "local-wb/cn:deepseek-v4.1-flash@local-gw@OpenAI-API-Compatible",
        "llm_setting": {"temperature": 0.1, "top_p": 0.3, "max_tokens": 2048, "thinking": False},
        "prompt_config": {"empty_response": "未找到相关的法规依据。"},
        "similarity_threshold": p["sim_threshold"],
        "vector_similarity_weight": p["vec_weight"],
        "top_n": p["top_n"],
        "top_k": p["top_k"],
    }
    d = call("POST", "/api/v1/chats", key, body)
    if d.get("code") != 0:
        raise RuntimeError(f"建助理失败: {d.get('message')}")
    return d["data"]["id"]


def ask(chat_id: str, q: str, key: str) -> dict:
    t0 = time.time()
    d = call(
        "POST", "/api/v1/chat/completions", key,
        {"chat_id": chat_id, "stream": False, "messages": [{"role": "user", "content": q}]},
    )
    data = d.get("data") or {}
    return {
        "answer": data.get("answer") or "",
        "elapsed": round(time.time() - t0, 2),
        "n_refs": len((data.get("reference") or {}).get("chunks") or []),
    }


def kw_score(case: dict, ans: str) -> float | None:
    kws = case.get("must_mention") or []
    if not kws:
        return None
    hit = sum(1 for k in kws if k.lower() in ans.lower())
    return hit / len(kws)


def run_config(kb: str, key: str, cases: list[dict], p: dict, tag: str) -> dict:
    ts = datetime.now().strftime("%H%M%S")
    chat = make_chat(kb, key, p, f"sweep-{tag}-{ts}")
    rows = []
    for i, case in enumerate(cases, 1):
        try:
            r = ask(chat, case["q_zh"], key)
        except Exception as e:  # noqa: BLE001
            print(f"    [{i}] ERR {e}", file=sys.stderr)
            continue
        ks = kw_score(case, r["answer"])
        rows.append({**r, "case_id": case["id"], "category": case["category"], "kw": ks})
        if i % 20 == 0:
            print(f"    {i}/{len(cases)}", flush=True)

    scored = [r for r in rows if r["kw"] is not None]
    kws = [r["kw"] for r in scored]
    summary = {
        "tag": tag,
        "params": p,
        "n": len(rows),
        "kw_mean": round(sum(kws) / len(kws), 4) if kws else 0,
        "kw_full": sum(1 for k in kws if k >= 1.0),
        "kw_zero": sum(1 for k in kws if k == 0),
        "avg_refs": round(sum(r["n_refs"] for r in rows) / len(rows), 2) if rows else 0,
        "avg_elapsed": round(sum(r["elapsed"] for r in rows) / len(rows), 2) if rows else 0,
    }
    out = OUT_DIR / f"sweep_{tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


CONFIGS = [
    ("baseline",       {"vec_weight": 0.3, "sim_threshold": 0.2, "top_n": 8, "top_k": 1024}),
    ("vecw50",         {"vec_weight": 0.5, "sim_threshold": 0.2, "top_n": 8, "top_k": 1024}),
    ("vecw70",         {"vec_weight": 0.7, "sim_threshold": 0.2, "top_n": 8, "top_k": 1024}),
    ("sim010",         {"vec_weight": 0.3, "sim_threshold": 0.1, "top_n": 8, "top_k": 1024}),
    ("topn20",         {"vec_weight": 0.3, "sim_threshold": 0.2, "top_n": 20, "top_k": 1024}),
    ("topk256",        {"vec_weight": 0.3, "sim_threshold": 0.2, "top_n": 8, "top_k": 256}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--vec-weight", type=float, default=0.3)
    ap.add_argument("--sim-threshold", type=float, default=0.2)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=1024)
    ap.add_argument("--sweep", action="store_true", help="跑完所有预置配置")
    args = ap.parse_args()

    key = api_key()
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))

    # 分层抽样：每个场景都取到，且总额受 limit 控制
    if args.limit and args.limit < len(cases):
        by_cat: dict[str, list[dict]] = {}
        for c in cases:
            by_cat.setdefault(c["category"], []).append(c)
        per = max(1, args.limit // len(by_cat))
        picked = []
        for cat, items in by_cat.items():
            picked.extend(items[:per])
        cases = picked[: args.limit]
        print(f"分层抽样 {len(cases)} 条，覆盖 {len(by_cat)} 个场景")

    configs = CONFIGS if args.sweep else [
        (args.tag, {
            "vec_weight": args.vec_weight,
            "sim_threshold": args.sim_threshold,
            "top_n": args.top_n,
            "top_k": args.top_k,
        })
    ]

    results = []
    for tag, p in configs:
        print(f"\n=== {tag}  {p} ===")
        s = run_config(args.kb, key, cases, p, tag)
        results.append(s)
        print(f"  kw_mean={s['kw_mean']}  full={s['kw_full']}  zero={s['kw_zero']}  "
              f"refs={s['avg_refs']}  {s['avg_elapsed']}s")

    print("\n===== 对比 =====")
    print(f"{'tag':12s} {'vec_w':>6s} {'sim':>5s} {'top_n':>6s} {'kw_mean':>8s} {'full':>5s} {'zero':>5s} {'elapsed':>8s}")
    for s in sorted(results, key=lambda x: -x["kw_mean"]):
        p = s["params"]
        print(f"{s['tag']:12s} {p['vec_weight']:6.1f} {p['sim_threshold']:5.1f} "
              f"{p['top_n']:6d} {s['kw_mean']:8.3f} {s['kw_full']:5d} {s['kw_zero']:5d} {s['avg_elapsed']:8.1f}")

    (OUT_DIR / "sweep_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n汇总写入 {OUT_DIR / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
