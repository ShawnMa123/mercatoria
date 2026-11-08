"""对 RAGFlow 知识库执行测试用例，并对比「思考开/关」的差异。

用法：
    # 冒烟测试（5 条）
    python run_eval.py --kb <id> --limit 5

    # 全量 + 思考开关对比
    python run_eval.py --kb <id> --full --thinking-compare

    # 只跑某一类
    python run_eval.py --kb <id> --category 出口管制

输出：
    corpus/eval/results_<ts>.jsonl     每条结果一行
    corpus/eval/summary_<ts>.json      汇总统计

设计要点：
- RAGFlow 的 chat API 用 `llm_setting.thinking` 控制思考开关
- 评测不依赖 LLM 判分，用「引用来源命中 + 关键词覆盖 + 拒答正确性」三个客观指标
- 拒答题判定：答案里出现「未找到 / 没有相关 / 无法回答 / not found / cannot answer」等
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from ragflow_config import api_key  # noqa: E402

BASE = "http://localhost:9380"
CASES_FILE = Path(__file__).resolve().parent.parent / "corpus" / "eval" / "test_cases.json"
OUT_DIR = CASES_FILE.parent

# 拒答标记：覆盖中英日三种表达。
# 注意 must 匹配「整段答案」，不能只看开头 —— 实测很多拒答
# 会先复述检索到的内容，再说明「没有相关规定」。
REFUSE_MARKERS = [
    # 中文
    "未找到", "没有找到", "没有相关", "无法回答", "无法确定", "不在本数据",
    "未涵盖", "超出", "没有提及", "无相关", "资料中未", "知识库中未",
    "没有任何相关", "并不涉及", "没有涉及", "不涉及", "没有对应内容",
    "均与", "无关", "未收录", "不包含",
    # 英文
    "not found", "no relevant", "cannot answer", "unable to", "not covered",
    "not mentioned", "out of scope", "no information", "does not contain",
    "do not contain", "does not include", "no content regarding",
    "not in the dataset", "not provided", "outside the scope",
    # 日文
    "含まれ", "該当", "範囲外",
]

# 各法域的来源文件名特征，用于判断「引用是否来自期望法域」
JURIS_HINT = {
    "cn-laws": ["海关法", "关税法", "对外贸易法", "出口管制法", "条例", "办法", "规定"],
    "ecfr": ["part-", "CFR", "§"],
    "wto": ["Article", "tfa", "gatt", "sps", "tbt", "adp", "cv_", "ro_", "lic_", "scm"],
    "uncitral": ["Article", "CISG"],
    "au": ["Section", "Customs Act", "Export Control Act"],
    "jp": ["第", "条", "関税", "輸出"],
}


def post(path: str, body: dict, key: str, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        BASE + path,
        json.dumps(body, ensure_ascii=False).encode("utf-8"),
        {"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def get(path: str, key: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def create_chat(kb: str, key: str, thinking: bool, name: str) -> str:
    """建一个专用问答助理（thinking 开关在建助理时确定）。"""
    body = {
        "name": name,
        "dataset_ids": [kb],
        "llm_id": "local-wb/cn:deepseek-v4.1-flash@local-gw@OpenAI-API-Compatible",
        "llm_setting": {
            "temperature": 0.1,
            "top_p": 0.3,
            # 思考模式下 reasoning 内容也计入 max_tokens。
            # 2048 会被思考过程吃光导致 completion_tokens=0、answer 为空。
            "max_tokens": 8192 if thinking else 2048,
            "thinking": thinking,
        },
        "prompt_config": {
            "empty_response": "未找到相关的法规依据。（检索无结果）",
            "show_quote": True,
        },
        "similarity_threshold": 0.2,
        "top_n": 8,
        "top_k": 1024,
    }
    d = post("/api/v1/chats", body, key)
    if d.get("code") != 0:
        raise RuntimeError(f"建助理失败: {d.get('message')}")
    return d["data"]["id"]


def ask(chat_id: str, question: str, key: str) -> dict:
    t0 = time.time()
    d = post(
        "/api/v1/chat/completions",
        {"chat_id": chat_id, "stream": False, "messages": [{"role": "user", "content": question}]},
        key,
    )
    dt = time.time() - t0
    data = d.get("data") or {}
    refs = (data.get("reference") or {}).get("chunks") or []
    return {
        "answer": data.get("answer") or "",
        "elapsed": round(dt, 2),
        "refs": [
            {
                "doc": c.get("document_keyword") or c.get("docnm_kwd") or "",
                "content": (c.get("content") or "")[:300],
                "similarity": c.get("similarity"),
            }
            for c in refs
        ],
    }


# RAGFlow 的兜底文案。实测 326 条答案含此句，其中 62% 是**追加在正常答案之后**的：
# 只要检索结果中有一个 chunk 低于阈值它就追加，与答案质量无关。
# 必须先剔除再判定拒答，否则会把大量正确回答误判成拒答。
RAGFLOW_FALLBACK = re.compile(
    r"the answer you are looking for is not found in the dataset!?",
    re.I,
)


def strip_fallback(answer: str) -> str:
    """去掉 RAGFlow 自动追加的兜底文案。"""
    return RAGFLOW_FALLBACK.sub("", answer).strip()


def is_refusal(answer: str) -> bool:
    """判定是否拒答（语义层面的拒答，不含系统兜底）。

    实测三个坑，都踩过：
    1. 只看开头 200 字符 → 漏判（拒答常先复述检索内容再说「没有相关规定」）
    2. 不剔除 RAGFlow 兜底文案 → 大量误判：62% 的正确回答后面被追加了
       "The answer you are looking for is not found in the dataset!"
    3. **最关键**：关思考时很多答案剔除兜底后是**完全空的**（模型一个字没输出），
       看起来像「拒答」，实际是生成失败。这类必须单独统计为 empty，
       不能算作拒答能力。

    因此本函数只判「模型自己写出的拒答」。
    """
    a = strip_fallback(answer).lower()
    if not a.strip():
        return False  # 空答案 ≠ 拒答，由 empty_answer 单独统计
    hits = sum(1 for m in REFUSE_MARKERS if m.lower() in a)
    return hits >= 1 and len(a) >= 20


# 「部分回答」标记：模型给出了相关内容，同时声明某一部分语料没有
PARTIAL_MARKERS = [
    "没有明确规定", "未规定", "没有直接规定", "未明确", "没有提到",
    "无法确认", "没有明确", "未作规定", "没有具体规定",
    "does not specify", "does not provide", "no specific",
]


def is_partial(answer: str) -> bool:
    """模型部分作答并说明缺口 —— 与「完全找不到」是两回事。"""
    a = strip_fallback(answer).lower()
    return any(m.lower() in a for m in PARTIAL_MARKERS)


def score(case: dict, res: dict) -> dict:
    """客观打分，不用 LLM 判分。"""
    ans = res["answer"]
    refs = res["refs"]

    # 1) 拒答正确性
    refused = is_refusal(ans)
    if case["expect"] == "refuse":
        return {
            "refuse_correct": refused,
            "keyword_hit": None,
            "has_refs": len(refs) > 0,
            "n_refs": len(refs),
            "elapsed": res["elapsed"],
        }

    # 2) 关键词覆盖
    kws = case.get("must_mention") or []
    hit = [k for k in kws if k.lower() in ans.lower()]
    kw_rate = len(hit) / len(kws) if kws else 0.0

    # 3) 引用是否存在
    #    三种「未答全」要分开统计：
    #    - full_refuse   : 完全拒答但引用了内容 → 检索或生成有问题
    #    - partial       : 部分作答 + 显式声明缺口 → 语料确实缺，属正常
    #    - answered      : 正常作答
    # 空答案判定必须剔除 RAGFlow 兜底文案 —— 否则「模型没输出、
    # 系统补了一句 not found」会被当成正常拒答，掩盖生成失败。
    clean = strip_fallback(ans)
    full_refuse = is_refusal(ans) and not is_partial(ans)
    return {
        "refuse_correct": None,
        "empty_answer": len(clean.strip()) < 10,
        "fallback_only": len(clean.strip()) < 10 and RAGFLOW_FALLBACK.search(ans or "") is not None,
        "full_refuse_with_refs": full_refuse and len(refs) > 0,
        "partial_with_gap": is_partial(ans),
        "keyword_hit": round(kw_rate, 3),
        "kw_hit_list": hit,
        "kw_miss_list": [k for k in kws if k not in hit],
        "has_refs": len(refs) > 0,
        "n_refs": len(refs),
        "elapsed": res["elapsed"],
        "ref_docs": sorted({r["doc"] for r in refs})[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    ap.add_argument("--full", action="store_true", help="跑全部")
    ap.add_argument("--category", default=None, help="只跑某类")
    ap.add_argument("--thinking-compare", action="store_true", help="同时跑思考开/关")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    args = ap.parse_args()

    key = api_key()
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not args.full and not args.limit:
        print("需要 --limit N 或 --full", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    modes = [True, False] if args.thinking_compare else [False]

    chats: dict[bool, str] = {}
    for t in modes:
        chats[t] = create_chat(
            args.kb, key, t, f"eval-{'thinking' if t else 'nothink'}-{ts}"
        )
        print(f"助理[thinking={t}]: {chats[t]}")

    out = OUT_DIR / f"results_{ts}.jsonl"
    fh = out.open("w", encoding="utf-8")

    summary: dict = {"kb": args.kb, "n_cases": len(cases), "modes": {}, "ts": ts}
    for t in modes:
        summary["modes"][str(t)] = {
            "n": 0, "refuse_ok": 0, "refuse_total": 0,
            "kw_sum": 0.0, "kw_n": 0, "with_refs": 0, "elapsed_sum": 0.0,
            "empty": 0, "truncated_refuse": 0, "partial": 0,
        }

    t_start = time.time()
    for i, case in enumerate(cases, 1):
        q = case["q_zh"] if args.lang == "zh" else case["q_en"]
        for t in modes:
            try:
                res = ask(chats[t], q, key)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(cases)}] thinking={t} ERR {e}", file=sys.stderr)
                continue
            sc = score(case, res)
            sc["elapsed"] = res["elapsed"]
            row = {
                "i": i, "case_id": case["id"], "category": case["category"],
                "question": q, "thinking": t,
                "answer": res["answer"], "score": sc,
                "elapsed": res["elapsed"],
                "refs": res["refs"],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

            s = summary["modes"][str(t)]
            s["n"] += 1
            s["elapsed_sum"] += res["elapsed"]
            if sc["refuse_correct"] is not None:
                s["refuse_total"] += 1
                s["refuse_ok"] += 1 if sc["refuse_correct"] else 0
            if sc["keyword_hit"] is not None:
                s["kw_sum"] += sc["keyword_hit"]
                s["kw_n"] += 1
            if sc["has_refs"]:
                s["with_refs"] += 1
            if sc.get("empty_answer"):
                s["empty"] += 1
            if sc.get("partial_with_gap"):
                s["partial"] += 1
            # 完全拒答却有引用 = 把「有依据但没匹配好」误报成「无结果」
            if case["expect"] != "refuse" and sc.get("full_refuse_with_refs"):
                s["truncated_refuse"] += 1

        if i % 10 == 0:
            el = time.time() - t_start
            print(f"  {i}/{len(cases)}  已用 {el:.0f}s", flush=True)

    fh.close()

    for t in modes:
        s = summary["modes"][str(t)]
        if s["n"]:
            s["avg_elapsed"] = round(s["elapsed_sum"] / s["n"], 2)
            s["keyword_rate"] = round(s["kw_sum"] / s["kw_n"], 4) if s["kw_n"] else None
            s["refuse_rate"] = (
                round(s["refuse_ok"] / s["refuse_total"], 4) if s["refuse_total"] else None
            )
            s["ref_hit_rate"] = round(s["with_refs"] / s["n"], 4)

    summ_path = OUT_DIR / f"summary_{ts}.json"
    summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 汇总 =====")
    for t in modes:
        s = summary["modes"][str(t)]
        print(f"\nthinking={t}  ({s['n']} 条)")
        print(f"  关键词覆盖率 : {s.get('keyword_rate')}")
        print(f"  有引用比例   : {s.get('ref_hit_rate')}")
        print(f"  拒答正确率   : {s.get('refuse_rate')}")
        print(f"  平均耗时     : {s.get('avg_elapsed')}s")
        print(f"  空答案       : {s.get('empty')}")
        print(f"  部分作答+声明缺口 : {s.get('partial')}  ← 语料确实缺，正常")
        print(f"  有引用却完全拒答  : {s.get('truncated_refuse')}  ← 越少越好")
    print(f"\n明细: {out}")
    print(f"汇总: {summ_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
