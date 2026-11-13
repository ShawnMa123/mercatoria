"""Read-only audit of the trade-law fixture against the current index.

For every question it retrieves the top-5 chunks exactly like
``tests/eval_trade_questions.py`` does and reports what came back, whether the
expected doc was hit, which keywords were present, and whether the row's
``corpus_gap`` flag agrees with reality.

Usage:
    python tools/audit_fixture.py [--index data/index/xxx.db] [--out docs/fixture-audit-<date>.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(ROOT))

from mercatoria.config import load_config  # noqa: E402
from mercatoria.retrieve import Retriever  # noqa: E402
from mercatoria.store import open_current, open_index  # noqa: E402

FIXTURE = ROOT / "tests" / "eval" / "trade_law_questions.json"


def _keywords(text: str, keywords: list[str]) -> list[str]:
    body = text.lower()
    return [kw for kw in keywords if str(kw).lower() in body]


def audit(cases: list[dict], retriever: Retriever, top_k: int = 5) -> list[dict]:
    rows: list[dict] = []
    for case in cases:
        expected_docs = {str(d) for d in case.get("expected_doc_ids") or []}
        keywords = [str(k) for k in case.get("expected_keywords") or []]
        result = retriever.retrieve([case["question"]], top_k=top_k)
        chunks = result["chunks"]
        rank = 0
        for index, chunk in enumerate(chunks, 1):
            if str(chunk.get("doc_id")) in expected_docs and _keywords(
                chunk.get("text") or "", keywords
            ):
                rank = index
                break
        best = chunks[0] if chunks else None
        rows.append(
            {
                "id": case["id"],
                "type": case.get("type", "answer"),
                "question": case["question"],
                "corpus_gap": bool(case.get("corpus_gap")),
                "coverage_note": case.get("coverage_note") or "",
                "expected_docs": sorted(expected_docs),
                "expected_keywords": keywords,
                "rank": rank,
                "hit": rank > 0,
                "confidence": (result.get("diagnostics") or {}).get("confidence"),
                "top": [
                    {
                        "rank": i,
                        "doc_id": chunk.get("doc_id"),
                        "article_id": (chunk.get("article_ids") or [None])[0],
                        "article_label": (chunk.get("articles") or [None])[0],
                        "doc_matches_expected": str(chunk.get("doc_id")) in expected_docs,
                        "keywords": _keywords(chunk.get("text") or "", keywords),
                    }
                    for i, chunk in enumerate(chunks, 1)
                ],
                "top1_has_keywords": bool(
                    best and _keywords(best.get("text") or "", keywords)
                ),
            }
        )
    return rows


def render(rows: list[dict], meta: dict) -> str:
    answered = [row for row in rows if row.get("type") != "no_answer"]
    no_answer = [row for row in rows if row.get("type") == "no_answer"]
    lines = [
        "# Fixture audit",
        "",
        f"- 生成日期：{meta['date']}",
        f"- 索引：`{meta['index']}`",
        f"- 题数：{len(rows)}",
        "- 命中判定与 `tests/eval_trade_questions.py` 一致：top-5 内同时命中 expected doc 与全部 expected keywords。",
        "",
        "## 汇总",
        "",
        f"- 有答案题 Hit@5：**{sum(1 for r in answered if r['hit']) / len(answered):.4f}**"
        f"（n={len(answered)}）",
        f"- `no_answer` 题：{len(no_answer)}，"
        f"低置信率 {sum(1 for r in no_answer if r['confidence'] == 'low') / len(no_answer):.4f}",
        f"- `corpus_gap=true` 的题：{', '.join(r['id'] for r in rows if r['corpus_gap']) or '无'}",
    ]
    mismatched = [r for r in answered if r["corpus_gap"] and r["hit"]]
    if mismatched:
        lines.append(
            "- ⚠ 标记为 gap 但已命中：" + ", ".join(r["id"] for r in mismatched)
        )
    gaps_unflagged = [r for r in answered if not r["corpus_gap"] and not r["hit"]]
    if gaps_unflagged:
        lines.append(
            "- ⚠ 未标记 gap 但未命中：" + ", ".join(r["id"] for r in gaps_unflagged)
        )
    lines.extend(["", "## 逐题 top-5", "", "| id | 命中 | rank | conf | gap | top-5 (doc:article) |", "|---|---|---|---|---|---|"])
    for row in answered + no_answer:
        top = "<br>".join(
            f"{item['rank']}. `{item['doc_id']}:{item['article_id']}`"
            + (f" ✔{','.join(item['keywords'])}" if item["keywords"] else "")
            for item in row["top"]
        )
        lines.append(
            f"| {row['id']} | {'✅' if row['hit'] else '❌'} | {row['rank'] or '-'} | "
            f"{row['confidence'] or '-'} | {row['corpus_gap']} | {top} |"
        )
    lines.extend(["", "## 期望与命中细节", ""])
    for row in rows:
        lines.append(f"### {row['id']}")
        lines.append("")
        lines.append(f"- 问题：{row['question']}")
        lines.append(f"- expected_doc_ids：`{row['expected_docs']}`")
        lines.append(f"- expected_keywords：`{row['expected_keywords']}`")
        lines.append(f"- 命中：{'是' if row['hit'] else '否'}（rank={row['rank'] or '-'}）")
        if row["coverage_note"]:
            lines.append(f"- coverage_note：{row['coverage_note']}")
        for item in row["top"]:
            flag = "doc✔" if item["doc_matches_expected"] else "doc✘"
            kw = ",".join(item["keywords"]) or "-"
            lines.append(
                f"  - {item['rank']}. {flag} `{item['doc_id']}` / `{item['article_id']}` "
                f"kw: {kw}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fixture", default=str(FIXTURE))
    args = ap.parse_args()

    cfg = load_config(args.config)
    index = open_index(args.index) if args.index else open_current(cfg.resolved_index_dir())
    retriever = Retriever(index, cfg)
    data = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    rows = audit(data["questions"], retriever)
    out = Path(args.out) if args.out else ROOT / "docs" / f"fixture-audit-{date.today().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, {"date": date.today().isoformat(), "index": str(index.path)}), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
