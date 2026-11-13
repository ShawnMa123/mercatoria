"""Generate ~1000 grounded questions from corpus articles.

The 60 externally-sourced questions stay first; the rest are template questions
built from actual article text so chunking experiments have broad coverage.
Generated rows are marked ``synthetic: true`` and keep the exact article_id.

Run inside the test container (corpus is copied there) or on the host:
  python tests/eval_generate_questions.py --out tests/eval/questions_1000.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from mercatoria.chunker import ChunkingConfig, chunk_document
from mercatoria.manifest import load_manifest
from mercatoria.tokenizer import tokenize

HERE = Path(__file__).resolve().parent
CJK_STOP = {
    "法律", "法规", "规定", "应当", "可以", "必须", "不得", "以及", "或者",
    "对于", "进行", "按照", "有关", "其他", "本条", "本法", "条例", "办法",
    "中国", "人民", "共和国", "国家", "国务院", "海关", "货物", "进口", "出口",
    "为了", "有下", "中华", "下列", "之一", "情形", "要求", "处理", "工作",
    "单位", "个人", "企业", "根据", "适用", "管理", "实施", "依法", "负责",
}
EN_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "shall", "section",
    "part", "under", "such", "other", "which", "have", "been", "into", "their",
    "any", "may", "not", "are", "was", "were", "will", "would", "must",
    "article", "paragraph", "subpart", "means", "person", "provide", "provides",
    "applies", "commencement", "provision", "general", "scope", "definitions",
    "each", "fixed", "cited", "title", "short", "cent", "per", "day",
}
NUMERAL = __import__("re").compile(r"^[第]?[零〇一二三四五六七八九十百千0-9]+[条章节款项]?$")


def _pick(terms: list[str], df: Counter[str], total: int, limit: int) -> list[str]:
    scored = []
    for term in set(terms):
        if term in CJK_STOP or term in EN_STOP or NUMERAL.match(term):
            continue
        freq = df.get(term, 0)
        if freq < 2 or freq > max(3, int(total * 0.25)):
            continue
        scored.append((freq, -len(term), term))
    scored.sort()
    return [term for _freq, _len, term in scored[:limit]]


def _keywords(
    text: str, language: str, df: Counter[str], total: int, limit: int = 3
) -> list[str]:
    if language == "en":
        words = [w for w in tokenize(text) if len(w) >= 4]
    else:
        try:
            import jieba.analyse

            words = [
                term
                for term in jieba.analyse.extract_tags(text, topK=12)
                if len(term) >= 2
            ]
        except ImportError:
            words = [t for t in tokenize(text) if len(t) == 2]
    return _pick(words, df, total, limit)


def _question(source: str, label: str, keywords: list[str], language: str, n: int) -> str:
    kw = "、".join(keywords) if language != "en" else " / ".join(keywords)
    if language == "en":
        templates = [
            f"What does {source} {label} provide about {kw}?",
            f"Under {source}, what is required for {kw}?",
            f"Which rules in {source} apply to {kw}?",
        ]
    else:
        templates = [
            f"{source}{label}对{kw}是怎么规定的？",
            f"根据{source}，关于{kw}应当如何处理？",
            f"{label}中关于{kw}的要求是什么？",
        ]
    return templates[n % len(templates)]


def generate(corpus_dir: Path, manifest_path: Path, target: int) -> list[dict]:
    manifest = load_manifest(manifest_path)
    cfg = ChunkingConfig()
    per_doc: dict[str, list[dict]] = defaultdict(list)
    all_records: list[tuple[dict, dict]] = []
    for doc in manifest["docs"]:
        path = corpus_dir / doc["path"]
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for rec in chunk_document(doc, raw, cfg):
            if rec.article_id.endswith((":doc", ":preamble")):
                continue
            if len(rec.text) < 80:
                continue
            item = {
                "doc": doc,
                "article_id": rec.article_id,
                "article_label": rec.article_label,
                "text": rec.text,
            }
            per_doc[doc["doc_id"]].append(item)
            all_records.append((doc, item))

    df: Counter[str] = Counter()
    for doc, item in all_records:
        language = doc.get("language") or "zh"
        terms = set(tokenize(item["text"])) if language != "en" else set(
            w for w in tokenize(item["text"]) if len(w) >= 4
        )
        df.update(terms)
    total_chunks = max(1, len(all_records))

    groups: dict[str, list[list[dict]]] = defaultdict(list)
    for doc_id, queue in sorted(per_doc.items()):
        jurisdiction = queue[0]["doc"].get("jurisdiction", "OTHER")
        groups[jurisdiction].append(queue)

    weights = {"CN": 0.38, "US": 0.25, "WTO": 0.14, "AU": 0.08, "JP": 0.07, "INT": 0.08}
    quotas = {
        jurisdiction: int(target * weight)
        for jurisdiction, weight in weights.items()
        if jurisdiction in groups
    }

    def take(jurisdiction: str, count: int) -> list[dict]:
        doc_queues = groups[jurisdiction]
        out: list[dict] = []
        while len(out) < count and any(doc_queues):
            for queue in doc_queues:
                if queue and len(out) < count:
                    out.append(queue.pop(0))
        return out

    selected: list[dict] = []
    for jurisdiction, quota in quotas.items():
        selected.extend(take(jurisdiction, quota))
    if len(selected) < target:
        leftovers: list[dict] = []
        for doc_queues in groups.values():
            for queue in doc_queues:
                leftovers.extend(queue)
                queue.clear()
        selected.extend(leftovers[: target - len(selected)])

    rows: list[dict] = []
    for item in selected[:target]:
        doc = item["doc"]
        language = doc.get("language") or "zh"
        keywords = _keywords(item["text"], language, df, total_chunks)
        if len(keywords) < 2:
            fallback = [
                t for t in tokenize(item["text"])
                if len(t) >= 2 and t not in CJK_STOP and t not in EN_STOP and not NUMERAL.match(t)
            ]
            keywords = list(dict.fromkeys(keywords + fallback))[:3]
        if len(keywords) < 2:
            continue
        rows.append(
            {
                "id": f"SYN-{len(rows) + 1:04d}",
                "category": (doc.get("applies_to") or [doc.get("jurisdiction", "other")])[0],
                "language": language,
                "question": _question(
                    doc.get("source_doc") or doc["doc_id"],
                    item["article_label"],
                    keywords,
                    language,
                    len(rows),
                ),
                "expected_doc_ids": [doc["doc_id"]],
                "expected_article_ids": [item["article_id"]],
                "expected_keywords": keywords,
                "source": "synthetic: corpus article",
                "source_url": doc["path"],
                "synthetic": True,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--manifest", default="corpus/manifest.json")
    ap.add_argument("--real", default=str(HERE / "eval" / "trade_law_questions.json"))
    ap.add_argument("--out", default=str(HERE / "eval" / "questions_1000.json"))
    ap.add_argument("--total", type=int, default=1000)
    args = ap.parse_args()

    real = json.loads(Path(args.real).read_text(encoding="utf-8"))["questions"]
    target = max(0, args.total - len(real))
    synthetic = generate(Path(args.corpus), Path(args.manifest), target)
    rows = real + synthetic
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"version": "2026-11-13", "note": f"{len(real)} real + {len(synthetic)} synthetic", "questions": rows},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"real": len(real), "synthetic": len(synthetic), "total": len(rows), "path": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
