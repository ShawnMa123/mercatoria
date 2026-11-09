"""Small tokenizer for BM25 and explainable term matches.

No jieba dependency: CJK text is indexed as overlapping 2-grams, which is a
surprisingly strong baseline for short legal terms (完税价格, 原产地, 反倾销).
English/number tokens are regular words. The same tokenizer is used for
documents and queries, so scoring is symmetric.
"""

from __future__ import annotations

import re
from collections import Counter

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]+")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    terms: list[str] = []
    for m in WORD_RE.finditer(text):
        token = m.group(0).strip("._/-")
        if len(token) >= 2 or token.isdigit():
            terms.append(token)
    for m in CJK_RE.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            terms.append(run)
            continue
        for i in range(len(run) - 1):
            terms.append(run[i : i + 2])
    return terms


def term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def extract_matches(query: str, text: str, limit: int = 12) -> list[dict[str, object]]:
    """Return query tokens that occur literally in text, with first offsets."""
    lowered = text.lower()
    seen: set[str] = set()
    matches: list[dict[str, object]] = []
    # Prefer longer terms first so a 3-gram is reported instead of its 2-grams.
    for term in sorted(set(tokenize(query)), key=lambda t: (-len(t), t)):
        if term in seen:
            continue
        pos = lowered.find(term)
        if pos < 0:
            continue
        seen.add(term)
        matches.append({"term": term, "start": pos, "end": pos + len(term)})
        if len(matches) >= limit:
            break
    matches.sort(key=lambda m: (int(m["start"]), -int(m["end"])))
    return matches


def query_terms(query: str, max_terms: int = 96) -> list[str]:
    """Deduplicated terms in first-seen order, capped for SQL comfort."""
    seen: set[str] = set()
    out: list[str] = []
    for term in tokenize(query):
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out
