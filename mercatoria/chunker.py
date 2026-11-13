"""Deterministic, configurable legal-text chunker.

The corpus is markdown with a deliberately simple structure:

* Chinese/Japanese laws use ``## 第一章`` for chapters and ``第X条`` lines for
  articles.
* eCFR uses ``## § 101.1`` for sections.
* WTO/CISG use ``## Article N``.
* AU uses ``## Section N``.

The chunker turns those headings into article-level chunks. It never
summarizes: ``text`` is a slice of the source body (HTML comments removed when
configured). ``chunk_id`` is content-derived, so a rebuild can be compared and
cited deterministically.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .config import ChunkingConfig

RE_HEADING = re.compile(r"^\s*(#{2,6})\s+(.*\S)\s*$")
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
RE_CN_ARTICLE = re.compile(
    r"^\s*第([零〇一二三四五六七八九十百千万0-9]+)条(?:[之の]([零〇一二三四五六七八九十0-9]+))?"
)
RE_SECTION = re.compile(r"^§\s*([0-9]+(?:\.[0-9]+)*[A-Za-z]?)\b")
RE_ARTICLE = re.compile(r"^Article\s+([0-9]+[A-Za-z]?|[IVXLCDM]+)\b", re.I)
RE_SECTION_EN = re.compile(r"^Section\s+([0-9]+[A-Za-z]?)\b", re.I)

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    article_id: str
    article_label: str
    text: str
    section: str
    part: int
    ordinal: int
    start: int
    end: int


def cn_to_int(value: str) -> int | None:
    """Convert a Chinese numeral (or Arabic digits) to an integer."""
    if value.isdigit():
        return int(value)
    total = 0
    section = 0
    number = 0
    for ch in value:
        if ch in CN_DIGITS:
            number = CN_DIGITS[ch]
        elif ch in CN_UNITS:
            unit = CN_UNITS[ch]
            if unit == 10000:
                total += (section + number) * unit
                section = 0
                number = 0
            else:
                section += (number or 1) * unit
                number = 0
        else:
            return None
    return total + section + number


def normalize_for_id(text: str) -> str:
    """NFC + collapse whitespace + strip. Used only for chunk_id."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_chunk_id(doc_id: str, article_id: str, text: str) -> str:
    payload = doc_id + "\x1f" + article_id + "\x1f" + normalize_for_id(text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def strip_comments(text: str) -> str:
    return RE_HTML_COMMENT.sub("", text)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_text(text: str, max_chars: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split a long article at sentence boundaries, returning (slice, start, end)."""
    if max_chars <= 0 or len(text) <= max_chars:
        start, end = _trim_span(text, 0, len(text))
        return [(text[start:end], start, end)] if start < end else []

    delimiters = set("。！？；\n.?!;")
    parts: list[tuple[str, int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        limit = min(start + max_chars, n)
        end = limit
        if limit < n:
            floor = start + max_chars // 2
            for i in range(limit, floor, -1):
                if text[i - 1] in delimiters:
                    end = i
                    break
        s, e = _trim_span(text, start, end)
        if s < e:
            parts.append((text[s:e], s, e))
        if end >= n:
            break
        next_start = max(end - max(0, overlap), start + 1)
        start = next_start
    return parts


def _article_from_label(
    label: str, doc_id: str, family: str = ""
) -> tuple[str, str] | None:
    """Return (article_id, normalized label) or None if not an article.

    ``family`` is the document jurisdiction. It prevents structural headings
    such as Japanese ``### Section 1`` or WTO ``## Section I`` from being
    mistaken for legal articles.
    """
    m = RE_CN_ARTICLE.match(label)
    if m:
        number = cn_to_int(m.group(1))
        if number is None:
            return None
        suffix = m.group(2)
        article_id = f"{doc_id}:{number}"
        if suffix:
            suffix_num = cn_to_int(suffix)
            article_id += f".{suffix_num if suffix_num is not None else suffix}"
        return article_id, m.group(0).strip()

    if family in {"CN", "JP"}:
        return None
    if family == "US":
        m = RE_SECTION.match(label)
        if m:
            return f"{doc_id}:{m.group(1)}", f"§ {m.group(1)}"
        return None
    if family == "AU":
        m = RE_SECTION_EN.match(label)
        if m:
            return f"{doc_id}:sec-{m.group(1)}", f"Section {m.group(1)}"
        return None
    if family in {"WTO", "INT"}:
        m = RE_ARTICLE.match(label)
        if m:
            return f"{doc_id}:art-{m.group(1)}", f"Article {m.group(1)}"
        return None

    m = RE_SECTION.match(label)
    if m:
        return f"{doc_id}:{m.group(1)}", f"§ {m.group(1)}"
    m = RE_ARTICLE.match(label)
    if m:
        return f"{doc_id}:art-{m.group(1)}", f"Article {m.group(1)}"
    m = RE_SECTION_EN.match(label)
    if m:
        return f"{doc_id}:sec-{m.group(1)}", f"Section {m.group(1)}"
    return None


def _article_from_heading(
    label: str, doc_id: str, family: str
) -> tuple[str, str] | None:
    """Article identity for an explicit markdown heading.

    ``_article_from_label`` refuses anything but ``第X条`` for CN/JP. That is
    right for body lines, but an explicit ``## § 101.1`` or ``## Article 1``
    heading is a structural marker: uploads default to ``jurisdiction="CN"``,
    so without this fallback every English article heading degrades to a
    section and a whole contract collapses into one ``:doc`` chunk.
    ``Section`` headings stay excluded (JP ``### Section 1 第一節``, WTO
    ``## Section I`` are containers, not articles).
    """
    art = _article_from_label(label, doc_id, family)
    if art or family not in {"CN", "JP"}:
        return art
    m = RE_SECTION.match(label)
    if m:
        return f"{doc_id}:{m.group(1)}", f"§ {m.group(1)}"
    m = RE_ARTICLE.match(label)
    if m:
        return f"{doc_id}:art-{m.group(1)}", f"Article {m.group(1)}"
    return None


def _section_from_label(label: str) -> str:
    label = re.sub(r"^#{2,6}\s*", "", label).strip()
    return label


def _drop_heading_marker(text: str, start: int) -> tuple[str, int]:
    m = re.match(r"^#{2,6}\s*", text)
    if not m:
        return text, start
    return text[m.end() :], start + m.end()


def _iter_starts(body: str, family: str = "") -> list[dict[str, Any]]:
    """Locate article and section starts with absolute character offsets."""
    starts: list[dict[str, Any]] = []
    offset = 0
    current_section = ""
    for line in body.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        heading = RE_HEADING.match(stripped)
        if heading:
            label = heading.group(2).strip()
            art = _article_from_heading(label, "", family)
            if art:
                starts.append(
                    {
                        "kind": "article",
                        "offset": offset,
                        "label": art[1],
                        "section": current_section,
                    }
                )
            else:
                current_section = _section_from_label(label)
                starts.append(
                    {
                        "kind": "section",
                        "offset": offset,
                        "label": current_section,
                        "section": current_section,
                    }
                )
        elif family in {"CN", "JP"}:
            # Chinese/Japanese articles are line starts (第X条), not headings.
            # For English families a body line beginning "Section 50A." is a
            # cross-reference, not a new article, so only headings count.
            art = _article_from_label(stripped, "", family)
            if art:
                starts.append(
                    {
                        "kind": "article",
                        "offset": offset,
                        "label": art[1],
                        "section": current_section,
                    }
                )
        offset += len(line)
    return starts


def _records_for_unit_article(
    doc_id: str, body: str, cfg: ChunkingConfig, family: str = ""
) -> list[ChunkRecord]:
    family = family or doc_id.split("-")[0].upper()
    starts = _iter_starts(body, family)
    if not starts:
        start, end = _trim_span(body, 0, len(body))
        if start >= end:
            return []
        article_id = f"{doc_id}:doc"
        text = body[start:end]
        if not cfg.include_heading:
            text, start = _drop_heading_marker(text, start)
        return [
            ChunkRecord(
                chunk_id=make_chunk_id(doc_id, article_id, text),
                doc_id=doc_id,
                article_id=article_id,
                article_label="document",
                text=text,
                section="",
                part=0,
                ordinal=0,
                start=start,
                end=end,
            )
        ]

    records: list[ChunkRecord] = []
    ordinal = 0
    used_article_ids: dict[str, int] = {}

    # Preamble (title/authority comments already removed by default).
    if not cfg.drop_preamble and starts[0]["offset"] > 0:
        s, e = _trim_span(body, 0, starts[0]["offset"])
        if s < e and (e - s) >= cfg.min_chars:
            text = body[s:e]
            if not cfg.include_heading:
                text, s = _drop_heading_marker(text, s)
            article_id = f"{doc_id}:preamble"
            records.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(doc_id, article_id, text),
                    doc_id=doc_id,
                    article_id=article_id,
                    article_label="preamble",
                    text=text,
                    section="",
                    part=0,
                    ordinal=ordinal,
                    start=s,
                    end=e,
                )
            )
            ordinal += 1

    for i, start_info in enumerate(starts):
        block_start = start_info["offset"]
        block_end = starts[i + 1]["offset"] if i + 1 < len(starts) else len(body)
        s, e = _trim_span(body, block_start, block_end)
        if s >= e:
            continue
        text = body[s:e]
        if start_info["kind"] == "section":
            if not cfg.include_heading:
                text, s = _drop_heading_marker(text, s)
            if cfg.drop_sections or len(text) < cfg.min_chars:
                continue
            article_id = f"{doc_id}:section-{ordinal}"
            records.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(doc_id, article_id, text),
                    doc_id=doc_id,
                    article_id=article_id,
                    article_label=start_info["label"],
                    text=text,
                    section=start_info["section"],
                    part=0,
                    ordinal=ordinal,
                    start=s,
                    end=e,
                )
            )
            ordinal += 1
            continue

        parsed = _article_from_heading(start_info["label"], doc_id, family)
        if not parsed:
            continue
        article_id, article_label = parsed
        # Supplementary provisions in Japanese laws restart article numbers.
        # Keep the first occurrence stable and disambiguate later ones.
        used_article_ids[article_id] = used_article_ids.get(article_id, 0) + 1
        if used_article_ids[article_id] > 1:
            article_id = f"{article_id}@{used_article_ids[article_id]}"
        if not cfg.include_heading:
            text, s = _drop_heading_marker(text, s)
        for part, (part_text, part_start, part_end) in enumerate(
            split_text(text, cfg.max_chars, cfg.overlap)
        ):
            if cfg.max_chunks_per_article and part >= cfg.max_chunks_per_article:
                break
            if len(part_text) < cfg.min_chars:
                continue
            records.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(doc_id, article_id, part_text),
                    doc_id=doc_id,
                    article_id=article_id,
                    article_label=article_label,
                    text=part_text,
                    section=start_info["section"],
                    part=part,
                    ordinal=ordinal,
                    start=s + part_start,
                    end=s + part_end,
                )
            )
            ordinal += 1
    if records:
        return records

    # A file with only non-article headings (or sections dropped by config)
    # must still become searchable: fall back to one whole-document chunk.
    start, end = _trim_span(body, 0, len(body))
    if start >= end:
        return []
    text = body[start:end]
    article_id = f"{doc_id}:doc"
    if not cfg.include_heading:
        text, start = _drop_heading_marker(text, start)
    return [
        ChunkRecord(
            chunk_id=make_chunk_id(doc_id, article_id, text),
            doc_id=doc_id,
            article_id=article_id,
            article_label="document",
            text=text,
            section="",
            part=0,
            ordinal=0,
            start=start,
            end=end,
        )
    ]


def _records_for_unit_section(
    doc_id: str, body: str, cfg: ChunkingConfig
) -> list[ChunkRecord]:
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in body.splitlines(keepends=True):
        m = RE_HEADING.match(line.rstrip("\r\n"))
        if m:
            headings.append((offset, m.group(2).strip()))
        offset += len(line)
    if not headings:
        return _records_for_unit_article(doc_id, body, cfg)

    records: list[ChunkRecord] = []
    for i, (start, label) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(body)
        s, e = _trim_span(body, start, end)
        if s >= e:
            continue
        text = body[s:e]
        if not cfg.include_heading:
            text, s = _drop_heading_marker(text, s)
        if len(text) < cfg.min_chars:
            continue
        article_id = f"{doc_id}:section-{i}"
        records.append(
            ChunkRecord(
                chunk_id=make_chunk_id(doc_id, article_id, text),
                doc_id=doc_id,
                article_id=article_id,
                article_label=_section_from_label(label),
                text=text,
                section=_section_from_label(label),
                part=0,
                ordinal=i,
                start=s,
                end=e,
            )
        )
    return records


def _records_for_unit_fixed(
    doc_id: str, body: str, cfg: ChunkingConfig
) -> list[ChunkRecord]:
    size = cfg.max_chars if cfg.max_chars > 0 else 2048
    records: list[ChunkRecord] = []
    for i, (text, start, end) in enumerate(split_text(body, size, cfg.overlap)):
        if len(text) < cfg.min_chars:
            continue
        article_id = f"{doc_id}:chunk-{i}"
        records.append(
            ChunkRecord(
                chunk_id=make_chunk_id(doc_id, article_id, text),
                doc_id=doc_id,
                article_id=article_id,
                article_label=f"chunk {i}",
                text=text,
                section="",
                part=i,
                ordinal=i,
                start=start,
                end=end,
            )
        )
    return records


def chunk_document(doc: dict[str, Any], raw_text: str, cfg: ChunkingConfig) -> list[ChunkRecord]:
    """Chunk one markdown document according to the configured unit."""
    body = strip_comments(raw_text) if cfg.strip_comments else raw_text
    doc_id = str(doc["doc_id"])
    family = str(doc.get("jurisdiction") or doc_id.split("-")[0]).upper()
    if cfg.unit == "fixed":
        return _records_for_unit_fixed(doc_id, body, cfg)
    if cfg.unit == "section":
        return _records_for_unit_section(doc_id, body, cfg)
    return _records_for_unit_article(doc_id, body, cfg, family)
