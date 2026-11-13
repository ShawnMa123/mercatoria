"""Parse uploaded files into article-oriented Markdown.

POC scope: ``.md`` / ``.txt`` / ``.docx`` / ``.pdf`` with a text layer.
No OCR, no legacy ``.doc``. Parsers only produce text; the existing chunker
still owns article boundaries and ``chunk_id``.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".docx", ".pdf"}

RE_CHAPTER = re.compile(r"^\s*(第[零〇一二三四五六七八九十百千万0-9]+章[^\n]*)$")
RE_EN_SECTION = re.compile(
    r"^\s*((?:Section|Article|§)\s+[0-9IVXLCDM][0-9A-Za-z.\-]*[^\n]*)$", re.I
)
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class ParseError(RuntimeError):
    pass


@dataclass
class ParsedDocument:
    text: str
    format: str
    pages: int = 0
    warnings: list[str] = field(default_factory=list)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError("text is not valid UTF-8 or GB18030")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_CONTROL.sub("", text)
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(line)
    return "\n".join(out).strip() + "\n"


def normalize_markdown(text: str) -> str:
    """Promote chapter/section lines to ``##`` headings.

    Chinese articles (``第X条``) stay as line starts: the chunker already treats
    them as article boundaries. Only chapters and English section/article labels
    need a heading marker.
    """
    out: list[str] = []
    for line in _clean(text).split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        if RE_CHAPTER.match(stripped) or RE_EN_SECTION.match(stripped):
            out.append("## " + stripped)
        else:
            out.append(line)
    return "\n".join(out).strip() + "\n"


def _docx_markdown(data: bytes, stem: str) -> ParsedDocument:
    try:
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ParseError("python-docx is not installed") from exc

    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - third-party parser errors
        raise ParseError(f"invalid DOCX: {exc}") from exc

    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            level = 0
            if "heading" in style or "标题" in style:
                match = re.search(r"(\d+)$", style)
                level = int(match.group(1)) if match else 2
            if level:
                blocks.append("#" * min(6, max(2, level)) + " " + text)
            elif RE_CHAPTER.match(text) or RE_EN_SECTION.match(text):
                blocks.append("## " + text)
            else:
                blocks.append(text)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows: list[list[str]] = []
            for row in table.rows:
                cells = [cell.text.replace("\n", " ").strip().replace("|", "\\|") for cell in row.cells]
                if any(cells):
                    rows.append(cells)
            if not rows:
                continue
            width = max(len(row) for row in rows)
            rows = [row + [""] * (width - len(row)) for row in rows]
            blocks.append("| " + " | ".join(rows[0]) + " |")
            blocks.append("| " + " | ".join("---" for _ in rows[0]) + " |")
            blocks.extend("| " + " | ".join(row) + " |" for row in rows[1:])

    text = "\n\n".join(blocks)
    if not text.strip():
        raise ParseError("DOCX contains no readable text")
    body = normalize_markdown(text)
    return ParsedDocument(text=f"<!-- document: {stem} -->\n\n{body}", format="docx")


def _pdf_markdown(data: bytes, stem: str) -> ParsedDocument:
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ParseError("PyMuPDF is not installed") from exc

    if not data.startswith(b"%PDF"):
        raise ParseError("file does not look like a PDF")
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"invalid PDF: {exc}") from exc

    try:
        if document.needs_pass:
            raise ParseError("encrypted PDF is not supported")
        pages: list[str] = []
        for number, page in enumerate(document, 1):
            page_text = page.get_text("text", sort=True)
            if page_text.strip():
                pages.append(f"<!-- page: {number} -->\n{page_text}")
        total = sum(len(page) for page in pages)
        if total < 20:
            raise ParseError("PDF has no text layer; OCR is not supported")
        text = "\n\n".join(pages)
        body = normalize_markdown(text)
        return ParsedDocument(
            text=f"<!-- document: {stem} -->\n\n{body}",
            format="pdf",
            pages=len(document),
        )
    finally:
        document.close()


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".doc":
        raise ParseError("legacy .doc is not supported; convert it to .docx")
    if suffix not in SUPPORTED:
        raise ParseError(f"unsupported file type: {suffix or '(none)'}")
    if not data:
        raise ParseError("empty file")
    stem = Path(filename).stem

    if suffix in {".md", ".markdown"}:
        text = _decode_text(data)
        return ParsedDocument(text=normalize_markdown(text), format="md")
    if suffix == ".txt":
        text = _decode_text(data)
        return ParsedDocument(text=normalize_markdown(text), format="txt")
    if suffix == ".docx":
        if not data.startswith(b"PK"):
            raise ParseError("file does not look like a DOCX")
        return _docx_markdown(data, stem)
    return _pdf_markdown(data, stem)
