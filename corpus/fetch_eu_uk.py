"""从欧盟 Cellar 与英国 legislation.gov.uk 抓取核心贸易法规。

用法：
    python fetch_eu_uk.py                # 全部
    python fetch_eu_uk.py --eu 32013R0952
    python fetch_eu_uk.py --uk uk-cross-border-trade-2018

输出：
    corpus/eu/{slug}.md
    corpus/eu/index.json
    corpus/uk/{slug}.md
    corpus/uk/index.json

设计要点：
- 欧盟官方直连 Cellar 无 Cloudflare/WAF：
  `publications.europa.eu/resource/celex/{celex}.ENG.xhtml` 会自动 303 到
  对象存储并返回 XHTML（OJ 的 CONVEX 排版），无需浏览器。
- OJ 的条文锚点是 `p.oj-ti-art`（"Article 1"），章节是
  `p.oj-ti-section-1/2`，正文是 `p.oj-normal`。输出 `## Article N` 与
  `### 章节` —— 与其它法域的分块约定一致。
- 英国 legislation.gov.uk 的 data.xml 在部分网络下会被 AWS WAF JS 挑战
  （HTTP 202 + 空 body）。此脚本会记录为 missing 而不是伪装成功；在可访问
  的网络或浏览器里先下载该文件再手工转换即可。
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

UA = "mercatoria-legal-research/0.1 (+legal retrieval POC)"
CELLAR = "https://publications.europa.eu/resource/celex"
UK_BASE = "https://www.legislation.gov.uk"

EU_DOCS: dict[str, tuple[str, str]] = {
    "32013R0952": ("eu-ucc-2013", "Regulation (EU) No 952/2013 (Union Customs Code)"),
    "32021R0821": ("eu-dual-use-2021", "Regulation (EU) 2021/821 (dual-use export controls)"),
}

UK_DOCS: dict[str, tuple[str, str]] = {
    # slug -> (doc_id, 标题)；路径在 main 里拼。
    "uk-cross-border-trade-2018": ("ukpga/2018/22", "Taxation (Cross-border Trade) Act 2018"),
}

# 只保留有结构意义的块；其它（注释、签名、修订史）不进语料。
BLOCK_CLASSES = {
    "oj-ti-art": "article",
    "oj-sti-art": "article_title",
    "oj-ti-section-1": "section",
    "oj-ti-section-2": "section",
    "oj-normal": "par",
    "oj-tbl-txt": "par",
    "oj-tbl-num": "par",
    "oj-ti-tbl": "par",
    "oj-bold": "par",
}

VOID_TAGS = {"br", "img", "hr", "meta", "link", "col", "input", "wbr"}


def fetch(url: str, retries: int = 3, timeout: int = 120, accept: str = "") -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        headers = {
            "User-Agent": UA,
            "Accept-Encoding": "gzip",
            "Accept-Language": "eng",
        }
        if accept:
            headers["Accept"] = accept
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                if not raw:
                    raise RuntimeError(f"空响应（WAF 挑战？）HTTP {resp.status}")
                return raw
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last}")


class BlockExtractor(HTMLParser):
    """把 OJ XHTML 拆成 (kind, text) 块。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._kind: str | None = None
        self._depth = 0
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in VOID_TAGS:
            return
        if self._kind is not None:
            self._depth += 1
            return
        classes = (dict(attrs).get("class") or "").split()
        kind = next((BLOCK_CLASSES[c] for c in classes if c in BLOCK_CLASSES), None)
        if kind:
            self._kind = kind
            self._depth = 1
            self._buf = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if self._kind is None:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text:
            self.blocks.append((self._kind, text))
        self._kind = None
        self._buf = []

    def handle_data(self, data: str) -> None:
        if self._kind is not None:
            self._buf.append(data)


def eu_to_markdown(html: str, title: str) -> tuple[str, int]:
    parser = BlockExtractor()
    parser.feed(html)
    parser.close()
    out: list[str] = [f"<!-- document: {title} -->", ""]
    n_article = 0
    for kind, text in parser.blocks:
        if kind == "article":
            out.extend(["", f"## {text}", ""])
            n_article += 1
        elif kind == "article_title":
            out.append(text)
        elif kind == "section":
            out.extend(["", f"### {text}", ""])
        else:
            out.append(text)
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md, n_article


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def uk_to_markdown(raw: bytes, title: str) -> tuple[str, int]:
    """legislation.gov.uk data.xml -> markdown（Section 一级）。"""
    root = ET.fromstring(raw)
    out: list[str] = [f"<!-- document: {title} -->", ""]
    n_section = 0
    for section in root.iter():
        if _local(section.tag) != "Section":
            continue
        number = ""
        heading = ""
        for child in section:
            tag = _local(child.tag)
            if tag == "Number" and not number:
                number = "".join(child.itertext()).strip()
            elif tag in {"Title", "Heading"} and not heading:
                heading = re.sub(r"\s+", " ", "".join(child.itertext())).strip()
        if not number:
            continue
        body: list[str] = []
        for child in section:
            if _local(child.tag) in {"Number", "Title", "Heading"}:
                continue
            text = re.sub(r"\s+", " ", "".join(child.itertext())).strip()
            if text:
                body.append(text)
        line = f"Section {number}" + (f" — {heading}" if heading else "")
        out.extend(["", f"## {line}", ""])
        out.extend(body)
        n_section += 1
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md, n_section


def _write_index(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_eu(only: set[str] | None) -> dict:
    out_dir = Path(__file__).parent / "eu"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"source": "EU Cellar (publications.europa.eu)", "fetched": [], "missing": []}
    for celex, (slug, title) in EU_DOCS.items():
        if only and celex not in only and slug not in only:
            continue
        url = f"{CELLAR}/{celex}.ENG.xhtml"
        try:
            raw = fetch(url, accept="application/xhtml+xml, text/html;q=0.9")
        except Exception as exc:  # noqa: BLE001
            print(f"  {slug}: {exc}", file=sys.stderr)
            index["missing"].append({"celex": celex, "slug": slug, "error": str(exc)})
            continue
        html = raw.decode("utf-8", errors="replace")
        md, n_article = eu_to_markdown(html, title)
        if n_article == 0:
            print(f"  {slug}: 未识别到 Article — 跳过", file=sys.stderr)
            index["missing"].append({"celex": celex, "slug": slug, "error": "no articles"})
            continue
        fname = f"{slug}.md"
        (out_dir / fname).write_text(md, encoding="utf-8")
        print(f"  {slug}: {n_article} articles, {len(md):,} chars -> {fname}")
        index["fetched"].append(
            {"celex": celex, "slug": slug, "file": fname, "title": title,
             "articles": n_article, "chars": len(md)}
        )
    _write_index(out_dir / "index.json", index)
    print(f"EU: {len(index['fetched'])} docs, {len(index['missing'])} missing")
    return index


def fetch_uk(only: set[str] | None) -> dict:
    out_dir = Path(__file__).parent / "uk"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"source": "legislation.gov.uk", "fetched": [], "missing": []}
    for slug, (path, title) in UK_DOCS.items():
        if only and slug not in only:
            continue
        url = f"{UK_BASE}/{path}/data.xml"
        try:
            raw = fetch(url, accept="application/xml")
            md, n_section = uk_to_markdown(raw, title)
        except Exception as exc:  # noqa: BLE001
            print(f"  {slug}: {exc}", file=sys.stderr)
            index["missing"].append({"slug": slug, "url": url, "error": str(exc)})
            continue
        if n_section == 0:
            print(f"  {slug}: 未识别到 Section — 跳过", file=sys.stderr)
            index["missing"].append({"slug": slug, "url": url, "error": "no sections"})
            continue
        fname = f"{slug}.md"
        (out_dir / fname).write_text(md, encoding="utf-8")
        print(f"  {slug}: {n_section} sections, {len(md):,} chars -> {fname}")
        index["fetched"].append(
            {"slug": slug, "url": url, "file": fname, "title": title,
             "sections": n_section, "chars": len(md)}
        )
    _write_index(out_dir / "index.json", index)
    print(f"UK: {len(index['fetched'])} docs, {len(index['missing'])} missing")
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eu", default="", help="逗号分隔的 CELEX 或 slug")
    ap.add_argument("--uk", default="", help="逗号分隔的 slug")
    args = ap.parse_args()
    eu_only = {x.strip() for x in args.eu.split(",") if x.strip()}
    uk_only = {x.strip() for x in args.uk.split(",") if x.strip()}
    want_eu = bool(eu_only) or not uk_only
    want_uk = bool(uk_only) or not eu_only
    if want_eu:
        fetch_eu(eu_only or None)
    if want_uk:
        fetch_uk(uk_only or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
