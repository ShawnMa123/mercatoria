"""从 WTO 法律文本页抓取协定正文，转成可导入 RAGFlow 的 markdown。

用法：
    python fetch_wto.py
    python fetch_wto.py --docs tfa_e,sps_e

输出：
    corpus/wto/{doc}.md         每个协定一个文件
    corpus/wto/index.json       抓取元数据

设计要点：
- WTO 页面是静态 HTML，无 JS 依赖，可直接正则去标签
- 条文锚点是 `<h3>Article N</h3>` 或 `<h2>Article N: Title</h2>`
- 输出用 `## Article N` 作为二级标题，匹配 RAGFlow laws 解析器的
  group 4（`Article [0-9]+`），与中文法律的「一条一块」对齐
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://www.wto.org/english/docs_e/legal_e"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# 文档 -> 标题
DOCS = {
    "tfa_e": "Agreement on Trade Facilitation",
    "sps_e": "Agreement on the Application of Sanitary and Phytosanitary Measures",
    "tbt_e": "Agreement on Technical Barriers to Trade",
    "adp_e": "Agreement on Implementation of Article VI (Anti-Dumping)",
    "cv_e": "Agreement on Implementation of Article VII (Customs Valuation)",
    "ro_e": "Agreement on Rules of Origin",
    "lic_e": "Agreement on Import Licensing Procedures",
    "scm_e": "Agreement on Subsidies and Countervailing Measures",
    "trims_e": "Agreement on Trade-Related Investment Measures",
    # GATT 1994 的正文即 GATT 1947 原文（经 1994 年谅解修订）。
    # 注意：gatt94_e.htm 只是索引页，正文在 gatt47_e.htm。
    "gatt47_e": "General Agreement on Tariffs and Trade 1994",
    # 早期文件的站点路径不统一：TRIPS 分页存于 27-trips_01_e.htm，
    # 保障措施是 sg_e.htm，农业协定是 ag_e.htm（fetch_document 负责跟进）。
    "dsu_e": "Understanding on Rules and Procedures Governing the Settlement of Disputes (DSU)",
    "sg_e": "Agreement on Safeguards",
    "ag_e": "Agreement on Agriculture",
    "27-trips_e": "Agreement on Trade-Related Aspects of Intellectual Property Rights",
}


def fetch(url: str, retries: int = 3, timeout: int = 60) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last}")


def _is_stub(html: str) -> bool:
    """WTO answers 200 with a generic nav shell for a missing legal text."""
    return len(html) <= 12_000 and "Article" not in html


def fetch_document(doc: str) -> tuple[str, list[str]]:
    """Fetch one legal text, following the `_01_e`/`_02_e` pagination.

    Several agreements (TRIPS, Agriculture, Safeguards) have no single-page
    URL: the unsuffixed path returns the site's generic "not found" shell,
    while the text is split across ``{stem}_01_e.htm``, ``_02_e``, ...
    """
    html = fetch(f"{BASE}/{doc}.htm")
    if not _is_stub(html):
        return html, [f"{BASE}/{doc}.htm"]
    stem = doc[:-2] if doc.endswith("_e") else doc  # 27-trips_e -> 27-trips
    pages: list[str] = []
    urls: list[str] = []
    for index in range(1, 31):
        url = f"{BASE}/{stem}_{index:02d}_e.htm"
        try:
            page = fetch(url)
        except Exception:  # noqa: BLE001
            break
        if _is_stub(page):
            break
        pages.append(page)
        urls.append(url)
    if not pages:
        return html, [f"{BASE}/{doc}.htm"]
    return "\n".join(pages), urls


class TextExtractor(HTMLParser):
    """把 HTML 转成「标题/段落」事件流。

    只保留正文区，跳过导航、脚本、样式。
    """

    BLOCK = {"p", "div", "li", "tr", "blockquote", "pre"}
    HEAD = {"h1", "h2", "h3", "h4", "h5", "h6"}
    SKIP = {"script", "style", "head", "nav"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._cur_head: list[str] | None = None
        self._cur_par: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag in self.HEAD:
            self._flush_par()
            self._cur_head = []
        elif tag in self.BLOCK:
            self._flush_par()
            self._cur_par = []
        elif tag == "br":
            if self._cur_par is not None:
                self._cur_par.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if tag in self.HEAD:
            self._flush_head()
        elif tag in self.BLOCK:
            self._flush_par()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._cur_head is not None:
            self._cur_head.append(data)
        elif self._cur_par is not None:
            self._cur_par.append(data)

    def _flush_head(self) -> None:
        if self._cur_head is None:
            return
        t = re.sub(r"\s+", " ", "".join(self._cur_head)).strip()
        if t:
            self.events.append(("head", t))
        self._cur_head = None

    def _flush_par(self) -> None:
        if self._cur_par is None:
            return
        t = re.sub(r"\s+", " ", "".join(self._cur_par)).strip()
        if t:
            self.events.append(("par", t))
        self._cur_par = None

    def close(self) -> None:  # noqa: A003
        self._flush_head()
        self._flush_par()
        super().close()


NAV_NOISE = re.compile(
    r"^(Home|About WTO|News|Trade topics|WTO membership|Documents|"
    r"External relations|Contact us|Site map|Search|Login|MyWTO|"
    r"WORLD TRADE ORGANIZATION|Fran[cç]ais|Espa[nñ]ol)$",
    re.I,
)
# 面包屑 / 站内导航行，如 "Home | About WTO | News & events | ..."
BREADCRUMB = re.compile(r"^(?:[^|]{1,40}\s*\|\s*){2,}", re.I)
# 页脚版权与"THE WTO AGREEMENTS:"类小标题
FOOTER = re.compile(
    r"^(THE WTO AGREEMENTS:.*|.*©\s*\d{4}.*|Page \d+ of \d+)$", re.I
)
ARTICLE_HEAD = re.compile(r"^Article\s+([0-9]+[A-Za-z]?|[IVXLC]+)\b", re.I)
SECTION_HEAD = re.compile(r"^(PART|SECTION|ANNEX|APPENDIX)\b", re.I)

# 目录区标志：形如 "Articles:" 的标题之后的一串条目
TOC_HEAD = re.compile(r"^Articles?:$", re.I)
# 目录条目：以 Article/Section/Annex 开头但很短且无正文（无句号结尾的长文本）
TOC_ENTRY = re.compile(
    r"^(Article\s+\S+|Section\s+\S+|Annex\s+\S+|PART\s+\S+)[:\s]", re.I
)


def html_to_markdown(html: str, title: str) -> tuple[str, int]:
    """转成 markdown。返回 (md, article 数)。"""
    ex = TextExtractor()
    ex.feed(html)
    ex.close()

    # 结构要点（实测验证）：
    # RAGFlow 的 `laws` 解析器（HierarchyTitleChunker）会把 H1 的文字
    # 复制进每一个子块。实测：有 H1 时每个 chunk 都带 "# 19 CFR Part 142"。
    # 因此文档标题放进 HTML 注释（单独成一块），顶层标题直接用 H2。
    out: list[str] = [f"<!-- document: {title} -->", ""]
    n_article = 0
    seen_content = False
    in_toc = False

    for kind, text in ex.events:
        if NAV_NOISE.match(text) or BREADCRUMB.match(text) or FOOTER.match(text):
            continue
        if len(text) < 2:
            continue

        # 目录区：从 "Articles:" 标题开始，到第一个真正的正文 Article 标题结束
        if kind == "head" and TOC_HEAD.match(text.strip()):
            in_toc = True
            continue
        if in_toc:
            # 目录里的条目既可能是 head 也可能是 par；正文的第一个 Article
            # 标题（head 且带正文标题文本）标志目录结束
            if kind == "head" and ARTICLE_HEAD.match(text):
                in_toc = False
            elif TOC_ENTRY.match(text) and len(text) < 160:
                continue
            elif kind == "par" and len(text) < 160:
                continue
            else:
                in_toc = False

        if kind == "head":
            m = ARTICLE_HEAD.match(text)
            if m:
                # 条文：用 ## 作为分块锚点
                out.append("")
                out.append(f"## {text}")
                out.append("")
                n_article += 1
                seen_content = True
            elif SECTION_HEAD.match(text):
                out.append("")
                out.append(f"### {text}")
                out.append("")
            elif text.upper() == title.upper() or "AGREEMENT" in text.upper():
                # 文档标题，跳过（已在 H1）
                continue
            else:
                out.append("")
                out.append(f"### {text}")
                out.append("")
        else:
            # 正文段落；跳过页眉页脚类短句
            if not seen_content and len(text) < 40:
                continue
            out.append(text)

    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md, n_article


def main() -> int:
    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "--docs":
        only = set(sys.argv[2].split(","))

    out_dir = Path(__file__).parent / "wto"
    out_dir.mkdir(parents=True, exist_ok=True)

    index: dict = {"source": "WTO legal texts", "base": BASE, "fetched": [], "missing": []}

    for doc, title in DOCS.items():
        if only and doc not in only:
            continue
        try:
            html, sources = fetch_document(doc)
        except Exception as e:  # noqa: BLE001
            print(f"  {doc}: {e}", file=sys.stderr)
            index["missing"].append({"doc": doc, "error": str(e)})
            continue

        md, n_art = html_to_markdown(html, title)
        if n_art == 0:
            print(f"  {doc}: 未识别到 Article — 跳过（可能是索引页）", file=sys.stderr)
            index["missing"].append({"doc": doc, "error": "no articles"})
            continue

        fname = f"{doc}.md"
        (out_dir / fname).write_text(md, encoding="utf-8")
        print(f"  {doc}: {n_art} articles, {len(md):,} chars -> {fname}")
        index["fetched"].append(
            {"doc": doc, "file": fname, "url": sources[0], "pages": len(sources),
             "articles": n_art, "chars": len(md)}
        )

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(x["articles"] for x in index["fetched"])
    print(f"\nWTO: {len(index['fetched'])} docs, {total} articles, {len(index['missing'])} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
