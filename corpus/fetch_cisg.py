"""从 UNCITRAL 抓取 CISG 正文，转成可导入 RAGFlow 的 markdown。

用法：
    python fetch_cisg.py

输出：
    corpus/uncitral/cisg.md
    corpus/uncitral/index.json

设计要点：
- UNCITRAL 页面本身不含条文，正文只在 PDF 里（19-09951_e_ebook.pdf）
- 站点对无 UA 的请求返回 403，必须带浏览器 UA
- 用 pymupdf 抽取 PDF 文本，再按 `Article N` 切条
- 输出用 `## Article N` 作二级标题，匹配 RAGFlow laws 解析器 group 4
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pymupdf

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

TITLE = "United Nations Convention on Contracts for the International Sale of Goods (CISG)"
PDF_URL = (
    "https://uncitral.un.org/sites/default/files/media-documents/"
    "uncitral/en/19-09951_e_ebook.pdf"
)

ARTICLE = re.compile(r"^\s*Article\s+(\d+[A-Za-z]?)\s*$", re.M)
# CISG 的章节标题
PART = re.compile(r"^\s*(PART\s+[IVX]+|CHAPTER\s+[IVX]+)\.?\s*(.*)$", re.M)


def fetch(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last}")


CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def pdf_to_text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    text = "\n".join(pages)
    # 去掉控制字符（该 PDF 有 2 后跟退格符 \x08 的排版残留）
    return CONTROL.sub("", text)


def find_convention_body(text: str) -> str:
    """定位公约正文起点。

    PDF 前部是序言、目录、UNCITRAL 介绍，正文从 "Article 1" 附近开始。
    以「最后一个 'Article 1' 之前最近的 'PART I' 或 'UNITED NATIONS CONVENTION'
    之后」为起点。
    """
    # 找所有 Article 1 的位置
    hits = [m.start() for m in re.finditer(r"^\s*Article\s+1\s*$", text, re.M)]
    if not hits:
        return text
    start = hits[0]
    # 往回找最近的 "PART I" 或 "Sphere of application"
    back = text.rfind("PART I", 0, start)
    if back == -1:
        back = text.rfind("Sphere of application", 0, start)
    if back != -1 and start - back < 3000:
        start = back

    # 找结尾：最后一条 Article 101 之后到 "Done at" / 签署地
    end = len(text)
    a101 = text.rfind("Article 101")
    if a101 != -1:
        tail = text[a101:]
        done = tail.find("Done at")
        if done != -1:
            end = a101 + done

    return text[start:end]


def to_markdown(body: str) -> tuple[str, int]:
    """把公约正文按 Article 切成 markdown。"""
    # 规范化：去掉页码行、页眉、页脚
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        # 纯页码（可能夹在正文段落之间）
        if re.fullmatch(r"\d{1,3}", s):
            continue
        # 页眉
        if re.fullmatch(r"(?i)united nations convention.*", s) and len(s) < 80:
            continue
        # 页脚 "Page N" / "N / 60" 形式
        if re.fullmatch(r"(?i)(page\s+)?\d{1,3}\s*(/|of)\s*\d{1,3}", s):
            continue
        lines.append(ln.rstrip())
    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body)

    # 找 Article 切分点
    marks = list(ARTICLE.finditer(body))
    if not marks:
        return f"# {TITLE}\n\n{body}\n", 0

    # 不用 H1：RAGFlow laws 解析器会把 H1 复制进每个子块（实测验证）
    out = [f"<!-- document: {TITLE} -->", ""]
    n = 0
    for i, m in enumerate(marks):
        seg_start = m.start()
        seg_end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        seg = body[seg_start:seg_end].strip()
        if not seg:
            continue
        first, _, rest = seg.partition("\n")
        num = m.group(1)
        out.append("")
        out.append(f"## Article {num}")
        out.append("")
        # 第一行是 "Article N"，其后的内容才是正文
        rest = rest.strip()
        if rest:
            for para in rest.split("\n\n"):
                para = para.strip()
                if para:
                    out.append(para)
        n += 1

    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md, n


def main() -> int:
    out_dir = Path(__file__).parent / "uncitral"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"source": "UNCITRAL", "fetched": [], "missing": []}

    try:
        pdf = fetch(PDF_URL)
    except Exception as e:  # noqa: BLE001
        print(f"  cisg: 下载失败 {e}", file=sys.stderr)
        index["missing"].append({"doc": "cisg", "error": str(e)})
        (out_dir / "index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1

    # 存档原始 PDF
    (out_dir / "cisg.pdf").write_bytes(pdf)

    text = pdf_to_text(pdf)
    body = find_convention_body(text)
    md, n_art = to_markdown(body)

    if n_art == 0:
        print("  cisg: 未识别到 Article", file=sys.stderr)
        index["missing"].append({"doc": "cisg", "error": "no articles"})
    else:
        (out_dir / "cisg.md").write_text(md, encoding="utf-8")
        print(f"  cisg: {n_art} articles, {len(md):,} chars -> cisg.md")
        index["fetched"].append(
            {"doc": "cisg", "file": "cisg.md", "articles": n_art, "chars": len(md)}
        )

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(x["articles"] for x in index["fetched"])
    print(f"\nUNCITRAL: {len(index['fetched'])} docs, {total} articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
