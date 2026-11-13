"""从 eCFR 抓取贸易相关条文，转成可直接导入 RAGFlow 的 markdown。

用法：
    python fetch_ecfr.py --title 19 --parts 0,10,11,12,18,19,24,54,101,102,113,141
    python fetch_ecfr.py --title 15 --parts 730,732,734,736,738,744,746,764

输出：
    corpus/ecfr/title-{N}/part-{P}.md    每卷每 part 一个文件
    corpus/ecfr/title-{N}/index.json     抓取元数据（供复现）

设计要点：
- eCFR 必须同时带 Accept 和 Accept-Encoding 两个头，否则 406
- DIV8 TYPE="SECTION" 是条文级单位，N 属性即节号
- 输出 md 用 `# 19 CFR 104` 作为一级标题、`## § 104.0 Scope` 作为二级标题
  这样 RAGFlow 的 laws 解析器（group 0 匹配 ^##）能正确识别为
  「一节一块」，与中文法律的「一条一块」对齐
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "https://www.ecfr.gov/api/versioner/v1"
UA = "mercatoria-corpus-builder/0.1 (+legal retrieval POC)"

# 抓取日期固定，保证可复现
DEFAULT_DATE = "2026-11-01"

# --title 未给 --parts 时使用的默认 part 清单。
# 19 = CBP 海关实务；15 = EAR 出口管制；31 = OFAC 制裁。
DEFAULT_PARTS: dict[int, str] = {
    19: (
        "0,7,10,11,12,18,19,24,54,101,102,103,111,112,113,114,115,118,122,"
        "123,125,127,128,132,133,134,141,142,143,144,145,146,147,148,149,151,"
        "152,158,159,161,162,163,165,171,172,173,174,177,181,182,190"
    ),
    15: "730,732,734,736,738,740,742,744,746,748,750,758,760,762,764,772",
    31: "501,510,535,542,560,587,591",
}


def fetch(url: str, retries: int = 3, timeout: int = 90) -> bytes:
    """带重试的 GET，eCFR 要求两个 Accept 头。"""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/xml",
                    "Accept-Encoding": "gzip",
                    "User-Agent": UA,
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw
        except urllib.error.HTTPError as e:
            # 404 表示该 part 在此日期不存在，不重试
            if e.code == 404:
                raise
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last}")


def text_of(el: ET.Element) -> str:
    """取元素全部文本，保留段落换行。"""
    parts: list[str] = []

    def walk(node: ET.Element, buf: list[str]) -> None:
        if node.text:
            buf.append(node.text)
        for child in node:
            walk(child, buf)
            if child.tail:
                buf.append(child.tail)

    buf: list[str] = []
    walk(el, buf)
    s = "".join(buf)
    # 折叠空白但保留段落
    s = re.sub(r"[ \t ]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_head(head: str) -> str:
    """把 § 101.0 Scope. 规整成单行。"""
    s = head.replace("§", "§").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def part_to_markdown(xml_bytes: bytes, title: int, part: str) -> tuple[str, int]:
    """把 eCFR part XML 转成 markdown。返回 (markdown, section 数)。"""
    root = ET.fromstring(xml_bytes)
    head_el = root.find("HEAD")
    part_title = clean_head(text_of(head_el)) if head_el is not None else f"Part {part}"

    # 结构要点（实测验证）：
    # RAGFlow 的 `laws` 解析器（HierarchyTitleChunker）会把 H1 的文字
    # **复制进每一个子块**。实测三种结构：
    #   A. 元信息在前 + H1  → 每个 chunk 都带 "# 19 CFR Part 142"
    #   B. H1 换成注释      → 元信息单独成一块，其余干净
    #   C. 完全无 H1        → 最干净，每个 chunk 只有自己的条文
    # 选 B：既保留文档级元信息（单独成块），又不污染条文块。
    meta_bits: list[str] = []
    for meta_tag in ("AUTH", "SOURCE"):
        el = root.find(meta_tag)
        if el is not None:
            meta = text_of(el)
            if meta:
                meta_bits.append(meta[:300])

    lines: list[str] = [f"<!-- document: 19 CFR Part {part} — {part_title} -->"]
    if meta_bits:
        lines.append("")
        lines.extend(f"<!-- {m} -->" for m in meta_bits)
    lines.append("")

    # SECTION 可能直接挂在 PART 下，也可能嵌在 SUBPART / SUBJGRP 里，
    # 因此必须递归遍历而不能只看根的子元素。
    n_sec = 0
    for sec in root.iter():
        if sec.tag != "DIV8" or sec.get("TYPE") != "SECTION":
            continue
        sh = sec.find("HEAD")
        if sh is None:
            continue
        sec_head = clean_head(text_of(sh))

        # 每条一节：用 ## 作为分块锚点（RAGFlow laws 解析器 group 0）
        lines.append(f"## {sec_head}")
        lines.append("")

        # 只取直接子元素，避免把嵌套 SECTION 的内容重复收录
        for child in sec:
            if child.tag == "HEAD":
                continue
            if child.tag == "DIV8" and child.get("TYPE") == "SECTION":
                continue
            body = text_of(child)
            if not body:
                continue
            for para in body.split("\n\n"):
                para = para.strip()
                if para:
                    lines.append(para)
        lines.append("")
        n_sec += 1

    return "\n".join(lines).rstrip() + "\n", n_sec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", type=int, required=True, help="CFR title 编号")
    ap.add_argument(
        "--parts",
        default="",
        help="逗号分隔的 part 编号，如 0,10,11；省略时用 DEFAULT_PARTS[title]",
    )
    ap.add_argument("--date", default=DEFAULT_DATE, help="快照日期 YYYY-MM-DD")
    ap.add_argument("--out", default=None, help="输出根目录，默认 corpus/ecfr")
    args = ap.parse_args()

    out_root = Path(args.out) if args.out else Path(__file__).parent / "ecfr"
    tdir = out_root / f"title-{args.title}"
    tdir.mkdir(parents=True, exist_ok=True)

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if not parts:
        parts = [p.strip() for p in DEFAULT_PARTS.get(args.title, "").split(",") if p.strip()]
    if not parts:
        ap.error(
            f"--parts is required for title {args.title} "
            f"(known titles: {sorted(DEFAULT_PARTS)})"
        )
    index: dict = {
        "source": "US eCFR",
        "title": args.title,
        "snapshot_date": args.date,
        "fetched": [],
        "missing": [],
    }

    for part in parts:
        url = f"{BASE}/full/{args.date}/title-{args.title}.xml?part={part}"
        try:
            raw = fetch(url)
        except urllib.error.HTTPError as e:
            print(f"  part {part}: HTTP {e.code} — 跳过", file=sys.stderr)
            index["missing"].append({"part": part, "http": e.code})
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  part {part}: {e} — 跳过", file=sys.stderr)
            index["missing"].append({"part": part, "error": str(e)})
            continue

        try:
            md, n_sec = part_to_markdown(raw, args.title, part)
        except ET.ParseError as e:
            print(f"  part {part}: XML 解析失败 {e}", file=sys.stderr)
            index["missing"].append({"part": part, "error": f"parse: {e}"})
            continue

        if n_sec == 0:
            print(f"  part {part}: 0 个 SECTION — 跳过", file=sys.stderr)
            index["missing"].append({"part": part, "error": "no sections"})
            continue

        fname = f"part-{part}.md"
        (tdir / fname).write_text(md, encoding="utf-8")
        print(f"  part {part}: {n_sec} sections, {len(md):,} chars -> {fname}")
        index["fetched"].append(
            {"part": part, "file": fname, "sections": n_sec, "chars": len(md)}
        )

    (tdir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(x["sections"] for x in index["fetched"])
    print(
        f"\nTitle {args.title}: {len(index['fetched'])} parts, {total} sections, "
        f"{len(index['missing'])} missing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
