"""从澳洲 Federal Register of Legislation 抓取法规，转成可导入 RAGFlow 的 markdown。

用法：
    python fetch_au.py

输出：
    corpus/au/{slug}.md
    corpus/au/{slug}.pdf       原始 PDF 存档
    corpus/au/index.json

设计要点：
- `api.prod.legislation.gov.au/v1/titles/{id}` 只给元数据
- 正文下载的正确形式是 OData 的 find 函数：
  `/v1/documents/find(registerId='...',type='Primary',format='Pdf',
                      uniqueTypeNumber=0,volumeNumber=N,rectificationVersionNumber=0)`
  直接返回二进制文件，不加 `/$value` 后缀（加了会 404）
- 一部法案可能分多卷（volumeNumber），需按卷分别下载再合并
- 条文格式：`Section 1`, `Part 1`, `Division 1`，或 `Item 1`
- 输出用 `## Section N` 作二级标题，匹配 RAGFlow laws 解析器 group 4
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pymupdf

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

API = "https://api.prod.legislation.gov.au/v1"

# titleId -> (slug, 说明)
TITLES = {
    "C1901A00006": ("customs-act-1901", "Customs Act 1901（海关法）"),
    # 注意：标题 ID 不是按年份数字直接拼的，需用
    # /v1/titles?$filter=contains(name,'...') 查出来
    "C2004A04997": ("customs-tariff-act-1995", "Customs Tariff Act 1995（关税税率法）"),
    "C2020A00012": ("export-control-act-2020", "Export Control Act 2020（出口管制法）"),
    "F2015L00375": ("customs-regulation-2015", "Customs Regulation 2015（海关条例）"),
    "C2011A00038": ("autonomous-sanctions-act-2011", "Autonomous Sanctions Act 2011（自主制裁法）"),
}


def fetch_json(url: str, retries: int = 3, timeout: int = 60) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last}")


def fetch_binary(url: str, retries: int = 3, timeout: int = 180) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}: {last}")


def latest_register(title_id: str) -> tuple[str, str, int]:
    """返回 (registerId, start, 最大卷号)。"""
    # OData 查询串含空格，必须编码，否则 urllib 拒绝
    params = urllib.parse.urlencode(
        {"$filter": f"titleId eq '{title_id}'", "$orderby": "start desc", "$top": "200"},
        quote_via=urllib.parse.quote,
    )
    d = fetch_json(f"{API}/documents?{params}")
    docs = d.get("value") or []
    if not docs:
        raise RuntimeError("无 document 记录")

    # 取最新的 start
    latest_start = max(x["start"] for x in docs)
    same = [x for x in docs if x["start"] == latest_start and x.get("type") == "Primary"]
    if not same:
        same = [x for x in docs if x["start"] == latest_start]
    reg = same[0]["registerId"]

    # 卷号约定不统一：
    # - 单卷法案 volumeNumber = 0
    # - 多卷法案 volumeNumber = 1..N
    # - 同一 registerId 可能既有 0 又有 1..N（0 是无效占位）
    # 因此取「Pdf 格式记录」的实际卷号集合；若存在 >0 的卷号，丢弃 0。
    pdf_vols = sorted(
        {int(x.get("volumeNumber") or 0) for x in same if x.get("format") == "Pdf"}
    )
    if not pdf_vols:
        pdf_vols = sorted({int(x.get("volumeNumber") or 0) for x in same})
    if any(v > 0 for v in pdf_vols):
        pdf_vols = [v for v in pdf_vols if v > 0] or pdf_vols

    return reg, latest_start, pdf_vols


def find_url(register_id: str, volume: int, fmt: str = "Pdf") -> str:
    return (
        f"{API}/documents/find(registerId='{register_id}',type='Primary',"
        f"format='{fmt}',uniqueTypeNumber=0,volumeNumber={volume},"
        f"rectificationVersionNumber=0)"
    )


CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 条文标题：正文里是 "1  Short title" / "126C  Regulations" 形式
# 条号后跟 1-4 个空格，再接标题（标题首字母大小写都可能有）
SECTION_LINE = re.compile(r"^(\d+[A-Z]{0,3})\s{1,4}(\S.{0,120})$")
# 章节结构
STRUCT_LINE = re.compile(
    r"^(Part\s+[IVXLC0-9]+[A-Z]?|Division\s+\d+[A-Z]?|Subdivision\s+\S+|"
    r"Schedule\s+\d+[A-Z]?|Chapter\s+\d+)\s*(.*)$"
)
# 目录行：带点线，或目录里的独立页码行
TOC_DOTS = re.compile(r"\.{4,}")
# 日期形式：“27 October 2026” / “1 January 2020”
DATE_LIKE = re.compile(
    r"^\d{1,2}\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b",
    re.I,
)


def pdf_to_lines(pdf_bytes: bytes) -> list[str]:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    lines: list[str] = []
    for page in doc:
        t = page.get_text("text")
        lines.extend(t.splitlines())
    doc.close()
    return lines


def lines_to_markdown(title: str, lines: list[str]) -> tuple[str, int]:
    """把 PDF 文本行转成 markdown。"""
    # 不用 H1：RAGFlow laws 解析器会把 H1 复制进每个子块（实测验证）
    out: list[str] = [f"<!-- document: {title} -->", ""]
    n_sec = 0
    buf: list[str] = []
    cur_head: str | None = None
    in_endnotes = False

    def flush() -> None:
        nonlocal buf, cur_head, n_sec
        if cur_head is None:
            buf = []
            return
        body = [x.strip() for x in buf if x.strip()]
        out.append("")
        out.append(f"## {cur_head}")
        out.append("")
        if body:
            out.extend(body)
        n_sec += 1
        buf = []
        cur_head = None

    for raw in lines:
        s = CONTROL.sub("", raw).strip()
        if not s:
            continue
        # 跳过页眉页脚
        if re.fullmatch(r"\d{1,4}", s):
            continue
        if re.fullmatch(r"(?i)(Federal Register of Legislation|Authorised Version).*", s):
            continue
        if re.fullmatch(r"(?i)Compilation No\.\s*\d+.*", s):
            continue
        if re.fullmatch(r"(?i)Compilation date:.*", s):
            continue
        # 目录行（带点线）
        if TOC_DOTS.search(s):
            continue
        # Endnotes 之后全是注释与修订史，不属条文。
        # 注意：目录里也有 "Endnotes" 条目，必须只认「已切出若干条文之后」的那次。
        if n_sec >= 10 and re.match(r"(?i)^(Endnotes?|Schedule\s+\d+\s*—?\s*Repeals?)$", s):
            in_endnotes = True
        if in_endnotes:
            continue

        m = STRUCT_LINE.match(s)
        if m:
            flush()
            out.append("")
            out.append(f"### {s}")
            out.append("")
            continue

        m = SECTION_LINE.match(s)
        # 判定为新条标题的条件：
        # - 条号后接的是标题短语（不是句子，不以句号结尾）
        # - 该行较短
        # - 标题不是日期（"27 October 2026" 会被误判成 Section 27）
        # - 标题不是纯数字
        if (
            m
            and len(s) < 140
            and not s.rstrip().endswith((".", ",", ";", ":"))
            and not s.startswith(("(", "（"))
            and not DATE_LIKE.match(s)
            and not m.group(2).isdigit()
        ):
            flush()
            cur_head = f"Section {m.group(1)} — {m.group(2)}".rstrip(" —")
            continue

        buf.append(s)

    flush()

    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md, n_sec


def main() -> int:
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--titles":
        only = set(sys.argv[2].split(","))

    out_dir = Path(__file__).parent / "au"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"source": "Australian Federal Register of Legislation",
                   "api": API, "fetched": [], "missing": []}

    for title_id, (slug, desc) in TITLES.items():
        if only and title_id not in only and slug not in only:
            continue
        try:
            reg, start, vols = latest_register(title_id)
        except Exception as e:  # noqa: BLE001
            print(f"  {slug}: 元数据失败 {e}", file=sys.stderr)
            index["missing"].append({"title_id": title_id, "slug": slug, "error": str(e)})
            continue

        all_lines: list[str] = []
        got = 0
        for v in vols:
            try:
                pdf = fetch_binary(find_url(reg, v))
            except Exception as e:  # noqa: BLE001
                print(f"  {slug} vol{v}: 下载失败 {e}", file=sys.stderr)
                continue
            all_lines.extend(pdf_to_lines(pdf))
            (out_dir / f"{slug}-vol{v}.pdf").write_bytes(pdf)
            got += 1

        if got == 0:
            index["missing"].append({"title_id": title_id, "slug": slug, "error": "no volumes"})
            continue

        md, n_sec = lines_to_markdown(f"{desc}", all_lines)
        (out_dir / f"{slug}.md").write_text(md, encoding="utf-8")
        print(f"  {slug}: {n_sec} sections, {len(md):,} chars, {got} vol(s) -> {slug}.md")
        index["fetched"].append({
            "title_id": title_id, "slug": slug, "register_id": reg,
            "start": start, "volumes": got, "sections": n_sec, "chars": len(md),
        })

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(x["sections"] for x in index["fetched"])
    print(f"\nAustralia: {len(index['fetched'])} acts, {total} sections, "
          f"{len(index['missing'])} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
