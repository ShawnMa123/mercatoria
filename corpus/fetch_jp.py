"""从日本 e-Gov 法令 API 抓取法规，转成可导入 RAGFlow 的 markdown。

用法：
    python fetch_jp.py
    python fetch_jp.py --laws 329AC0000000061

输出：
    corpus/jp/{slug}.md
    corpus/jp/index.json

设计要点：
- e-Gov API v2 无需认证，`GET /api/2/law_data/{law_id}` 返回 JSON
- JSON 里的 `law_full_text` 是 XML 转成的树，标签保留
- 条文锚点是 `tag=Article`，其 `children` 里 `tag=ArticleTitle` 给出「第十五条」
- 输出用 `## 第十五条` 作二级标题，匹配 RAGFlow laws 解析器 group 1
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://laws.e-gov.go.jp/api/2"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# law_id -> (输出 slug, 中文说明)
LAWS = {
    "329AC0000000061": ("kazei-hou", "関税法（关税法）"),
    "143AC0000000054": ("kazei-teiritsu-hou", "関税定率法（关税定率法）"),
    "324AC0000000228": ("gaiamei-hou", "外国為替及び外国貿易法（外汇及外贸法 / 外為法）"),
    "324CO0000000378": ("yushutsu-boueki-kanri-rei", "輸出貿易管理令（出口贸易管理令）"),
    "329CO0000000150": ("kanzei-hou-sekourei", "関税法施行令（关税法施行令）"),
    "329CO0000000155": ("kanzei-teiritsu-hou-sekourei", "関税定率法施行令（关税定率法施行令）"),
    "335AC0000000036": ("kanzei-zantei-sochi-hou", "関税暫定措置法（关税暂定措施法）"),
    "355CO0000000260": ("gaikoku-kawase-rei", "外国為替令（外汇令）"),
    "324CO0000000414": ("yunyu-boueki-kanri-rei", "輸入貿易管理令（进口贸易管理令）"),
}


def fetch_json(url: str, retries: int = 3, timeout: int = 120) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last}")


def node_text(node: dict | str) -> str:
    """递归取出节点下所有文本。"""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    parts: list[str] = []
    for child in node.get("children") or []:
        parts.append(node_text(child))
    return "".join(parts)


SKIP_TAGS = {
    "ArticleCaption",  # 条文見出し（如「（課税物件の範囲）」），保留更有用
}


def render_article(art: dict) -> str:
    """渲染一个 Article 节点为 markdown 正文。"""
    lines: list[str] = []

    def walk(n: dict, depth: int = 0) -> None:
        tag = n.get("tag", "")
        kids = n.get("children") or []

        if tag == "ArticleTitle":
            return  # 已在标题行处理
        if tag in ("Chapter", "Section", "Part", "Subsection"):
            return  # 由上层结构处理

        if tag == "Paragraph":
            num = (n.get("attr") or {}).get("Num", "")
            body = node_text(n).strip()
            if body:
                prefix = f"{num}　" if num else ""
                lines.append(prefix + body)
            return

        if tag in ("Item",):
            num = (n.get("attr") or {}).get("Num", "")
            body = node_text(n).strip()
            if body:
                lines.append(f"{num}　{body}" if num else body)
            return

        if tag == "ArticleCaption":
            cap = node_text(n).strip()
            if cap:
                lines.append(cap)
            return

        # 其他容器递归
        for c in kids:
            if isinstance(c, dict):
                walk(c, depth + 1)

    walk(art)
    return "\n".join(x for x in lines if x.strip())


def law_to_markdown(data: dict) -> tuple[str, int]:
    """把整部法令转成 markdown。返回 (md, article 数)。"""
    ri = data.get("revision_info") or {}
    li = data.get("law_info") or {}
    title = ri.get("law_title") or li.get("law_id") or "Unknown"

    meta: list[str] = []
    for k, label in (
        ("law_num", "法令番号"),
        ("promulgation_date", "公布日"),
        ("amendment_promulgate_date", "改正公布日"),
        ("amendment_enforcement_date", "施行日"),
    ):
        v = li.get(k) or ri.get(k)
        if v:
            meta.append(f"- {label}: {v}")

    # 不用 H1：RAGFlow laws 解析器会把 H1 复制进每个子块（实测验证）
    out: list[str] = [f"<!-- document: {title} -->", ""]
    if meta:
        out.extend(meta)
        out.append("")

    n_art = 0
    ft = data.get("law_full_text") or {}

    def walk_container(n: dict) -> None:
        nonlocal n_art
        tag = n.get("tag", "")
        kids = n.get("children") or []

        if tag in ("Chapter", "Section", "Part", "Subsection"):
            num = (n.get("attr") or {}).get("Num", "")
            # 章節标题在子节点 ChapterTitle / SectionTitle
            t = ""
            for c in kids:
                if isinstance(c, dict) and c.get("tag") in (
                    "ChapterTitle",
                    "SectionTitle",
                    "PartTitle",
                ):
                    t = node_text(c).strip()
                    break
            if num or t:
                out.append("")
                out.append(f"### {tag} {num}　{t}".rstrip())
                out.append("")

        if tag == "Article":
            num = (n.get("attr") or {}).get("Num", "")
            atitle = ""
            for c in kids:
                if isinstance(c, dict) and c.get("tag") == "ArticleTitle":
                    atitle = node_text(c).strip()
                    break
            heading = atitle or f"第{num}条"
            body = render_article(n)
            out.append("")
            out.append(f"## {heading}")
            out.append("")
            if body:
                out.append(body)
            n_art += 1
            return

        for c in kids:
            if isinstance(c, dict):
                walk_container(c)

    walk_container(ft)

    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md, n_art


def main() -> int:
    only = None
    if len(sys.argv) > 2 and sys.argv[1] == "--laws":
        only = set(sys.argv[2].split(","))

    out_dir = Path(__file__).parent / "jp"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"source": "Japan e-Gov 法令 API v2", "base": BASE,
                   "fetched": [], "missing": []}

    for law_id, (slug, desc) in LAWS.items():
        if only and law_id not in only and slug not in only:
            continue
        url = f"{BASE}/law_data/{law_id}"
        try:
            data = fetch_json(url)
        except Exception as e:  # noqa: BLE001
            print(f"  {slug}: {e}", file=sys.stderr)
            index["missing"].append({"law_id": law_id, "slug": slug, "error": str(e)})
            continue

        md, n_art = law_to_markdown(data)
        if n_art == 0:
            print(f"  {slug}: 未识别到 Article — 跳过", file=sys.stderr)
            index["missing"].append({"law_id": law_id, "slug": slug, "error": "no articles"})
            continue

        fname = f"{slug}.md"
        (out_dir / fname).write_text(md, encoding="utf-8")
        title = (data.get("revision_info") or {}).get("law_title", "")
        print(f"  {slug}: {n_art} articles, {len(md):,} chars -> {fname}  [{desc}]")
        index["fetched"].append({
            "law_id": law_id, "slug": slug, "file": fname,
            "title": title, "articles": n_art, "chars": len(md),
        })

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(x["articles"] for x in index["fetched"])
    print(f"\nJapan: {len(index['fetched'])} laws, {total} articles, "
          f"{len(index['missing'])} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
