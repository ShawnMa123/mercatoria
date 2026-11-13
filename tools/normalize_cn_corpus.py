"""把 LawRefBook 的中文法规 markdown 规整成适合 RAGFlow `laws` 解析器的结构。

实测背景：
    RAGFlow 的 `laws` 解析器（HierarchyTitleChunker）会把 H1 的文字
    **复制进每一个子块**。LawRefBook 的原始文件是：

        # 中华人民共和国关税法
        2024年4月26日 第十四届全国人民代表大会常务委员会第九次会议通过
        <!-- INFO END -->

        第一章 总则
        第一条 ……

    直接导入时「中华人民共和国关税法」这句话会出现在每个 chunk 开头，
    既浪费 context 又稀释向量区分度（实测 60% 文档受影响）。

处理：
    1. H1 及 INFO END 之前的元信息 → 转成 HTML 注释（单独成块，不污染条文块）
    2. 章标题保持原样（解析器的 group 1 正则能识别 `第X章`）

用法：
    python normalize_cn_corpus.py --src <LawRefBook 目录> --out corpus/cn
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

# 与 import_to_ragflow.py 保持一致的收录清单
CN_ALLOW = [
    "海关法", "关税法", "进出口关税条例", "对外贸易法",
    "进出口商品检验法", "出口管制法", "两用物项出口管制条例",
    "货物进出口管理条例", "进出口货物原产地条例", "反倾销条例",
    "反补贴条例", "外汇管理条例", "海关事务担保条例",
    "技术进出口管理条例", "核出口管制条例", "导弹及相关物项和技术出口管制条例",
    "核两用品及相关技术出口管制条例", "反倾销和反补贴条例",
    "海关行政处罚实施条例",
    "海关稽查条例", "知识产权海关保护条例", "海关审定进出口货物完税价格办法",
    "海关进出口货物征税管理办法", "海关进出口货物申报管理规定",
    "海关进出口货物商品归类管理规定", "海关预裁定管理暂行办法",
    "海关综合保税区管理办法", "海关注册登记和备案企业信用管理办法",
    "海关报关单位备案管理规定",
    "关于非优惠原产地规则中实质性改变标准的规定", "海关RCEP项下进出口货物原产地管理办法",
    "保障措施条例", "两用物项和技术进出口许可证管理办法",
    "两用物项和技术出口通用许可管理办法",
    "最高人民法院关于审理信用证纠纷案件若干问题的规定",
    "最高人民法院关于审理独立保函纠纷案件若干问题的规定",
]
EXCLUDE_DIRS = {"案例", ".git", ".venv", "node_modules"}

INFO_END = "<!-- INFO END -->"


def normalize(text: str) -> str:
    """H1 + 元信息块 → HTML 注释。"""
    lines = text.splitlines()

    # 找第一个非空行是否为 H1
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not lines[i].startswith("# "):
        return text  # 没有 H1，原样返回

    title = lines[i][2:].strip()

    # 找 INFO END（元信息结束标记）
    end = None
    for j in range(i + 1, min(len(lines), i + 60)):
        if INFO_END in lines[j]:
            end = j
            break

    if end is None:
        # 没有 INFO END：只把 H1 换成注释，其后内容保持
        new = lines[:i] + [f"<!-- document: {title} -->"] + lines[i + 1 :]
        return "\n".join(new)

    meta = [ln.strip() for ln in lines[i + 1 : end] if ln.strip()]

    head = [f"<!-- document: {title} -->"]
    if meta:
        head.append("")
        head.extend(f"<!-- {m} -->" for m in meta)
    head.append("")

    # INFO END 之后的内容才是正文
    body = lines[end + 1 :]
    # 去掉开头连续空行
    k = 0
    while k < len(body) and not body[k].strip():
        k += 1
    body = body[k:]

    return "\n".join(head + body)


def collect(src: Path) -> list[Path]:
    files: dict[str, tuple[str, Path]] = {}
    for p in src.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if not any(k in p.name for k in CN_ALLOW):
            continue
        stem = p.stem
        if stem.endswith(")") and "(" in stem:
            title, _, date = stem.rpartition("(")
            date = date.rstrip(")")
        else:
            title, date = stem, ""
        prev = files.get(title)
        if prev is None or date > prev[0]:
            files[title] = (date, p)
    return [p for _, p in files.values()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=os.environ.get("LAWREFBOOK_ROOT", ""),
        help="LawRefBook 目录（也可用 LAWREFBOOK_ROOT 环境变量指定）",
    )
    ap.add_argument("--out", default="corpus/cn")
    args = ap.parse_args()
    if not args.src:
        ap.error("须指定 --src 或设置 LAWREFBOOK_ROOT 环境变量")

    src = Path(args.src)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    files = collect(src)
    index = []
    for p in sorted(files):
        text = p.read_text(encoding="utf-8", errors="replace")
        fixed = normalize(text)
        (out / p.name).write_text(fixed, encoding="utf-8")
        n_art = len(re.findall(r"^第[零一二三四五六七八九十百千0-9]+条", fixed, re.M))
        index.append({"file": p.name, "source": str(p), "articles": n_art,
                      "chars": len(fixed)})
        print(f"  {p.name[:44]:46s} {n_art:4d} 条  {len(fixed):>8,d} 字")

    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共 {len(index)} 份，条文合计 {sum(x['articles'] for x in index)}")
    print(f"输出目录 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
