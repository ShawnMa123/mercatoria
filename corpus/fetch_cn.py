"""从 LawRefBook 镜像抓取中文法规正文，规整后放入 corpus/cn/。

用法：
    python fetch_cn.py                 # 抓取 CATALOG 中全部
    python fetch_cn.py --only 保障措施  # 按文件名关键词筛选

输出：
    corpus/cn/{名称}({日期}).md   —— 与既有语料同一命名约定，
                                     mercatoria/manifest.py 从文件名取生效日期

为什么用 LawRefBook：国家法律法规数据库（flk.npc.gov.cn）返回的是 SPA 空壳，
正文需浏览器执行 JS；LawRefBook 是逐条全文的 git 镜像，可直接 raw 下载，
且本仓库既有中文语料本来就来自它（见 tools/normalize_cn_corpus.py）。

设计要点：
- H1 与 INFO END 之前的元信息转成 HTML 注释（normalize_cn_corpus.normalize），
  否则法规标题会复制进每个条文块。
- 仓库路径必须逐字匹配（文件名带发布日期）。抓不到只记录 missing，不静默跳过。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = "https://raw.githubusercontent.com/LawRefBook/Laws/master/"
UA = "mercatoria-corpus-builder/0.1 (+legal retrieval POC)"

# LawRefBook 仓库路径 -> corpus/cn 下的文件名（保持“名称(日期).md”约定）
CATALOG: dict[str, str] = {
    # 海关监管、估价、归类
    "行政法规/海关稽查条例(2022-03-29).md": "海关稽查条例(2022-03-29).md",
    "行政法规/知识产权海关保护条例(2018-03-19).md": "知识产权海关保护条例(2018-03-19).md",
    "部门规章/海关总署/海关审定进出口货物完税价格办法(2013-12-25).md":
        "海关审定进出口货物完税价格办法(2013-12-25).md",
    "部门规章/海关总署/海关进出口货物征税管理办法(2024-10-28).md":
        "海关进出口货物征税管理办法(2024-10-28).md",
    "部门规章/海关总署/海关进出口货物申报管理规定(2018-11-23).md":
        "海关进出口货物申报管理规定(2018-11-23).md",
    "部门规章/海关总署/海关进出口货物商品归类管理规定(2021-09-06).md":
        "海关进出口货物商品归类管理规定(2021-09-06).md",
    "部门规章/海关总署/海关预裁定管理暂行办法(2017-12-26).md":
        "海关预裁定管理暂行办法(2017-12-26).md",
    "部门规章/海关总署/海关综合保税区管理办法(2022-01-01).md":
        "海关综合保税区管理办法(2022-01-01).md",
    "部门规章/海关总署/海关注册登记和备案企业信用管理办法(2021-09-13).md":
        "海关注册登记和备案企业信用管理办法(2021-09-13).md",
    "部门规章/海关总署/海关报关单位备案管理规定(2021-11-19).md":
        "海关报关单位备案管理规定(2021-11-19).md",
    # 原产地与自贸协定
    "部门规章/海关总署/关于非优惠原产地规则中实质性改变标准的规定(2018-04-28).md":
        "关于非优惠原产地规则中实质性改变标准的规定(2018-04-28).md",
    "部门规章/海关总署/海关《区域全面经济伙伴关系协定》项下进出口货物原产地管理办法(2021-11-23).md":
        "海关RCEP项下进出口货物原产地管理办法(2021-11-23).md",
    # 贸易救济：补齐三大支柱中的《保障措施条例》
    "行政法规/保障措施条例(2004-03-31).md": "保障措施条例(2004-03-31).md",
    # 出口管制许可
    "部门规章/商务部/两用物项和技术进出口许可证管理办法(2005-12-31).md":
        "两用物项和技术进出口许可证管理办法(2005-12-31).md",
    "部门规章/商务部/两用物项和技术出口通用许可管理办法(2009-05-13).md":
        "两用物项和技术出口通用许可管理办法(2009-05-13).md",
    # 国际结算
    "司法解释/最高人民法院关于审理信用证纠纷案件若干问题的规定(2020-12-29).md":
        "最高人民法院关于审理信用证纠纷案件若干问题的规定(2020-12-29).md",
    "司法解释/最高人民法院关于审理独立保函纠纷案件若干问题的规定(2020-12-29).md":
        "最高人民法院关于审理独立保函纠纷案件若干问题的规定(2020-12-29).md",
}


def fetch(path: str, retries: int = 3, timeout: int = 60) -> str:
    url = RAW + urllib.request.quote(path)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"抓取失败 {url}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔的文件名关键词")
    args = ap.parse_args()
    only = [x.strip() for x in args.only.split(",") if x.strip()]

    sys.path.insert(0, str(ROOT / "tools"))
    from normalize_cn_corpus import normalize  # noqa: PLC0415

    out_dir = Path(__file__).parent / "cn"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"source": "LawRefBook (github.com/LawRefBook/Laws)",
                   "fetched": [], "missing": []}

    for repo_path, filename in CATALOG.items():
        if only and not any(key in filename for key in only):
            continue
        try:
            raw = fetch(repo_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  {filename}: {exc}", file=sys.stderr)
            index["missing"].append({"path": repo_path, "file": filename, "error": str(exc)})
            continue
        text = normalize(raw)
        if not text.lstrip().startswith("<!-- document:"):
            print(f"  {filename}: 未识别到法规标题 — 跳过", file=sys.stderr)
            index["missing"].append({"path": repo_path, "file": filename, "error": "no title"})
            continue
        (out_dir / filename).write_text(text, encoding="utf-8")
        print(f"  {filename}: {len(text):,} chars")
        index["fetched"].append(
            {"path": repo_path, "file": filename, "chars": len(text)}
        )

    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"中文语料: {len(index['fetched'])} 篇, {len(index['missing'])} 缺失")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
