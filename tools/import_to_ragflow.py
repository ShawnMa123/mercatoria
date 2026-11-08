"""把语料批量导入 RAGFlow 知识库。

用法：
    python import_to_ragflow.py --kb <kb_id> --dry-run
    python import_to_ragflow.py --kb <kb_id> --go

设计要点：
- 上传用 multipart，每批不超过 8 个文件（避免单请求过大）
- 上传后统一触发 ingest（POST /api/v1/documents/ingest，需要 doc_ids + run）
- 轮询文档状态直到 COMPLETED 或超时
- 目标库必须已用 `laws` 解析器创建；本脚本会校验并拒绝其他解析器
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

BASE_HOST = "localhost"
BASE_PORT = 9380

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus"
# 中文法规的原始出处。仓库不随附这批文件（见 corpus/README），
# 需要时用 --laws-root 指定，或设 LAWREFBOOK_ROOT 环境变量。
LAWS_ROOT = Path(os.environ.get("LAWREFBOOK_ROOT", ""))

# 中文法规：与旧库同一批（旧库 50 份，这里按名称匹配全部）
CN_EXCLUDE_DIRS = {"案例", ".git", ".venv", "node_modules"}


from ragflow_config import api_key  # noqa: E402


def api(method: str, path: str, key: str, body: dict | None = None, timeout: int = 300):
    conn = http.client.HTTPConnection(BASE_HOST, BASE_PORT, timeout=timeout)
    headers = {"Authorization": "Bearer " + key}
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn.request(method, path, data, headers)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return r.status, {"raw": raw[:400]}


def upload_batch(dataset_id: str, paths: list[Path], key: str) -> dict:
    boundary = "----mercatoria" + uuid.uuid4().hex
    body = b""
    for p in paths:
        body += ("--%s\r\n" % boundary).encode()
        body += (
            'Content-Disposition: form-data; name="file"; filename="%s"\r\n' % p.name
        ).encode("utf-8")
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += p.read_bytes() + b"\r\n"
    body += ("--%s--\r\n" % boundary).encode()

    conn = http.client.HTTPConnection(BASE_HOST, BASE_PORT, timeout=600)
    conn.request(
        "POST",
        f"/api/v1/datasets/{dataset_id}/documents",
        body,
        {
            "Content-Type": "multipart/form-data; boundary=%s" % boundary,
            "Authorization": "Bearer " + key,
        },
    )
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    return json.loads(raw.decode("utf-8", "replace"))


def collect_corpus() -> list[Path]:
    """收集全部待导入文件。

    中文部分刻意收紧：语料范围与切分策略是两个变量，
    同时放开会让评测分不清是哪个在起作用（见交接文档 §7.4）。
    """
    files: list[Path] = []

    # 1) 中文法规（LawRefBook）—— 只收核心贸易法，不按关键词泛匹配
    cn_allow = [
        "海关法", "关税法", "进出口关税条例", "对外贸易法",
        "进出口商品检验法", "出口管制法", "两用物项出口管制条例",
        "货物进出口管理条例", "进出口货物原产地条例", "反倾销条例",
        "反补贴条例", "外汇管理条例", "海关事务担保条例",
        "技术进出口管理条例", "核出口管制条例", "导弹及相关物项和技术出口管制条例",
        "核两用品及相关技术出口管制条例", "反倾销和反补贴条例",
        "海关行政处罚实施条例", "进出口关税条例",
    ]
    # 优先用 tools/normalize_cn_corpus.py 规整过的副本
    # （去掉 H1，避免 RAGFlow laws 解析器把文件名复制进每个 chunk）
    cn_dir = CORPUS_ROOT / "cn"
    if cn_dir.exists():
        files.extend(sorted(cn_dir.glob("*.md")))
    elif LAWS_ROOT.exists():
        # 兜底：直接用原始 LawRefBook 目录（结构未规整，效果会差）
        latest: dict[str, tuple[str, Path]] = {}
        for p in LAWS_ROOT.rglob("*.md"):
            if any(part in CN_EXCLUDE_DIRS for part in p.parts):
                continue
            if not any(k in p.name for k in cn_allow):
                continue
            stem = p.stem
            if stem.endswith(")") and "(" in stem:
                title, _, date = stem.rpartition("(")
                date = date.rstrip(")")
            else:
                title, date = stem, ""
            prev = latest.get(title)
            if prev is None or date > prev[0]:
                latest[title] = (date, p)
        files.extend(p for _, p in latest.values())

    # 2) 我抓的外文语料
    for sub in ("ecfr/title-19", "ecfr/title-15", "wto", "jp", "au", "uncitral"):
        d = CORPUS_ROOT / sub
        if d.exists():
            files.extend(sorted(d.glob("*.md")))

    return sorted(set(files))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="目标知识库 ID")
    ap.add_argument("--batch", type=int, default=8, help="每批文件数")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go", action="store_true", help="确认执行")
    ap.add_argument("--skip-upload", action="store_true", help="只触发解析")
    args = ap.parse_args()

    if not args.dry_run and not args.go:
        print("需要 --dry-run 或 --go", file=sys.stderr)
        return 2

    key = api_key()

    # 校验目标库的解析器
    st, d = api("GET", f"/api/v1/datasets?page=1&page_size=100", key)
    target = None
    for x in (d.get("data") or []):
        if x.get("id") == args.kb:
            target = x
            break
    if not target:
        print(f"找不到知识库 {args.kb}", file=sys.stderr)
        return 2
    parser = target.get("parser_id") or target.get("chunk_method")
    print(f"目标库: {target.get('name')}  parser_id={parser}")
    if parser not in ("laws",):
        print(f"⚠️ 目标库解析器是 {parser}，不是 laws。中止。", file=sys.stderr)
        return 2

    files = collect_corpus()
    by_group: dict[str, int] = {}
    for p in files:
        try:
            g = p.relative_to(CORPUS_ROOT).parts[0]
        except ValueError:
            g = "cn-laws"
        by_group[g] = by_group.get(g, 0) + 1

    print(f"\n待导入 {len(files)} 个文件：")
    for g, n in sorted(by_group.items()):
        print(f"  {g}: {n}")

    if args.dry_run:
        print("\n[dry-run] 未执行任何写入。")
        return 0

    # 分批上传
    uploaded: list[str] = []
    batches = [files[i : i + args.batch] for i in range(0, len(files), args.batch)]
    for idx, batch in enumerate(batches, 1):
        try:
            r = upload_batch(args.kb, batch, key)
        except Exception as e:  # noqa: BLE001
            print(f"  批次 {idx}/{len(batches)} 上传异常: {e}", file=sys.stderr)
            continue
        if r.get("code") != 0:
            print(f"  批次 {idx}: code={r.get('code')} {r.get('message')}", file=sys.stderr)
            continue
        for x in r.get("data") or []:
            uploaded.append(x["id"])
        print(f"  批次 {idx}/{len(batches)}: +{len(r.get('data') or [])} 文件")

    print(f"\n已上传 {len(uploaded)} 个文档，触发解析…")
    for i in range(0, len(uploaded), 50):
        chunk = uploaded[i : i + 50]
        st, r = api(
            "POST",
            "/api/v1/documents/ingest",
            key,
            {"doc_ids": chunk, "run": "1"},
        )
        print(f"  ingest {i+1}-{i+len(chunk)}: HTTP {st} code={r.get('code')}")

    Path("imported_doc_ids.json").write_text(
        json.dumps(uploaded, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n文档 ID 已写入 imported_doc_ids.json（{len(uploaded)} 个）")
    print("用 tools/check_import.py --kb <id> 轮询进度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
