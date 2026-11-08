"""轮询 RAGFlow 知识库的导入进度，并汇总 chunk 统计。

用法：
    python check_import.py --kb <kb_id>
    python check_import.py --kb <kb_id> --wait
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request

from ragflow_config import api_key  # noqa: E402

BASE = "http://localhost:9380"


def get(path: str, key: str, timeout: int = 120):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--wait", action="store_true", help="等到全部完成")
    ap.add_argument("--interval", type=int, default=15)
    args = ap.parse_args()

    key = api_key()
    last = None
    while True:
        docs: list[dict] = []
        page = 1
        while True:
            d = get(
                f"/api/v1/datasets/{args.kb}/documents?page={page}&page_size=100", key
            )
            batch = d["data"]["docs"]
            docs.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        states: dict[str, int] = {}
        chunks = 0
        for x in docs:
            s = x.get("ingestion_status") or "?"
            states[s] = states.get(s, 0) + 1
            chunks += x.get("chunk_count") or 0

        line = f"文档 {len(docs)} | chunk {chunks} | " + " ".join(
            f"{k}={v}" for k, v in sorted(states.items())
        )
        if line != last:
            print(line, flush=True)
            last = line

        running = any(
            s in ("RUNNING", "UNSTART", "PENDING", "STARTING") for s in states
        )
        if not args.wait or not running:
            break
        time.sleep(args.interval)

    # 各文档 chunk 数分布
    rows = sorted(
        ((x.get("chunk_count") or 0, x.get("name") or "?") for x in docs), reverse=True
    )
    out = io.open("import_stats.txt", "w", encoding="utf-8")
    for c, n in rows:
        out.write(f"{c}\t{n}\n")
    out.close()
    print(f"\n明细已写入 import_stats.txt（{len(rows)} 行）")
    print("chunk 最多的 5 个：")
    for c, n in rows[:5]:
        print(f"  {c:5d}  {n}")
    print("chunk 最少的 5 个：")
    for c, n in rows[-5:]:
        print(f"  {c:5d}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
