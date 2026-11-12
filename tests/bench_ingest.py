"""Benchmark a batch upload of N small Markdown laws through the HTTP API.

Run from the test container:
  python tests/bench_ingest.py --base http://host.docker.internal:8080 --files 200
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid


def _multipart(files: list[tuple[str, bytes]], fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = "----mercatoria" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for filename, data in files:
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
                f'filename="{filename}"\r\nContent-Type: text/markdown\r\n\r\n'
            ).encode()
        )
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _post(base: str, path: str, body: bytes, content_type: str, timeout: float) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--files", type=int, default=100)
    ap.add_argument("--jurisdiction", default="CN")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    files: list[tuple[str, bytes]] = []
    for i in range(args.files):
        body = (
            f"<!-- document: 压测法规 {args.tag}{i} -->\n"
            f"## 第一章 总则\n\n"
            f"第一条 压测法规 {args.tag}{i} 适用于进口货物的征收管理。\n\n"
            f"第二条 进口货物的完税价格以成交价格为基础确定。\n\n"
            f"第三条 出口经营者应当依法提交许可证。\n\n"
            f"第四条 违反规定的，依照海关法处理。\n"
        )
        files.append((f"bench-{i:03d}.md", body.encode("utf-8")))

    body, content_type = _multipart(
        files, {"jurisdiction": args.jurisdiction, "applies_to": "customs"}
    )
    started = time.perf_counter()
    result = _post(args.base, "/v1/documents", body, content_type, timeout=300)
    upload_done = time.perf_counter()
    job_id = result["job_id"]
    while True:
        job = _get(args.base, f"/v1/jobs/{job_id}")
        if job["status"] != "running":
            break
        time.sleep(0.5)
    finished = time.perf_counter()
    print(
        json.dumps(
            {
                "files": args.files,
                "upload_seconds": round(upload_done - started, 2),
                "ingest_seconds": round(finished - upload_done, 2),
                "total_seconds": round(finished - started, 2),
                "status": job["status"],
                "result": job.get("result"),
                "error": job.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if job["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
