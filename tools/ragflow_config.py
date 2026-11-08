"""评测工具链的共享配置：RAGFlow 地址与 API Key。

为什么单独抽出来：
    原来 6 个脚本各自硬编码了 KEY_FILE = Path(r"C:\\MWB\\Code\\...\\ragflow-cheatsheet.md")，
    那个文件里存着明文 API Key。仓库要公开，路径和密钥都不能进版本控制。

取值优先级：
    1. 环境变量 RAGFLOW_API_KEY / RAGFLOW_BASE
    2. 仓库根目录的 .env 文件（已在 .gitignore 中）
    3. 都没有 → 报错并说明怎么配

.env 格式（三行，无引号）：
    RAGFLOW_BASE=http://localhost:9380
    RAGFLOW_API_KEY=ragflow-xxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_BASE = "http://localhost:9380"


def _load_env_file() -> None:
    """把 .env 里的键值读进 os.environ，不覆盖已存在的环境变量。"""
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def base_url() -> str:
    """RAGFlow 的 API 地址。"""
    _load_env_file()
    return os.environ.get("RAGFLOW_BASE", DEFAULT_BASE).rstrip("/")


def api_key() -> str:
    """RAGFlow 的 API Key。缺失时给出可操作的错误信息。"""
    _load_env_file()
    key = os.environ.get("RAGFLOW_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError(
        f"未配置 RAGFlow API Key。两种方式任选其一：\n"
        f"  1. 设置环境变量 RAGFLOW_API_KEY=ragflow-xxxx\n"
        f"  2. 在 {ENV_FILE} 里写一行 RAGFLOW_API_KEY=ragflow-xxxx\n"
        f"（该文件已在 .gitignore 中，不会被提交）"
    )
