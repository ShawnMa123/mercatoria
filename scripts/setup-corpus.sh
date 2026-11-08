#!/usr/bin/env bash
# 重建本地语料（corpus/）。
#
# 为什么需要这个脚本：本仓库不随附语料文件 —— 法规文本的转载受各来源的
# 使用条款约束（中文法规来自 LawRefBook，WTO 协定有其自己的版权声明）。
# 仓库只保留抓取脚本，语料由使用者自行从原始来源获取。
#
# 用法：
#   bash scripts/setup-corpus.sh           # 抓外文语料（美/澳/日/WTO/CISG）
#   bash scripts/setup-corpus.sh --check   # 只检查现有语料是否齐全
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="$ROOT/corpus"
PY="$ROOT/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "找不到虚拟环境：$ROOT/.venv" >&2
    echo "先创建并安装依赖：" >&2
    echo "  python -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [ "${1:-}" = "--check" ]; then
    echo "检查语料完整性…"
    missing=0
    for spec in "ecfr/title-19:44" "ecfr/title-15:8" "wto:10" "au:3" "jp:4" "uncitral:1"; do
        dir="${spec%%:*}"; want="${spec##*:}"
        got=$(find "$CORPUS/$dir" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
        status="OK"
        [ "$got" -lt "$want" ] && { status="缺 $((want - got)) 个"; missing=1; }
        printf "  %-16s %3s/%-3s %s\n" "$dir" "$got" "$want" "$status"
    done
    cn=$(find "$CORPUS/cn" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-16s %3s      %s\n" "cn" "$cn" "（需自行从 LawRefBook 获取）"
    exit "$missing"
fi

echo "抓取外文语料（约 10 分钟，视网络）…"
cd "$CORPUS"
"$PY" fetch_ecfr.py --title 19
"$PY" fetch_ecfr.py --title 15
"$PY" fetch_wto.py
"$PY" fetch_au.py
"$PY" fetch_jp.py
"$PY" fetch_cisg.py

echo
echo "完成。中文法规（corpus/cn/）不在此脚本范围内："
echo "  法规原文需从 LawRefBook 或其他来源自行获取，"
echo "  放到 corpus/cn/ 后可用 tools/normalize_cn_corpus.py 规范化。"
echo
bash "$0" --check
