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
    for spec in "ecfr/title-19:51" "ecfr/title-15:16" "ecfr/title-31:6" \
                "wto:14" "au:5" "jp:9" "eu:2" "uk:1" "cn:39" "uncitral:1"; do
        dir="${spec%%:*}"; want="${spec##*:}"
        got=$(find "$CORPUS/$dir" -name '*.md' 2>/dev/null | wc -l | tr -d ' ' || true)
        got=${got:-0}
        status="OK"
        if [ "$got" -lt "$want" ]; then
            status="缺 $((want - got)) 个"
            missing=1
        fi
        printf "  %-16s %3s/%-3s %s\n" "$dir" "$got" "$want" "$status"
    done
    exit "$missing"
fi

echo "抓取语料（约 15 分钟，视网络）…"
cd "$CORPUS"
"$PY" fetch_ecfr.py --title 19
"$PY" fetch_ecfr.py --title 15
"$PY" fetch_ecfr.py --title 31
"$PY" fetch_wto.py
"$PY" fetch_au.py
"$PY" fetch_jp.py
"$PY" fetch_eu_uk.py
"$PY" fetch_cn.py
"$PY" fetch_cisg.py

echo
echo "完成。若 uk/ 为空：legislation.gov.uk 对非浏览器客户端返回 AWS WAF"
echo "挑战（HTTP 202 空 body），需在可访问的网络用浏览器下载 data.xml 后手工转换。"
echo
bash "$0" --check
