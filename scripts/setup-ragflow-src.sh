#!/usr/bin/env bash
# 重建 RAGFlow 源码副本（ragflow-src/）。
#
# 为什么需要这个脚本：本仓库不随附 RAGFlow 源码。它是 Apache-2.0 许可的
# 第三方项目，随附一份 67MB 的副本既不必要也容易与上游脱节。
# 仓库只保留「我们改了哪几处」的记录，见 ragflow-src/UPSTREAM.md。
#
# 用法：
#   bash scripts/setup-ragflow-src.sh            # 拉取上游并重放我们的改动
#   bash scripts/setup-ragflow-src.sh --verify   # 只校验现有副本的版本
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/ragflow-src"
UPSTREAM="https://github.com/infiniflow/ragflow.git"
COMMIT="3c6d2255af06164f0818d0959af10de0afd8b8fa"

if [ "${1:-}" = "--verify" ]; then
    if [ ! -d "$DEST/internal" ]; then
        echo "ragflow-src/ 不存在或不完整" >&2
        exit 1
    fi
    echo "ragflow-src/ 已存在：$(find "$DEST" -type f | wc -l | tr -d ' ') 个文件"
    echo "期望的上游 commit：$COMMIT"
    echo "（副本没有 .git，无法自动核对版本，改动记录见 ragflow-src/UPSTREAM.md）"
    exit 0
fi

if [ -d "$DEST/internal" ]; then
    echo "ragflow-src/ 已存在，跳过。要重建先删掉它。"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "克隆上游 $COMMIT（浅克隆）…"
git clone --filter=blob:none --no-checkout "$UPSTREAM" "$TMP/up"
git -C "$TMP/up" checkout "$COMMIT"

echo "复制后端与前端…"
mkdir -p "$DEST"
for item in internal cmd conf docker sdk tools web go.mod go.sum LICENSE AGENTS.md README.md; do
    [ -e "$TMP/up/$item" ] && cp -r "$TMP/up/$item" "$DEST/"
done

echo
echo "完成：$(find "$DEST" -type f | wc -l | tr -d ' ') 个文件"
echo
echo "注意：这是纯净的上游副本，**不含我们的改动**。"
echo "我们的改动记录在 ragflow-src/UPSTREAM.md 的表格里，需要按表手工重放："
echo "  1. internal/ingestion/component/schema/chunker.go    ChunkTokenCap 512 → 2048"
echo "  2. internal/ingestion/component/chunker/title_token_cap_test.go  同步断言"
echo "  3. internal/engine/elasticsearch/chunk.go            sortByScore 并列按 chunk_id"
echo "  4. internal/service/nlp/retrieval.go                 idxScores 并列按 chunk_id"
echo "  5. internal/engine/elasticsearch/chunk_helpers_test.go  新增并列排序单测"
