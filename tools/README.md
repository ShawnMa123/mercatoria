# 工具

## 导入

```bash
# 建 laws 解析器知识库（一次性）。
# 仓库不提供建库脚本：在 RAGFlow 界面新建知识库，解析方法选 `laws`；
# import_to_ragflow.py 会校验目标库的 parser_id，不匹配会拒绝导入。

# 导入语料（先 dry-run 看清单）
python tools/import_to_ragflow.py --kb <kb_id> --dry-run
python tools/import_to_ragflow.py --kb <kb_id> --go

# 轮询进度
python tools/check_import.py --kb <kb_id> --wait
```

## 评测

## 本地服务（推荐）

```bash
# 先建索引（见根 README），然后直接跑本地检索评测
python tools/eval_local.py --config config.example.json --limit 50
python tools/eval_local.py --config config.example.json --tag local-final

# 参数扫描（不需要重建索引）
for w in 0.3 0.5 0.7; do
  python tools/eval_local.py --limit 50 --vec-weight $w --tag w$w
done
```

`eval_retrieval.py` 保留为 RAGFlow 基线；`eval_local.py` 走
`mercatoria.retrieve`，复用同一批用例和 strict/loose 判据。

## RAGFlow 基线

```bash
# 生成用例
python tools/gen_test_cases.py

# 冒烟
python tools/run_eval.py --kb <id> --limit 5

# 思考开关对比
python tools/run_eval.py --kb <id> --limit 20 --thinking-compare

# 全量
python tools/run_eval.py --kb <id> --full --thinking-compare
```

## 关键设计

**语料准备必须用 `laws` 解析器。**
`general`（别名 `naive`）是 512 token 窗口切块，会把一条法条切散、跨条合并，
导致下游无法判断某个 chunk 对应哪一条。实测《海关法》用 `general` 切出 33 块
（切在句中、跨 4 条），用 `laws` 切出 117 块且 98.3% 是完整法条。

**测试用例不要用「情形N」等无语义后缀凑数。**
实测这类后缀会被模型当作实体去匹配，把检索带偏，
然后模型看到不相关的内容就拒答 —— 表现为「检索失败」，实际是用例构造错误。

**思考开关（`llm_setting.thinking`）的真实影响：**
- 耗时约 3.5–4 倍（19s vs 5s）
- 开思考时 `max_tokens` 必须给足（≥8192），否则 reasoning 内容吃光配额，
  `completion_tokens=0`、answer 为空
- 准确率差异在小样本下不稳定，需 ≥50 条才能看出趋势
