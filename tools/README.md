# 工具

## 导入

```bash
# 建 laws 解析器知识库（一次性）
python tools/create_kb.py

# 导入语料（先 dry-run 看清单）
python tools/import_to_ragflow.py --kb <kb_id> --dry-run
python tools/import_to_ragflow.py --kb <kb_id> --go

# 轮询进度
python tools/check_import.py --kb <kb_id> --wait
```

## 评测

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
