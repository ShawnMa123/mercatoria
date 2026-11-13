# tools/ 使用约定

## 评测纪律（v2 起）

| 集合 | 文件 | 用途 | 纪律 |
|---|---|---|---|
| dev | `corpus/eval/test_cases.json` | 调参、消融 | 可以反复跑，可以据此改代码 |
| hold-out | `tests/eval/holdout_cases.json` | 里程碑验收 | **只看不改**；每次运行往 `docs/holdout-log.md` 追加一行（日期 / commit / 指标） |
| 真题 | `tests/eval/trade_law_questions.json` | 业务口径验收 | fixture 改动必须附原文依据（题号 / 变更 / `doc_id:article_id`） |

- hold-out 由 `python tools/make_holdout.py` 按类别分层抽样生成（固定 seed），生成后 dev 运行默认排除这些 id；
  `python tools/eval_local.py --holdout` 才读它们。
- 禁止用 hold-out 的失败样本反推调参；发现问题的题应写进 `docs/holdout-log.md` 而不是直接改代码。
- 任何会改变指标口径的改动（判分规则、fixture、语料）必须与检索算法改动分开提交，并在 commit message 里写明"是否改变评测口径"。

## 常用工具

| 工具 | 用途 |
|---|---|
| `eval_local.py` | 本地检索评测（`--legacy-scoring` 旧口径 / `--fusion` 多查询 / `--workers N` 并发 / `--no-quota` 关配额 / `--jurisdiction-filter` 走法域过滤） |
| `eval_retrieval.py` | RAGFlow 侧评测（`score_case` 被 `eval_local` 复用） |
| `profile_retrieve.py` | 分阶段耗时 profile，输出 `docs/profile-<日期>.md` |
| `audit_fixture.py` | 只读审计真题 fixture 的 top-5 命中，输出 `docs/fixture-audit-<日期>.md` |
| `make_holdout.py` | 生成冻结 hold-out |
| `make_quota_cases.py` | 生成配额精度评测用例 |
| `eval_quota.py` | 配额策略对比（off / hard / soft） |
| `check_corpus.py` | 语料完整性检查（`scripts/setup-corpus.sh --check` 的实现） |
