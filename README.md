# mercatoria

**进出口贸易法规检索服务** —— 为 `smart-edit` 的 Agent 3 提供可引用的法规原文片段。

> `lex mercatoria`（商人法）指跨国商事交易自发形成的规则体系：不由任何国家立法机关制定，
> 而由商人实践、行业惯例与国际公约构成。这个名字同时界定了语料范围 ——
> 不装刑法、婚姻法、行政诉讼法，不是因为没做，而是它们不属 lex mercatoria。

## 这是什么，不是什么

**是**：`queries + filters → chunks` 的检索服务。

**不是**：
- ❌ RAG 库 —— 检索策略（切分/向量/rerank）是本仓库的自由
- ❌ 问答系统 —— 生成答案、判断合规是 `smart-edit` 里 Agent 3 的 `answer` 节点
- ❌ 法律意见 —— 本服务返回「可被引用的原文片段」，不是「结论」

**另一个区分**：通用文档检索引擎只回答「我找到了什么」，
`mercatoria` 还要回答「我**不**覆盖什么」（`corpus.jurisdictions` / `corpus.status`）。

## 语料范围

| 类别 | 内容 | 上游 |
|---|---|---|
| 进出口贸易 | 关税法、进出口商品检验法、原产地规则、反倾销/反补贴、两用物项出口管制 | LawRefBook、US eCFR Title 19 |
| 国际经贸规则 | GATT 1994、TBT、SPS、TFA、反倾销、海关估价、CISG | WTO、UNCITRAL |
| 银行规定 | 商业银行法、反洗钱法、外汇管理条例、担保制度、独立保函司法解释 | LawRefBook、eCFR Title 31 |
| 出口管制 / 制裁 | EAR（Title 15）、OFAC（Title 31） | US eCFR |

**明确不装**：刑法、婚姻家庭、行政诉讼、劳动法、知识产权、地方性法规（除非直接涉及贸易）。

## 文档

设计文档存放在本地 `docs/` 目录，**暂不随本仓库公开**。包括：契约 v1（字段表、
`chunk_id` 算法、不变式清单）、交接文档、上游数据源实测记录、RAGFlow 复用评估。

下面「三条最硬的约束」是其中对实现影响最大的部分，直接写在这里以免失传。

## 三条最硬的约束

这三点错了，故障会在**演示时**才暴露，且症状指向下游：

**1. `chunk_id` 必须内容派生且稳定**

```
chunk_id = sha256(doc_id + "\x1f" + article_id + "\x1f" + normalize(text))[:16]
```

重新索引就变 ID → 已落库的引用全部失效 → Agent 4 `citation_check` 莫名 fail。

**2. 过滤必须在召回之前**

先召回 top-5 再过滤 → 实际有依据却返回空 → 被误判成「检索无结果」。

**3. `text` 必须是原文，不得摘要**

`cite_check` 做实体覆盖校验，摘要会让所有 claim 降级成「依据不足」。

详见本地 `docs/contract.md`（暂不公开）。

## 目录结构

```text
mercatoria/
├── tools/                评测与导入工具链
├── scripts/              环境重建脚本
├── corpus/               仅保留抓取脚本与说明（语料本身不入库）
│   ├── README.md         语料来源与规模
│   └── fetch_*.py        从原始来源抓取
├── docs/                 设计文档（本地保留，不公开）
├── ragflow-src/          RAGFlow 源码副本（本地保留，不公开）
└── openspec/             规格与变更管理（本地保留，不公开）
```

> [!IMPORTANT]
> **本仓库只包含工具链代码，不包含语料与第三方源码。**
>
> | 目录 | 是否入库 | 原因 | 如何重建 |
> |---|---|---|---|
> | `corpus/` 语料文件 | ❌ | 法规文本转载受各来源条款约束 | `bash scripts/setup-corpus.sh` |
> | `ragflow-src/` | ❌ | Apache-2.0 第三方项目，67MB | `bash scripts/setup-ragflow-src.sh` |
> | `docs/` | ❌ | 暂不公开 | — |
> | `openspec/` | ❌ | 仅本地开发管理 | — |
>
> 换机器或重新 clone 后，先跑两个 setup 脚本再跑评测。
> 语料抓取约需 10 分钟；RAGFlow 源码重建后需按
> `ragflow-src/UPSTREAM.md` 的表格手工重放我们的改动。

## 快速开始

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip

cp .env.example .env                            # 填入 RAGFlow 地址与 API Key
bash scripts/setup-corpus.sh                    # 抓语料（约 10 分钟）
bash scripts/setup-ragflow-src.sh               # 拉 RAGFlow 源码（仅需改源码时）

# 跑评测
python tools/eval_retrieval.py --kb <kb_id> --tag baseline
```

> [!NOTE]
> API Key 通过环境变量或 `.env` 提供，不再硬编码在脚本里。
> `.env` 已在 `.gitignore` 中。

## 评测

```bash
# 纯检索评测（不调 LLM，539 条约 20 分钟）
python tools/eval_retrieval.py --kb <kb_id> --vec-weight 0.5 --tag final

# 含生成质量与思考开关对比（慢）
python tools/run_eval.py --kb <kb_id> --full --thinking-compare

# 参数扫描
python tools/param_sweep.py --kb <kb_id> --sweep
```

评测方法与失败分析见本地 `docs/eval-report.md`（暂不公开）。

> [!WARNING]
> **单次评测不可信。** 检索存在分数并列，同一查询连续运行可能给出不同的
> top-N。仓库中的排序已修正（并列时按 `chunk_id`），但**任何参数对比都应
> 重复 3 次取均值**，而不是比较单次结果。

## 下游对接

`smart-edit` 的 Agent 3 在 `retrieve` 节点调用本服务。
需要补齐的 `Chunk` 类型字段清单见本地 `docs/contract.md` §10（暂不公开）。
