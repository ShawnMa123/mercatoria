# Mercatoria · 跨境贸易法规检索后端 / Cross-border Trade Law Retrieval Backend

> **基于 RAGFlow 的后端实现**：沿用 RAGFlow `laws` 解析器的分块契约与知识库导入约定，
> 检索、过滤、重排、引用与问答由本服务实现，可独立部署，也可作为 RAGFlow 生态的检索后端。
>
> **A RAGFlow-based backend implementation**: keeps RAGFlow's `laws` chunking contract and
> import conventions, and serves retrieval, filtering, reranking, citation and answer
> generation as a standalone backend.

`Python 3.11` · `FastAPI` · `SQLite` · `NumPy` · `CPU-only` · `deterministic chunk ids`

**中文** · [English](#english)

## 是什么

面向跨境贸易合规场景的**法条级检索后端**：一句自然语言问题，返回可引用的法规原文、
完整法条、交叉引用与置信度信号。核心性质：

- **可引用** — `text` 是原文，`chunk_id` 内容派生，重建后引用不变。
- **可解释** — 每次响应带 `funnel`、`zero_reason`、`score_parts`、`timings_ms`。
- **可运营** — 配置热更新、索引多版本原子切换、上传/重建异步 job、schema 版本化。
- **零 GPU** — embedding 走 OpenAI 兼容网关，检索为内存倒排 + NumPy。

当前规模：**143 篇法规 / 11,562 chunks / 8,191 条文**；本地检索 p50 **28 ms**，
缓存命中 **0.24 ms**；dev Hit@5 **0.8610**、hold-out Hit@5 **0.9000**、
真题 Hit@5 **0.9831**。

## 文档索引 / Documentation

| 文档 / Document | 内容 / Contents |
|---|---|
| [架构设计 / Architecture](documentation/architecture.md) | 分层、检索流水线、索引生命周期、设计决策、数据模型 |
| [API 接口 / API Reference](documentation/api.md) | 全部端点、字段、默认值、请求/响应示例、错误码 |
| [评测与基准 / Benchmark](documentation/benchmark.md) | 口径、dev/hold-out/真题指标、消融、延迟、压测、回答质量 |
| [部署与测试 / Deployment & Testing](documentation/deployment-testing.md) | 安装、配置、语料、建索引、运行、运维、测试流程 |

## 快速开始 / Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # POSIX: .venv/bin/python

bash scripts/setup-corpus.sh          # 可选：抓取语料（约 15 分钟）
python -m mercatoria.cli manifest
python -m mercatoria.cli rebuild
python -m mercatoria.cli serve --host 0.0.0.0 --port 8080

curl -s localhost:8080/health
curl -s localhost:8080/v1/retrieve -H 'Content-Type: application/json' \
  -d '{"queries":["进口货物的完税价格由哪些项目组成？"],"top_k":3,"filters":{"jurisdictions":["CN"]}}'
```

接口速览：`/health`、`/v1/config`、`/v1/retrieve`、`/v1/articles/{id}`、
`/v1/articles/batch`、`/v1/chat`、`/v1/documents*`、`/v1/admin/*`、`/v1/jobs/{id}`。
完整参数见 [API 参考](documentation/api.md)。

## 设计亮点 / Design highlights

- **Filter-first**：法域 → 业务类型 → 时效 → 文档 → 条文，逐级过滤并写入漏斗诊断。
- **双路召回 + 配额**：内存 BM25 与向量矩阵融合；多法域比较按配额保留槽位（覆盖率 1/8 → 8/8）。
- **内容派生 ID**：`sha256(doc_id | article_id | normalize(text))[:16]`，引用可复核。
- **索引文本解耦**：向量/BM25 用 `[法典 | 章节 | 条号]\n原文`，返回值仍是原文。
- **原子发布**：候选 tmp → `os.replace` → 提交前完整 prepare → control head 单事务 CAS 追加 generation → 安装快照；`current.txt` 只是导出。
- **单写者**：`data/documents/write.lock` 在打开 control/registry 之前取得，第二个实例直接拒绝启动。
- **只读置信度**：`high|medium|low` 附在 `diagnostics`，不改变候选顺序。

## 仓库结构 / Layout

```text
mercatoria/     服务实现（api / retrieve / store / chunker / config / manifest / ingest）
corpus/         抓取脚本 + manifest（法规原文不入库）
documentation/  架构、API、基准、部署与测试文档
tools/          评测、profile、审计、压测、hold-out 生成
tests/          单测 + 真题 fixture + 冻结 hold-out
data/           索引、上传、缓存（gitignored）
```

> 原始语料、索引/数据库、向量文件一律不入库；`corpus/` 仅保留抓取脚本与元数据。
> Raw statutes, indexes/databases and vector files are never committed.

## 边界 / Boundaries

上传解析支持 md/txt/docx/pdf（有文本层，无 OCR）。引用解析覆盖同文档条文号、区间与
`前条/上条/下条/前款`、英文 `section/§/Article`，跨文档仅解析书名号全名匹配；项级引用
保留在 `unresolved`。服务无内建鉴权，请置于网关之后。检索置信度与 `corpus_gap` 是可靠性
信号，不构成法律意见。

---

# English

## What it is

An **article-level retrieval backend** for cross-border trade compliance: a natural-language
question in, citable statute text, whole articles, cross-references and a confidence signal
out.

- **Citable** — raw `text`, content-addressed `chunk_id`, stable across rebuilds.
- **Explainable** — `funnel`, `zero_reason`, `score_parts`, `timings_ms` on every response.
- **Operable** — config hot-reload, atomic index activation, async ingest/rebuild jobs,
  versioned schema.
- **Zero GPU** — embeddings via an OpenAI-compatible gateway; retrieval is in-memory BM25
  plus NumPy.

Current corpus: **143 statutes / 11,562 chunks / 8,191 articles**; local p50 **28 ms**,
cache-hit p50 **0.24 ms**; dev Hit@5 **0.8610**, hold-out Hit@5 **0.9000**, trade Hit@5
**0.9831**.

## Documentation

| Document | Contents |
|---|---|
| [Architecture](documentation/architecture.md) | layers, retrieval pipeline, index lifecycle, decisions, data model |
| [API Reference](documentation/api.md) | all endpoints, fields, defaults, examples, status codes |
| [Benchmark](documentation/benchmark.md) | metric definitions, dev/hold-out/trade numbers, ablation, latency, stress, answer quality |
| [Deployment & Testing](documentation/deployment-testing.md) | install, config, corpus, index build, run, operations, test procedures |

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

bash scripts/setup-corpus.sh
python -m mercatoria.cli manifest
python -m mercatoria.cli rebuild
python -m mercatoria.cli serve --host 0.0.0.0 --port 8080

curl -s localhost:8080/health
```

Endpoints: `/health`, `/v1/config`, `/v1/retrieve`, `/v1/articles/{id}`, `/v1/articles/batch`,
`/v1/chat`, `/v1/documents*`, `/v1/admin/*`, `/v1/jobs/{id}`.

## Design highlights

- Filter-first funnel with per-stage diagnostics.
- In-memory BM25 + dense vectors; jurisdiction quotas widen the pool before reserving slots.
- Content-addressed ids: `sha256(doc_id | article_id | normalize(text))[:16]`.
- Index text (`[statute | chapter | article] + text`) decoupled from the returned payload.
- Atomic publish: candidate `tmp → os.replace`, full prepare before the commit, a CAS generation
  appended to the control head, then the snapshot swap; `current.txt` is an export only.
- Single writer: `data/documents/write.lock` is taken before any writable store is opened, so a
  second instance refuses to start instead of serving a stale head.
- Read-only confidence signal (`high|medium|low`) that never reorders results.

## Boundaries

md/txt/docx/pdf with a text layer (no OCR). Reference parsing covers same-document article
numbers, ranges, `前条/上条/下条/前款`, English `section/§/Article`, and cross-document only by
exact 《title》. Item-level references remain `unresolved`. No built-in auth — deploy behind a
gateway; confidence and `corpus_gap` are reliability signals, not legal advice.
