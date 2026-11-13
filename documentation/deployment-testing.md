# 部署与测试指南 / Deployment & Testing Guide

> 适用版本：schema v2，索引目录 `data/index`，Python 3.11。
> Applies to: schema v2, index directory `data/index`, Python 3.11.

---

## 中文

### 1. 前置条件

| 项 | 要求 |
|---|---|
| Python | 3.11（`.venv`） |
| 内存 | 最小 1 GB；11.5k chunks 实测索引归因约 254 MB，建议 2 GB |
| CPU | x86_64 / arm64 均可；无 GPU 依赖 |
| 网络 | 必须能访问 embedding 网关；rerank / LLM 可选 |
| 磁盘 | 索引约 156 MB/11.5k chunks；上传与缓存另计 |
| 端口 | 默认 8080（HTTP） |

### 2. 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
# POSIX
.venv/bin/python -m pip install -r requirements.txt
```

### 3. 配置

```bash
cp config.example.json config.json     # 按需修改
```

- 密钥与端点建议用环境变量注入（不落盘）：`EMBEDDING_BASE`、`EMBEDDING_MODEL`、
  `EMBEDDING_API_KEY`、`RERANK_*`、`LLM_*`、`MERCATORIA_*`（完整列表见 `config.example.json`
  与 `README.md` 的配置参考）。
- 关键默认：`retrieval.top_k=5`、`vector_weight=0.8`、`bm25_saturation=8`、
  `query_stopwords=true`、`multi_query_bonus=0`、`rerank.enabled=false`、
  `confidence={high:0.8, medium:0.64, min_gap:0.02}`。
- `embedding.dim` 可留 0（自动探测）；设错会导致启动校验失败。

### 4. 语料准备

```bash
bash scripts/setup-corpus.sh          # 抓取（约 15 分钟，视网络）
bash scripts/setup-corpus.sh --check  # 校验数量；UK 为可选源
```

| 来源 | 预期 | 说明 |
|---|---|---|
| eCFR Title 19 / 15 / 31 | 51 / 16 / 6 parts | 快照日期固定 2026-11-01；part 542 在该日期 404 |
| WTO | 14 篇 | TRIPS/AG/SG 为分页文本，脚本自动跟进 |
| 日本 e-Gov | 9 部 | 按 `law_id` 直取 |
| 澳洲 | 5 部 | 通过 OData `documents/find` 下载 PDF 再抽取 |
| EU Cellar | 2 部 | `celex/*.ENG.xhtml` 官方直连 |
| 中文（LawRefBook 镜像） | 39 篇 | 解析 H1 + INFO END |
| CISG | 1 篇 | PDF 抽取 |
| UK | 0 篇（可选） | legislation.gov.uk 有 WAF；改用浏览器下载 + `corpus/convert_uk_xml.py` |

> 语料原文不入库：`corpus/{cn,ecfr,wto,jp,au,eu,uk,uncitral}/` 与 `data/` 均在 `.gitignore` 中；
> 仓库只保留抓取脚本、`manifest.json`（元数据）与评测 fixture。

### 5. 构建索引

```bash
python -m mercatoria.cli manifest         # 生成 corpus/manifest.json
python -m mercatoria.cli rebuild          # 全量索引，结束自动激活
```

- 首次构建需要为每个 chunk 调 embedding（批量 16、并发 4、带重试与 SQLite 缓存）；
  11.5k chunks 在网关正常时约 1–10 分钟。
- 重建会按每个 chunk 的 `index_text` 哈希复用旧向量；只有文本真的变了才重新 embedding。
- 产物：`data/index/index-<ts>-<hash>.db` + `current.txt`；旧版本保留，可回滚。
- 只想冒烟：`python -m mercatoria.cli rebuild --limit 20 --no-activate`。

### 6. 运行

```bash
# 方式 A：CLI
python -m mercatoria.cli serve --host 0.0.0.0 --port 8080

# 方式 B：uvicorn
uvicorn mercatoria.api:app --host 0.0.0.0 --port 8080 --workers 1

# 方式 C：Docker
docker compose -f docker-compose.test.yml up -d --build
curl -s localhost:8080/health
```

单 worker 即可满足当前规模；多 worker 会各自加载索引（内存 × N）。

### 7. 反向代理与安全

- 服务**无内建鉴权**：放在 Nginx/Ingress/API Gateway 之后，启用鉴权与限流。
- 建议超时：检索 60 s、聊天 300 s（生成可能较慢）、上传 600 s。
- 请求体上限与 `max_upload_bytes` / `max_batch_bytes` 保持一致，避免网关先截断。
- 不暴露 `/v1/admin/*` 给终端用户。

### 8. 运维

| 操作 | 方式 |
|---|---|
| 健康检查 | `GET /health`（503 表示未建/为空/schema 过新） |
| 配置热更新 | `PUT /v1/config`（检索/重排即时生效；分片/模型变更置 `pending_rebuild`） |
| 重建 | `POST /v1/admin/rebuild` → `GET /v1/jobs/{id}`，结束后自动 reload |
| 切换/回滚 | `GET /v1/admin/indexes` → `POST /v1/admin/activate/{version}` |
| 上传/删除/改元数据 | `/v1/documents*`，全部异步 job |
| 缓存 | 响应 256 条、查询向量 1024 条、embedding 持久化缓存；`PUT /v1/config` 会清空前两者 |
| 降级 | `fallback_bm25_on_embed_error=true` 时网关失败退化为 BM25，响应标记 `degraded` |

**升级流程**：拉代码 → 装依赖 → `rebuild`（新索引原子落盘）→ `activate`/自动 reload →
`/health` 校验 `schema_version`。旧库可直接打开（`index_schema_outdated=true` 提示重建）。

### 9. 容量与备份

- 向量占用 ≈ `chunks × dim × 4` 字节（11.5k × 1024 × 4 ≈ 45 MB）。
- 索引内存 ≈ 向量 + 倒排 + 条文文本；10× 语料预计约 2.5 GB、加载 ~17 s。
- 备份：`data/index/`（可重建）、`data/documents/registry.db` 与 `data/uploads/`
  （上传文档，**必须备份**）。`data/embedding-cache.db` 可丢（重建会重新计算）。

### 10. 故障排查

| 现象 | 排查 |
|---|---|
| `/health` 503 `not_built` | 未建索引：`python -m mercatoria.cli rebuild` |
| `/health` 503 `index_schema_outdated` | 旧 schema：重建索引 |
| 503 提示 “newer than this build” | 代码版本旧于索引：升级代码或重建 |
| 检索 500 `embedding request failed` | 网关不可达/超时；检查 `EMBEDDING_BASE`，或开启 BM25 降级 |
| 上传 413 | 超过 `max_upload_bytes` / `max_batch_bytes`（默认 20 MB / 200 MB） |
| 上传 409 | 已有 ingest/rebuild job 在跑：`GET /v1/jobs/{id}` |
| 检索结果为空且 `zero_reason=no_as_of` | `as_of` 过滤：检查日期的 `effective_from/to` |
| 结果缺少期望法条 | 看 `diagnostics.funnel` 与 `requested_uncovered`；必要时重建 manifest |
| 启动慢 | 索引加载 ~1.7 s 属正常；更大语料考虑 CSR/float16 |

### 11. 测试指南

#### 11.1 单元测试（无需网络）

```bash
python -m unittest discover -s tests -q
# 135 tests，覆盖：chunk_id 稳定性、切分、检索/过滤/配额/置信度、缓存与并发、
# schema 兼容、引用解析、API 路由与错误码、语料检查、fixture 一致性。
```

#### 11.2 语料完整性

```bash
python tools/check_corpus.py --corpus corpus
# 缺少必需源 → 退出码 1；仅缺 uk → 退出码 0（可选源）
```

#### 11.3 检索评测（dev / hold-out）

```bash
# dev（默认已排除 100 题 hold-out）
python tools/eval_local.py --config config.example.json --workers 8 --tag dev

# hold-out（只在里程碑跑，不据此调参）
python tools/eval_local.py --config config.example.json --holdout --tag holdout

# 多查询融合、旧判分、法域过滤 + 配额对比
python tools/eval_local.py --fusion --tag fusion
python tools/eval_local.py --legacy-scoring --tag legacy
python tools/eval_local.py --jurisdiction-filter --tag cov-quota
python tools/eval_local.py --jurisdiction-filter --no-quota --tag cov-noquota
```

输出：`corpus/eval/local_<tag>_<ts>.json`，含 `hit5/hit10/mrr/by_category/coverage5` 与口径字段。

#### 11.4 真题与无答案

```bash
# 先启动服务
python -m mercatoria.cli serve --port 8080

python tests/eval_trade_questions.py --base http://localhost:8080 --out data/eval/trade.json
# 汇总包含 hit5 / mrr / corpus_gap / no_answer_low_rate / answer_false_low_rate
```

#### 11.5 回答质量（需要 LLM）

```bash
python tools/eval_answers_local.py --base http://localhost:8080 --ids FT-001,FT-054,FT-109
# 指标：引用有效率、编造引用、拒答率、关键词覆盖、生成延迟
```

#### 11.6 性能与压力

```bash
python tools/profile_retrieve.py --sample 200        # 分阶段 p50/p90/p99
python tools/stress_retrieve.py --endpoint http://127.0.0.1:8080 --threads 8 --seconds 60
python tools/index_stats.py                          # 加载时间 / 内存 / 外推
```

#### 11.7 配额与 rerank 评估

```bash
python tools/eval_quota.py          # off / hard / soft 三策略
python tools/eval_rerank_policy.py --sample 100     # off / always / gated / topN
```

#### 11.8 消融流程

1. `config.example.json` 派生变体：`query_stopwords=false`、`bm25_saturation=0`、
   `expand_cross_language=false`、`multi_query_bonus=0`。
2. Context Header 需独立索引：复制配置、`index_dir=data/index-noheader`、
   `chunking.index_context_header=false`，然后 `--config <file> rebuild`。
3. 每行跑 dev 539 + 真题 59，记录 Hit@5/Hit@10/MRR 与分类别差值（≥2pp 的单列说明）。

#### 11.9 fixture 审计

```bash
python tools/audit_fixture.py       # 逐题 top-5、命中、置信度 → docs/fixture-audit-*.md
```

#### 11.10 测试数据纪律

- 调参只看 dev；hold-out 只在验收跑，并记录到运行日志。
- fixture 改动必须附原文依据（题号 / 变更 / `doc_id:article_id`）。
- 改变指标口径（判分、fixture、语料）的提交与检索算法提交分开。
- 原始语料、索引/数据库、向量文件一律不入库。

---

## English

### 1. Prerequisites

Python 3.11; ≥1 GB RAM (2 GB recommended; index attribution ~254 MB at 11.5k chunks);
no GPU; network access to the embedding gateway; rerank/LLM optional; disk ~156 MB per
11.5k-chunk index (plus uploads and caches); default port 8080.

### 2. Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # POSIX: .venv/bin/python
```

### 3. Configure

`cp config.example.json config.json` and edit. Preferred: inject endpoints/secrets through
environment variables (`EMBEDDING_*`, `RERANK_*`, `LLM_*`, `MERCATORIA_*`) so keys never hit
disk. Defaults: `top_k=5`, `vector_weight=0.8`, `bm25_saturation=8`, `query_stopwords=true`,
`multi_query_bonus=0`, `rerank.enabled=false`, `confidence={high:0.8, medium:0.64}`.
Leave `embedding.dim=0` to auto-detect.

### 4. Corpus

```bash
bash scripts/setup-corpus.sh          # ~15 minutes depending on network
bash scripts/setup-corpus.sh --check  # UK is optional
```

Sources: eCFR 19/15/31 (51/16/6 parts, snapshot 2026-11-01), WTO 14, Japan e-Gov 9,
Australia 5, EU Cellar 2, Chinese LawRefBook 39, CISG 1, UK 0 (WAF; manual download +
`corpus/convert_uk_xml.py`).
Raw statute text, indexes, databases and vectors are never committed (`.gitignore`).

### 5. Build the index

```bash
python -m mercatoria.cli manifest
python -m mercatoria.cli rebuild
```

First build embeds every chunk (batch 16, concurrency 4, retries, SQLite cache); rebuilds
reuse vectors per chunk by hashing the exact `index_text`. Output:
`data/index/index-<ts>-<hash>.db` + `current.txt`; older versions are kept for rollback.
Smoke build: `rebuild --limit 20 --no-activate`.

### 6. Run

```bash
python -m mercatoria.cli serve --host 0.0.0.0 --port 8080
uvicorn mercatoria.api:app --host 0.0.0.0 --port 8080 --workers 1
docker compose -f docker-compose.test.yml up -d --build && curl -s localhost:8080/health
```

One worker is enough at this scale; each extra worker loads its own index copy.

### 7. Proxy and security

No built-in auth: put it behind Nginx/Ingress/API gateway with auth and rate limits.
Suggested timeouts: retrieval 60 s, chat 300 s, upload 600 s. Align gateway body limits with
`max_upload_bytes` / `max_batch_bytes`. Do not expose `/v1/admin/*` publicly.

### 8. Operations

Health `GET /health`; hot config `PUT /v1/config`; rebuild `POST /v1/admin/rebuild` +
`GET /v1/jobs/{id}`; rollback `POST /v1/admin/activate/{version}`; caches are 256 responses,
1024 query vectors and a persistent embedding cache; optional BM25 fallback on gateway
failure (`fallback_bm25_on_embed_error`).
Upgrade: pull → install → rebuild (atomic) → activate/reload → check `/health`.
Old indexes still open (`index_schema_outdated=true`).

### 9. Capacity and backup

Vectors ≈ `chunks × dim × 4` bytes (≈45 MB at 11.5k × 1024). 10× corpus ≈ 2.5 GB index
memory and ~17 s load. Back up `data/index/` (rebuildable), and **must** back up
`data/documents/registry.db` + `data/uploads/`; the embedding cache is disposable.

### 10. Troubleshooting

| Symptom | Action |
|---|---|
| `/health` 503 `not_built` | build the index |
| 503 `index_schema_outdated` | rebuild the index |
| 503 “newer than this build” | upgrade the code or rebuild |
| 500 `embedding request failed` | check gateway/base or enable BM25 fallback |
| 413 on upload | increase `max_upload_bytes` / `max_batch_bytes` |
| 409 on upload | a job is running; check `/v1/jobs/{id}` |
| Empty results `zero_reason=no_as_of` | check `effective_from/to` vs `as_of` |
| Missing expected article | inspect `diagnostics.funnel` and `requested_uncovered` |

### 11. Testing

```bash
python -m unittest discover -s tests -q          # 135 tests, no network
python tools/check_corpus.py --corpus corpus     # required sources; UK optional
python tools/eval_local.py --workers 8 --tag dev # dev (hold-out excluded)
python tools/eval_local.py --holdout --tag holdout
python tools/eval_local.py --fusion --legacy-scoring --jurisdiction-filter ...
python tests/eval_trade_questions.py --base http://localhost:8080 --out data/eval/trade.json
python tools/eval_answers_local.py --base http://localhost:8080 --ids FT-001,FT-054
python tools/profile_retrieve.py --sample 200
python tools/stress_retrieve.py --endpoint http://127.0.0.1:8080 --threads 8 --seconds 60
python tools/index_stats.py
python tools/eval_quota.py
python tools/eval_rerank_policy.py --sample 100
python tools/audit_fixture.py
```

Ablation: derive config variants (`query_stopwords=false`, `bm25_saturation=0`,
`expand_cross_language=false`, `multi_query_bonus=0`); Context Header needs a separate index
(`index_dir=data/index-noheader`, `index_context_header=false`), then run dev + trade per row.

Discipline: tune on dev only; run hold-out at milestones and log it; fixture changes need source-text evidence
(`doc_id:article_id`); keep metric-口径 changes in separate commits; never
commit corpus text, databases or vector files.
