# mercatoria

**进出口贸易法规检索服务** —— 为 `smart-edit` 的 Agent 3 提供可引用的法规原文片段。

> `lex mercatoria`（商人法）指跨国商事交易自发形成的规则体系：不由任何国家立法机关制定，
> 而由商人实践、行业惯例与国际公约构成。这个名字同时界定了语料范围 ——
> 不装刑法、婚姻法、行政诉讼法，不是因为没做，而是它们不属 lex mercatoria。

## 这是什么，不是什么

**是**：`queries + filters → chunks` 的检索服务；额外的 `/ask` 是可选便利层，
直接返回「带引用的答案 + 匹配信息」，也可以关掉让 Agent 自己生成。

**不是**：
- ❌ RAG 库 —— 检索策略（切分/向量/rerank）是本仓库的自由
- ❌ 合规判断/法律意见 —— 返回「可被引用的原文片段」和基于片段的答案，不是「能做/不能做」的结论
- ❌ 必须依赖 `smart-edit` 的 `answer` 节点 —— `/ask` 可选用，Agent 仍可直接消费 `/retrieve`

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
├── mercatoria/           轻量检索/问答服务（API、索引、分片）
├── tools/                评测与导入工具链
├── scripts/              环境重建脚本
├── corpus/               抓取脚本、manifest 与说明（语料本身不入库）
│   ├── README.md         语料来源与规模
│   ├── manifest.json     稳定 doc_id + 法域/时效/业务类型
│   └── fetch_*.py        从原始来源抓取
├── docs/                 设计文档（本地保留，不公开）
├── ragflow-src/          RAGFlow 源码副本（本地保留，不公开）
└── openspec/             规格与变更管理（本地保留，不公开）
```

> [!IMPORTANT]
> **本仓库包含工具链与服务代码，不包含语料与第三方源码。**
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

## 轻量 API 服务（推荐运行时）

RAGFlow 保留为评测基线；运行时使用本仓库的 `mercatoria` 包：
一个 Python 进程 + SQLite 索引 + NumPy 暴力向量检索。embedding 与
answer 走 OpenAI-compatible 外部网关，rerank 走 `gte-rerank-v2` 的 DashScope
原生接口（可关闭），不在本地加载模型，不需要 GPU。

```bash
# 安装（numpy / fastapi / uvicorn）
.venv/Scripts/pip install -e .

# 1. 生成稳定 doc_id + 法规元数据
python -m mercatoria --config config.example.json manifest

# 2. 建索引（--limit 用于冒烟；全量约 7.5k chunk）
python -m mercatoria --config config.example.json rebuild --limit 2
python -m mercatoria --config config.example.json rebuild

# 3. 启动 API（无前端；FastAPI 的 /docs 仅作调试）
python -m mercatoria --config config.example.json serve --port 8080
```

```bash
# 纯检索：返回 chunks + matched_terms / matched_fields + diagnostics
curl -s localhost:8080/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["完税价格由哪些项目组成"],"top_k":5,
       "fields":{"business_type":"customs","country":"CN"}}'

# 检索 + 可选答案：answer.enabled=false 时只返回匹配信息
curl -s localhost:8080/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"完税价格由哪些项目组成？","top_k":5}'
```

索引默认写到 `data/index/`，`current.txt` 指向活动版本；重建成功后才切换。
改分片参数必须重建，改 `vector_weight/top_k` 不需要。当前默认
`vector_weight=0.8`（本地 539 条：Hit@5 **0.8126**、Hit@10 **0.8794**、
MRR **0.6890**；RAGFlow v2 基线为 0.783 / 0.859 / 0.656。59 条分层样本上
0.5→0.8136、0.7→0.8983、0.8→0.9153 Hit@5。这是本地 BM25+cosine 的标定，
不能直接套用 RAGFlow 的 0.5。正式使用前建议用自己的用例再扫一遍）。embedding 按
`(model, text)` 缓存到 `data/embedding-cache.db`，重建时只算变化过的 chunk。
完整配置见 `config.example.json`。

### Rerank（gte-rerank-v2）

`rerank.enabled=true` 时，检索先取 `candidates`（默认 40）条候选，再交给
cross-encoder 逐对打分，最终 `top_k` 按 rerank 分数排序。rerank 失败不报错，
自动回退到 BM25+cosine 顺序，原因写在 `diagnostics.rerank.error`。
`gte-rerank-v2` 只提供在 DashScope 原生路径
（`/api/v1/services/rerank/text-rerank/text-rerank`），不走 magpie 的
`/v1/rerank`；key 用 `RERANK_API_KEY` 注入，不写入配置文件。
每个 chunk 的 `score_parts` 同时给出 `first_stage`（混合分）和 `rerank`（原始分），
两者不可直接比较。

| 端点 | 用途 |
|---|---|
| `POST /retrieve`（别名 `/v1/retrieve`） | 纯检索：chunks + matched_terms/matched_fields + diagnostics |
| `POST /chat`（别名 `/v1/chat`） | 对话式问答：接受 `messages` 历史，返回 answer + citations + **references 原文** |
| `POST /ask`（别名 `/v1/ask`） | 单轮问答，与 `/chat` 同一实现；`answer.enabled=false` 只返回检索 + references |
| `GET /v1/config` / `PUT /v1/config` | 查看/更新检索、生成、分片配置 |
| `POST /v1/admin/rebuild` | 后台重建索引；`background=false` 同步冒烟 |
| `GET /v1/admin/rebuild/{job_id}` | 重建进度 |
| `POST /v1/admin/activate/{version}` | 切换活动索引 |
| `GET /health` | 活动索引、chunk 数、文档数 |

对话式请求与响应（`references` 是检索到的原文全文，不只是摘要）：

```bash
curl -s localhost:8080/v1/chat -H 'Content-Type: application/json' -d '{
  "messages": [
    {"role": "user", "content": "进口货物的完税价格由哪些项目组成？"}
  ],
  "top_k": 5,
  "filters": {"jurisdictions": ["CN"]}
}'
```

```jsonc
{
  "question": "进口货物的完税价格由哪些项目组成？",
  "retrieval_query": "进口货物的完税价格由哪些项目组成？",
  "answer": "进口货物完税价格以成交价格为基础… [f583078a58c014ca]",
  "answer_status": "answered",
  "citations": [
    {"chunk_id": "f583078a58c014ca", "article_id": "cn-customs-law-2021:55",
     "article_label": "第五十五条", "quote": "…", "text": "第五十五条 进出口货物的完税价格…"}
  ],
  "references": [
    {"chunk_id": "f583078a58c014ca", "rank": 1, "source_doc": "中华人民共和国海关法",
     "article_id": "cn-customs-law-2021:55", "article_label": "第五十五条",
     "jurisdiction": "CN", "score": 0.406691, "rerank_score": 0.406691,
     "cited": true, "text": "第五十五条 进出口货物的完税价格…（原文全文）"}
  ],
  "chunks": ["…完整检索结果，含 matched_terms / score_parts…"],
  "diagnostics": {"rerank": {"enabled": true, "applied": true}},
  "coverage": {"unavailable": false, "notes": []}
}
```

多轮对话把之前轮次放进 `messages` 即可；最后一轮是短追问（如「它什么时候生效？」）时，
服务会自动把上一轮用户问题拼进检索 query。也可以显式传 `retrieval_query` 覆盖。

## 文件入库（md / docx / pdf）

上传后后台解析、归一化为条文 Markdown、自动重建索引；同一批次只重建一次。
支持 `.md` / `.markdown` / `.txt` / `.docx` / `.pdf`（PDF 必须有文本层）。
旧版 `.doc` 需先转 `.docx`；扫描件不做 OCR，会以 `failed` 明确返回。

```bash
curl -s -X POST localhost:8080/v1/documents \
  -F "files=@法规.md" \
  -F "files=@办法.docx" \
  -F "files=@part-101.pdf" \
  -F "jurisdiction=CN" \
  -F "applies_to=customs,tariff" \
  -F "effective_from=2026-01-01"
```

返回 `job_id`，轮询进度：

```bash
curl -s localhost:8080/v1/jobs/<job_id>
```

```jsonc
{
  "status": "done",
  "progress": {"stage": "index", "done": 95, "total": 95},
  "result": {
    "indexed": 3,
    "chunks": 7202,
    "documents_total": 95,
    "documents": [{"doc_id": "cn-upload-…", "ok": true, "format": "docx"}]
  }
}
```

文档生命周期：

| 接口 | 用途 |
|---|---|
| `GET /v1/documents` | 列表 + 状态 + `chunk_count` |
| `GET /v1/documents/{id}` | 详情 |
| `DELETE /v1/documents/{id}` | 删除原文件/解析产物并重建索引 |
| `POST /v1/documents/{id}/reindex` | 重新解析并重建 |
| `GET /v1/documents/{id}/chunks` | 该文档的 chunk 列表 |

同一文件重复上传按 `sha256` 幂等，返回已有 `doc_id`，不重复 embedding。
索引重建只 embedding 新增 chunk：活动索引里 `chunk_id` 未变的向量直接复用。

```bash
# 容器内跑全部测试（推荐，也是验收方式）
docker build --target test -t mercatoria:test-suite .
docker run --rm mercatoria:test-suite
```

Agent 也可以不走 HTTP，直接进程内调用：

```python
from mercatoria import load_config, open_current, Retriever

cfg = load_config("config.example.json")
retriever = Retriever(open_current(cfg.resolved_index_dir()), cfg)
result = retriever.retrieve(["完税价格由哪些项目组成"], top_k=5)
```

> [!NOTE]
> 跨语言检索仍是已知短板：中文问 WTO/CISG 等英文语料时，向量召回可能不足。
> 服务保留了纯检索接口，后续可在 `retrieve` 前加语言路由/查询翻译，不影响本层契约。

## Docker 测试镜像

镜像内置 `corpus/`（92 文档 / 7,196 chunks）和活动索引（`qwen3.7-text-embedding`，
1024 维），启动即可检索。embedding 与 chat 通过 `host.docker.internal:3425`
访问宿主 magpie；rerank 直连 DashScope。

```bash
# .env 提供 RERANK_API_KEY
docker compose -f docker-compose.test.yml up -d --build
curl -s localhost:8080/health

# 改 embedding 模型或分片参数后重建内置索引
docker run --rm --add-host host.docker.internal:host-gateway \
  -e EMBEDDING_BASE=http://host.docker.internal:3425/v1 \
  --entrypoint python mercatoria:test \
  -m mercatoria --config config.json rebuild
```

镜像只复制 `data/index/current.txt` 指向的活动索引，历史版本不会进入镜像。
`data/` 挂在 `mercatoria-data` 命名卷上；首次启动会把镜像里的索引种子复制进去，
上传的文档、注册表和后续重建都持久化在卷里。

```bash
# 用 60 道真实来源题目做检索验收（题目在 tests/eval/trade_law_questions.json）
docker run --rm --add-host host.docker.internal:host-gateway \
  mercatoria:test-suite \
  python tests/eval_trade_questions.py \
    --base http://host.docker.internal:8080 --answer-limit 5

# 200 文件批量入库压测
docker run --rm --add-host host.docker.internal:host-gateway \
  mercatoria:test-suite \
  python tests/bench_ingest.py \
    --base http://host.docker.internal:8080 --files 200
```

### POC 实测（2026-11-12，CPU-only）

| 指标 | 实测 |
|---|---|
| 容器空闲内存 | 216 MiB |
| 92 文档 / 7,196 chunks 启动 | 就绪，无重建 |
| 200 个小文件批量入库 | **8.6 s**（上传 0.02 s + 解析/索引 8.6 s，新增 600 chunks） |
| 60 题检索（含 rerank，冷缓存） | p50 **1.41 s**，p95 **4.33 s** |
| 同一查询重复 | p50 **~4 ms**（LRU 命中） |
| 60 题 Hit@5 | **0.9333**；剔除 6 道语料缺口后 **1.00（54/54）** |
| 60 题 MRR | **0.8381** |
| 真实问答（3 题） | 10.7–18.1 s，citations 4–5 条，references 5 条全文 |
| Docker 测试 | **69 项全部通过** |

6 道 `corpus_gap` 题是语料本身未覆盖：公式定价、出口加工区内销原产地、
海关商品编号、出口收汇核销、WTO DSU、以及仅存在于已被《关税法》替代的
2017 年关税条例中的境外修理规则。它们保留在测试集里，不计入 covered 指标。

> [!IMPORTANT]
> 调分片会改变 chunk 边界与 `chunk_id`。下游引用请同时保存
> `article_id` + `index.version`；`chunk_id` 只代表某次索引里的段落。

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
