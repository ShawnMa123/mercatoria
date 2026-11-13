# API 接口使用 / API Reference

> 所有接口返回 JSON（上传为 `multipart/form-data`），UTF-8；日期 `YYYY-MM-DD`；
> 无内建鉴权，请置于网关之后。错误体为 `{"detail": "..."}`，503 为
> `{"error": "...", "corpus": {...}}`。
>
> All endpoints speak JSON (uploads use `multipart/form-data`); dates are `YYYY-MM-DD`;
> there is no built-in auth. Errors are `{"detail": ...}`; 503 returns
> `{"error": ..., "corpus": ...}`.

---

## 中文

### 1. 端点总表

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 健康、活动索引、schema、配置摘要 |
| GET / PUT | `/v1/config` | 读取 / 热更新配置 |
| POST | `/retrieve`、`/v1/retrieve` | 纯检索 |
| GET | `/v1/articles/{article_id}` | 完整法条 + 邻条 + 交叉引用 |
| POST | `/v1/articles/batch` | 批量取法条（≤32） |
| POST | `/chat`、`/v1/chat`、`/ask`、`/v1/ask` | 检索 + 可选生成（带引用） |
| POST | `/v1/documents` | 批量上传并异步入库 |
| GET | `/v1/documents` | 文档列表（`?status=`） |
| GET | `/v1/documents/{doc_id}` | 文档详情 |
| PATCH | `/v1/documents/{doc_id}` | 修改元数据并重建 |
| DELETE | `/v1/documents/{doc_id}` | 删除并重建 |
| POST | `/v1/documents/{doc_id}/reindex` | 单文档重解析/重索引 |
| GET | `/v1/documents/{doc_id}/chunks` | 分页查看 chunk |
| POST | `/v1/admin/manifest` | 从语料目录重建 manifest |
| POST | `/v1/admin/rebuild` | 重建索引（后台/同步） |
| GET | `/v1/jobs/{job_id}` | job 状态 |
| GET | `/v1/admin/rebuild/{job_id}` | 同上（别名） |
| POST | `/v1/admin/activate/{version}` | 切换活动索引 |
| GET | `/v1/admin/indexes` | 列出索引与当前版本 |
| GET | `/` | 服务名与端点清单 |

### 2. `GET /health`

```json
{
  "status": "ready",
  "pending_rebuild": false,
  "index": {
    "version": "20261113-112706-384123-72d4a2df2417",
    "chunks": 11562, "documents": 143,
    "embedding_model": "my-team-2k/qwen3.7-text-embedding",
    "chunking_hash": "72d4a2df2417", "built_at": "2026-11-13T03:27:19+00:00",
    "schema_version": 2, "index_schema_outdated": false
  },
  "config": {"retrieval": {"top_k": 5}}
}
```

503：`{"status": "not_built" | "empty", "error": "...", "corpus": {...}?}`。

### 3. `GET|PUT /v1/config`

`GET` → `{"config": {...}, "active_index": "...", "pending_rebuild": false}`。
`PUT` 为**部分更新**：只传要改的 section/key，未知 key 忽略，`"save": true` 落盘到 `config.json`。
修改 `chunking`/`embedding` 置 `pending_rebuild=true`；任何修改都会清空检索缓存并同步
`rerank_enabled`。

```json
{"retrieval": {"top_k": 8, "confidence": {"medium": 0.6}},
 "rerank": {"enabled": true, "candidates": 20}, "save": false}
```

### 4. `POST /v1/retrieve`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `queries` | `string[]`/`string` | — | 1–16；也接受 `question` |
| `top_k` | int | `retrieval.top_k` | 1–100 |
| `filters` | object | `{}` | 见下表 |
| `fields` | object | `{}` | 兼容映射 |
| `rerank` | bool | 服务端默认 | 单请求开关 |
| `per_jurisdiction_k` | int | 无 | 0–100；0 关闭配额 |

`filters`：`jurisdictions[]`、`applies_to[]`、`as_of`（`YYYY-MM-DD`）、`doc_ids[]`、
`article_ids[]`、`per_jurisdiction_k`。

`fields` 兼容：`business_type→applies_to`、`jurisdiction|country→jurisdictions`、
`date|as_of→as_of`、`doc_ids`、`article_ids`。

```json
{"queries": ["进口货物的完税价格由哪些项目组成？"], "top_k": 5,
 "filters": {"jurisdictions": ["CN"]}}
```

响应字段：

| 字段 | 内容 |
|---|---|
| `chunks[]` | `chunk_id`、`doc_id`、`source_doc`、`text`、`score`、`score_parts{vector,keyword,vector_norm,keyword_norm,first_stage,hybrid,rerank,fusion_bonus}`、`articles[]`、`article_ids[]`、`article_offsets[]`、`effective_from`、`effective_to`、`jurisdictions[]`、`applies_to[]`、`superseded_by`、`matched_by[]`、`matched_queries[]`、`matched_terms[]`、`matched_fields{}`、`rank` |
| `corpus` | `status`、`version`、`snapshot_date`、`documents`、`article_count`、`chunk_count`、`jurisdictions[]`、`precision{}`、`built_at`、`embedding_model` |
| `temporal` | `as_of`、`by_jurisdiction{J: {mode, note}}` |
| `index` | `version`、`built_at`、`chunk_count`、`embedding_model`、`chunking_hash` |
| `field_matches` | `jurisdictions{requested,matched,uncovered}`、`applies_to{...}`、`as_of{requested,precision}` |
| `diagnostics` | 见下 |

`diagnostics`：

| key | 说明 |
|---|---|
| `funnel` | `total / after_jurisdiction / after_applies_to / after_as_of / after_doc / after_article / after_retrieval` |
| `zero_reason` | `no_jurisdiction`、`no_business_type`、`no_temporal`、`no_doc`、`no_article`、`no_match`、`null` |
| `requested_uncovered` | `jurisdictions[]`、`applies_to[]` |
| `rerank` | `enabled/applied/model/candidates/blend_weight/fusion/latency_ms/error` |
| `per_jurisdiction_k` / `quota_slots` | 配额与各法域实际槽位 |
| `embedding_cache_hits` | 命中查询向量缓存的条数 |
| `degraded` | `["embedding"]` 表示网关失败降级 BM25 |
| `confidence` | `high | medium | low`（只读） |
| `warnings` | 命中已被替代法规时的提示 |
| `timings_ms` | `filter / embed / bm25 / vector / rank / total` |
| `cache_hit` | `true` 表示响应来自进程内缓存 |

### 5. `GET /v1/articles/{article_id}`

`article_id` 形如 `cn-customs-law-2021:55`、`us-cfr-19-101:101.1`、`wto-dsu:art-17`；
路径参数支持 `/`。查询参数：`neighbors`（0–5）、`follow_refs`（0–2）。

响应：`article_id`、`doc_id`、`source_doc`、`article_label`、`section`、`jurisdiction`、
`applies_to`、`effective_from`、`effective_to`、`superseded_by`、`text`、
`chunks[{chunk_id,part,ordinal,token_count,text}]`、`neighbors{previous[],next[]}`、
`references{resolved[{text,article_id,article_label}], unresolved[{text}]}`、
`followed[]`（同 neighbors 结构 + `superseded_by`，上限 20）、`truncated`。

```json
{"article_id": "cn-customs-law-2021:58",
 "references": {"resolved": [
   {"text": "第五十六条", "article_id": "cn-customs-law-2021:56", "article_label": "第五十六条"},
   {"text": "第五十七条", "article_id": "cn-customs-law-2021:57", "article_label": "第五十七条"}],
   "unresolved": []},
 "followed": [{"article_id": "cn-customs-law-2021:56", "chunk_ids": ["..."], "text": "..."}],
 "truncated": false}
```

错误：400（参数越界）、404（未知 id）、503（无索引）。

### 6. `POST /v1/articles/batch`

请求：`{"article_ids": ["..."], "neighbors": 0, "follow_refs": 0}`（≤32）。
响应：`{"articles": [...], "missing": [...], "count": n}`；未知 id 进 `missing`，不报错。

### 7. `POST /v1/chat`（别名 `/chat`、`/v1/ask`、`/ask`）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `messages` / `question` | array / string | — | 取最后一条 user |
| `retrieval_query` | string | — | 显式检索串，优先级最高 |
| `rewrite_query` | bool | true | false 时不拼接历史问题 |
| `queries` | string[] | — | 直接指定多路检索 |
| `top_k`/`filters`/`fields`/`rerank`/`per_jurisdiction_k` | — | — | 同 `/v1/retrieve` |
| `answer` | object | 服务端配置 | `{"enabled": false}` 只检索 |

响应 = 检索响应 + `answer`、`answer_status`、`citations[]`、`references[]`、`error`、
`question`、`retrieval_query`、`messages`、`coverage{unavailable, notes}`。

`answer_status`：`answered`（有引用）、`partial`（提到依据不足但有引用）、
`unsupported`（无引用）、`unavailable`（空检索/拒答）、`skipped`（生成关闭）、`error`。

`citations[]`：`chunk_id`、`doc_id`、`article_id`、`article_label`、`source_doc`、`quote`、`text`。
`references[]`：`chunk_id`、`rank`、`doc_id`、`source_doc`、`article_id`、`article_label`、
`jurisdiction`、`effective_from`、`effective_to`、`score`、`rerank_score`、`cited`、`text`、
`article_offsets`。

### 8. 文档管理

**上传** `POST /v1/documents`（multipart）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `files` | 必填 | ≤200；md/markdown/txt/docx/pdf（无 OCR） |
| `jurisdiction` | `CN` | 决定分块家族 |
| `language` | CN→`zh`，JP→`ja`，其余 `en` | |
| `precision` | CN→`exact`，其余 `degraded` | 影响 `as_of` 过滤 |
| `applies_to` | 空 | 逗号/分号分隔 |
| `effective_from` / `effective_to` | 空 | `YYYY-MM-DD` |
| `source_doc` | 文件名 | 展示名 |

响应：`{"job_id": "...", "status": "running", "documents": ["a.md"]}`；重复内容按 sha256
去重（`duplicate: true`）。上限：单文件 20 MB、单批 200 MB；超限 413；有 job 在跑 409。

**列表** `GET /v1/documents?status=ready` → `{"documents": [...], "count": n}`。
记录字段：`doc_id`、`filename`、`source_doc`、`jurisdiction`、`language`、`precision`、
`applies_to[]`、`effective_from/to`、`sha256`、`status`
（`queued|parsing|parsed|indexing|ready|failed`）、`chunk_count`、`parsed_path`、
`original_path`、`error`。

`GET /v1/documents/{doc_id}` 详情；`PATCH` 可改元数据（触发重建）；`DELETE` 删除并重建；
`POST /v1/documents/{doc_id}/reindex` 重解析。
`GET /v1/documents/{doc_id}/chunks?limit=100&offset=0`（limit ≤500）→
`{"doc_id","total","offset","limit","chunks":[{chunk_id,article_id,article_label,section,part,text,token_count,effective_from,effective_to}]}`。

### 9. 管理端点

| 端点 | 请求 | 响应 |
|---|---|---|
| `POST /v1/admin/manifest` | 无 | `{"documents": n, "path": "corpus/manifest.json"}` |
| `POST /v1/admin/rebuild` | `{"background": true, "activate": true, "chunking"?: {}, "embedding"?: {}}` | 后台 `{job_id,status,pending_rebuild}`；`background=false` 返回完整 job |
| `GET /v1/jobs/{job_id}` | — | `job_id`、`kind`（`rebuild|ingest|delete|reindex|update`）、`status`（`running|done|error`）、`started_at`、`finished_at`、`result`、`error`、`progress{stage,done,total}` |
| `POST /v1/admin/activate/{version}` | — | `{"active": "<version>"}` |
| `GET /v1/admin/indexes` | — | `{"indexes": [...], "current": "..."}` |

### 10. 错误码

| 码 | 场景 |
|---|---|
| 400 | 参数非法（空/超 16 查询、`top_k` 越界、日期格式、`neighbors`/`follow_refs` 越界、无文件） |
| 404 | 未知 article/doc/job/索引版本 |
| 409 | 已有 ingest/rebuild job 在跑 |
| 413 | 超过单文件/单批大小限制 |
| 500 | embedding 网关不可恢复错误（未开降级） |
| 503 | 索引未构建/为空/由更新 schema 写入 |

---

## English

### 1. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | health, active index, schema |
| GET / PUT | `/v1/config` | read / hot-update config |
| POST | `/retrieve`, `/v1/retrieve` | retrieval only |
| GET | `/v1/articles/{article_id}` | full article + neighbours + references |
| POST | `/v1/articles/batch` | batch fetch (≤32) |
| POST | `/chat`, `/v1/chat`, `/ask`, `/v1/ask` | retrieval + optional generation |
| POST | `/v1/documents` | batch upload, async ingest |
| GET | `/v1/documents` | list (`?status=`) |
| GET/PATCH/DELETE | `/v1/documents/{doc_id}` | detail / metadata update / delete |
| POST | `/v1/documents/{doc_id}/reindex` | re-parse and re-index |
| GET | `/v1/documents/{doc_id}/chunks` | paginated chunks |
| POST | `/v1/admin/manifest` | rebuild manifest |
| POST | `/v1/admin/rebuild` | rebuild index |
| GET | `/v1/jobs/{job_id}` | job status |
| GET | `/v1/admin/rebuild/{job_id}` | alias |
| POST | `/v1/admin/activate/{version}` | switch active index |
| GET | `/v1/admin/indexes` | index files + current |
| GET | `/` | service info |

### 2. `/health`

`200` with `status`, `pending_rebuild`, `index{version,chunks,documents,embedding_model,
chunking_hash,built_at,schema_version,index_schema_outdated}`, `config`.
`503` with `{"status": "not_built"|"empty", "error": ..., "corpus": {...}?}`.

### 3. `/v1/config`

`GET` → `{config, active_index, pending_rebuild}`. `PUT` merges partial sections; unknown
keys ignored; `"save": true` persists to `config.json`. Chunking/embedding changes set
`pending_rebuild=true`; every update clears retrieval caches and re-syncs `rerank_enabled`.

### 4. `POST /v1/retrieve`

| Field | Type | Default | Notes |
|---|---|---|---|
| `queries` | `string[]`/`string` | — | 1–16; `question` accepted |
| `top_k` | int | `retrieval.top_k` | 1–100 |
| `filters` | object | `{}` | `jurisdictions[]`, `applies_to[]`, `as_of`, `doc_ids[]`, `article_ids[]`, `per_jurisdiction_k` |
| `fields` | object | `{}` | legacy aliases (`business_type`, `jurisdiction|country`, `date`) |
| `rerank` | bool | server default | per-request override |
| `per_jurisdiction_k` | int | none | 0 disables quotas |

Response: `schema_version`, `chunks[]`, `corpus`, `temporal`, `index`, `field_matches`,
`diagnostics`.
Chunk fields: `chunk_id, doc_id, source_doc, text, score, score_parts{vector,keyword,
vector_norm,keyword_norm,first_stage,hybrid,rerank,fusion_bonus}, articles[], article_ids[],
article_offsets[], effective_from, effective_to, jurisdictions[], applies_to[],
superseded_by, matched_by[], matched_queries[], matched_terms[], matched_fields{}, rank`.
Diagnostics: `funnel`, `zero_reason`, `requested_uncovered`, `rerank`, `per_jurisdiction_k`,
`quota_slots`, `embedding_cache_hits`, `degraded[]`, `confidence`, `warnings[]`,
`timings_ms{filter,embed,bm25,vector,rank,total}`, `cache_hit`.
`zero_reason ∈ {no_jurisdiction,no_business_type,no_temporal,no_doc,no_article,no_match}`.

### 5. `/v1/articles/{article_id}`

Path accepts `/`; query `neighbors` (0–5), `follow_refs` (0–2).
Response adds `references{resolved[{text,article_id,article_label}], unresolved[{text}]}`,
`followed[]` (cap 20) and `truncated`. `400` on bad ranges, `404` unknown id, `503` no index.

### 6. `/v1/articles/batch`

`{"article_ids": [...], "neighbors": 0, "follow_refs": 0}` (≤32) →
`{articles, missing, count}`.

### 7. `/v1/chat`

Fields: `messages|question`, `retrieval_query`, `rewrite_query` (default true), `queries`,
`top_k`, `filters`, `fields`, `rerank`, `per_jurisdiction_k`, `answer{enabled}`.
Adds `answer`, `answer_status` (`answered|partial|unsupported|unavailable|skipped|error`),
`citations[]`, `references[]`, `error`, `question`, `retrieval_query`, `messages`,
`coverage{unavailable,notes}`. Citation ids must be 16-hex `chunk_id`s from the retrieved set;
invalid ids are dropped.

### 8. Documents

Upload (multipart): `files` (≤200; md/markdown/txt/docx/pdf), `jurisdiction` (CN),
`language` (auto), `precision` (auto), `applies_to`, `effective_from`, `effective_to`,
`source_doc`. Returns `{job_id,status,documents}`; sha256 dedupe; 20 MB/file, 200 MB/batch.
Records: `doc_id, filename, source_doc, jurisdiction, language, precision, applies_to,
dates, sha256, status, chunk_count, parsed_path, original_path, error`.
Chunk listing: `?limit=100&offset=0` (limit ≤500).

### 9. Admin

`manifest`, `rebuild{background,activate,chunking?,embedding?}`, `jobs` (kind/status/
progress/result/error), `activate/{version}`, `indexes`.

### 10. Status codes

400 invalid input; 404 unknown resource; 409 job already running; 413 size limits;
500 unrecoverable embedding error; 503 index missing/empty/newer schema.
