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
| GET | `/health`、`/health/ready` | 健康、快照/索引版本、head 与在线 generation 是否一致、schema、配置摘要 |
| GET | `/health/live` | 存活（不依赖索引） |
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
| POST | `/v1/admin/manifest` | 从语料目录重建 manifest（管理互斥，原子替换） |
| POST | `/v1/admin/rebuild` | 重建索引（后台/同步） |
| GET | `/v1/jobs/{job_id}` | job 状态 |
| GET | `/v1/admin/rebuild/{job_id}` | 同上（别名） |
| POST | `/v1/admin/activate/{snapshot_version}` | 回滚到保留的完整状态 |
| POST | `/v1/admin/indexes/{index_version}/activate` | 发布已构建的候选索引 |
| GET | `/v1/admin/snapshots` | 历史状态（generation）列表 |
| GET | `/v1/admin/indexes` | 列出索引文件、当前版本与 staged 候选 |
| GET | `/` | 服务名与端点清单 |

### 2. `GET /health`

```json
{
  "status": "ready",
  "snapshot_version": "3",
  "index_version": "20261113-112706-384123-72d4a2df2417",
  "chunks": 11562,
  "documents": 143,
  "head_generation": 3,
  "online_generation": 3,
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

`ready` 只在**已提交的 head 与本进程在线快照一致**时返回。head 已推进但在线快照未跟上时，健康检查会先做一次确定性 reload；仍不一致则返回
503 `{"status": "not_ready", "error": "...", "head_generation": 3, "online_generation": 2}`，
不会把旧快照当作当前状态。未建索引（head 为空）返回 503 `not_built`。
已发布空库是 `ready`（`chunks=0`），与未建有区别。

### 3. `GET|PUT /v1/config`

`GET` → `{"config": {...}, "config_version": "...", "desired_config": {...}|null, "desired_config_version": "...", "active_index": "...", "snapshot_version": "3", "pending_rebuild": false}`。
`config` 是**当前在线快照实际使用的配置**（来自已提交的 control state）；
`desired_config` 仅在存在待重建的 build 变更时给出运维意图。

`PUT` 为**部分更新**：只传要改的 section/key；未知字段、类型不符（如 `top_k: "5"`、
`enabled: "false"`）、越界或非法端点 URL 一律 422，且不产生任何部分应用。
接受的配置**一律持久化**到 control DB：runtime 更新与该次接受的 desired 意图在**同一个控制事务**内提交
（head 与 `desired_config` 不会一前一后分裂），仅 pending build 的更新是单行原子 upsert。
`"save": true` 额外把**完整部署配置**（目录、manifest 路径、上传限额 + 非敏感 section）原子写入
`config_path`（同目录临时文件 + `os.replace`）；该文件不含任何 api_key，`save: false` 不会让更新在重启后消失。
文件写入失败只是 `warnings`（已提交状态不回滚）；命令返回体新增 `warnings: []`。

嵌套对象按字段深度合并，不会把 `confidence` 之类的 dataclass 换成 dict。
仅改检索/回答参数时，在**当前活动 build** 上发布新 `snapshot_version`（复用 `index_version`），
待重建的 chunking/embedding 变更不会随之泄漏进在线状态。
改 `chunking`、`embedding.base/model/dim/model_revision/space_version` 属于 build 身份变更：只记录为期望状态并置
`pending_rebuild=true`，需显式 rebuild 才生效；`embedding.timeout/retries`（以及凭证轮换）属于运行参数，
立即重建客户端并发布，且不会使已有向量失效。

**管理互斥**：config、rebuild、上传、删除、元数据、reindex、manifest 生成、候选激活与回滚同一时刻只允许一个在飞。
并发管理写返回 409（且不改动任何状态或文件）：

```json
{"detail": "management operation 'ingest' is in progress; cannot start 'config'",
 "error": {"code": "management_busy", "retryable": true,
           "details": {"current": "ingest", "requested": "config"}}}
```


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

**删除是逻辑发布**：`DELETE` 只把记录标为 `deleted`（原文件、解析稿、parsed revision 全部保留），
等新 manifest 发布成功后才从在线索引消失；**提交前**重建失败会恢复原状态。
提交完成后（control head 已推进）的附属失败（导出、内存安装）不会回滚业务状态：
导出失败记为 `warnings`，安装失败让 job 变成 `error` 且 `result.committed=true`，readiness 返回 503，重启按 DB 恢复。

**列表** `GET /v1/documents?status=ready` → `{"documents": [...], "count": n}`；已逻辑删除的记录不出现，
详情返回 404。
记录字段：`doc_id`、`filename`、`source_doc`、`jurisdiction`、`language`、`precision`、
`applies_to[]`、`effective_from/to`、`sha256`、`status`
（`queued|parsing|parsed|indexing|ready|failed`；内部另有 `deleted`）、`chunk_count`、`parsed_path`、
`original_path`、`error`。

`GET /v1/documents/{doc_id}` 详情；`PATCH` 可改元数据（触发重建）；`DELETE` 删除并重建；
`POST /v1/documents/{doc_id}/reindex` 重解析。
`GET /v1/documents/{doc_id}/chunks?limit=100&offset=0`（limit ≤500）→
`{"doc_id","total","offset","limit","chunks":[{chunk_id,article_id,article_label,section,part,text,token_count,effective_from,effective_to}]}`。

### 9. 管理端点

| 端点 | 请求 | 响应 |
|---|---|---|
| `POST /v1/admin/manifest` | 无 | `{"documents": n, "path": "corpus/manifest.json"}`；与其它管理操作互斥（忙时 409，不触碰文件），文件用临时文件 + `os.replace` 原子替换，读取方只会看到旧的完整 JSON 或新的完整 JSON |
| `POST /v1/admin/rebuild` | `{"background": true, "activate": true, "chunking"?: {}, "embedding"?: {}}` | 后台 `{job_id,status,pending_rebuild}`；`background=false` 返回完整 job。job `result` 含 `committed`、`installed`、`snapshot_version`、`warnings` |
| `GET /v1/jobs/{job_id}` | — | `job_id`、`kind`（`rebuild|ingest|delete|reindex|update`）、`status`（`running|done|error`）、`started_at`、`finished_at`、`result`、`error`、`progress{stage,done,total}` |
| `POST /v1/admin/activate/{snapshot_version}` | — | 追加新 generation，恢复目标的索引、manifest 与该 generation 持久化的配置；未知 404，索引已淘汰/不可加载 409 |
| `POST /v1/admin/indexes/{index_version}/activate` | — | 发布候选索引：读取该次 build 自己记录的 manifest、build 配置与 base generation；base 已过期 409 且状态不变 |
| `GET /v1/admin/snapshots` | — | `{"current": {...}, "snapshots": [...]}`，每项含 `snapshot_version`、`generation`、`index_version`、`manifest_id`、`build_profile_id`、`runtime_config_id`、`config_persisted`、`created_at` |
| `GET /v1/admin/indexes` | — | `{"indexes": [...], "current": "...", "snapshot_version": "3", "staged": [...]}` |

候选激活要点：

- 候选只存在于磁盘，`activate=false` 构建后重启也**不会**自动激活；迁移只认显式 `current.txt` 指针，
  没有指针即 `not_built`，指针损坏 fail closed，已有 head 后 `current.txt` 不再参与判断。
- 候选文件内记录了 `{manifest, build config, base generation}`；激活用这份记录 prepare（完整校验并构造客户端），
  base 过期返回 409，不会用"当前 manifest"替代，也不会用当前 head 生成新 base 强行覆盖。
- **缺少候选记录、缺少 base 字段或 manifest 记录无效一律 409**，提示重新构建；不按文件年龄/名字或当前 head 猜 base。
  需要恢复旧索引时走 `activate/{snapshot_version}` 回滚（目标的完整 ControlState），不提供隐式 force 通道。
  记录中的 `base_generation: null` 只表示"显式期望空库"，与"缺信息"不同。
- vec/chunk 必须逐条对应：同数量但错 ID、缺失、多余、NaN/Inf、维度不符都拒发，拒绝后 head 与在线版本不变。

### 10. 错误码

| 码 | 场景 |
|---|---|
| 400 | 参数非法（空/超 16 查询、`top_k` 越界、日期格式、`neighbors`/`follow_refs` 越界、无文件） |
| 404 | 未知 article/doc/job/索引版本 |
| 409 | 另一管理操作在飞（`management_busy`）；发布/回滚目标不可用、候选 base 过期、候选缺少自身证据或文档集合/配置不可用；候选索引 chunk/vector 不一致 |
| 422 | 配置字段未知、类型不符、越界或端点 URL 非法 |
| 413 | 超过单文件/单批大小限制 |
| 500 | embedding 网关不可恢复错误（未开降级） |
| 503 | 索引未构建/为空/由更新 schema 写入；发布已提交但本进程未能安装（`not_ready`，见 `/health`）；数据接口发现 head 与在线快照不一致且无法恢复 |

**数据接口的就绪门**：`/v1/retrieve`、`/v1/chat`、`/v1/articles/*`、`/v1/documents/{id}/chunks` 等证据读取不只看
`/health`：每次请求在短发布门内确认"已提交 head == 在线快照"再取得租约；不一致时先做一次确定性 reload，
仍失败即 503，不会把旧快照当作当前状态返回。已持有租约的在途请求可以用旧快照完成。

---

## English

### 1. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/health/ready` | health, snapshot/index version, head-vs-online agreement, schema |
| GET | `/health/live` | liveness, index-independent |
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
| POST | `/v1/admin/manifest` | rebuild manifest (management-exclusive, atomic replace) |
| POST | `/v1/admin/rebuild` | rebuild index |
| GET | `/v1/jobs/{job_id}` | job status |
| GET | `/v1/admin/rebuild/{job_id}` | alias |
| POST | `/v1/admin/activate/{snapshot_version}` | roll back to a retained state |
| POST | `/v1/admin/indexes/{index_version}/activate` | publish a built candidate |
| GET | `/v1/admin/snapshots` | retained generations |
| GET | `/v1/admin/indexes` | index files, current and staged candidates |
| GET | `/` | service info |

### 2. `/health`

`200` with `status`, `snapshot_version`, `index_version`, `chunks`, `documents`,
`head_generation`, `online_generation`, `pending_rebuild`,
`index{version,chunks,documents,embedding_model,chunking_hash,built_at,
schema_version,index_schema_outdated}`, `config`.

`ready` requires the committed head and the online snapshot to agree. When the head has moved but
this process has not installed it, readiness retries one deterministic reload; if the mismatch
persists it returns `503 {"status": "not_ready", "head_generation": 3, "online_generation": 2}`.
It never reports a stale snapshot as current. A published but empty index is `200` with
`status="ready"` and `chunks=0`: retrieve returns `chunks=[]` and chat answers `unavailable`.
`503 {"status": "not_built"}` means no control state was ever published — a different condition,
not an empty library.

### 3. `/v1/config`

`GET` → `{config, config_version, desired_config, desired_config_version, active_index,
snapshot_version, pending_rebuild}`. `config` is the configuration the online snapshot actually
serves (resolved from the committed control state); `desired_config` is the operator's intent and
is only present while a rebuild is pending.

`PUT` merges partial sections; unknown fields, wrong types (`top_k: "5"`, `enabled: "false"`),
out-of-range values or an invalid endpoint URL return 422 and apply nothing. Accepted values are
always persisted in the control DB: a runtime publication and the accepted desired intent are
committed in **one control transaction**, and a pending-build-only update is a single atomic row
upsert. `"save": true` additionally writes the **complete deployment config** (directories, manifest
path, upload limits and the non-sensitive sections) atomically to `config_path`
(same-directory temp file + `os.replace`); that file never contains an api_key, and `save: false`
no longer means "temporary". A failed file write is reported in `warnings` and does not undo the
committed state.
Nested objects merge field by field, so a dataclass such as `confidence` is never replaced by a
dict. Retrieval/answer-only changes publish a new `snapshot_version` on the **active** build over
the same `index_version`, and a pending chunking/embedding change does not leak into it.
Chunking and `embedding.base/model/dim/model_revision/space_version` are build identity: they are
recorded as desired state with `pending_rebuild=true` until an explicit rebuild publishes a matching
index. Runtime client parameters such as `embedding.timeout`/`retries` (and credential rotation)
rebuild the client and publish immediately without invalidating vectors.

**Management exclusivity**: config, rebuild, upload, delete, metadata, reindex, manifest
generation, candidate activation and rollback are serialized. A concurrent management write returns
409 `management_busy` before changing anything (no state and no file is touched):

```json
{"detail": "management operation 'ingest' is in progress; cannot start 'config'",
 "error": {"code": "management_busy", "retryable": true,
           "details": {"current": "ingest", "requested": "config"}}}
```

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
`DELETE` is a logical publish: the record is marked `deleted` (files, parsed text and revision
stay on disk), the document leaves the index only when the new manifest publishes, and a failed
rebuild restores the previous status. Deleted records are hidden from the list and return 404.
Records: `doc_id, filename, source_doc, jurisdiction, language, precision, applies_to,
dates, sha256, status, chunk_count, parsed_path, original_path, error`.
Chunk listing: `?limit=100&offset=0` (limit ≤500).

### 9. Admin

`manifest`, `rebuild{background,activate,chunking?,embedding?}`, `jobs` (kind/status/
progress/result/error), `snapshots`, `indexes`, `activate/{snapshot_version}` (append a
generation restoring that state; 404 unknown, 409 unavailable target) and
`indexes/{index_version}/activate`.

Candidate activation reads the manifest, build config and base generation recorded by that build
and CAS-checks the base. A candidate without a record, without a `base_generation` field or with an
invalid manifest record is refused with 409 and no state change; there is no implicit base from the
current head. `base_generation: null` means "the head must still be empty", not "unknown". Use
`activate/{snapshot_version}` to restore a retained state.

### 10. Status codes

400 invalid input; 404 unknown resource; 409 another management operation in flight
(`management_busy`), publish/rollback target unusable, candidate base expired, candidate missing its
own evidence, or chunk/vector alignment invalid; 413 size limits; 422 invalid config
field/type/range/URL; 500 unrecoverable embedding error; 503 index missing or written by a newer
schema, a publication committed but this process could not install it (`not_ready`, see `/health`),
or a data route could not line up the committed head with an online snapshot.

Evidence reads (`/v1/retrieve`, `/v1/chat`, `/v1/articles/*`, `/v1/documents/{id}/chunks`) gate on the
committed head themselves: they confirm head == online snapshot inside the short publication gate and
pin that snapshot before releasing the gate. A mismatch triggers one deterministic reload and then
503; the old snapshot is never served to a new request. A request that already holds a lease
completes on its snapshot.
