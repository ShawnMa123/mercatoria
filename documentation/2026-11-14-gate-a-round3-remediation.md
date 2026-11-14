# Gate A 第三轮修复报告（H01-H03）

日期：2026-11-14  
Review 基线：`52175dc`（分支 `fix/gate-a-round2`）。  
本轮分支：`fix/gate-a-round3`，从 `52175dc` 直接切出，未 amend/rebase 任何既有提交。  
范围：只做 Task D1-D3（H01-H03）。未开展 M1-M4，未开始 rerank 批次 B；rerank 默认仍为
`enabled=false`、`gte-rerank-v2`、`weighted`、`weight=0.5`（本轮未触碰 config/rerank 代码）。

## 0. 环境与约束确认

- 全部测试在临时目录（`tempfile.TemporaryDirectory`）内完成；DB、cache、config、manifest、uploads 均为临时路径。
- 使用 fake embedder；未调用真实/付费网关，未启动或重启任何服务，未监听端口（CLI 用例是离线子进程，无 socket）。
- 未修改 `ragflow-src/`，未 push/部署。
- 四份交接文档（含本轮 handoff）保持未跟踪状态，未改动他人代码。

## 1. 提交清单

| 提交 | 内容 |
|---|---|
| `9e1cd69` | fix(control): committed-view reads and a real commit/pin boundary (H01) |
| `e47efec` | fix(ingest): one post-commit boundary for the whole projection tail (H02) |
| `d3392aa` | fix(manifest): one owner for the manifest write entry points (H03) |
| `8545165` | docs: round 3 behaviour (control reads, commit boundary, manifest writes) |
| `04e8eb4` | fix(control): close the committed-view reader under its own lock |
| 本文件所在提交 | docs: round 3 commit list |

既有历史（`075b0cf`、`43cabd2`、`6393fd2`、`abc2d5f`、`52175dc`）未变更。

## 2. H01 共享 SQLite 连接把未提交 head 当作已提交状态

### 2.1 根因

`ControlStore` 只有一个连接（`check_same_thread=False`、autocommit）。`publish()` 在同一连接上
`BEGIN IMMEDIATE` → 写 `control_states` / `control_head` / `desired_config` → `COMMIT`，而
`head_generation()`、`head()`、`get()`、`states()`、`desired_config()` 也读这条连接，于是请求线程
在事务提交前就读到了新 head（含新 desired）。`serving_lease` 用这个 head 触发 `reload` 并安装，
事务随后回滚，流程里就出现"在线 generation 大于已提交 head"。

### 2.2 修复（文件 → 函数）

- `mercatoria/control.py`
  - `ControlStore.__init__` / `_connect`：写连接（只在写事务内使用）与第二条 `PRAGMA query_only=ON`
    只读连接；只读连接不开启任何事务。
  - `head_generation` / `head` / `get` / `states` / `desired_config`：全部走 `_read()`（只读连接 +
    `_read_lock`）。`head()` 改为一条 JOIN 语句，head 行与它指向的 state 不会来自两次不同提交。
  - `_head_on(conn)`：事务内部读当前 head 的私有无锁 helper（`publish` 用它做 CAS），不再递归调用
    公开读方法（否则会在非重入锁上自锁）。
  - 类文档写明连接所有权与锁顺序。
- `mercatoria/publication.py`
  - `PublicationCoordinator.publish`：`self.gate` 覆盖"控制事务提交 → `export_current` → 安装快照"，
    `prepare`（打开索引、校验、构造 retriever/embedding/rerank 客户端）仍在门外。
  - `PublicationCoordinator.__init__` 注释改写为真实的锁顺序契约。
- `mercatoria/api.py`：`Service.serving_lease` docstring 改为描述实际边界（读 head + pin 与提交互斥）。

锁顺序（两处注释一致，写死在文档里）：

```text
1. PublicationCoordinator.gate          发布/租约边界（覆盖提交与 head-check+pin）
2. ControlStore._write_lock             同一时刻一个控制事务
3. ControlStore._read_lock              同一时刻一个已提交视图读
路径：gate -> write_lock（publish） 或 gate -> read_lock（head check + pin）
不存在：控制锁 -> 等待 gate。reload 在拿 gate 之前释放读锁，因此不嵌套。
```

### 2.3 复现证据（暂停在 `_write_desired` 内、随后 `OSError` 回滚）

基线 `52175dc`：

```text
uncommitted_head_seen = 2
online_before = 2
leased_generation = [2]
writer_error = OSError rollback forced
committed_head_after_rollback = 1
online_after_rollback = 2
```

本轮修复后（同一探针）：

```text
uncommitted_head_seen = 1
online_before = 1
leased_generation = []
writer_error = OSError rollback forced
committed_head_after_rollback = 1
online_after_rollback = 1
leased_after = [1]
```

（探针为一次性脚本，运行后已删除；同样的构造固化在 `tests/test_gate_a_round3.py`。）

### 2.4 测试映射

| 要求 | 测试 | 基线结果 | 现在 |
|---|---|---|---|
| 未提交 generation 不被租约 | `test_request_never_leases_an_uncommitted_generation` | FAIL（事务未提交时有租约完成） | OK |
| 回滚后 head/online 一致且可读 | `test_rollback_never_leaves_online_ahead_of_control` | FAIL（同上，online 会领先） | OK |
| 提交后单一版本边界 | `test_commit_then_lease_has_a_single_version_boundary` | OK（前向验收） | OK |
| 公开控制读取不含未提交状态 | `test_control_reads_do_not_expose_uncommitted_state` | FAIL（`head_generation` 返回 2） | OK |
| 并发租约不丢已提交 generation | `test_concurrent_leases_never_miss_a_committed_generation` | OK（前向验收） | OK |

测试线程全部使用 `Event` + 明确 join 超时（`finish()` 在 `finally` 语义下释放 Event 并 join），
没有 sleep 轮询；唯一的定时等待是"必须不发生的事件"的有界等待（2 s），失败路径会立刻返回。
主线程从不持有写锁去同步调用可能阻塞的读取。

## 3. H02 commit 后 registry 枚举异常触发反向补偿

### 3.1 根因

`rebuild_with_uploads` 在 `build_index` 提交后直接 `for record in registry.list()`，该调用在既有
try/warning 之外。异常穿出后，`start_delete` / `start_update` 的 `except Exception` 把它当成提交前
失败，恢复旧 registry 状态；而 head/在线索引已经推进，业务状态与权威 manifest 相反。

### 3.2 修复（文件 → 函数）

- `mercatoria/ingest.py`
  - 新增 `READY_STATUSES`、`_finish_committed()`、`_project_after_commit()`：chunk 统计、枚举
    （调用**和**遍历）、逐条 `registry.update`、以及兜底包装各自独立 guard；失败写入
    `result["warnings"]`，返回的仍是同一份已提交结果。
  - `rebuild_with_uploads` 与 `ingest_batch` 都改走该边界（upload/delete/update/reindex 共用一份实现）。
    最终 progress 仍在边界内 guard。
- `mercatoria/api.py`
  - `start_delete` / `start_update` 的补偿改为按结构化结果判断：仅当
    `committed_failure(exc) is None`（确认未提交）才恢复旧状态；`PostCommitError` 与任何携带
    committed 结果的异常都不再触发反向补偿。判断依据是发布链路传出的结果，不是异常类型或猜测 head。

### 3.3 证据

删除一条已就绪文档，注入"提交后 `registry.list()` 抛 `OSError`"：

- 基线：job `status=error`、`result=null`（未标注 committed）、registry 行被恢复为 `ready`，
  重启后文档复活。
- 现在：job 结果为已提交（`committed: true`）并带
  `registry enumeration failed after commit: ...` 警告；registry 保持 `deleted`；head manifest 与在线
  索引都不含该文档；`/health/ready` 200；重启后仍为 `deleted`。
- 元数据版本：新 jurisdiction 同时存在于 registry 与 head manifest。
- 对照：提交前 `publish_built` 失败仍恢复旧状态，head 与在线索引不变（证明没有把错误一概当已提交）。

### 3.4 测试映射

| 要求 | 测试 | 基线 | 现在 |
|---|---|---|---|
| 提交后枚举失败不回滚删除 | `test_registry_enumeration_failure_after_commit_does_not_undo_delete` | FAIL（committed 丢失、状态被恢复） | OK |
| 遍历阶段失败同样安全 | `test_registry_iteration_failure_after_commit_does_not_undo_delete` | FAIL | OK |
| 元数据更新不被恢复旧值 | `test_metadata_update_survives_registry_enumeration_failure` | FAIL | OK |
| 提交前失败仍需补偿 | `test_precommit_publication_failure_still_reverts_registry_state` | OK（对照，必须保持） | OK |
| 重启不复活已删除内容 | `test_restart_does_not_revive_deleted_document_after_projection_failure` | FAIL | OK |

## 4. H03 manifest 生成绕过管理门和 CLI 目录锁

### 4.1 修复（文件 → 函数）

- `mercatoria/manifest.py`：`write_manifest` 改为同目录临时文件 + `fsync` + `os.replace`，失败时删除
  临时文件并重新抛出；目标文件只会是旧的完整 JSON 或新的完整 JSON。
- `mercatoria/api.py`：新增 `Service.write_manifest()`，用管理 token（kind=`manifest`）包住
  corpus 读取 + 原子写；`POST /v1/admin/manifest` 只调用它。忙时由 `ManagementBusy` handler 返回 409
  且不改动文件；失败时 `finally` 释放 token。
- `mercatoria/cli.py`：`_cmd_manifest` 先取 `DirectoryLock(cfg.resolved_write_lock_path())`，
  锁忙则 stderr 报错并 `return 1`，不触碰 manifest。`--out` 不能绕过：锁保护的是这条命令读取
  corpus/manifest 输入的那个数据目录。

### 4.2 证据

- 基线：手动持有 management token（kind=`rebuild`）后 `POST /v1/admin/manifest` 返回 **200** 并写出文件
  （owner 仍是 rebuild）。
- 现在：同一条件返回 **409**（`error.code=management_busy`），manifest bytes 与 `st_mtime_ns` 均不变；
  释放 token 后同一请求 200。
- 注入写失败后管理 token 立即释放，后续 manifest 请求 200。
- 基线 CLI：目录锁被占用时仍 `returncode=0` 并重写 manifest。
- 现在 CLI：锁被占用时 `returncode=1`、stderr 含 `in use`、文件 bytes 不变；释放后同命令 0 退出。

### 4.3 测试映射

| 要求 | 测试 | 基线 | 现在 |
|---|---|---|---|
| API 忙时 409 且不改文件 | `test_manifest_api_is_rejected_while_management_is_busy` | FAIL（200，文件被改写） | OK |
| 失败后释放管理权 | `test_manifest_api_releases_management_on_failure` | OK（本就不取 token，属前向验收） | OK |
| CLI 第二进程拒绝锁 | `test_cli_manifest_refuses_a_locked_data_directory` | FAIL（0 退出并写文件） | OK |
| 写入失败保持旧 manifest | `test_manifest_write_is_atomic_on_failure` | ERROR（模块未导入 os，即原地写） | OK |

## 5. 测试真实结果

命令（Windows PowerShell）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_gate_a_round3.py' -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

实际输出（本轮最后一次运行）：

- `test_gate_a_round3.py`：**Ran 14 tests in 5.668s — OK**。
- 全量：**Ran 248 tests in 18.949s — OK**（基线 234 + 本轮新增 14）。
- 失败注入的中间结果：H01 修复前 5 个用例中 3 个失败（另 2 个是前向验收，前后都通过）；
  H02 修复前 5 个中 4 个失败（另 1 个是必须保持通过的提交前对照）；H03 修复前 4 个中 2 失败 1 error
  （另 1 个是前向验收）。逐项转为通过。
- 警告：既有 Starlette/httpx 弃用警告（`StarletteDeprecationWarning: Using httpx with
  starlette.testclient is deprecated`）；另有既有 CLI 锁测试写入 stderr 的预期提示文本，非失败。
- 无真实网关调用，无端口监听，无生产目录访问。

## 6. 连接与锁说明（实现现状）

- **写连接** `ControlStore.conn`：`isolation_level=None`，只在 `BEGIN IMMEDIATE ... COMMIT` 内使用；
  由 `_write_lock` 串行。WAL + `busy_timeout=5000`。
- **读连接** `ControlStore._read_conn`：`PRAGMA query_only=ON`；每个公开读一条语句、不开事务；由
  `_read_lock` 串行。SQLite 保证另一条连接只看到已提交快照。
- **发布门** `PublicationCoordinator.gate`（`RLock`）：覆盖提交、export 与安装；`prepare`、
  `load_state`、检索与生成都在门外。
- **锁顺序**：gate → write_lock / read_lock，且从不反向；控制事务内部读 head 走 `_head_on(conn)`。
- **进程级**：`data/documents/write.lock` 仍由服务在打开任何可写 store 之前获取；CLI `manifest` 与
  `rebuild` 共用该锁；API 的 manifest 入口在服务进程内，因此只由管理 token 协调。
- **管理 token**：新增 kind `manifest`，与 config/rebuild/ingest/delete/update/reindex/activate/rollback 同一把。

## 7. 兼容性

- 无 schema 迁移：control `PRAGMA user_version` 仍为 2，索引 schema 不变；已有 DB/索引/缓存直接可用。
- 公开 HTTP 接口不变；行为收紧两处：`POST /v1/admin/manifest` 忙时由 200 变 409；
  `python -m mercatoria manifest` 在锁忙时非 0 退出。两者都是本轮要求。
- `write_manifest` 签名/返回值不变；临时文件为同目录 `.manifest.json.*.tmp`，失败即清理。
- `ControlStore` 内部字段 `_lock` 改名为 `_write_lock`，无仓库内外部使用者（已 grep 确认）。
- 关闭后的控制读取由 `sqlite3.ProgrammingError` 变为 `ControlError("control store is closed")`。
- job 结果语义不变：仍以 `committed`/`installed`/`snapshot_version`/`warnings` 表达；区别是提交后的
  投影失败不再返回 `result: null`。

## 8. 已知限制与未测风险

- 发布门是**进程内**的；多 worker 部署仍不受支持（既有边界）。多进程一致性只靠目录锁 + 各自连接。
- 提交/export/安装都在门内，数据请求会在这段时间等待发布门；`current.txt` 写很慢的极端磁盘未压测。
- 只读连接与写连接同进程；跨进程写竞争仍依赖 `busy_timeout=5000`，超时后的 `SQLITE_BUSY` 行为未改。
- 提交后投影失败（含枚举失败）只记 warning：`chunk_count`/`status` 可能滞后到下一次发布，这是既有语义，
  现在覆盖了枚举与遍历。
- `_project_after_commit` 对 registry 整体 `list()`；超大 registry 的时间/内存未测。
- 结果组装分支（`result["version"]` 等）理论上不抛异常，未单独注入故障。
- 未做进程崩溃注入（commit 后、install 前崩溃仍走既有 `PostCommitError`/readiness 路径，本轮未新增验证）。
- Windows 上 `os.replace` 遇到目标文件被持续打开的行为未测；本轮只在单进程内注入替换失败。

## 9. 结论

H01-H03 的失败用例均已先复现、再修复、再回归；全量 248 测试通过。停在 Gate A，等待独立 Review，
不视为 Gate A 已通过。
