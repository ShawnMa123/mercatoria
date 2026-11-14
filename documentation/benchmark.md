# 评测与基准 / Benchmark & Quality

> 指标口径（重要）：`Hit@5` 指 top-5 中存在“条文块”且命中全部 `must_mention` 关键词。
> v2 起长条文 `part>=1` 也算条文块；该口径变化单独计量（+2.04pp），避免与检索算法混淆。
>
> Metric definition: `Hit@5` requires a top-5 chunk that is an article block containing every
> `must_mention` keyword. Since v2, `part>=1` chunks of long articles also count; this scoring
> change is measured separately (+2.04pp) so retrieval changes stay attributable.

---

## 中文

### 1. 数据与口径

| 集合 | 规模 | 用途 |
|---|---|---|
| dev `corpus/eval/test_cases.json`（去除 hold-out） | 439 | 调参 |
| hold-out `tests/eval/holdout_cases.json` | 100 | 里程碑验收，不据以调参 |
| 真题 `tests/eval/trade_law_questions.json` | 59 有答案 + 18 `no_answer` | 业务口径 |
| 跨法域配额 `tests/eval/quota_cases.json` | 24 | 配额代价 |
| 引用样例 `tests/test_references_accuracy.py` | 32 | 解析准确率 |

### 2. 当前结果（当前索引：143 篇 / 11,562 chunks）

| 指标 | 数值 |
|---|---|
| dev Hit@5 / Hit@10 / Hit@5(loose) / MRR | **0.8610** / 0.8884 / 0.8861 / 0.7287 |
| hold-out Hit@5 / Hit@10 / MRR | **0.9000** / 0.9500 / 0.7857 |
| 真题 Hit@5 / MRR（59 有答案） | **0.9831** / 0.8446 |
| 真题 60 题口径（含 FT-054） | 0.9667 |
| 跨法域覆盖率@5（配额开/关） | **1.0000** / 0.1250 |
| `no_answer` 低置信率 | 0.8333 |
| 有答案题误标 `low` | 0.0678 |
| 引用解析准确率 | 32/32 |

**分类别（dev，hit5）**：原产地/保税/外汇/担保/稽查信用/商品归类等 30+ 类别 = 1.000；
贸易救济 0.950、出口管制 0.900、报关 0.889、澳洲 0.889、CISG 深度 0.850、关税 0.743、
日本 0.714、WTO 0.556、国际货物买卖 0.500、跨法域比较 0.125（关键词“差异”只在少数条文
中偶然出现，严格口径不可达；真实目标用覆盖率衡量，见上表）。

### 3. 消融（dev 539 / 同一索引）

| 编号 | 配置 | Hit@5 | Hit@10 | MRR | 真题 Hit@5 |
|---|---|---|---|---|---|
| A0 | 旧判分 + 全部新特性 | 0.8479 | 0.8813 | 0.7041 | 0.9831 |
| A1 | 新判分 + 全部开启（基线） | **0.8683** | 0.8998 | 0.7393 | 0.9831 |
| A2 | 关闭 Context Header（独立索引） | 0.8664 | 0.9091 | 0.7556 | 0.9661 |
| A3 | 关闭查询停用词 | 0.8683 | 0.8980 | 0.7401 | 0.9831 |
| A4 | 关闭 BM25 饱和归一 | 0.8683 | 0.8998 | 0.7393 | 0.9831 |
| A5 | 关闭术语表扩展 | 0.8627 | 0.8998 | 0.7421 | 0.9831 |
| A6-on | 多查询加分 0.03（fusion 模式） | 0.7978 | 0.8683 | 0.6417 | — |
| A6-off | 多查询加分 0（fusion 模式） | 0.8015 | 0.8664 | 0.6502 | — |

结论：判分口径 +2.04pp（A0→A1）；Context Header 提升 Hit@10/MRR 与真题 Hit@5；
停用词与 BM25 饱和中性；术语表提升真题 MRR；多查询加分为负 → 默认 0。

### 4. 延迟与容量

| 项 | 数值 |
|---|---|
| 本地检索 p50（fake embed，11.5k chunks） | **28 ms**（filter 0.6 / BM25 2.4 / vector 11.2 / rank 0.2） |
| 缓存命中 p50 | **0.24 ms** |
| 索引加载 | 1.7 s；向量 45.2 MB；进程归因内存 254 MB（tracemalloc） |
| 冷调用 embedding 网关 | 实测受网关影响，常见 0.2–2.3 s/次（不在本地耗时内） |
| 并发 8 线程 × 60 s | 60.2 rps，p50 124 ms，p99 311 ms，0 错误 |
| 并发 32 线程 × 60 s | 57.2 rps，p50 548 ms，p99 908 ms，0 错误，RSS +0.4 MB |
| 10× 语料外推 | 向量 ~452 MB，索引内存 ~2.5 GB，加载 ~17 s（建议届时改 CSR / float16） |

### 5. rerank 策略（n=100）

| 策略 | Hit@5 | MRR | p50 | p95 |
|---|---|---|---|---|
| off | 0.9000 | 0.7679 | 32 ms | 45 ms |
| always | 0.9100 | 0.8007 | 462 ms | 820 ms |
| gated（非 high 才重排） | 0.9100 | 0.8004 | 448 ms | 844 ms |
| topN（只重排前 5） | 0.9100 | 0.8007 | 458 ms | 686 ms |

默认仍关闭：Agent 调用对延迟更敏感；需要更优排序时按请求 `"rerank": true`。

#### 5.1 上游 rerank 模型可用性（2026-11-14 实测）

`GET /api/v1/models` 列出平台 373 个模型，其中 rerank 类 4 个：

| 模型 | 标价（每百万 token） | 本账号实测 |
|---|---|---|
| `gte-rerank-v2`（当前默认） | 0.8 | 可用 |
| `qwen3-rerank` | 0.5 | **403 Model.AccessDenied** |
| `qwen3.7-text-rerank` | 0.5 | **403 Model.AccessDenied** |
| `qwen3-vl-rerank` | 0.5（图像 token） | **403 Model.AccessDenied** |

`compatible-mode/v1/rerank` 在该网关上不存在（404），只能用 dashscope 原生路径。
复测 `gte-rerank-v2`（n=100，seed=7，与上表同口径，2026-11-14）：

| 策略 | Hit@5 | MRR | p50 | p95 |
|---|---|---|---|---|
| off | 0.9000 | 0.7679 | 45 ms | 75 ms |
| always | 0.9100 | 0.8007 | 544 ms | 1168 ms |

质量与 11-13 记录逐位一致；延迟劣化约 18%（p50 462→544 ms）/ 42%（p95 820→1168 ms），
说明网关侧变慢，而非代码路径变化。

同一批 100 题上复核 fusion（`always`，`candidates=10`）：

| fusion | Hit@5 | MRR | p50 |
|---|---|---|---|
| `weighted` 0.5（默认） | 0.9100 | **0.8007** | 531 ms |
| `weighted` 0.8 | 0.8900 | 0.7917 | 522 ms |
| `rerank`（纯重排分） | 0.8900 | 0.7726 | 496 ms |
| `max` | 0.9000 | 0.7706 | 479 ms |

纯重排反而掉 2.0pp Hit@5 / 2.8pp MRR：第一阶段分数带有重排模型看不到的精确法条信号，
所以即便换成更强的 reranker，融合权重也不应设为“只用重排分”。

### 6. 回答质量（抽样 n=16）

| 指标 | 数值 |
|---|---|
| 引用有效率（cited ⊆ retrieved） | 100% |
| 编造引用 | 0 |
| 无答案拒答 | 5/6 |
| 有答案题误拒答 | 0 |
| 关键词全覆盖 / 任一命中 | 80% / 100% |
| 引用段落覆盖全部关键词 | 70% |
| 生成 p50 / p95 | 8.7 s / 18.2 s |

### 7. 语料

143 篇 / 11,562 chunks / 8,191 条文：US 58、CN 39、WTO 14、JP 9、AU 5、EU 2、INT 1。
日本与 WTO 的条文条目数较多（政令/协定分条），CN 多为短条文。

---

## English

### 1. Datasets

| Set | Size | Purpose |
|---|---|---|
| dev `corpus/eval/test_cases.json` minus hold-out | 439 | tuning |
| hold-out `tests/eval/holdout_cases.json` | 100 | milestone acceptance only |
| trade `tests/eval/trade_law_questions.json` | 59 answered + 18 `no_answer` | business-facing |
| quota cases | 24 | quota precision cost |
| reference samples | 32 | parser accuracy |

### 2. Current results (143 docs / 11,562 chunks)

| Metric | Value |
|---|---|
| dev Hit@5 / Hit@10 / loose / MRR | **0.8610** / 0.8884 / 0.8861 / 0.7287 |
| hold-out Hit@5 / Hit@10 / MRR | **0.9000** / 0.9500 / 0.7857 |
| trade Hit@5 / MRR (59 answered) | **0.9831** / 0.8446 |
| trade, 60-question convention | 0.9667 |
| cross-jurisdiction coverage@5 (quota on/off) | **1.0000** / 0.1250 |
| no_answer low-confidence rate | 0.8333 |
| answered cases falsely marked low | 0.0678 |
| reference resolution accuracy | 32/32 |

Category highlights: 30+ categories at 1.000 (origin, bonded, FX, guarantee, credit,
classification); trade remedy 0.950, export control 0.900, declaration 0.889, Australia
0.889, CISG depth 0.850, tariff 0.743, Japan 0.714, WTO 0.556, international sale 0.500,
cross-jurisdiction 0.125 (the strict keyword “差异” only appears incidentally; the real
target is coverage, shown above).

### 3. Ablation (dev 539, same index)

| Run | Config | Hit@5 | Hit@10 | MRR | Trade Hit@5 |
|---|---|---|---|---|---|
| A0 | legacy scoring + all features | 0.8479 | 0.8813 | 0.7041 | 0.9831 |
| A1 | v2 scoring + all on (baseline) | **0.8683** | 0.8998 | 0.7393 | 0.9831 |
| A2 | Context Header off (separate index) | 0.8664 | 0.9091 | 0.7556 | 0.9661 |
| A3 | query stopwords off | 0.8683 | 0.8980 | 0.7401 | 0.9831 |
| A4 | BM25 saturation off | 0.8683 | 0.8998 | 0.7393 | 0.9831 |
| A5 | glossary expansion off | 0.8627 | 0.8998 | 0.7421 | 0.9831 |
| A6-on | multi-query bonus 0.03 (fusion) | 0.7978 | 0.8683 | 0.6417 | — |
| A6-off | multi-query bonus 0 (fusion) | 0.8015 | 0.8664 | 0.6502 | — |

Scoring change contributes +2.04pp; Context Header helps Hit@10/MRR and trade Hit@5;
stopwords and BM25 saturation are neutral; the glossary helps trade MRR; the multi-query
bonus is negative and defaults to 0.

### 4. Latency and capacity

| Item | Value |
|---|---|
| Local retrieval p50 (fake embed, 11.5k chunks) | **28 ms** (filter 0.6 / BM25 2.4 / vector 11.2 / rank 0.2) |
| Cache-hit p50 | **0.24 ms** |
| Index load | 1.7 s; vectors 45.2 MB; traced process memory 254 MB |
| Cold embedding gateway | 0.2–2.3 s per call observed (outside local compute) |
| 8 threads × 60 s | 60.2 rps, p50 124 ms, p99 311 ms, 0 errors |
| 32 threads × 60 s | 57.2 rps, p50 548 ms, p99 908 ms, 0 errors, RSS +0.4 MB |
| 10× corpus projection | ~452 MB vectors, ~2.5 GB index memory, ~17 s load (switch to CSR/float16 then) |

### 5. Rerank policies (n=100)

| Policy | Hit@5 | MRR | p50 | p95 |
|---|---|---|---|---|
| off | 0.9000 | 0.7679 | 32 ms | 45 ms |
| always | 0.9100 | 0.8007 | 462 ms | 820 ms |
| gated (only non-high) | 0.9100 | 0.8004 | 448 ms | 844 ms |
| topN (first 5) | 0.9100 | 0.8007 | 458 ms | 686 ms |

Rerank stays off by default; enable per request when ranking quality matters more than
latency.

#### 5.1 Upstream rerank models (measured 2026-11-14)

`GET /api/v1/models` lists 373 platform models, 4 of them rerankers: `gte-rerank-v2`
(0.8 per million tokens, callable), `qwen3-rerank`, `qwen3.7-text-rerank` and
`qwen3-vl-rerank` (0.5, all three answer `403 Model.AccessDenied` for this key).
`compatible-mode/v1/rerank` does not exist on this gateway (404); only the native
dashscope path works. Re-running `gte-rerank-v2` at n=100/seed=7 reproduced the table
above digit for digit (off 0.9000/0.7679, always 0.9100/0.8007) while latency drifted
18% at p50 (462→544 ms) and 42% at p95 (820→1168 ms) — gateway-side, not code.
Fusion was re-checked on the same 100 questions (`always`, `candidates=10`): `weighted` 0.5
0.9100/0.8007 (best), `weighted` 0.8 0.8900/0.7917, rerank-only 0.8900/0.7726, `max`
0.9000/0.7706. Pure rerank costs 2.0pp Hit@5 and 2.8pp MRR, so the first-stage score carries
signal the cross-encoder cannot see; do not switch to rerank-only fusion even with a stronger
model.

### 6. Answer quality (sample n=16)

| Metric | Value |
|---|---|
| Citation validity (cited ⊆ retrieved) | 100% |
| Fabricated citations | 0 |
| Refusal on no_answer | 5/6 |
| False refusal on answered | 0 |
| Keyword all-hit / any-hit | 80% / 100% |
| Cited passages cover all keywords | 70% |
| Generation p50 / p95 | 8.7 s / 18.2 s |

### 7. Corpus

143 statutes / 11,562 chunks / 8,191 articles: US 58, CN 39, WTO 14, JP 9, AU 5, EU 2,
INT 1.
