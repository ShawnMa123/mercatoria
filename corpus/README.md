# 外文语料

> 抓取日期：**2026-11-07**
> 用途：导入 RAGFlow，与中文法规一起做跨境业务场景测试
> 抓取脚本可直接重跑，保证可复现

## 总量

| 法域 | 源 | 文档 | 条文 | 字符 |
|---|---|---|---|---|
| 🇺🇸 US | eCFR Title 19（海关） | 44 parts | **2072** | 5.4 M |
| 🇺🇸 US | eCFR Title 15（EAR 出口管制） | 8 parts | 81 | 0.44 M |
| 🇺🇸 US | WTO 协定（GATT/TFA/SPS/TBT/ADP/CV/RO/LIC/SCM/TRIMS） | 10 docs | 192 | 0.71 M |
| 🇦🇺 AU | Customs Act 1901 | 1 act | **376** | 365 K |
| 🇦🇺 AU | Customs Tariff Act 1995 | 1 act | 54 | 38 K |
| 🇦🇺 AU | Export Control Act 2020 | 1 act | **464** | 520 K |
| 🇯🇵 JP | 関税法 | 1 law | **530** | 225 K |
| 🇯🇵 JP | 関税定率法 | 1 law | 139 | 60 K |
| 🇯🇵 JP | 外為法（外国為替及び外国貿易法） | 1 law | 213 | 75 K |
| 🇯🇵 JP | 輸出貿易管理令 | 1 law | 63 | 12 K |
| 🌐 | UNCITRAL CISG | 1 doc | 101 | 82 K |
| **合计** | | **72** | **4285** | **7.5 M** |

## 目录结构

```text
corpus/
├── fetch_ecfr.py       eCFR 抓取（--title N --parts a,b,c）
├── fetch_wto.py        WTO 抓取
├── fetch_cisg.py       CISG 抓取（PDF 抽取）
├── fetch_jp.py         日本 e-Gov 抓取
├── fetch_au.py         澳洲 Federal Register 抓取
├── ecfr/title-19/      44 个 part-*.md
├── ecfr/title-15/      8 个 part-*.md
├── wto/                10 个 *_e.md
├── jp/                 4 个法令 md
├── au/                 3 个法案 md（+ 原始 PDF 存档）
└── uncitral/           cisg.md（+ 原始 PDF）
```

每个目录下有 `index.json` 记录抓取元数据（快照日期、来源 URL、条文数）。

## 格式设计：为什么用 `##` 二级标题

所有输出统一用 **`## <条文标识>`** 作为条文分隔：

| 法域 | 条文标题形式 |
|---|---|
| US eCFR | `## § 101.1 Definitions.` |
| WTO | `## Article 1: Publication and Availability of Information` |
| CISG | `## Article 1` |
| AU | `## Section 27 — Short title` |
| JP | `## 第一条` |

不是随意选的。RAGFlow 的 `laws` 解析器默认层级正则为：

```json
group 0: ["^#[^#]", "^##[^#]", "^###[^#]", "^####[^#]"]     ← markdown 标题
group 1: ["第X编", "第X章", "第X节", "第X条", "（一）"]        ← 中国/日本法
group 4: ["PART ...", "Chapter ...", "Section N", "Article N"]  ← 英美法/条约
```

**`##` 在 group 0 里优先级最高**，所以无论条文是中文、日文还是英文，
只要用 `##` 标记，`laws` 解析器都会切出「一条一块」。

这保证所有语料在 RAGFlow 里的切块行为一致，便于统一评测。

## 重跑

```bash
# 需要 .venv（pymupdf 用于 PDF 抽取）
cd corpus
PY=../.venv/Scripts/python.exe

# 美国 eCFR
$PY fetch_ecfr.py --title 19 --parts 0,7,10,11,12,18,19,24,54,101,102,103,111,112,113,114,115,118,122,123,125,127,128,132,133,134,141,142,143,144,145,146,147,148,149,151,152,158,159,161,162,163,165,172
$PY fetch_ecfr.py --title 15 --parts 730,732,734,736,738,744,746,764

# WTO
$PY fetch_wto.py

# CISG
$PY fetch_cisg.py

# 日本
$PY fetch_jp.py

# 澳洲
$PY fetch_au.py
```

## 各源的坑（均为实测踩到）

### eCFR

| 坑 | 处理 |
|---|---|
| 少任一请求头返回 **406** | 同时带 `Accept: application/xml` + `Accept-Encoding: gzip` |
| `DIV8 SECTION` 嵌在 `SUBPART` 里 | 递归遍历，不能只看根的直系子元素 |
| part 104 在 2026-11-01 版本不存在 | 已跳过 |

### WTO

| 坑 | 处理 |
|---|---|
| `gatt94_e.htm` 是**索引页**，无正文 | 改用 `gatt47_e.htm`（GATT 1994 = 1947 原文 + 1994 谅解） |
| 页面有目录区、面包屑、页脚 | 三重过滤 |

### 日本 e-Gov

| 坑 | 处理 |
|---|---|
| `keyword` 搜索每页只回 11 条，命中数却报 1384 | 不用搜索，直接按已知 `law_id` 取 `law_data` |
| `law_full_text` 是 XML 转的 JSON 树 | 递归遍历 `tag=Article`，标题在 `ArticleTitle` 子节点 |

**已知 law_id**：

| 法令 | ID |
|---|---|
| 関税法 | `329AC0000000061` |
| 関税定率法 | `143AC0000000054` |
| 外国為替及び外国貿易法 | `324AC0000000228` |
| 輸出貿易管理令 | `324CO0000000378` |

### 澳洲 Federal Register

**这是最难的一个，正文路径绕了很久。**

| 尝试 | 结果 |
|---|---|
| `api.prod.legislation.gov.au/v1/titles/{id}` | ✅ 只给元数据 |
| `api.prod.legislation.gov.au/v1/documents/{registerId}` | ❌ 404 |
| `www.legislation.gov.au/{registerId}/latest/text` | ❌ SPA 空壳，返回「could not be loaded」 |
| `www.legislation.gov.au/{registerId}/latest/downloads` | ❌ 返回 HTML |
| **`/v1/documents/find(registerId='...',type='Primary',format='Pdf',uniqueTypeNumber=0,volumeNumber=N,rectificationVersionNumber=0)`** | ✅ **直接返回 PDF 二进制** |

三个额外陷阱：

1. **加的 `/$value` 后缀会 404** —— 必须不加后缀直接请求
2. **卷号约定不统一** —— 单卷法案 `volumeNumber=0`，多卷是 `1..N`；同一 registerId 可能既有 0 又有 1..N（0 是无效占位）
3. **标题 ID 不能按年份拼** —— `Customs Tariff Act 1995` 的 ID 是 `C2004A04997` 而不是 `C1995A00359`，必须用 `/v1/titles?$filter=contains(name,'...')` 查

PDF 文本解析的坑：

| 坑 | 处理 |
|---|---|
| 开头几十页是目录 | 过滤带点线（`....`）的行 |
| 目录里也有 `Endnotes` 条目 | 只认「已切出 ≥10 条之后」的那次 |
| 页眉日期 `27 October 2026` 被当成 `Section 27` | 用日期正则排除 |

### UNCITRAL CISG

| 坑 | 处理 |
|---|---|
| 不带浏览器 UA 返回 **403** | 带 UA |
| 页面本身无条文，正文只在 PDF 里 | 下载 `19-09951_e_ebook.pdf` 用 pymupdf 抽取 |
| PDF 里页码 `2` 后跟退格符 `\x08` | 过滤控制字符 |
| PDF 前部有序言、目录、介绍 | 定位「最后一个 PART I」到「Done at」之间 |

## 未抓取的源

| 源 | 原因 |
|---|---|
| UK legislation.gov.uk | 实测 202 空响应，反爬（见 `docs/upstream-legal-sources.md`） |
| EU EUR-Lex | 同上；正确路径是 CELLAR SPARQL，成本高 |
| ICC 规则（UCP 600 / URDG 758） | 付费出版物，有版权，公开源不存在 |

## 语料范围说明

US 部分刻意**不是全量**。Title 19 共 83 个 part，取了 44 个与贸易单据审核相关的；
排除的是船只、岛国领地、政府雇员等无关内容（全量灌入会稀释召回信号）。

Title 15 只取 EAR 核心 8 个 part，Title 31（OFAC 制裁）尚未抓取。
