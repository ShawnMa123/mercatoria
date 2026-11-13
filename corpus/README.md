# 外文语料

> 抓取日期：**2026-11-13**（eCFR 快照 2026-11-01）
> 用途：导入 RAGFlow，与中文法规一起做跨境业务场景测试
> 抓取脚本可直接重跑，保证可复现

## 总量

| 法域 | 源 | 文档 | 条文 | 字符 |
|---|---|---|---|---|
| 🇺🇸 US | eCFR Title 19（CBP 海关实务） | 51 parts | **2373** | 5.6 M |
| 🇺🇸 US | eCFR Title 15（EAR 出口管制） | 16 parts | 179 | 1.5 M |
| 🇺🇸 US | eCFR Title 31（OFAC 制裁） | 6 parts | 440 | 0.7 M |
| 🌐 WTO | 协定正文（GATT/TFA/SPS/TBT/ADP/CV/RO/LIC/SCM/TRIMS/DSU/SG/AG/TRIPS） | 14 docs | 308 | 0.9 M |
| 🇦🇺 AU | 海关法、关税税率法、出口管制法、海关条例、自主制裁法 | 5 acts | **1126** | 1.1 M |
| 🇯🇵 JP | 关税法、关税定率法、外為法、通关/进口贸易管理令及 3 部配套政令 | 9 laws | **1828** | 2.1 M |
| 🇪🇺 EU | 欧盟海关法典 952/2013、两用物项条例 2021/821 | 2 regs | 320 | 0.9 M |
| 🇨🇳 CN | 法律、行政法规、海关总署/商务部规章、司法解释 | 39 docs | 1650 | 0.6 M |
| 🌐 | UNCITRAL CISG | 1 doc | 101 | 0.08 M |
| **合计** | | **143** | **8325** | **13.6 M** |

UK 目前为 0 篇：`legislation.gov.uk` 对非浏览器客户端返回 AWS WAF 挑战
（HTTP 202 + 空 body），脚本会如实记录为 missing，见 `fetch_eu_uk.py`。

## 目录结构

```text
corpus/
├── fetch_ecfr.py       eCFR 抓取（--title N [--parts a,b,c]，省略 parts 用默认清单）
├── fetch_wto.py        WTO 抓取（自动跟进 _01_e/_02_e 分页）
├── fetch_cisg.py       CISG 抓取（PDF 抽取）
├── fetch_jp.py         日本 e-Gov 抓取
├── fetch_au.py         澳洲 Federal Register 抓取
├── fetch_eu_uk.py      欧盟 Cellar 抓取 / 英国 legislation.gov.uk
├── fetch_cn.py         中文法规（LawRefBook 镜像）抓取 + 规整
├── ecfr/title-19/      51 个 part-*.md
├── ecfr/title-15/      16 个 part-*.md
├── ecfr/title-31/      6 个 part-*.md
├── wto/                14 个 *_e.md
├── jp/                 9 个法令 md
├── au/                 5 个法案 md（+ 原始 PDF 存档）
├── eu/                 2 个条例 md
├── cn/                 39 个法规 md
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

# 美国 eCFR（省略 --parts 时使用脚本内 DEFAULT_PARTS）
$PY fetch_ecfr.py --title 19
$PY fetch_ecfr.py --title 15
$PY fetch_ecfr.py --title 31

# WTO
$PY fetch_wto.py

# CISG
$PY fetch_cisg.py

# 日本
$PY fetch_jp.py

# 澳洲
$PY fetch_au.py

# 欧盟 / 英国
$PY fetch_eu_uk.py

# 中文法规（LawRefBook 镜像）
$PY fetch_cn.py

# 或者一次跑完并检查数量
bash ../scripts/setup-corpus.sh
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
| 関税法施行令 | `329CO0000000150` |
| 関税定率法施行令 | `329CO0000000155` |
| 関税暫定措置法 | `335AC0000000036`（**不是** `341AC…`，会 404） |
| 外国為替令 | `355CO0000000260` |
| 輸入貿易管理令 | `324CO0000000414` |

另有坑：`/api/2/laws?law_title=...` 搜索可用，但返回的 `law_id` 才是
`/law_data/{id}` 的合法输入；按年份臆造的 ID 一律 404。

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
| UK legislation.gov.uk | `data.xml` 返回 202 + 空 body（AWS WAF JS 挑战），换 UA/Accept 均无效；需在可访问网络用浏览器下载后手工转换 |
| ICC 规则（UCP 600 / URDG 758） | 付费出版物，有版权，公开源不存在 |
| 国家法律法规数据库 flk.npc.gov.cn | 返回 SPA 空壳，正文要执行 JS；中文语料改走 LawRefBook 镜像（`fetch_cn.py`） |

EU 不再是缺口：`publications.europa.eu/resource/celex/{celex}.ENG.xhtml`
是官方直连（303 → Cellar 对象存储），无 WAF，`fetch_eu_uk.py` 可直接抓。

### 欧盟 Cellar / 中文镜像的坑

| 坑 | 处理 |
|---|---|
| EU XHTML 是 CONVEX 排版，条文在 `p.oj-ti-art`，正文在 `p.oj-normal` | 按 class 抽取块，`## Article N` / `### 章节` |
| UCC 的 288 条分散在 chapter/section 层级里 | 章节用 `###`，条文用 `##`，与其它法域一致 |
| LawRefBook 的文件名带发布日期 | 保留 `名称(YYYY-MM-DD).md`，manifest 从文件名取生效日期 |
| LawRefBook 文件以 H1 + INFO END 开头 | `normalize_cn_corpus.normalize()` 转成 HTML 注释 |

## 语料范围说明

US 部分刻意**不是全量**。Title 19 共 83 个 part，取了 51 个与贸易单据审核相关的；
排除的是船只、岛国领地、政府雇员等无关内容（全量灌入会稀释召回信号）。

Title 15 取 EAR 核心 16 个 part（新增 740 许可例外、742 管制政策、748/750 许可
申请与审批、758 清关与 EEI、760 反抵制、762 记录保存、772 术语定义）；
Title 31 新增 OFAC 6 个 part（501 程序与处罚、510/535/560/587/591 制裁条例；
part 542 在该快照日期 404）。

## 已知缺口（v2 复核）

| 缺口 | Real-world 来源 | 处理方式 |
|---|---|---|
| FT-008 公式定价公告 | 海关总署公告 2021 年第 44 号《关于公式定价进口货物完税价格确定有关问题的公告》。`customs.gov.cn` 对脚本返回瑞数 JS 反爬壳，LawRefBook 不镜像公告 | **人工下载**公告正文 → 存为 `corpus/cn/公式定价公告(2021-09-01).md`（首行 `<!-- document: ... -->`）→ `python tools/normalize_cn_corpus.py` 规整 → `python -m mercatoria.cli manifest` 重建 manifest → 重建索引。完成后把 fixture 里该题 `corpus_gap` 改回 `false` 并复核 |
| 英国 legislation.gov.uk | AWS WAF JS 挑战（HTTP 202 空 body） | 浏览器下载 `ukpga/2018/22/data.xml` → `python corpus/convert_uk_xml.py --input data.xml --slug uk-cross-border-trade-2018 --title "Taxation (Cross-border Trade) Act 2018"`。`check_corpus.py` 把 `uk/` 视为可选源 |
| 出口收汇核销（FT-054） | 2012-08-01 起取消核销单，2012 年公告已于 2025 年废止，现行法无正向条文 | 已改为 `no_answer` 用例，不再计入 `corpus_gap` |
