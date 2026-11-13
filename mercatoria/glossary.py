"""Bilingual/trilingual glossary for cross-language retrieval.

Chinese questions about WTO/CISG/Japan law must reach English and Japanese
corpus text. A full translation model adds a network call to every query; for
the POC, append the equivalents of known legal terms so BM25 and the
multilingual embedding both see them.

``GLOSSARY`` maps Chinese trade/customs terms to English; ``GLOSSARY_JA`` maps
the same concepts to the Japanese wording used in e-Gov statutes. Both are
substring-matched against the query, longest term first, so 特惠关税 wins over
关税.
"""

from __future__ import annotations

import re

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# Keep entries short and domain-specific. Order matters only for readability.
GLOSSARY: dict[str, str] = {
    # --- WTO core principles -------------------------------------------
    "国民待遇": "national treatment",
    "最惠国待遇": "most-favoured-nation treatment",
    "国内税": "internal taxes",
    "关税同盟": "customs union",
    "自由贸易区": "free-trade area",
    "自由贸易协定": "free trade agreement",
    "透明度": "transparency",
    "非关税措施": "non-tariff measure",
    "数量限制": "quantitative restriction",
    "国营贸易": "state trading",
    "政府采购": "government procurement",
    "服务贸易": "trade in services",
    "电子商务": "electronic commerce",
    # --- dispute settlement --------------------------------------------
    "争端解决": "dispute settlement",
    "争端解决机构": "Dispute Settlement Body",
    "上诉机构": "Appellate Body",
    "专家组": "panel",
    "磋商": "consultations",
    "谅解": "understanding",
    "仲裁": "arbitration",
    "报复": "retaliation",
    "中止减让": "suspension of concessions",
    # --- trade remedy ---------------------------------------------------
    "反倾销": "anti-dumping",
    "反补贴": "countervailing subsidy",
    "保障措施": "safeguard measures",
    "保障措施协定": "Agreement on Safeguards",
    "补贴": "subsidy",
    "倾销": "dumping",
    "损害": "injury",
    "严重损害": "serious injury",
    "产业损害": "industry injury",
    "因果关系": "causal link",
    "正常价值": "normal value",
    "出口价格": "export price",
    "倾销幅度": "dumping margin",
    "临时措施": "provisional measures",
    "价格承诺": "price undertaking",
    "期终复审": "sunset review",
    "反规避": "anti-circumvention",
    "可诉补贴": "actionable subsidy",
    "禁止性补贴": "prohibited subsidy",
    "专向性": "specificity",
    "调查": "investigation",
    # --- agriculture / standards ----------------------------------------
    "农业协定": "Agreement on Agriculture",
    "关税化": "tariffication",
    "最低进口价格": "minimum import price",
    "技术壁垒": "technical barriers to trade",
    "卫生与植物卫生": "sanitary and phytosanitary",
    "动植物卫生检疫": "sanitary and phytosanitary",
    "技术法规": "technical regulations",
    "合格评定": "conformity assessment",
    "国际标准": "international standard",
    "等效性": "equivalence",
    "相互承认": "mutual recognition",
    # --- customs valuation ----------------------------------------------
    "海关估价": "customs valuation",
    "成交价格": "transaction value",
    "完税价格": "customs value",
    "保险费": "insurance",
    "运费": "freight",
    "佣金": "commission",
    "经纪费": "brokerage",
    "容器费用": "cost of containers",
    "包装费用": "packing costs",
    "特许权使用费": "royalties and licence fees",
    "特殊关系": "special relationship",
    "相同货物": "identical goods",
    "类似货物": "similar goods",
    "倒扣价格": "deductive value",
    "计算价格": "computed value",
    "合理方法": "fall-back method",
    # --- origin ----------------------------------------------------------
    "原产地": "rules of origin",
    "实质性改变": "substantial transformation",
    "原产地证书": "certificate of origin",
    "原产地声明": "declaration of origin",
    "经核准出口商": "approved exporter",
    "累积规则": "cumulation",
    "直接运输": "direct transport",
    "增值": "value added",
    "微小加工": "minimal operations",
    # --- classification / tariff ----------------------------------------
    "税则归类": "tariff classification",
    "归类": "classification",
    "关税配额": "tariff-rate quota",
    "进口许可": "import licensing",
    "进口配额": "import quota",
    "商品编码": "HS code",
    "协调制度": "harmonized system",
    "关税壁垒": "tariff barrier",
    "关税税率": "tariff rate",
    "普通税率": "general rate",
    "协定税率": "conventional rate",
    "特惠税率": "preferential rate",
    "最惠国税率": "most-favoured-nation rate",
    "特惠关税": "preferential tariff",
    "反倾销税": "anti-dumping duty",
    "反补贴税": "countervailing duty",
    # --- customs practice ------------------------------------------------
    "报关": "customs declaration",
    "报关单": "customs declaration form",
    "报关行": "customs broker",
    "申报": "declaration",
    "过境": "transit",
    "保税": "bonded",
    "保税仓库": "bonded warehouse",
    "退税": "drawback",
    "出口退税": "export drawback",
    "暂准进口": "temporary admission",
    "滞纳金": "late payment surcharge",
    "滞报金": "late declaration fee",
    "稽查": "post-clearance audit",
    "清关": "customs clearance",
    "放行": "release",
    "查验": "inspection of goods",
    "缉私": "anti-smuggling",
    "罚款": "fine",
    "没收": "forfeiture",
    "行政复议": "administrative reconsideration",
    "异议": "protest",
    "海关监管": "customs supervision",
    "加工贸易": "processing trade",
    "综合保税区": "comprehensive bonded zone",
    "内销": "domestic sale",
    "核销": "verification",
    "转关": "transit customs",
    "减免税": "duty reduction and exemption",
    "保证金": "deposit",
    "担保": "guarantee",
    "海关事务担保": "customs guarantee",
    "信用管理": "credit management",
    "认证企业": "authorized enterprise",
    "失信企业": "discredited enterprise",
    # --- inspection / quarantine ----------------------------------------
    "检验": "inspection",
    "检疫": "quarantine",
    "商品检验": "commodity inspection",
    "质量许可": "quality licence",
    "强制性国家标准": "mandatory national standard",
    "认证认可": "certification and accreditation",
    # --- export control / sanctions -------------------------------------
    "出口管制": "export control",
    "出口许可": "export licence",
    "两用物项": "dual-use items",
    "许可证": "licence",
    "最终用户": "end user",
    "最终用途": "end use",
    "管制编码": "control code",
    "管制清单": "Commerce Control List",
    "许可例外": "License Exception",
    "视同出口": "deemed export",
    "最终用户清单": "Entity List",
    "未经验证清单": "Unverified List",
    "外国直接产品规则": "Foreign Direct Product Rule",
    "安全保障贸易": "security export control",
    "全面管制": "catch-all control",
    "记录保存": "recordkeeping",
    "电子出口信息": "electronic export information",
    "反抵制": "anti-boycott",
    "制裁": "sanctions",
    "禁运": "embargo",
    "冻结资产": "asset freeze",
    # --- contracts / CISG -----------------------------------------------
    "根本违约": "fundamental breach",
    "损害赔偿": "damages",
    "宣告合同无效": "avoidance of the contract",
    "货物销售": "sale of goods",
    "卖方": "seller",
    "买方": "buyer",
    "不可抗力": "force majeure",
    "豁免": "exemption",
    # --- finance / settlement -------------------------------------------
    "信用证": "letter of credit",
    "独立保函": "independent guarantee",
    "外汇": "foreign exchange",
    "收汇": "export proceeds",
    "结汇": "settlement of foreign exchange",
    "跨境支付": "cross-border payment",
    "保险单": "insurance policy",
    "提单": "bill of lading",
    "国际贸易术语": "Incoterms",
    "离岸价": "free on board",
    "到岸价": "cost insurance and freight",
    "完税后交货": "delivered duty paid",
    "转让定价": "transfer pricing",
    # --- general risk wording used by the eval set -----------------------
    "风险评估": "risk assessment",
    "科学依据": "scientific evidence",
    "知识产权边境措施": "border measures for intellectual property",
    "预裁定": "advance ruling",
    "世界海关组织": "World Customs Organization",
}

# Chinese term -> Japanese wording used by e-Gov statutes.
GLOSSARY_JA: dict[str, str] = {
    "关税": "関税",
    "关税法": "関税法",
    "关税定率法": "関税定率法",
    "关税暂定措施法": "関税暫定措置法",
    "关税同盟": "関税同盟",
    "关税配额": "関税割当",
    "关税估价": "関税評価",
    "报关": "通関",
    "通关": "通関",
    "报关行": "通関業者",
    "保税": "保税",
    "保税仓库": "保税蔵置場",
    "出口退税": "戻し税",
    "稽查": "事後調査",
    "原产地": "原産地",
    "原产地证书": "原産地証明書",
    "进口许可": "輸入承認",
    "进口贸易管理令": "輸入貿易管理令",
    "出口贸易管理令": "輸出貿易管理令",
    "外汇": "外国為替",
    "外汇法": "外為法",
    "出口管制": "輸出管理",
    "安全保障贸易": "安全保障貿易管理",
    "特惠关税": "特恵関税",
    "许可例外": "許可例外",
    "管制清单": "規制リスト",
    "最终用户": "最終需要者",
    "两用物项": "安全保障貿易管理",
    "制裁": "制裁",
    "最惠国待遇": "最恵国待遇",
    "国民待遇": "内国民待遇",
    "自由贸易协定": "自由貿易協定",
    "技术法规": "技術基準",
    "检验": "検査",
    "检疫": "検疫",
    "完税价格": "課税価格",
    "成交价格": "取引価格",
    "申报": "申告",
    "反倾销": "不当廉売",
    "反补贴": "補助金相殺",
    "保障措施": "セーフガード",
}

# Longest term first so 特惠关税 is expanded before its substring 关税.
_TERM_ORDER: list[tuple[str, str]] = sorted(
    [*GLOSSARY.items(), *GLOSSARY_JA.items()],
    key=lambda item: (-len(item[0]), item[0]),
)


def expand_query(query: str, limit: int = 16) -> str:
    if not CJK.search(query):
        return query
    additions: list[str] = []
    seen: set[str] = set()
    for term, translation in _TERM_ORDER:
        if term not in query or translation in seen:
            continue
        seen.add(translation)
        additions.append(translation)
        if len(additions) >= limit:
            break
    if not additions:
        return query
    return query + " " + " ".join(additions)
