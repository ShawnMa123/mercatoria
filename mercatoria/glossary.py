"""Small bilingual glossary for cross-language retrieval.

Chinese questions about WTO/CISG must reach English corpus text. A full
translation model adds a network call to every query; for the POC, append the
English equivalents of known legal terms so BM25 and the multilingual embedding
both see them.
"""

from __future__ import annotations

import re

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# Keep entries short and domain-specific. Order matters only for readability.
GLOSSARY: dict[str, str] = {
    "国民待遇": "national treatment",
    "最惠国待遇": "most-favoured-nation treatment",
    "国内税": "internal taxes",
    "关税同盟": "customs union",
    "自由贸易区": "free-trade area",
    "争端解决": "dispute settlement",
    "上诉机构": "Appellate Body",
    "专家组": "panel",
    "反倾销": "anti-dumping",
    "反补贴": "countervailing subsidy",
    "保障措施": "safeguard measures",
    "补贴": "subsidy",
    "倾销": "dumping",
    "损害": "injury",
    "因果关系": "causal link",
    "正常价值": "normal value",
    "出口价格": "export price",
    "原产地": "rules of origin",
    "实质性改变": "substantial transformation",
    "税则归类": "tariff classification",
    "关税配额": "tariff-rate quota",
    "进口许可": "import licensing",
    "技术壁垒": "technical barriers to trade",
    "卫生与植物卫生": "sanitary and phytosanitary",
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
    "根本违约": "fundamental breach",
    "损害赔偿": "damages",
    "宣告合同无效": "avoidance of the contract",
    "货物销售": "sale of goods",
    "卖方": "seller",
    "买方": "buyer",
    "出口管制": "export control",
    "两用物项": "dual-use items",
    "许可证": "licence",
    "最终用户": "end user",
    "最终用途": "end use",
    "管制编码": "control code",
    "预裁定": "advance ruling",
    "过境": "transit",
    "保税": "bonded",
    "退税": "drawback",
    "滞纳金": "late payment surcharge",
    "报关": "customs declaration",
    "申报": "declaration",
    "检验": "inspection",
    "检疫": "quarantine",
}


def expand_query(query: str, limit: int = 16) -> str:
    if not CJK.search(query):
        return query
    additions: list[str] = []
    seen: set[str] = set()
    for term, translation in GLOSSARY.items():
        if term in query and translation not in seen:
            seen.add(translation)
            additions.append(translation)
            if len(additions) >= limit:
                break
    if not additions:
        return query
    return query + " " + " ".join(additions)
