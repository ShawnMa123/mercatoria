"""生成跨境贸易法规测试用例。

设计原则（基于实测踩坑）：
1. **不用「情形N」等无语义后缀凑数** —— 实测会污染检索，
   模型会把"情形2"当作实体去匹配，导致召回不相关内容后拒答
2. 每条用例是真实业务场景的自然表述
3. 期望关键词取自法条中必然出现的术语，而不是宽泛的业务词

输出：
    corpus/eval/test_cases.json
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "corpus" / "eval" / "test_cases.json"

CN, US, WTO, CISG, AU, JP = "cn-laws", "ecfr", "wto", "uncitral", "au", "jp"

CASES: list[dict] = []


def add(
    cat,
    sub,
    zh,
    en,
    jurisdiction,
    keywords,
    expect="answer",
    expected_jurisdictions=None,
):
    case = {
        "id": len(CASES) + 1,
        "category": cat,
        "subcategory": sub,
        "q_zh": zh,
        "q_en": en,
        "jurisdiction": jurisdiction,
        "must_mention": keywords,
        "expect": expect,
    }
    if expected_jurisdictions:
        case["expected_jurisdictions"] = expected_jurisdictions
    CASES.append(case)


# ══ 1. 报关与申报 ══════════════════════════════════════════════
for zh, en in [
    ("一般贸易进口报关需要提交哪些单证？", "What documents are required for a general trade import declaration?"),
    ("进口货物报关单上需要填写哪些项目？", "What items must be filled in on an import declaration form?"),
    ("报关单填错了可以修改吗？需要满足什么条件？", "Can a filed customs declaration be amended, and under what conditions?"),
    ("货物到港后多久内必须向海关申报？逾期会怎样？", "Within how long must goods be declared after arrival, and what happens if late?"),
    ("可以由报关企业代理报关吗？需要什么授权手续？", "May a customs broker declare on my behalf, and what authorization is needed?"),
    ("进口货物商品归类申报错误会有什么后果？", "What are the consequences of misclassifying goods in an import declaration?"),
    ("海关对申报内容有疑问时可以要求补充申报吗？", "May customs require a supplementary declaration when it has doubts?"),
    ("货物实际进出口前可以申请预归类吗？有效期多久？", "Can an advance classification ruling be requested before shipment, and how long is it valid?"),
    ("报关时提供的发票金额与真实成交价格不一致会如何处理？", "How is a discrepancy between the declared invoice value and the true transaction value handled?"),
    ("进口货物分批到港可以分批报关吗？", "May goods arriving in instalments be declared in separate batches?"),
    ("暂时进口货物报关适用什么程序？", "What procedure applies to declaring temporarily imported goods?"),
    ("报关时漏报运费和保险费会被如何认定？", "How is omission of freight and insurance from a declaration treated?"),
    ("知识产权权利人可以申请海关对进口货物采取保护措施吗？", "Can an IP rights holder request customs protection measures for imports?"),
    ("保税仓库货物出库报关的流程是什么？", "What is the declaration procedure for goods leaving a bonded warehouse?"),
    ("过境货物需要向海关申报吗？", "Must transit goods be declared to customs?"),
    ("跨境电商零售进口商品如何报关？", "How are cross-border e-commerce retail imports declared?"),
    ("报关单证需要保存多久？", "How long must customs declaration records be retained?"),
    ("进口货物收货人未按期申报，海关如何处置？", "How does customs handle a consignee who fails to declare on time?"),
    ("海关查验发现申报货物与实际不符，程序上如何处理？", "What is the procedure when inspection reveals goods differ from the declaration?"),
    ("报关时商品编号填错但货物已放行，还能纠正吗？", "Can a misclassified commodity code be corrected after the goods have been released?"),
]:
    add("报关", "申报义务", zh, en, [CN, US], ["申报"])

for zh, en in [
    ("低报进口货物成交价格逃避关税，海关会如何处罚？", "What penalties apply for under-declaring transaction value to evade duty?"),
    ("伪报货物名称进出口属于什么性质的行为？", "What is the nature of falsely declaring the description of imported goods?"),
    ("申报不实但未造成税款损失的，还会被处罚吗？", "Is a false declaration still punishable if no duty loss results?"),
    ("进口货物数量申报少于实际数量，如何处理？", "How is a declaration of fewer goods than actually imported handled?"),
    ("瞒报货物原产地的法律后果是什么？", "What are the legal consequences of concealing the origin of goods?"),
    ("申报价格明显低于同类货物，海关可以启动价格质疑吗？", "May customs raise a valuation doubt where the declared price is clearly below comparable goods?"),
    ("走私行为与违规行为的界限在哪里？", "Where is the line between smuggling and a regulatory violation?"),
    ("一年内多次申报不实会被加重处罚吗？", "Are repeated false declarations within one year punished more severely?"),
    ("申报不实案件由哪个部门管辖？", "Which authority has jurisdiction over false declaration cases?"),
    ("主动披露申报错误可以减免处罚吗？", "Can voluntary disclosure of a declaration error reduce penalties?"),
]:
    add("报关", "申报不实", zh, en, [CN, US], ["申报"])

# ══ 2. 关税与完税价格 ══════════════════════════════════════════
for zh, en in [
    ("海关审定完税价格时，成交价格需要满足哪些条件？", "What conditions must the transaction value satisfy for customs valuation?"),
    ("进口货物的完税价格是否包含运费和保险费？", "Does the customs value of imports include freight and insurance?"),
    ("买卖双方存在特殊关系时，成交价格还能作为完税价格吗？", "Can the transaction value still be used where buyer and seller are related?"),
    ("相同货物成交价格法和类似货物成交价格法如何适用？", "How do the identical goods and similar goods valuation methods apply?"),
    ("海关可以使用计算价格法确定完税价格吗？", "May customs use the computed value method for customs valuation?"),
    ("进口货物在境内转售的收益需要计入完税价格吗？", "Must proceeds from resale of imported goods be included in the customs value?"),
    ("买方承担的模具费用是否应计入完税价格？", "Should tooling costs borne by the buyer be included in the customs value?"),
    ("特许权使用费在什么情况下应计入完税价格？", "When should royalties and licence fees be included in the customs value?"),
    ("完税价格申报不实被海关重新估价的，税款如何补缴？", "How is duty recovered when customs revalues after a false valuation?"),
    ("进口货物的包装费用是否计入完税价格？", "Are packing costs included in the customs value?"),
]:
    add("关税", "完税价格", zh, en, [CN, US, WTO], ["完税价格"])

for zh, en in [
    ("进出口货物应适用哪一天实施的税率？", "Which date's tariff rate applies to imported goods?"),
    ("最惠国税率、协定税率和普通税率分别适用什么情形？", "When do MFN, agreement and general rates apply?"),
    ("暂定税率和配额税率如何优先适用？", "How do provisional and tariff-quota rates take priority?"),
    ("反倾销税和反补贴税与普通关税如何叠加？", "How do anti-dumping and countervailing duties interact with ordinary customs duty?"),
    ("原产于与我国签订自贸协定的国家的货物适用什么税率？", "What rate applies to goods originating in countries with which a free trade agreement exists?"),
    ("进口货物适用税率发生调整的，以何时为准？", "When a tariff rate changes, which point in time governs?"),
    ("出口关税的征收范围包括哪些商品？", "Which goods are subject to export duty?"),
    ("进口环节增值税和消费税由谁征收？", "Who collects import VAT and consumption tax?"),
    ("船舶吨税与关税是什么关系？", "What is the relationship between tonnage dues and customs duty?"),
    ("加工贸易进口料件内销时如何补征税款？", "How is duty recovered when processing trade materials are sold domestically?"),
]:
    add("关税", "税率适用", zh, en, [CN, US, WTO], ["税率"])

for zh, en in [
    ("多缴的关税可以申请退还吗？期限是多长？", "Can overpaid customs duty be refunded, and within what period?"),
    ("少征或漏征的关税，海关追征的期限是多久？", "Within what period may customs recover underpaid duty?"),
    ("因纳税义务人违反规定造成少征税款的，追征期有何不同？", "How does the recovery period differ where the shortfall results from the declarant's violation?"),
    ("海关发现多征税款应当如何处理？", "How should customs handle over-collected duty it discovers?"),
    ("申请退还关税需要提交什么材料？", "What materials are needed to apply for a duty refund?"),
    ("退运货物的已征税款可以退还吗？", "Can duty already paid be refunded for returned goods?"),
    ("进口货物因品质问题索赔后，税款可以调整吗？", "Can duty be adjusted after a quality claim on imported goods?"),
    ("海关补征税款是否需要加收滞纳金？", "Is late payment interest charged on duty recovered by customs?"),
    ("关税缓缴需要什么条件？", "What conditions apply to deferring duty payment?"),
    ("纳税义务人对海关征税决定不服可以申请复议吗？", "May a duty payer seek review of a customs duty decision?"),
]:
    add("关税", "退还补征", zh, en, [CN, US], ["退还", "补征"])

# ══ 3. 原产地规则 ══════════════════════════════════════════════
for zh, en in [
    ("什么叫实质性改变？如何判定？", "What is substantial transformation and how is it determined?"),
    ("税则归类改变标准在原产地认定中如何适用？", "How does the change-in-tariff-classification criterion apply to origin determination?"),
    ("区域价值成分标准如何计算？", "How is the regional value content criterion calculated?"),
    ("微小加工或处理是否赋予原产地资格？", "Do minimal operations or processing confer origin?"),
    ("简单组装能否使产品获得原产地资格？", "Can simple assembly confer origin on a product?"),
    ("累积规则在原产地认定中起什么作用？", "What role does the cumulation rule play in origin determination?"),
    ("直接运输规则对原产地资格有什么影响？", "What effect does the direct transport rule have on origin qualification?"),
    ("经过第三国转运的货物还能享受协定税率吗？", "Can goods transshipped through a third country still enjoy agreement rates?"),
    ("完全获得产品的原产地如何认定？", "How is the origin of wholly obtained products determined?"),
    ("进口成分超过一定比例是否丧失原产地资格？", "Does exceeding a threshold of imported content disqualify origin?"),
]:
    add("原产地", "实质性改变", zh, en, [CN, US, WTO], ["原产地"])

for zh, en in [
    ("申请原产地证书需要提交什么材料？", "What materials are needed to apply for a certificate of origin?"),
    ("海关如何核查原产地证书的真实性？", "How does customs verify the authenticity of a certificate of origin?"),
    ("原产地证书丢失或损毁可以补发吗？", "Can a lost or damaged certificate of origin be reissued?"),
    ("原产地证书的有效期是多久？", "How long is a certificate of origin valid?"),
    ("原产地声明与证书的法律效力有何区别？", "What is the legal difference between an origin declaration and a certificate?"),
    ("出口货物原产地证书由哪个机构签发？", "Which body issues certificates of origin for exports?"),
    ("海关对原产地证书有疑问时可以请求境外核查吗？", "May customs request foreign verification when it doubts a certificate of origin?"),
    ("优惠原产地证书与一般原产地证书有何区别？", "What distinguishes a preferential certificate of origin from an ordinary one?"),
    ("原产地证书上的信息与实际货物不符会怎么处理？", "How is a mismatch between the certificate and the actual goods handled?"),
    ("企业可以自行出具原产地声明吗？", "May an enterprise issue its own origin declaration?"),
]:
    add("原产地", "原产地证书", zh, en, [CN, US], ["原产地"])

# ══ 4. 出口管制 ═══════════════════════════════════════════════
for zh, en in [
    ("出口两用物项需要办理什么许可手续？", "What licence is required to export dual-use items?"),
    ("未经许可出口管制物项会承担什么法律责任？", "What legal liability arises from exporting controlled items without a licence?"),
    ("出口管制清单由哪个部门制定和调整？", "Which authority sets and adjusts the export control list?"),
    ("如何判断某项货物是否属于管制范围？", "How do I determine whether goods fall within the control scope?"),
    ("出口经营者对最终用户负有哪些核实义务？", "What end-user verification obligations does an exporter have?"),
    ("最终用户证明需要包含哪些内容？", "What must an end-user certificate contain?"),
    ("管制物项的出口许可申请流程是什么？", "What is the licence application process for controlled items?"),
    ("出口管制是否适用于技术出口？", "Does export control apply to technology exports?"),
    ("明知用于大规模杀伤性武器仍出口的后果是什么？", "What are the consequences of exporting while knowing the items are destined for WMD?"),
    ("转口、再出口管制物项需要许可吗？", "Is a licence required for transit or re-export of controlled items?"),
    ("出口管制是否适用于跨国公司内部调拨？", "Does export control apply to intra-group transfers within a multinational?"),
    ("违反出口管制的企业会被列入黑名单吗？", "Can enterprises violating export controls be placed on a watchlist?"),
    ("海关在出口环节如何查验管制物项？", "How does customs inspect controlled items at the export stage?"),
    ("未经许可出口可能用于军事目的的民用产品如何处理？", "How is unlicensed export of civil products potentially usable for military purposes handled?"),
    ("出口管制的域外适用如何理解？", "How should extraterritorial application of export controls be understood?"),
]:
    add("出口管制", "两用物项", zh, en, [CN, US, JP], ["出口管制", "许可"])

for zh, en in [
    ("如《出口管制法》对管制物项规定了哪些许可类型？", "What licence types does export control law provide for controlled items?"),
    ("出口许可证的有效期和使用次数如何规定？", "How long is an export licence valid and how many times may it be used?"),
    ("许可申请被驳回后可以申请重新审查吗？", "May an applicant seek reconsideration after a licence denial?"),
    ("列入管控名单的企业会受哪些限制？", "What restrictions apply to enterprises on a control list?"),
    ("临时管制措施如何适用？", "How do interim control measures apply?"),
    ("出口管制清单与两用物项清单是什么关系？", "What is the relationship between the export control list and the dual-use list?"),
    ("海关对出口管制物项的归类有疑问如何处理？", "How are classification doubts on controlled export items resolved?"),
    ("出口经营者如何建立内部合规制度？", "How should exporters establish internal compliance programmes?"),
]:
    add("出口管制", "管制清单", zh, en, [CN, US], ["管制"])

# ══ 5. 检验检疫 ═══════════════════════════════════════════════
for zh, en in [
    ("进口商品检验的流程是什么？", "What is the inspection process for imported goods?"),
    ("检验不合格的进口货物如何处理？", "How are imported goods that fail inspection handled?"),
    ("进口商品必须由指定机构检验吗？", "Must imported goods be inspected by a designated body?"),
    ("进口旧机电产品的检验要求有哪些？", "What inspection requirements apply to used mechanical and electrical products?"),
    ("进口食品需要办理哪些检验手续？", "What inspection procedures apply to imported food?"),
    ("进口化妆品需要进行哪些检验？", "What inspections are required for imported cosmetics?"),
    ("进口汽车如何办理检验手续？", "How are imported vehicles inspected?"),
    ("进口棉花检验的特殊要求是什么？", "What special requirements apply to cotton inspection?"),
    ("进口计量器具如何检验？", "How are imported measuring instruments inspected?"),
    ("进口可用作原料的固体废物有什么检验要求？", "What inspection requirements apply to imported solid waste usable as raw material?"),
]:
    add("检验检疫", "进口检验", zh, en, [CN, US], ["检验"])

for zh, en in [
    ("进境动植物及其产品需要办理哪些检疫审批手续？", "What quarantine approval is required for importing animals, plants and their products?"),
    ("进境水果的检疫要求是什么？", "What are the quarantine requirements for imported fruit?"),
    ("进境种子的检疫审批流程是什么？", "What is the quarantine approval process for imported seeds?"),
    ("进境动物隔离检疫场的设立条件是什么？", "What are the conditions for establishing a quarantine station for imported animals?"),
    ("进境木质包装需要什么检疫处理？", "What quarantine treatment is required for imported wood packaging?"),
    ("携带宠物入境需要哪些检疫手续？", "What quarantine formalities apply to bringing pets into the country?"),
    ("进境水生动物的检疫监督如何进行？", "How is quarantine supervision of imported aquatic animals carried out?"),
    ("检疫处理不合格的进境货物如何处理？", "How are imported goods failing quarantine treatment handled?"),
    ("进境动植物检疫审批的有效期是多久？", "How long is quarantine approval for imported animals and plants valid?"),
    ("国境卫生检疫对入境交通工具的查验要求是什么？", "What inspection requirements apply to inbound transport under frontier health quarantine?"),
]:
    add("检验检疫", "动植物检疫", zh, en, [CN, AU], ["检疫"])

# ══ 6. 保税与特殊监管区域 ═════════════════════════════════════
for zh, en in [
    ("保税货物的监管要求是什么？", "What are the supervision requirements for bonded goods?"),
    ("擅自处置保税货物会有什么法律后果？", "What are the legal consequences of disposing of bonded goods without authorization?"),
    ("保税仓库的设立条件有哪些？", "What are the conditions for establishing a bonded warehouse?"),
    ("保税货物内销时如何补税？", "How is duty recovered when bonded goods are sold domestically?"),
    ("保税货物加工后出口需要办理什么手续？", "What procedures apply to exporting bonded goods after processing?"),
    ("保税货物可以抵押或质押吗？", "May bonded goods be mortgaged or pledged?"),
    ("保税仓库货物盘点差异如何申报处理？", "How are inventory discrepancies at a bonded warehouse declared and handled?"),
    ("保税物流中心可以开展哪些业务？", "What business may a bonded logistics centre conduct?"),
    ("加工贸易保税料件的单耗如何核定？", "How is the unit consumption of bonded materials under processing trade determined?"),
    ("保税货物的存储期限是多久？", "How long may bonded goods be stored?"),
]:
    add("保税", "保税货物", zh, en, [CN, US], ["保税"])

for zh, en in [
    ("综合保税区内的企业可以开展哪些业务？", "What business may enterprises in a comprehensive bonded zone conduct?"),
    ("货物从保税区进入境内区外如何征税？", "How are goods taxed when moving from a bonded zone into the domestic market?"),
    ("保税区内企业间的货物转让需要报关吗？", "Must goods transferred between enterprises within a bonded zone be declared?"),
    ("自由贸易试验区与综合保税区有何不同？", "How do pilot free trade zones differ from comprehensive bonded zones?"),
    ("保税区内货物可以自由流转吗？", "May goods circulate freely within a bonded zone?"),
    ("区内企业进口自用设备可以免税吗？", "May enterprises in a bonded zone import capital equipment free of duty?"),
    ("保税区货物退运境外如何办理？", "How are bonded zone goods returned overseas handled?"),
    ("海关特殊监管区域之间的货物结转如何办理？", "How are goods transferred between special customs supervision areas?"),
]:
    add("保税", "特殊监管区", zh, en, [CN], ["保税"])

# ══ 7. 反倾销反补贴 ═══════════════════════════════════════════
for zh, en in [
    ("反倾销调查的申请主体和立案条件是什么？", "Who may apply for an anti-dumping investigation and what are the initiation conditions?"),
    ("倾销幅度如何确定？", "How is the dumping margin determined?"),
    ("国内产业损害如何认定？", "How is injury to the domestic industry determined?"),
    ("倾销与损害之间的因果关系如何证明？", "How is causation between dumping and injury established?"),
    ("反倾销临时措施最长可以实施多久？", "How long may provisional anti-dumping measures last?"),
    ("反倾销税如何征收？由谁征收？", "How and by whom are anti-dumping duties collected?"),
    ("价格承诺在反倾销程序中的作用是什么？", "What role do price undertakings play in anti-dumping proceedings?"),
    ("反倾销调查期限是多长？", "How long does an anti-dumping investigation take?"),
    ("反倾销复审的启动条件是什么？", "What triggers an anti-dumping review?"),
    ("规避反倾销措施的行为如何认定？", "How is circumvention of anti-dumping measures determined?"),
    ("反倾销措施针对的倾销产品如何界定？", "How is the dumped product defined for anti-dumping measures?"),
]:
    add("贸易救济", "反倾销", zh, en, [CN, WTO], ["反倾销"])

for zh, en in [
    ("反补贴措施针对哪些补贴行为？", "Which subsidy practices are subject to countervailing measures?"),
    ("专向性补贴如何认定？", "How is a specific subsidy identified?"),
    ("补贴金额如何计算？", "How is the amount of subsidy calculated?"),
    ("反补贴临时措施的实施条件是什么？", "What conditions apply to provisional countervailing measures?"),
    ("反补贴税与反倾销税可以同时征收吗？", "May countervailing and anti-dumping duties be imposed simultaneously?"),
    ("禁止性补贴包括哪些类型？", "What types of subsidy are prohibited?"),
    ("可诉补贴与不可诉补贴如何区分？", "How are actionable and non-actionable subsidies distinguished?"),
    ("反补贴调查中的磋商程序如何进行？", "How do consultations proceed in a countervailing investigation?"),
    ("发展中国家在反补贴措施上享有特殊待遇吗？", "Do developing countries receive special treatment in countervailing measures?"),
]:
    add("贸易救济", "反补贴", zh, en, [CN, WTO], ["补贴"])

# ══ 8. 外汇与担保 ═════════════════════════════════════════════
for zh, en in [
    ("货物贸易跨境收付汇需要办理哪些手续？", "What procedures apply to cross-border payment for trade in goods?"),
    ("外汇管理局如何监管货物贸易外汇收支？", "How does the foreign exchange administration supervise trade payments?"),
    ("进口付汇需要提交哪些单据？", "What documents are required for import payment?"),
    ("出口收汇期限有何规定？", "Are there rules on the deadline for receiving export proceeds?"),
    ("企业外汇分类管理等级如何划分？", "How are enterprises classified under the foreign exchange classification system?"),
    ("跨境人民币结算与外币结算有何不同？", "How does cross-border RMB settlement differ from foreign currency settlement?"),
    ("服务贸易项下付汇需要什么材料？", "What materials are needed for payments under trade in services?"),
    ("资本项目外汇收支与经常项目有何区别？", "How do capital account transactions differ from current account ones?"),
    ("个人年度购汇额度是多少？", "What is the annual foreign exchange purchase quota for individuals?"),
    ("违反外汇管理规定会受到什么处罚？", "What penalties apply to violations of foreign exchange rules?"),
]:
    add("外汇", "跨境收付", zh, en, [CN], ["外汇"])

for zh, en in [
    ("海关事务担保的方式有哪些？", "What forms of customs security are accepted?"),
    ("担保金额如何确定？", "How is the amount of security determined?"),
    ("哪些情形下可以申请担保放行？", "In what circumstances may goods be released against security?"),
    ("保证金和保函有何区别？", "What is the difference between a cash deposit and a guarantee letter?"),
    ("担保期限届满后如何处理？", "What happens when the security period expires?"),
    ("担保人可以变更吗？", "May the guarantor be changed?"),
    ("税收保全措施的适用条件是什么？", "What conditions apply to tax preservation measures?"),
    ("担保物权实现的程序是什么？", "What is the procedure for enforcing security?"),
    ("关税保证保险可以作为担保方式吗？", "May duty guarantee insurance serve as security?"),
    ("担保放行后未按期补办手续的后果是什么？", "What are the consequences of failing to complete formalities after release against security?"),
]:
    add("担保", "海关事务担保", zh, en, [CN, US], ["担保"])

# ══ 9. CISG 国际货物买卖 ══════════════════════════════════════
for zh, en in [
    ("根据 CISG，国际货物销售合同何时成立？", "Under the CISG, when is a contract of sale concluded?"),
    ("CISG 下要约与承诺的规则是什么？", "What rules govern offer and acceptance under the CISG?"),
    ("逾期承诺是否有效？", "Is a late acceptance effective?"),
    ("对要约作出实质性变更的回复构成反要约吗？", "Does a reply materially altering an offer constitute a counter-offer?"),
    ("CISG 是否适用于营业地在不同国家的当事人之间的合同？", "Does the CISG apply to contracts between parties with places of business in different States?"),
    ("当事人可以排除 CISG 的适用吗？", "May the parties exclude the application of the CISG?"),
    ("CISG 不适用的销售有哪些？", "Which sales are excluded from the CISG?"),
    ("CISG 是否适用于服务合同？", "Does the CISG apply to service contracts?"),
    ("CISG 对合同形式有要求吗？", "Does the CISG impose formal requirements on contracts?"),
    ("CISG 下货物风险何时转移？", "When does risk pass under the CISG?"),
]:
    add("国际货物买卖", "合同成立", zh, en, [CISG], ["CISG"])

for zh, en in [
    ("根据 CISG，货物不符合同时买方有哪些救济？", "Under the CISG, what remedies does the buyer have for non-conforming goods?"),
    ("买方检验货物和通知不符的义务是什么？", "What are the buyer's obligations to examine goods and give notice of non-conformity?"),
    ("根本违约与一般违约的区别在哪？", "What distinguishes a fundamental breach from an ordinary breach?"),
    ("CISG 下卖方可以补救交货不符吗？", "May the seller cure a non-conforming delivery under the CISG?"),
    ("买方要求实际履行的条件是什么？", "What conditions apply to the buyer's right to require performance?"),
    ("CISG 下的损害赔偿如何计算？", "How are damages calculated under the CISG?"),
    ("减价权在 CISG 中如何行使？", "How is the right to reduce the price exercised under the CISG?"),
    ("货物风险转移后出现损坏由谁承担？", "Who bears loss after risk has passed?"),
    ("CISG 下的免责事由包括哪些？", "What exemptions from liability exist under the CISG?"),
    ("买方解除合同的条件是什么？", "What conditions allow the buyer to avoid the contract?"),
    ("CISG 下的预期违约如何救济？", "What remedies exist for anticipatory breach under the CISG?"),
    ("卖方交付替代货物的条件是什么？", "What conditions apply to delivery of substitute goods?"),
]:
    add("国际货物买卖", "货物相符", zh, en, [CISG], ["CISG"])

# ══ 10. WTO 规则 ══════════════════════════════════════════════
for zh, en in [
    ("GATT 1994 的国民待遇原则对进口产品的国内税有什么要求？", "What does GATT 1994 national treatment require for internal taxes on imports?"),
    ("最惠国待遇原则如何适用于关税措施？", "How does MFN treatment apply to tariff measures?"),
    ("关税约束义务对成员国的税率调整有何限制？", "How does tariff binding constrain members' rate changes?"),
    ("数量限制在 GATT 下一般是否允许？有哪些例外？", "Are quantitative restrictions generally permitted under GATT, and what exceptions exist?"),
    ("一般例外条款包括哪些情形？", "What situations fall under the general exceptions clause?"),
    ("安全例外条款的适用范围是什么？", "What is the scope of the security exceptions clause?"),
    ("技术法规与标准的区别是什么？", "What is the difference between technical regulations and standards?"),
    ("TBT 协定对成员国技术法规的制定有何要求？", "What does the TBT Agreement require of members' technical regulations?"),
    ("SPS 协定下卫生检疫措施的合法性如何判断？", "How is the legitimacy of sanitary measures assessed under the SPS Agreement?"),
    ("透明度原则要求成员国公布哪些贸易措施？", "What trade measures must members publish under the transparency principle?"),
]:
    add("WTO", "基本原则", zh, en, [WTO], ["GATT"])

for zh, en in [
    ("《贸易便利化协定》对成员国通关程序有哪些要求？", "What does the Trade Facilitation Agreement require of members' clearance procedures?"),
    ("预裁定制度在贸易便利化协定中如何规定？", "How does the TFA provide for advance rulings?"),
    ("单一窗口建设属于贸易便利化的哪项义务？", "Under which TFA obligation does single window fall?"),
    ("海关合作与信息交换的规定是什么？", "What are the provisions on customs cooperation and information exchange?"),
    ("过境自由原则对成员国的要求是什么？", "What does the freedom of transit principle require of members?"),
    ("贸易便利化协定的特殊与差别待遇如何体现？", "How does special and differential treatment appear in the TFA?"),
    ("海关估价协定规定的估价方法有哪几种？", "What valuation methods does the Customs Valuation Agreement provide?"),
    ("进口许可程序协定对许可程序有何约束？", "How does the Import Licensing Agreement constrain licensing procedures?"),
    ("原产地规则协定对成员国的要求是什么？", "What does the Rules of Origin Agreement require of members?"),
    ("反倾销协定对调查程序有何要求？", "What does the Anti-Dumping Agreement require of investigation procedures?"),
]:
    add("WTO", "具体协定", zh, en, [WTO], ["Article"])

# ══ 11. 日本法规 ══════════════════════════════════════════════
for zh, en in [
    ("日本の関税法における輸入許可の手続きはどのようなものですか。",
     "What is the import permit procedure under Japan's Customs Act?"),
    ("関税法上の課税価格はどのように決定されますか。",
     "How is the dutiable value determined under Japan's Customs Act?"),
    ("関税の納期限はいつですか。",
     "When is the deadline for paying customs duty in Japan?"),
    ("関税の更正と賦課決定の違いは何ですか。",
     "What is the difference between correction and assessment of duty in Japan?"),
    ("保税地域にはどのような種類がありますか。",
     "What types of bonded area exist under Japanese customs law?"),
    ("関税の減免制度にはどのようなものがありますか。",
     "What duty reduction and exemption regimes exist in Japan?"),
    ("輸入貨物の検査はどのように行われますか。",
     "How is the inspection of imported goods conducted in Japan?"),
    ("関税定率法における原産地規則はどのように定められていますか。",
     "How are rules of origin provided under Japan's Customs Tariff Act?"),
    ("不服申立ての手続きはどうなっていますか。",
     "What is the procedure for filing an objection in Japan?"),
    ("密輸に対する罰則はどのようになっていますか。",
     "What penalties apply to smuggling under Japanese law?"),
]:
    add("日本", "关税", zh, en, [JP], ["関税"])

for zh, en in [
    ("外国為替及び外国貿易法に基づく輸出規制の仕組みは？",
     "What is the export control framework under Japan's Foreign Exchange and Foreign Trade Act?"),
    ("輸出貿易管理令に基づく許可が必要な貨物は？",
     "Which goods require a licence under Japan's Export Trade Control Order?"),
    ("包括許可制度とはどのようなものですか。",
     "What is Japan's bulk licence system?"),
    ("最終需要者確認の義務はどのように定められていますか。",
     "How are end-user confirmation obligations provided in Japan?"),
    ("輸出管理の対象となる技術は？",
     "Which technologies are subject to Japanese export control?"),
    ("キャッチオール規制とは何ですか。",
     "What is Japan's catch-all control?"),
    ("対内直接投資の審査手続きは？",
     "What is the review procedure for inward direct investment in Japan?"),
    ("外為法違反の罰則は？",
     "What penalties apply for violating Japan's Foreign Exchange Act?"),
]:
    add("日本", "外為法", zh, en, [JP], ["輸出"])

# ══ 12. 澳洲法规 ══════════════════════════════════════════════
for en in [
    "Under the Australian Customs Act 1901, what are the requirements for entering goods for home consumption?",
    "What is an import declaration under the Australian Customs Act 1901?",
    "How is the customs value of imported goods determined in Australia?",
    "What licences or permissions are required to move goods into a warehouse in Australia?",
    "What are the powers of customs officers to examine goods in Australia?",
    "How does Australia's Customs Act deal with goods that are prohibited imports?",
    "What duty concessions are available under Australian customs law?",
    "What is the procedure for appealing a customs decision in Australia?",
    "How are transit goods treated under the Australian Customs Act?",
    "What are the record-keeping obligations for importers in Australia?",
]:
    add("澳洲", "海关", en, en, [AU], ["Customs"])

for en in [
    "Under the Australian Export Control Act 2020, what permits are required to export prescribed goods?",
    "What are the export control rules for agricultural products in Australia?",
    "Who issues export permits under Australian export control law?",
    "What are the fit and proper person requirements for export licence holders in Australia?",
    "What are the inspection and audit powers under Australia's export control legislation?",
    "What sanctions apply for unlicensed exports under Australian law?",
    "How does Australia regulate the export of live animals?",
    "What are the requirements for exporting food under Australian export control law?",
]:
    add("澳洲", "出口管制", en, en, [AU], ["export"])

# ══ 13. 跨法域比较 ════════════════════════════════════════════
for cat, zh, en, expected in [
    ("报关", "中国海关法和美国 CFR 对报关代理人资格的要求有什么不同？",
     "How do Chinese customs law and the US CFR differ on customs broker licensing?", ["CN", "US"]),
    ("关税", "中国、美国与 WTO 对完税价格中运费保险费的处理规则有何差异？",
     "How do China, the US and the WTO differ on treatment of freight and insurance in customs valuation?",
     ["CN", "US", "WTO"]),
    ("原产地", "中国原产地条例与 WTO 原产地规则协定在实质性改变标准上有何差异？",
     "How do China's origin regulations and the WTO ROO Agreement differ on substantial transformation?",
     ["CN", "WTO"]),
    ("出口管制", "中国出口管制法和美国 EAR 在许可要求上的主要差异是什么？",
     "What are the main differences between China's Export Control Law and the US EAR licensing regime?",
     ["CN", "US"]),
    ("检验检疫", "中国和澳洲对进境动植物产品的检疫审批要求有何不同？",
     "How do China and Australia differ on quarantine approval for imported animal and plant products?",
     ["CN", "AU"]),
    ("贸易救济", "中国反倾销条例与 WTO 反倾销协定在倾销认定上的差异是什么？",
     "How does China's anti-dumping regulation differ from the WTO Anti-Dumping Agreement on dumping determination?",
     ["CN", "WTO"]),
    ("国际货物买卖", "CISG 与国内法在货物瑕疵担保责任上有何区别？",
     "How does the CISG differ from domestic law on liability for defective goods?", ["INT", "CN"]),
    ("关税", "日本的課税価格認定与中国的完税价格审定规则有何不同？",
     "How does Japan's dutiable value determination differ from China's customs valuation rules?",
     ["JP", "CN"]),
]:
    add("跨法域比较", cat, zh, en, [CN, US, WTO, CISG], ["差异"],
        expected_jurisdictions=expected)

# ══ 14. 边界/拒答（超出语料范围）══════════════════════════════
for zh, en in [
    ("离婚时夫妻共同财产如何分割？", "How is marital property divided in a divorce?"),
    ("故意杀人罪的量刑标准是什么？", "What is the sentencing standard for intentional homicide?"),
    ("有限责任公司章程的必备条款有哪些？", "What are the mandatory clauses of a limited liability company's articles?"),
    ("劳动合同解除需要提前多久通知？", "How much advance notice is required to terminate an employment contract?"),
    ("发明专利申请的实质审查流程是什么？", "What is the substantive examination process for an invention patent?"),
    ("房屋租赁合同纠纷应当向哪个法院起诉？", "Which court has jurisdiction over a residential lease dispute?"),
    ("遗产继承的法定顺序是什么？", "What is the statutory order of inheritance?"),
    ("交通事故责任如何认定？", "How is liability determined in a traffic accident?"),
]:
    add("边界拒答", "超出语料范围", zh, en, [], [], expect="refuse")

# ══ 15. 综合场景 ══════════════════════════════════════════════
for zh, en in [
    ("我们从越南进口一批纺织品，经香港转口，最终运到上海。整个链条涉及哪些海关手续和原产地认定问题？",
     "We import textiles from Vietnam, transshipped through Hong Kong to Shanghai. What customs procedures and origin issues arise?"),
    ("一批货物在海运途中受损，进口时已投保。报关时完税价格怎么申报？损失部分能否减征关税？",
     "Goods were damaged in transit and insured. How should the value be declared, and can duty be reduced for the damaged portion?"),
    ("我们打算以加工贸易方式进口原材料，成品全部出口。需要办什么手续？如果部分内销又该怎么补税？",
     "We plan to import raw materials under processing trade with all output exported. What procedures apply, and how is duty paid on domestic sales?"),
    ("海外客户要求我们出具原产地声明而不是原产地证书。这两种方式在法律效力和操作上有什么区别？",
     "An overseas customer asks for an origin declaration rather than a certificate. How do they differ legally and operationally?"),
    ("进口货物被海关扣留查验，产生了仓储费用。这笔费用由谁承担？扣留期限有规定吗？",
     "Imported goods were detained for inspection, incurring storage costs. Who bears these costs, and are there limits on detention?"),
    ("我们出口的一批设备被境外客户转卖给第三国，这属于再出口吗？需要履行什么义务？",
     "Equipment we exported was resold by the overseas customer to a third country. Is this a re-export, and what obligations apply?"),
    ("进口货物的成交价格因买方与卖方是关联企业而被海关质疑，我们应如何应对？",
     "Customs has questioned our transaction value because buyer and seller are related. How should we respond?"),
    ("跨境电商保税进口模式下，商品从保税仓发货给消费者，涉及哪些海关和税收环节？",
     "Under the cross-border e-commerce bonded import model, what customs and tax steps apply when goods ship from a bonded warehouse to consumers?"),
    ("同一批货物同时涉及反倾销税和进口环节增值税，税款计算顺序是什么？",
     "A consignment is subject to both anti-dumping duty and import VAT. In what order are the amounts calculated?"),
    ("我们想申请海关高级认证企业资格，需要满足哪些条件？认证后能享受什么便利？",
     "We want to apply for Authorised Economic Operator status. What conditions apply, and what benefits does it bring?"),
]:
    add("综合场景", "业务流程", zh, en, [CN, US], ["海关"])

# ══ 16. 出口报关与退税 ════════════════════════════════════════
for zh, en in [
    ("出口货物报关需要提交哪些单证？", "What documents are required for an export declaration?"),
    ("出口货物的申报期限是如何规定的？", "What is the deadline for declaring export goods?"),
    ("出口退税的申请条件是什么？", "What conditions apply to applying for export tax refund?"),
    ("出口货物退关如何处理？", "How is an export cancellation handled?"),
    ("出口加工区货物出口的报关流程是什么？", "What is the declaration procedure for goods exported from an export processing zone?"),
    ("出口货物的报关单与进口报关单有何不同？", "How does an export declaration differ from an import declaration?"),
    ("出口货物退税申报需要哪些凭证？", "What vouchers are needed for an export tax refund claim?"),
    ("出口收汇核销制度现在如何实施？", "How is the export proceeds verification system implemented today?"),
    ("出口货物的原产地证在报关时起什么作用？", "What role does the certificate of origin play in export declaration?"),
    ("出口货物被境外退运，已退税的如何处理？", "How is a refund already granted handled when exported goods are returned?"),
]:
    add("出口", "报关退税", zh, en, [CN], ["出口"])

# ══ 17. 海关稽查与信用管理 ════════════════════════════════════
for zh, en in [
    ("海关稽查的启动条件和程序是什么？", "What triggers a customs audit and what is the procedure?"),
    ("海关稽查的期限范围是多久？", "What period may a customs audit cover?"),
    ("企业在海关稽查中负有哪些配合义务？", "What cooperation obligations do enterprises have during a customs audit?"),
    ("海关可以查阅企业账簿吗？", "May customs inspect enterprise accounting records?"),
    ("高级认证企业的认证标准是什么？", "What are the criteria for Authorised Enterprise accreditation?"),
    ("失信企业会被采取哪些惩戒措施？", "What punitive measures apply to discredited enterprises?"),
    ("企业信用等级如何修复？", "How can an enterprise's credit rating be restored?"),
    ("AEO 互认对企业有什么实际好处？", "What practical benefits does AEO mutual recognition bring?"),
    ("海关对企业分类管理的依据是什么？", "On what basis does customs classify enterprises?"),
    ("企业主动披露违规可以减轻处罚吗？", "Can voluntary disclosure reduce penalties for an enterprise?"),
]:
    add("稽查信用", "海关稽查", zh, en, [CN], ["海关"])

# ══ 18. 税收征管细节 ══════════════════════════════════════════
for zh, en in [
    ("关税的纳税义务人是谁？", "Who is liable to pay customs duty?"),
    ("关税的纳税期限是多少天？", "How many days are allowed for duty payment?"),
    ("逾期缴纳税款会加收多少滞纳金？", "What late payment surcharge applies to overdue duty?"),
    ("纳税义务人可以提供担保延期缴税吗？", "May a duty payer provide security to defer payment?"),
    ("海关可以采取哪些税收保全措施？", "What tax preservation measures may customs take?"),
    ("强制执行措施包括哪些？", "What enforcement measures are available?"),
    ("纳税义务人对税款有异议怎么办？", "What can a duty payer do if it disputes the duty assessed?"),
    ("关税的纳税地点在哪里？", "Where is duty paid?"),
    ("进口环节海关代征税包括哪些？", "What taxes does customs collect on behalf of others at import?"),
    ("减免税货物在监管期内转让需要补税吗？", "Is duty payable when duty-exempt goods are transferred within the supervision period?"),
]:
    add("关税", "税收征管", zh, en, [CN], ["纳"])

# ══ 19. 商品归类 ══════════════════════════════════════════════
for zh, en in [
    ("商品归类的基本原则是什么？", "What are the basic principles of tariff classification?"),
    ("税则号列如何确定？", "How is the tariff heading determined?"),
    ("归类总规则包括哪几条？", "What are the General Rules for the Interpretation of the HS?"),
    ("成套货物如何归类？", "How are goods put up in sets classified?"),
    ("功能机组如何归类？", "How are functional units classified?"),
    ("未完成品或拆散件如何归类？", "How are incomplete or unassembled goods classified?"),
    ("对归类有争议可以申请预裁定吗？", "May an advance ruling be requested for a classification dispute?"),
    ("归类错误的补税期限如何确定？", "How is the recovery period determined for a classification error?"),
    ("海关化验结果与申报归类不符如何处理？", "How is a mismatch between laboratory results and the declared classification handled?"),
    ("商标或品牌对商品归类有影响吗？", "Does brand or trademark affect classification?"),
]:
    add("商品归类", "税则归类", zh, en, [CN, US], ["归类"])

# ══ 20. 贸易便利化实践 ════════════════════════════════════════
for zh, en in [
    ("提前申报模式适用于哪些情形？", "To which situations does advance declaration apply?"),
    ("两步申报的具体流程是什么？", "What is the two-step declaration process?"),
    ("汇总征税的适用条件是什么？", "What conditions apply to consolidated duty payment?"),
    ("关税保证保险与保证金有何区别？", "How does duty guarantee insurance differ from a cash deposit?"),
    ("单一窗口可以办理哪些业务？", "What services can be handled through the single window?"),
    ("原产地证书自助打印需要什么条件？", "What conditions are needed for self-printing certificates of origin?"),
    ("海关预裁定申请的有效期和适用范围是什么？", "What are the validity and scope of a customs advance ruling?"),
    ("经核准出口商制度是什么？", "What is the approved exporter system?"),
    ("集中申报与逐票申报的区别是什么？", "How does consolidated declaration differ from per-shipment declaration?"),
    ("免除查验没有问题外贸企业吊装移位仓储费用政策是什么？", "What is the policy on waiving handling and storage fees for compliant enterprises?"),
]:
    add("贸易便利化", "通关模式", zh, en, [CN], ["申报"])

# ══ 21. 英文场景补充（面向 US/AU/WTO）═════════════════════════
for cat, en, juris in [
    ("US报关", "What is the process for filing an entry summary with US Customs?", [US]),
    ("US报关", "What is the difference between a formal and an informal entry in the United States?", [US]),
    ("US报关", "What is Customs Bonded Warehouse procedure in the US?", [US]),
    ("US报关", "What are the recordkeeping requirements under 19 CFR Part 163?", [US]),
    ("US关税", "How does US Customs determine the transaction value of imported merchandise?", [US]),
    ("US关税", "What is the Merchandise Processing Fee in US customs?", [US]),
    ("US关税", "How does the US apply the Harmonized Tariff Schedule?", [US]),
    ("US关税", "What is the protest procedure for US customs decisions?", [US]),
    ("US原产地", "What are the US rules of origin for preferential tariff treatment?", [US]),
    ("US原产地", "How does US Customs determine country of origin marking requirements?", [US]),
    ("US出口管制", "What is the Commerce Control List under the EAR?", [US]),
    ("US出口管制", "What is an export licence exception under the EAR?", [US]),
    ("US出口管制", "What are the deemed export rules under the EAR?", [US]),
    ("US出口管制", "Who is a US person for export control purposes?", [US]),
    ("US出口管制", "What is the Entity List and what effect does listing have?", [US]),
    ("US保税", "What are the requirements for a US foreign trade zone?", [US]),
    ("US保税", "How are goods admitted to a US bonded warehouse?", [US]),
    ("US担保", "What forms of customs bond are accepted by US Customs?", [US]),
    ("US担保", "What is the difference between a single entry and a continuous bond in the US?", [US]),
    ("WTO", "What are the transparency obligations under the TBT Agreement?", [WTO]),
    ("WTO", "How does the SPS Agreement require risk assessment?", [WTO]),
]:
    add(cat, "英文场景", en, en, juris, ["customs"])

# ══ 22. 日本场景补充 ══════════════════════════════════════════
for zh, en in [
    ("日本の輸入通関手続きの流れを教えてください。", "What is the flow of import customs clearance procedures in Japan?"),
    ("日本の関税の税率種類にはどのようなものがありますか。", "What types of customs duty rates exist in Japan?"),
    ("日本の特恵関税制度とは何ですか。", "What is Japan's preferential tariff system?"),
    ("日本の輸入禁止品目にはどのようなものがありますか。", "Which items are prohibited from import into Japan?"),
    ("日本の税関検査はどのように行われますか。", "How are Japanese customs examinations conducted?"),
    ("日本の関税評価における加算要素は何ですか。", "What additions are made in Japanese customs valuation?"),
    ("日本の輸入貨物の原産地証明はどう取得しますか。", "How does one obtain a certificate of origin for imports into Japan?"),
    ("日本の保税蔵置場の利用条件は？", "What are the conditions for using a bonded storage area in Japan?"),
]:
    add("日本", "通关实务", zh, en, [JP], ["関税"])

# ══ 23. 英文综合业务问法（覆盖 US/AU 多角度）══════════════════
for cat, en, juris, kw in [
    ("US综合", "How do I import goods into the United States for the first time? What are the key steps?", [US], ["entry"]),
    ("US综合", "What happens if imported goods are not declared within the required time in the US?", [US], ["entry"]),
    ("US综合", "How does US Customs handle detained or seized merchandise?", [US], ["seizure"]),
    ("US综合", "What is the prior notice requirement for imported food in the US?", [US], ["notice"]),
    ("US综合", "How are antidumping and countervailing duties collected in the US?", [US], ["duty"]),
    ("US综合", "What is the role of a customs broker in the United States?", [US], ["broker"]),
    ("US综合", "How does the US Importer Security Filing requirement work?", [US], ["filing"]),
    ("US综合", "What are the penalties for fraudulent customs declarations in the US?", [US], ["penalty"]),
    ("US综合", "How does US Customs value related-party transactions?", [US], ["value"]),
    ("US综合", "What is the US customs ruling request procedure?", [US], ["ruling"]),
    ("AU综合", "How do I lodge an import declaration in Australia?", [AU], ["declaration"]),
    ("AU综合", "What duties and taxes apply to imports into Australia?", [AU], ["duty"]),
    ("AU综合", "What is a customs broker licence in Australia and who needs one?", [AU], ["broker"]),
    ("AU综合", "How does Australia's Customs Act treat goods temporarily imported?", [AU], ["temporary"]),
    ("AU综合", "What are Australia's rules for importing food and agricultural products?", [AU], ["food"]),
    ("AU综合", "How does Australia regulate exports of controlled goods?", [AU], ["export"]),
    ("AU综合", "What record keeping is required for Australian customs purposes?", [AU], ["record"]),
    ("AU综合", "What is the Australian Trusted Trader programme?", [AU], ["trader"]),
    ("WTO综合", "How does the WTO dispute settlement process work for trade disputes?", [WTO], ["dispute"]),
    ("WTO综合", "What is the role of the WTO Committee on Trade Facilitation?", [WTO], ["committee"]),
    ("WTO综合", "How are tariff quotas administered under WTO rules?", [WTO], ["quota"]),
    ("WTO综合", "What disciplines apply to export subsidies under the SCM Agreement?", [WTO], ["subsid"]),
    ("WTO综合", "How does the WTO Agreement on Agriculture treat market access?", [WTO], ["agricultur"]),
    ("CISG综合", "Under the CISG, what is the place of delivery when the contract does not specify it?", [CISG], ["delivery"]),
    ("CISG综合", "How does the CISG deal with a contract for goods to be manufactured?", [CISG], ["manufactur"]),
    ("CISG综合", "What is the CISG rule on interest on overdue payments?", [CISG], ["interest"]),
    ("CISG综合", "How does the CISG treat a party's right to suspend performance?", [CISG], ["suspend"]),
    ("CISG综合", "What is the preservation duty of the buyer on rejected goods under the CISG?", [CISG], ["preserv"]),
    ("CISG综合", "How does the CISG address mitigation of loss?", [CISG], ["mitigat"]),
    ("CISG综合", "What is the CISG rule on avoidance for non-payment?", [CISG], ["avoid"]),
]:
    add(cat, "英文实务", en, en, juris, kw)

# ══ 24. 中文多角度追问（同一法条的实务变体）══════════════════
for cat, zh, en, juris, kw in [
    ("关税", "纳税义务人可以通过什么方式查询自己应缴的关税？", "How can a duty payer check the duty payable?", [CN], ["关"]),
    ("关税", "海关作出的完税价格审定决定是否必须书面通知？", "Must customs notify its valuation decision in writing?", [CN], ["完税价格"]),
    ("关税", "进口货物的汇率应当按哪一天的汇率折算？", "Which exchange rate date applies to imported goods?", [CN, US], ["汇率"]),
    ("报关", "报关企业接受委托后可以转委托吗？", "May a customs broker sub-delegate its engagement?", [CN], ["报关"]),
    ("报关", "无纸化报关需要满足什么条件？", "What conditions are needed for paperless declaration?", [CN], ["报关"]),
    ("报关", "报关单的申报日期如何确定？", "How is the declaration date determined?", [CN], ["报关"]),
    ("原产地", "原产地证书上的第三方发票如何填报？", "How is a third-party invoice reflected on a certificate of origin?", [CN], ["原产地"]),
    ("原产地", "从保税区出口的货物原产地如何认定？", "How is origin determined for goods exported from a bonded zone?", [CN], ["原产地"]),
    ("出口管制", "出口管制许可申请需要多长时间审批？", "How long does export control licence approval take?", [CN], ["许可"]),
    ("出口管制", "列入管制清单的物项出口到任何国家都需要许可吗？", "Do listed controlled items require a licence for export to any country?", [CN, US], ["许可"]),
    ("检验检疫", "进口食品的境外生产企业必须注册吗？", "Must overseas manufacturers of imported food be registered?", [CN], ["注册"]),
    ("检验检疫", "进口预包装食品标签有什么要求？", "What labelling requirements apply to imported prepackaged food?", [CN], ["标签"]),
    ("反倾销", "反倾销调查中利害关系方有哪些权利？", "What rights do interested parties have in an anti-dumping investigation?", [CN, WTO], ["利害关系"]),
    ("反倾销", "反倾销税追溯征收的条件是什么？", "What conditions allow retroactive collection of anti-dumping duty?", [CN, WTO], ["追溯"]),
    ("外汇", "跨境贸易人民币结算试点企业的条件是什么？", "What conditions apply to enterprises settling cross-border trade in RMB?", [CN], ["人民币"]),
    ("外汇", "出口收入存放境外的条件是什么？", "What conditions apply to retaining export proceeds abroad?", [CN], ["境外"]),
    ("担保", "担保财产被查封后海关如何处理？", "How does customs handle security assets that are seized?", [CN], ["担保"]),
    ("保税", "保税维修业务的适用范围是什么？", "What is the scope of bonded repair business?", [CN], ["保税"]),
    ("保税", "保税展示交易的监管要求是什么？", "What supervision requirements apply to bonded exhibition and trading?", [CN], ["保税"]),
    ("商品归类", "预裁定决定在全国范围内有效吗？", "Is an advance ruling valid nationwide?", [CN], ["裁定"]),
]:
    add(cat, "实务追问", zh, en, juris, kw)

# ══ 25. 场景化长问（贴近真实单证审核）═════════════════════════
for cat, zh, en, juris, kw in [
    ("场景", "我方从德国进口数控机床，合同价格 FOB 汉堡，海运至上海。报关时完税价格应如何申报？运费保险费是否必须计入？",
     "We import CNC machines from Germany on FOB Hamburg terms, shipped to Shanghai. How should the customs value be declared, and must freight and insurance be included?",
     [CN, US], ["完税价格"]),
    ("场景", "客户在信用证中要求提交原产地证书，但货物经第三国转运。这份证书还能用于享受协定税率吗？",
     "The L/C requires a certificate of origin, but goods transit a third country. Can that certificate still support agreement rate treatment?",
     [CN, WTO], ["原产地"]),
    ("场景", "我司拟出口一批可用于无人机的陀螺仪，客户在东南亚。如何判断是否需要办理两用物项出口许可？",
     "We plan to export gyroscopes usable in drones to a Southeast Asian customer. How do we determine whether a dual-use export licence is required?",
     [CN, US, JP], ["两用物项"]),
    ("场景", "进口的一批冻肉在口岸查验时发现检疫不合格，海关要求退运。已缴的关税能退吗？滞港费用谁承担？",
     "A frozen meat consignment fails quarantine inspection at the port and customs orders return. Can duty already paid be refunded, and who bears the demurrage?",
     [CN, AU], ["检疫"]),
    ("场景", "我们把进口的免税设备租给关联企业使用，监管期未满。这样操作合规吗？需要补税吗？",
     "We leased duty-exempt imported equipment to an affiliate before the supervision period expired. Is this compliant, and is duty payable?",
     [CN], ["免税"]),
    ("场景", "境外供应商报的成交价格明显低于同期同类货物，海关启动价格质疑程序。我们需要准备哪些证明材料？",
     "The overseas supplier's price is clearly below comparable goods, and customs has initiated a valuation doubt. What evidence should we prepare?",
     [CN, US], ["完税价格"]),
    ("场景", "我司以加工贸易方式进口的料件因订单取消需要内销，如何计算应补缴的税款并办理手续？",
     "Materials imported under processing trade must be sold domestically due to a cancelled order. How is the duty calculated and what formalities apply?",
     [CN], ["内销"]),
    ("场景", "一批货物同时被认定构成走私和违反外汇管理规定，两个部门的处罚可以并罚吗？",
     "A consignment is found to involve both smuggling and a foreign exchange violation. May both authorities impose penalties?",
     [CN], ["走私"]),
    ("场景", "客户要求我们承担进口国的反倾销税，这笔税在合同价格条款中如何约定才有效？",
     "The customer asks us to bear anti-dumping duty in the importing country. How should this be addressed in the price clause?",
     [CN, WTO], ["反倾销"]),
    ("场景", "我们出口的设备在境外使用后需要返修再出口，进出境时适用什么监管方式？",
     "Equipment we exported needs repair abroad and re-export. What supervision mode applies on exit and re-entry?",
     [CN], ["修理"]),
    ("场景", "进口货物因运输延误错过了关税配额年度，还能享受配额内税率吗？",
     "Imported goods missed the tariff quota year due to transport delay. Can the in-quota rate still apply?",
     [CN, WTO], ["配额"]),
    ("场景", "海关认定我们的商品归类错误并要求补税三年，这个追溯期有法律依据吗？",
     "Customs found our classification wrong and demands three years of back duty. Is there a legal basis for that lookback period?",
     [CN], ["归类"]),
    ("场景", "我们从韩国进口化妆品，成品需在境内分装后销售。分装是否改变原产地？",
     "We import cosmetics from Korea and repackage them domestically before sale. Does repackaging change the origin?",
     [CN], ["原产地"]),
    ("场景", "企业被列入出口管制管控名单后，已签合同能否继续履行？在途货物如何处理？",
     "After being placed on an export control list, may existing contracts still be performed, and how are goods in transit handled?",
     [CN, US], ["管控名单"]),
    ("场景", "进口货物在保税仓存储超过一年，是否会被强制处理？如何申请延期？",
     "Imported goods have been stored in a bonded warehouse for over a year. Will they be compulsorily disposed of, and how can an extension be sought?",
     [CN], ["保税"]),
    ("场景", "我司代理客户报关，客户提供的发票被证实伪造。报关企业需要承担什么责任？",
     "We declared on behalf of a client whose invoice proved to be forged. What liability does the customs broker bear?",
     [CN], ["报关"]),
    ("场景", "进口二手设备被认定为固体废物，海关要求退运。退运前的仓储和处置费用由谁承担？",
     "Used equipment was classified as solid waste and customs ordered return. Who bears storage and disposal costs before return?",
     [CN], ["固体废物"]),
    ("场景", "跨境电子商务零售进口商品的单次交易限值和年度限值是多少？超限值如何征税？",
     "What are the per-transaction and annual limits for cross-border e-commerce retail imports, and how are excess amounts taxed?",
     [CN], ["跨境"]),
    ("场景", "我们想申请成为经核准出口商自行出具原产地声明，需要满足哪些条件？",
     "We want to become an approved exporter issuing our own origin declarations. What conditions must be met?",
     [CN], ["原产地"]),
    ("场景", "进口货物的买卖合同适用 CISG，但双方约定适用中国法。CISG 还能适用吗？",
     "The sale contract falls under the CISG, but the parties chose Chinese law. Does the CISG still apply?",
     [CISG, CN], ["CISG"]),
    ("场景", "卖方在 CISG 下交付的货物有轻微瑕疵，买方能否直接宣告合同无效？",
     "Under the CISG the seller delivered goods with a minor defect. May the buyer simply avoid the contract?",
     [CISG], ["CISG"]),
    ("场景", "我方作为买方在 CISG 下发现货物不符，但已超过两年未通知卖方。还能主张权利吗？",
     "As buyer under the CISG we discovered non-conformity but gave no notice for over two years. Can we still claim?",
     [CISG], ["CISG"]),
    ("场景", "同一批进口货物被两个部门分别认定为固体废物和禁止进口货物，应适用哪个规定？",
     "The same imported goods were classified by two authorities as solid waste and as prohibited imports. Which rule applies?",
     [CN], ["固体废物"]),
    ("场景", "我司进口的货物在运输途中因承运人过失灭失，已投保一切险。报关前货物不存在了，还需要申报吗？",
     "Imported goods were lost in transit through the carrier's fault and were fully insured. Must they still be declared even though they no longer exist?",
     [CN], ["申报"]),
    ("场景", "客户在境外要求我们把出口货物的原产地标为中国，但实际生产在越南。这样申报有什么风险？",
     "The customer asks us to mark origin as China though production is in Vietnam. What are the risks of such a declaration?",
     [CN], ["原产地"]),
    ("场景", "我们出口的技术资料包含受管制的软件代码，通过邮件发送。这属于出口管制范围吗？",
     "The technical materials we export contain controlled software code, sent by email. Does this fall within export control?",
     [CN, US], ["出口管制"]),
]:
    add(cat, "单证审核", zh, en, juris, kw)

# ══ 26. 美国 eCFR 深度覆盖（补足英文样本量）════════════════════════════
# 每条都对着一个具体 part，便于定位召回问题
for cat, en, kw in [
    # Part 101 总则
    ("US定义", "What does 19 CFR Part 101 define as the customs territory of the United States?", ["customs territory"]),
    ("US定义", "How does 19 CFR Part 101 define the term business day for customs purposes?", ["business day"]),
    # Part 111 报关代理人
    ("US报关代理", "What are the licence requirements for a customs broker under 19 CFR Part 111?", ["licen"]),
    ("US报关代理", "What examination must a customs broker pass under 19 CFR Part 111?", ["examination"]),
    ("US报关代理", "What are the duties of a customs broker to its client under 19 CFR Part 111?", ["duty"]),
    ("US报关代理", "How can a customs broker licence be revoked or suspended?", ["revok"]),
    ("US报关代理", "What recordkeeping is required of customs brokers under 19 CFR Part 111?", ["record"]),
    # Part 113 担保
    ("US担保", "What is the minimum bond amount required under 19 CFR Part 113?", ["bond"]),
    ("US担保", "What are the conditions of a customs bond under 19 CFR Part 113?", ["bond"]),
    ("US担保", "What happens if a bond is breached under 19 CFR Part 113?", ["breach"]),
    # Part 134 原产地标记
    ("US原产地", "What are the country of origin marking requirements under 19 CFR Part 134?", ["marking"]),
    ("US原产地", "What goods are exempt from country of origin marking requirements?", ["exempt"]),
    ("US原产地", "What is the ultimate purchaser rule in 19 CFR Part 134?", ["purchaser"]),
    # Part 141/142 报关
    ("US报关", "What is required to make entry of merchandise under 19 CFR Part 141?", ["entry"]),
    ("US报关", "What are the time limits for entry under 19 CFR Part 141?", ["time"]),
    ("US报关", "What is the entry process for imported goods under 19 CFR Part 142?", ["entry"]),
    ("US报关", "What is a special entry procedure under 19 CFR Part 143?", ["entry"]),
    # Part 144 保税仓
    ("US保税", "What is a warehouse entry under 19 CFR Part 144?", ["warehouse"]),
    ("US保税", "How are goods withdrawn from a bonded warehouse under 19 CFR Part 144?", ["withdraw"]),
    # Part 149 进口商安全申报
    ("US安全申报", "What is the Importer Security Filing requirement under 19 CFR Part 149?", ["filing"]),
    ("US安全申报", "What data elements must be included in an Importer Security Filing?", ["data"]),
    # Part 151/152 查验估值
    ("US查验", "How does customs examine and sample merchandise under 19 CFR Part 151?", ["examin"]),
    ("US估值", "How is merchandise appraised under 19 CFR Part 152?", ["apprais"]),
    ("US估值", "What are the alternative valuation methods under 19 CFR Part 152?", ["value"]),
    # Part 158/159 减免清算
    ("US清算", "How are duties liquidated under 19 CFR Part 159?", ["liquidat"]),
    ("US清算", "What is the protest period after liquidation?", ["protest"]),
    ("US减免", "When is duty relief available for damaged merchandise under 19 CFR Part 158?", ["relief"]),
    # Part 162/163 执法与记录
    ("US执法", "What search and seizure powers does customs have under 19 CFR Part 162?", ["seizure"]),
    ("US执法", "What are the recordkeeping requirements under 19 CFR Part 163?", ["record"]),
    ("US执法", "How long must importers retain records under 19 CFR Part 163?", ["retain"]),
    # Part 165 反规避
    ("US贸易救济", "What is the evasion investigation process under 19 CFR Part 165?", ["evasion"]),
    ("US贸易救济", "How is an allegation of duty evasion filed under 19 CFR Part 165?", ["allegation"]),
    # Part 172 违约金
    ("US处罚", "What are liquidated damages under 19 CFR Part 172?", ["damages"]),
    ("US处罚", "How can liquidated damages be mitigated or remitted?", ["mitigat"]),
    # Part 19/18/24 其他
    ("US保税", "What rules apply to container stations under 19 CFR Part 19?", ["container"]),
    ("US转运", "What is transportation in bond under 19 CFR Part 18?", ["bond"]),
    ("US财务", "What accounting procedures apply to customs under 19 CFR Part 24?", ["account"]),
    # Title 15 EAR
    ("US出口管制", "What is the scope of the Export Administration Regulations in 15 CFR Part 730?", ["scope"]),
    ("US出口管制", "What is the general prohibition in 15 CFR Part 736?", ["prohibit"]),
    ("US出口管制", "How is the Commerce Country Chart used under 15 CFR Part 738?", ["country chart"]),
    ("US出口管制", "What are the end-use and end-user controls in 15 CFR Part 744?", ["end-user"]),
    ("US出口管制", "What embargoes and other special controls apply under 15 CFR Part 746?", ["embargo"]),
    ("US出口管制", "What are the recordkeeping and enforcement provisions in 15 CFR Part 764?", ["record"]),
]:
    add(cat, "美国深度", en, en, [US], kw)

# ══ 27. 澳洲深度覆盖 ═══════════════════════════════════════════════
for cat, en, kw in [
    ("AU报关", "What is the meaning of 'entry for home consumption' in the Australian Customs Act 1901?", ["home consumption"]),
    ("AU报关", "What is an information report under the Australian Customs Act?", ["information"]),
    ("AU报关", "How are goods valued for duty purposes in Australia?", ["value"]),
    ("AU报关", "What is a Tariff Concession Order in Australia?", ["concession"]),
    ("AU报关", "What is the Australian Customs Tariff Act and how does it set rates?", ["tariff"]),
    ("AU报关", "How does Australia treat goods imported for temporary use?", ["temporary"]),
    ("AU保税", "What are the requirements for a licensed warehouse in Australia?", ["warehouse"]),
    ("AU保税", "How are goods moved under bond within Australia?", ["bond"]),
    ("AU执法", "What are the seizure and forfeiture provisions of the Australian Customs Act?", ["seizure"]),
    ("AU执法", "What penalties apply for false declarations under the Australian Customs Act?", ["penalt"]),
    ("AU出口", "What goods are prescribed for export control under the Australian Export Control Act 2020?", ["prescribed"]),
    ("AU出口", "What is an export permit and who can apply in Australia?", ["permit"]),
    ("AU出口", "What are the conditions for holding an Australian export licence?", ["licence"]),
    ("AU出口", "How does Australia regulate the export of live animals?", ["live animal"]),
    ("AU出口", "What enforcement powers exist under the Australian Export Control Act?", ["enforce"]),
]:
    add(cat, "澳洲深度", en, en, [AU], kw)

# ══ 28. WTO 深度覆盖 ═══════════════════════════════════════════════
for cat, en, kw in [
    ("WTO GATT", "What is the most-favoured-nation obligation in Article I of GATT 1994?", ["Article I"]),
    ("WTO GATT", "What does Article III of GATT 1994 require on national treatment?", ["Article III"]),
    ("WTO GATT", "What is the tariff binding obligation in Article II of GATT 1994?", ["binding"]),
    ("WTO GATT", "What are the general exceptions in Article XX of GATT 1994?", ["Article XX"]),
    ("WTO GATT", "What is the security exception in Article XXI of GATT 1994?", ["Article XXI"]),
    ("WTO GATT", "What does Article XI of GATT 1994 say about quantitative restrictions?", ["quantitative"]),
    ("WTO GATT", "What is the freedom of transit rule in Article V of GATT 1994?", ["transit"]),
    ("WTO TFA", "What does Article 1 of the Trade Facilitation Agreement require on publication?", ["publication"]),
    ("WTO TFA", "What does Article 3 of the TFA require on advance rulings?", ["advance ruling"]),
    ("WTO TFA", "What does Article 7 of the TFA require on release and clearance?", ["clearance"]),
    ("WTO TFA", "What does Article 11 of the TFA cover on transit?", ["transit"]),
    ("WTO SPS", "What is the requirement for scientific justification under the SPS Agreement?", ["scientific"]),
    ("WTO SPS", "How does the SPS Agreement treat international standards?", ["standard"]),
    ("WTO TBT", "What is the code of good practice under the TBT Agreement?", ["code of good practice"]),
    ("WTO TBT", "What does the TBT Agreement require on conformity assessment?", ["conformity"]),
    ("WTO AD", "How is dumping defined in the Anti-Dumping Agreement?", ["dumping"]),
    ("WTO AD", "What are the procedural requirements for an anti-dumping investigation?", ["investigation"]),
    ("WTO CV", "What are the six valuation methods under the Customs Valuation Agreement?", ["method"]),
    ("WTO CV", "What is transaction value under the Customs Valuation Agreement?", ["transaction value"]),
    ("WTO RO", "What disciplines apply to rules of origin under the ROO Agreement?", ["origin"]),
    ("WTO LIC", "What is the difference between automatic and non-automatic licensing?", ["automatic"]),
    ("WTO SCM", "What are prohibited subsidies under the SCM Agreement?", ["prohibited"]),
    ("WTO SCM", "What is the de minimis threshold under the SCM Agreement?", ["de minimis"]),
    ("WTO TRIMS", "What investment measures are prohibited under TRIMS?", ["investment"]),
    ("WTO Agri", "How does the Agreement on Agriculture treat market access?", ["market access"]),
]:
    add(cat, "WTO深度", en, en, [WTO], kw)

# ══ 29. CISG 深度覆盖 ═════════════════════════════════════════════
for n, kw in [
    (1, ["place of business"]),
    (2, ["personal, family"]),
    (6, ["party autonomy"]),
    (11, ["writing"]),
    (14, ["offer"]),
    (18, ["acceptance"]),
    (19, ["materially"]),
    (21, ["late acceptance"]),
    (25, ["fundamental breach"]),
    (31, ["delivery"]),
    (33, ["time for delivery"]),
    (35, ["conformity"]),
    (38, ["examine"]),
    (39, ["notice"]),
    (45, ["remedies"]),
    (46, ["performance"]),
    (49, ["avoid"]),
    (50, ["reduce the price"]),
    (74, ["damages"]),
    (77, ["mitigate"]),
    (78, ["interest"]),
    (79, ["impediment"]),
    (81, ["restitution"]),
    (88, ["preserve"]),
]:
    add("CISG深度", "条文", f"CISG 第 {n} 条规定了什么？",
        f"What does Article {n} of the CISG provide?", [CISG], kw)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(CASES, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter

    print(f"用例总数: {len(CASES)}")
    print("\n按场景：")
    for k, v in Counter(c["category"] for c in CASES).most_common():
        print(f"  {k:12s} {v:4d}")
    print("\n按法域：")
    for k, v in Counter(j for c in CASES for j in c["jurisdiction"]).most_common():
        print(f"  {k:12s} {v:4d}")
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
