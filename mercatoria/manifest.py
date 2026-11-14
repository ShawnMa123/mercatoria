"""Corpus manifest: stable document identity + per-law metadata.

The fetched corpus deliberately has almost no metadata: a markdown file and a
directory. This module derives a stable, explicit manifest from that layout. It
is intentionally rule-based so a fresh clone can regenerate it without a
database. Article-level metadata comes from the chunker; document-level
metadata lives here.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "2026-11-09.1"

CN_SLUGS = {
    "两用物项出口管制条例(2024-09-30).md": "cn-dual-use-export-control-2024",
    "关税法(2024-04-26).md": "cn-tariff-law-2024",
    "出口管制法(2020-10-17).md": "cn-export-control-law-2020",
    "反倾销和反补贴条例(1997-03-25).md": "cn-ad-cvd-1997",
    "反倾销条例(2004-03-31).md": "cn-anti-dumping-2004",
    "反补贴条例(2004-03-31).md": "cn-countervailing-2004",
    "外汇管理条例(2008-08-05).md": "cn-foreign-exchange-2008",
    "对外贸易法(2025-12-27).md": "cn-foreign-trade-law-2025",
    "导弹及相关物项和技术出口管制条例(2002-08-22).md": "cn-missile-export-control-2002",
    "技术进出口管理条例(2020-11-29).md": "cn-tech-import-export-2020",
    "核两用品及相关技术出口管制条例(2007-01-26).md": "cn-nuclear-dual-use-2007",
    "核出口管制条例(2006-11-09).md": "cn-nuclear-export-control-2006",
    "海关事务担保条例(2018-03-19).md": "cn-customs-guarantee-2018",
    "海关对检举或协助查获违反海关法案件有功人员的奖励办法(2010-11-26).md": "cn-customs-reward-2010",
    "海关法(2021-04-29).md": "cn-customs-law-2021",
    "海关法行政处罚实施细则(2004-11-01).md": "cn-customs-penalty-rules-2004",
    "海关行政处罚实施条例(2022-03-29).md": "cn-customs-penalty-2022",
    "货物进出口管理条例(2024-03-10).md": "cn-goods-import-export-2024",
    "进出口关税条例(2017-03-01).md": "cn-import-export-tariff-2017",
    "进出口商品检验法(2021-04-29).md": "cn-inspection-law-2021",
    "进出口商品检验法实施条例(2026-08-08).md": "cn-inspection-regulations-2026",
    "进出口货物原产地条例(2019-03-02).md": "cn-origin-2019",
    # 海关监管、估价、归类与信用管理
    "海关稽查条例(2022-03-29).md": "cn-customs-audit-2022",
    "知识产权海关保护条例(2018-03-19).md": "cn-ipr-border-2018",
    "海关审定进出口货物完税价格办法(2013-12-25).md": "cn-customs-valuation-measures-2013",
    "海关进出口货物征税管理办法(2024-10-28).md": "cn-customs-duty-collection-2024",
    "海关进出口货物申报管理规定(2018-11-23).md": "cn-customs-declaration-2018",
    "海关进出口货物商品归类管理规定(2021-09-06).md": "cn-customs-classification-2021",
    "海关预裁定管理暂行办法(2017-12-26).md": "cn-customs-advance-ruling-2017",
    "海关综合保税区管理办法(2022-01-01).md": "cn-comprehensive-bonded-zone-2022",
    "海关注册登记和备案企业信用管理办法(2021-09-13).md": "cn-customs-credit-2021",
    "海关报关单位备案管理规定(2021-11-19).md": "cn-customs-broker-filing-2021",
    # 原产地、自贸协定与贸易救济
    "关于非优惠原产地规则中实质性改变标准的规定(2018-04-28).md":
        "cn-nonpreferential-origin-2018",
    "海关RCEP项下进出口货物原产地管理办法(2021-11-23).md": "cn-rcep-origin-2021",
    "保障措施条例(2004-03-31).md": "cn-safeguard-2004",
    # 出口管制许可
    "两用物项和技术进出口许可证管理办法(2005-12-31).md":
        "cn-dual-use-license-measures-2005",
    "两用物项和技术出口通用许可管理办法(2009-05-13).md": "cn-dual-use-general-license-2009",
    # 国际结算
    "最高人民法院关于审理信用证纠纷案件若干问题的规定(2020-12-29).md": "cn-lc-dispute-2020",
    "最高人民法院关于审理独立保函纠纷案件若干问题的规定(2020-12-29).md":
        "cn-independent-guarantee-2020",
}

CN_APPLIES_TO = {
    "cn-dual-use-export-control-2024": ["export_control"],
    "cn-tariff-law-2024": ["tariff", "customs"],
    "cn-export-control-law-2020": ["export_control"],
    "cn-ad-cvd-1997": ["trade_remedy"],
    "cn-anti-dumping-2004": ["trade_remedy"],
    "cn-countervailing-2004": ["trade_remedy"],
    "cn-foreign-exchange-2008": ["foreign_exchange"],
    "cn-foreign-trade-law-2025": ["foreign_trade"],
    "cn-missile-export-control-2002": ["export_control"],
    "cn-tech-import-export-2020": ["technology_trade", "export_control"],
    "cn-nuclear-dual-use-2007": ["export_control"],
    "cn-nuclear-export-control-2006": ["export_control"],
    "cn-customs-guarantee-2018": ["guarantee", "customs"],
    "cn-customs-reward-2010": ["customs"],
    "cn-customs-law-2021": ["customs", "declaration", "penalty"],
    "cn-customs-penalty-rules-2004": ["customs", "penalty"],
    "cn-customs-penalty-2022": ["customs", "penalty"],
    "cn-goods-import-export-2024": ["foreign_trade", "import_export"],
    "cn-import-export-tariff-2017": ["tariff", "customs"],
    "cn-inspection-law-2021": ["inspection"],
    "cn-inspection-regulations-2026": ["inspection"],
    "cn-origin-2019": ["origin"],
    "cn-customs-audit-2022": ["customs", "audit"],
    "cn-ipr-border-2018": ["customs", "intellectual_property"],
    "cn-customs-valuation-measures-2013": ["tariff", "customs", "customs_valuation"],
    "cn-customs-duty-collection-2024": ["tariff", "customs"],
    "cn-customs-declaration-2018": ["customs", "declaration"],
    "cn-customs-classification-2021": ["customs", "classification"],
    "cn-customs-advance-ruling-2017": ["customs", "advance_ruling"],
    "cn-comprehensive-bonded-zone-2022": ["customs", "bonded", "origin"],
    "cn-customs-credit-2021": ["customs", "credit"],
    "cn-customs-broker-filing-2021": ["customs", "declaration"],
    "cn-nonpreferential-origin-2018": ["origin", "customs"],
    "cn-rcep-origin-2021": ["origin", "customs", "rcep"],
    "cn-safeguard-2004": ["trade_remedy"],
    "cn-dual-use-license-measures-2005": ["export_control", "customs"],
    "cn-dual-use-general-license-2009": ["export_control"],
    "cn-lc-dispute-2020": ["settlement", "trade_finance"],
    "cn-independent-guarantee-2020": ["settlement", "guarantee", "trade_finance"],
}

# Passage date is not always the effective date. Override the few laws where
# the filename carries the adoption/publication date instead.
CN_EFFECTIVE_FROM = {
    "cn-tariff-law-2024": "2024-12-01",
    "cn-export-control-law-2020": "2020-12-01",
    "cn-dual-use-export-control-2024": "2024-12-01",
    "cn-anti-dumping-2004": "2004-06-01",
    "cn-countervailing-2004": "2004-06-01",
}

# 旧 doc_id -> 承接它的现行 doc_id。检索返回旧法时据此告诉 Agent 该看哪部。
SUPERSEDED_BY = {
    "cn-ad-cvd-1997": "cn-anti-dumping-2004",
    "cn-customs-penalty-rules-2004": "cn-customs-penalty-2022",
    "cn-import-export-tariff-2017": "cn-tariff-law-2024",
}

CN_EFFECTIVE_TO = {
    # Superseded by the 2004 anti-dumping / countervailing regulations.
    "cn-ad-cvd-1997": "2004-03-31",
    # Superseded by the 2022 customs penalty regulation.
    "cn-customs-penalty-rules-2004": "2022-03-29",
    # Superseded by the Tariff Law (effective 2024-12-01).
    "cn-import-export-tariff-2017": "2024-12-01",
}

WTO_APPLIES_TO = {
    "gatt47_e": ["wto", "tariff", "trade"],
    "tfa_e": ["wto", "trade_facilitation"],
    "sps_e": ["wto", "inspection"],
    "tbt_e": ["wto", "standards"],
    "adp_e": ["wto", "trade_remedy"],
    "cv_e": ["wto", "customs_valuation"],
    "ro_e": ["wto", "origin"],
    "lic_e": ["wto", "import_licensing"],
    "scm_e": ["wto", "trade_remedy"],
    "trims_e": ["wto", "investment"],
    "dsu_e": ["wto", "dispute_settlement"],
    "sg_e": ["wto", "trade_remedy"],
    "ag_e": ["wto", "agriculture"],
    "27-trips_e": ["wto", "intellectual_property"],
}

# Stems whose generated doc_id would be unreadable (wto-27-trips, wto-sg).
WTO_DOC_IDS = {
    "gatt47_e": "wto-gatt-1994",
    "sg_e": "wto-safeg",
    "27-trips_e": "wto-trips",
}

ECFR_APPLIES_TO = {
    "15": ["export_control"],
    "19": ["customs", "tariff", "origin", "entry"],
    "31": ["sanctions", "export_control", "foreign_exchange"],
}

JP_APPLIES_TO = {
    "kazei-hou": ["customs"],
    "kazei-teiritsu-hou": ["tariff", "customs"],
    "gaiamei-hou": ["foreign_exchange", "export_control"],
    "yushutsu-boueki-kanri-rei": ["export_control"],
    "kanzei-hou-sekourei": ["customs"],
    "kanzei-teiritsu-hou-sekourei": ["tariff", "customs"],
    "kanzei-zantei-sochi-hou": ["tariff", "origin"],
    "gaikoku-kawase-rei": ["foreign_exchange", "export_control"],
    "yunyu-boueki-kanri-rei": ["import_export", "customs"],
}

AU_APPLIES_TO = {
    "customs-act-1901": ["customs"],
    "customs-tariff-act-1995": ["tariff", "customs"],
    "export-control-act-2020": ["export_control"],
    "customs-regulation-2015": ["customs", "tariff"],
    "autonomous-sanctions-act-2011": ["sanctions", "export_control"],
}

EU_APPLIES_TO = {
    "eu-ucc-2013": ["customs", "tariff", "origin"],
    "eu-dual-use-2021": ["export_control"],
}

UK_APPLIES_TO = {
    "uk-cross-border-trade-2018": ["customs", "tariff"],
}

RE_DOC_COMMENT = re.compile(r"<!--\s*document:\s*(.*?)\s*-->", re.I | re.S)
RE_DATE_IN_NAME = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")


def _first_document_title(path: Path, fallback: str) -> str:
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return fallback
    m = RE_DOC_COMMENT.search(head)
    return m.group(1).strip() if m else fallback


def _doc(
    path: Path,
    corpus_dir: Path,
    *,
    doc_id: str,
    jurisdiction: str,
    language: str,
    precision: str,
    applies_to: list[str],
    effective_from: str | None,
    effective_to: str | None = None,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    rel = path.relative_to(corpus_dir).as_posix()
    return {
        "doc_id": doc_id,
        "path": rel,
        "source_doc": _first_document_title(path, path.stem),
        "jurisdiction": jurisdiction,
        "language": language,
        "precision": precision,
        "applies_to": applies_to,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "snapshot_date": snapshot_date,
    }


def _index_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _wto_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    metas = _index_json(corpus_dir / "wto" / "index.json")
    snapshot = metas.get("snapshot_date") or "2026-11-01"
    out = []
    for path in sorted((corpus_dir / "wto").glob("*.md")):
        stem = path.stem
        doc_id = WTO_DOC_IDS.get(stem) or f"wto-{stem.replace('_e', '')}"
        out.append(
            _doc(
                path,
                corpus_dir,
                doc_id=doc_id,
                jurisdiction="WTO",
                language="en",
                precision="snapshot",
                applies_to=WTO_APPLIES_TO.get(stem, ["wto"]),
                effective_from=None,
                snapshot_date=snapshot,
            )
        )
    return out


def _ecfr_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    out = []
    for title_dir in sorted((corpus_dir / "ecfr").glob("title-*")):
        title = title_dir.name.removeprefix("title-")
        metas = _index_json(title_dir / "index.json")
        snapshot = metas.get("snapshot_date") or "2026-11-01"
        for path in sorted(title_dir.glob("part-*.md")):
            part = path.stem.removeprefix("part-")
            out.append(
                _doc(
                    path,
                    corpus_dir,
                    doc_id=f"us-cfr-{title}-{part}",
                    jurisdiction="US",
                    language="en",
                    precision="degraded",
                    applies_to=ECFR_APPLIES_TO.get(title, ["customs"]),
                    effective_from=snapshot,
                    snapshot_date=snapshot,
                )
            )
    return out


def _au_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    metas = _index_json(corpus_dir / "au" / "index.json")
    fetched = {item.get("slug"): item for item in metas.get("fetched", []) if isinstance(item, dict)}
    out = []
    for path in sorted((corpus_dir / "au").glob("*.md")):
        slug = path.stem
        meta = fetched.get(slug, {})
        start = str(meta.get("start") or "2026-11-01")[:10]
        out.append(
            _doc(
                path,
                corpus_dir,
                doc_id=f"au-{slug}",
                jurisdiction="AU",
                language="en",
                precision="degraded",
                applies_to=AU_APPLIES_TO.get(slug, ["customs"]),
                effective_from=start,
                snapshot_date=start,
            )
        )
    return out


def _jp_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    metas = _index_json(corpus_dir / "jp" / "index.json")
    fetched = {item.get("file"): item for item in metas.get("fetched", []) if isinstance(item, dict)}
    out = []
    for path in sorted((corpus_dir / "jp").glob("*.md")):
        meta = fetched.get(path.name, {})
        slug = meta.get("slug") or path.stem
        out.append(
            _doc(
                path,
                corpus_dir,
                doc_id=f"jp-{slug}",
                jurisdiction="JP",
                language="ja",
                precision="degraded",
                applies_to=JP_APPLIES_TO.get(slug, ["customs"]),
                effective_from="2026-11-01",
                snapshot_date="2026-11-01",
            )
        )
    return out


def _cn_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((corpus_dir / "cn").glob("*.md")):
        doc_id = CN_SLUGS.get(path.name)
        if not doc_id:
            # Unknown Chinese law: keep it stable, but never guess a human slug.
            doc_id = "cn-" + path.stem.split("(")[0].strip().replace(" ", "-")
        m = RE_DATE_IN_NAME.search(path.name)
        effective_from = CN_EFFECTIVE_FROM.get(doc_id) or (m.group(1) if m else None)
        out.append(
            _doc(
                path,
                corpus_dir,
                doc_id=doc_id,
                jurisdiction="CN",
                language="zh",
                precision="exact",
                applies_to=CN_APPLIES_TO.get(doc_id, []),
                effective_from=effective_from,
                effective_to=CN_EFFECTIVE_TO.get(doc_id),
            )
        )
    return out


def _eu_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((corpus_dir / "eu").glob("*.md")):
        slug = path.stem
        out.append(
            _doc(
                path,
                corpus_dir,
                doc_id=slug,
                jurisdiction="EU",
                language="en",
                precision="snapshot",
                applies_to=EU_APPLIES_TO.get(slug, ["customs"]),
                effective_from=None,
                snapshot_date="2026-11-01",
            )
        )
    return out


def _uk_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((corpus_dir / "uk").glob("*.md")):
        slug = path.stem
        out.append(
            _doc(
                path,
                corpus_dir,
                doc_id=slug,
                jurisdiction="UK",
                language="en",
                precision="degraded",
                applies_to=UK_APPLIES_TO.get(slug, ["customs"]),
                effective_from="2026-11-01",
                snapshot_date="2026-11-01",
            )
        )
    return out


def _cisg_docs(corpus_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((corpus_dir / "uncitral").glob("*.md")):
        out.append(
            _doc(
                path,
                corpus_dir,
                doc_id="cisg",
                jurisdiction="INT",
                language="en",
                precision="snapshot",
                applies_to=["cisg", "contract"],
                effective_from=None,
                snapshot_date="1988-01-01",
            )
        )
    return out


def build_manifest(corpus_dir: str | Path) -> dict[str, Any]:
    corpus = Path(corpus_dir).resolve()
    docs: list[dict[str, Any]] = []
    for part in (
        _cn_docs,
        _ecfr_docs,
        _wto_docs,
        _au_docs,
        _jp_docs,
        _eu_docs,
        _uk_docs,
        _cisg_docs,
    ):
        docs.extend(part(corpus))
    docs.sort(key=lambda d: d["doc_id"])
    known = {doc["doc_id"] for doc in docs}
    for doc in docs:
        replacement = SUPERSEDED_BY.get(doc["doc_id"], "")
        # Point at the replacement only when it is actually in the corpus.
        doc["superseded_by"] = replacement if replacement in known else ""
    return {
        "manifest_version": MANIFEST_VERSION,
        "corpus_dir": ".",
        "documents": len(docs),
        "docs": docs,
    }


def write_manifest(corpus_dir: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Regenerate the manifest and replace ``out_path`` atomically.

    The manifest is a build input, read by the API, the CLI and the ingest
    path. Writing it in place let a concurrent reader parse half a document, so
    the JSON is written to a temporary file next to the target and swapped in
    with one ``os.replace``: readers see either the old file or the complete new
    one, and a failure leaves the previous manifest untouched.
    """
    manifest = build_manifest(corpus_dir)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    handle, temporary = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not data.get("docs"):
        raise ValueError(f"manifest has no docs: {p}")
    return data
