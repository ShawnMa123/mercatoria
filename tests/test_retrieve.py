from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mercatoria.config import AppConfig
from mercatoria.manifest import load_manifest
from mercatoria.retrieve import Retriever, _apply_jurisdiction_quota, _quota_for
from mercatoria.store import build_index, open_current


def fake_embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        lower = text.lower()
        if "关税" in text or "tariff" in lower:
            row = [1.0, 0.0, 0.0]
        elif "原产地" in text or "origin" in lower:
            row = [0.0, 1.0, 0.0]
        else:
            row = [0.0, 0.0, 1.0]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        (self.corpus / "cn").mkdir(parents=True)
        (self.corpus / "us").mkdir(parents=True)
        (self.corpus / "cn" / "test.md").write_text(
            "<!-- document: 测试关税法 -->\n"
            "第一条 关税的征收适用本法。\n\n"
            "第二条 原产地规则另行规定。\n",
            encoding="utf-8",
        )
        (self.corpus / "us" / "test.md").write_text(
            "<!-- document: Test Tariff -->\n"
            "## § 101.0 Tariff scope.\n\nThis section covers tariff entry.\n",
            encoding="utf-8",
        )
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": "test",
                    "docs": [
                        {
                            "doc_id": "cn-test-law",
                            "path": "cn/test.md",
                            "source_doc": "测试关税法",
                            "jurisdiction": "CN",
                            "language": "zh",
                            "precision": "exact",
                            "applies_to": ["customs"],
                            "effective_from": "2024-01-01",
                            "effective_to": None,
                            "snapshot_date": None,
                        },
                        {
                            "doc_id": "us-test-101",
                            "path": "us/test.md",
                            "source_doc": "Test Tariff",
                            "jurisdiction": "US",
                            "language": "en",
                            "precision": "degraded",
                            "applies_to": ["customs"],
                            "effective_from": "2026-11-01",
                            "effective_to": None,
                            "snapshot_date": "2026-11-01",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.cfg = AppConfig(
            corpus_dir=str(self.corpus),
            index_dir=str(self.root / "index"),
            manifest_path=str(self.manifest_path),
            documents_dir=str(self.root / "documents"),
        )
        self.cfg.embedding.dim = 3
        self.cfg.embedding.batch_size = 2
        self.cfg.retrieval.vector_weight = 0.5
        self.manifest = load_manifest(self.manifest_path)
        self.index_dir = self.root / "index"
        build_index(self.manifest, self.cfg, self.index_dir, embed_fn=fake_embed)
        self.retriever = Retriever(open_current(self.index_dir), self.cfg, embed_fn=fake_embed)

    def tearDown(self) -> None:
        self.retriever.index.conn.close()
        self.tmp.cleanup()

    def test_retrieves_article_and_explains_match(self) -> None:
        result = self.retriever.retrieve(["关税"], top_k=2)
        self.assertEqual(result["corpus"]["status"], "ready")
        self.assertGreaterEqual(len(result["chunks"]), 1)
        first = result["chunks"][0]
        self.assertIn("关税", first["text"])
        self.assertEqual(first["article_ids"], ["cn-test-law:1"])
        self.assertTrue(any(m["term"] == "关税" for m in first["matched_terms"]))
        self.assertIn("score_parts", first)
        self.assertEqual(result["diagnostics"]["funnel"]["after_retrieval"], len(result["chunks"]))

    def test_jurisdiction_filter_is_intersection(self) -> None:
        result = self.retriever.retrieve(["关税"], filters={"jurisdictions": ["CN", "SG"]})
        self.assertTrue(result["chunks"])
        self.assertTrue(all("CN" in c["jurisdictions"] for c in result["chunks"]))
        self.assertEqual(result["diagnostics"]["requested_uncovered"]["jurisdictions"], ["SG"])

    def test_no_jurisdiction(self) -> None:
        result = self.retriever.retrieve(["关税"], filters={"jurisdictions": ["SG"]})
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["diagnostics"]["zero_reason"], "no_jurisdiction")

    def test_business_type_filter_and_universal_semantics(self) -> None:
        result = self.retriever.retrieve(["关税"], filters={"applies_to": ["export_control"]})
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["diagnostics"]["zero_reason"], "no_business_type")
        result = self.retriever.retrieve(["关税"], filters={"applies_to": ["customs"]})
        self.assertTrue(result["chunks"])

    def test_temporal_degraded_is_not_date_filtered(self) -> None:
        result = self.retriever.retrieve(
            ["tariff"], filters={"jurisdictions": ["US"], "as_of": "2023-01-01"}
        )
        self.assertTrue(result["chunks"])
        self.assertTrue(all("US" in c["jurisdictions"] for c in result["chunks"]))

    def test_temporal_exact_filter(self) -> None:
        # CN is exact; the US document is degraded and therefore is not
        # date-filtered. Restrict to CN to exercise no_temporal.
        result = self.retriever.retrieve(
            ["关税"], filters={"jurisdictions": ["CN"], "as_of": "2023-01-01"}
        )
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["diagnostics"]["zero_reason"], "no_temporal")
        result = self.retriever.retrieve(
            ["关税"], filters={"jurisdictions": ["CN"], "as_of": "2025-01-01"}
        )
        self.assertTrue(result["chunks"])

    def test_fields_are_mapped_and_echoed(self) -> None:
        result = self.retriever.retrieve(
            ["关税"], fields={"business_type": "customs", "country": "CN"}
        )
        self.assertTrue(result["chunks"])
        self.assertEqual(result["field_matches"]["jurisdictions"]["requested"], ["CN"])
        self.assertEqual(result["field_matches"]["applies_to"]["requested"], ["customs"])
        self.assertEqual(result["chunks"][0]["matched_fields"]["requested_jurisdictions"], ["CN"])

    def test_chunk_ids_are_stable_across_rebuild(self) -> None:
        ids1 = [c["chunk_id"] for c in self.retriever.index.chunks]
        self.retriever.index.conn.close()
        build_index(self.manifest, self.cfg, self.index_dir, embed_fn=fake_embed)
        index2 = open_current(self.index_dir)
        ids2 = [c["chunk_id"] for c in index2.chunks]
        index2.conn.close()
        self.assertEqual(ids1, ids2)

    def test_empty_query_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.retrieve(["  "])

    def test_rerank_promotes_when_first_stage_is_weak(self) -> None:
        def fake_rerank(query: str, docs: list[str], top_n: int):
            pairs = []
            for i, doc in enumerate(docs):
                score = 10.0 if "原产地" in doc else 0.1
                pairs.append((i, score))
            return pairs, 1.5

        retriever = Retriever(
            self.retriever.index, self.cfg, embed_fn=fake_embed, rerank_fn=fake_rerank
        )
        result = retriever.retrieve(["zzz"], top_k=2)
        self.assertTrue(result["diagnostics"]["rerank"]["applied"])
        self.assertEqual(result["diagnostics"]["rerank"]["model"], self.cfg.rerank.model)
        self.assertIn("原产地", result["chunks"][0]["text"])
        self.assertIsNotNone(result["chunks"][0]["score_parts"]["rerank"])

    def test_max_fusion_keeps_exact_first_stage_match(self) -> None:
        self.cfg.rerank.fusion = "max"
        self.cfg.rerank.weight = 0.8

        def fake_rerank(query: str, docs: list[str], top_n: int):
            return [(i, 10.0 if "原产地" in d else 1.0) for i, d in enumerate(docs)], 1.0

        retriever = Retriever(
            self.retriever.index, self.cfg, embed_fn=fake_embed, rerank_fn=fake_rerank
        )
        result = retriever.retrieve(["关税"], top_k=2)
        self.assertIn("关税", result["chunks"][0]["text"])

    def test_request_can_disable_rerank(self) -> None:
        def fake_rerank(query: str, docs: list[str], top_n: int):
            return [(i, 10.0) for i in range(len(docs))], 1.0

        retriever = Retriever(
            self.retriever.index, self.cfg, embed_fn=fake_embed, rerank_fn=fake_rerank
        )
        result = retriever.retrieve(["关税"], top_k=2, use_rerank=False)
        self.assertFalse(result["diagnostics"]["rerank"]["enabled"])
        self.assertFalse(result["diagnostics"]["rerank"]["applied"])
        self.assertIsNone(result["chunks"][0]["score_parts"]["rerank"])

    def test_cache_key_covers_every_scoring_parameter(self) -> None:
        cfg = self.cfg
        base = self.retriever._cache_key(["q"], 5, {}, {}, False)
        mutations = [
            ("retrieval", "vector_weight", cfg.retrieval.vector_weight + 0.1),
            ("retrieval", "similarity_threshold", 0.25),
            ("retrieval", "bm25_k1", cfg.retrieval.bm25_k1 + 0.5),
            ("retrieval", "bm25_b", cfg.retrieval.bm25_b + 0.1),
            ("retrieval", "multi_query_bonus", 0.05),
            ("retrieval", "expand_cross_language", False),
            ("rerank", "candidates", 99),
            ("rerank", "fusion", "max"),
            ("rerank", "weight", 0.9),
            ("rerank", "top_n", 7),
            ("rerank", "min_score", 0.5),
            ("rerank", "model", "other-reranker"),
        ]
        for section, key, value in mutations:
            target = getattr(cfg, section)
            old = getattr(target, key)
            setattr(target, key, value)
            try:
                self.assertNotEqual(
                    base,
                    self.retriever._cache_key(["q"], 5, {}, {}, False),
                    f"{section}.{key} is missing from the cache key",
                )
            finally:
                setattr(target, key, old)
        self.assertEqual(base, self.retriever._cache_key(["q"], 5, {}, {}, False))

    def test_clear_cache_drops_entries(self) -> None:
        self.retriever.retrieve(["关税"], top_k=2)
        self.assertTrue(self.retriever._cache)
        self.retriever.clear_cache()
        self.assertFalse(self.retriever._cache)

    def test_bm25_runs_from_memory_not_sqlite(self) -> None:
        index = self.retriever.index
        self.assertTrue(index.postings)
        self.assertTrue(index.terms)
        self.assertTrue(index.idf)
        # The connection is closed once the index is loaded, so no request
        # thread can issue SQL against it.
        with self.assertRaises(sqlite3.ProgrammingError):
            index.conn.execute("SELECT 1")
        scores = index.bm25_scores(set(index.chunk_ids), ["关税"], 1.5, 0.75)
        self.assertTrue(scores)
        self.assertTrue(all(value > 0 for value in scores.values()))

    def test_doc_and_article_filters_run_before_scoring(self) -> None:
        result = self.retriever.retrieve(["原产地"], filters={"doc_ids": ["cn-test-law"]})
        self.assertTrue(result["chunks"])
        self.assertTrue(all(c["doc_id"] == "cn-test-law" for c in result["chunks"]))
        result = self.retriever.retrieve(
            ["原产地"], filters={"article_ids": ["cn-test-law:2"]}
        )
        self.assertEqual([c["article_ids"] for c in result["chunks"]], [["cn-test-law:2"]])
        self.assertEqual(result["diagnostics"]["funnel"]["after_article"], 1)
        missing = self.retriever.retrieve(["原产地"], filters={"doc_ids": ["nope"]})
        self.assertEqual(missing["chunks"], [])
        self.assertEqual(missing["diagnostics"]["zero_reason"], "no_doc")
        missing = self.retriever.retrieve(
            ["原产地"], filters={"article_ids": ["cn-test-law:99"]}
        )
        self.assertEqual(missing["diagnostics"]["zero_reason"], "no_article")

    def test_fields_map_doc_and_article_ids(self) -> None:
        result = self.retriever.retrieve(["关税"], fields={"doc_ids": ["us-test-101"]})
        self.assertTrue(result["chunks"])
        self.assertTrue(all(c["doc_id"] == "us-test-101" for c in result["chunks"]))

    def test_multi_query_match_earns_a_bounded_bonus(self) -> None:
        self.cfg.retrieval.multi_query_bonus = 0.03
        self.retriever.clear_cache()
        result = self.retriever.retrieve(["关税", "关税的征收适用本法"], top_k=5)
        hit = next(c for c in result["chunks"] if c["article_ids"] == ["cn-test-law:1"])
        self.assertEqual(len(hit["matched_queries"]), 2)
        self.assertAlmostEqual(hit["score_parts"]["fusion_bonus"], 0.03, places=6)
        single = self.retriever.retrieve(["关税"], top_k=5)
        self.assertEqual(single["chunks"][0]["score_parts"]["fusion_bonus"], 0.0)

    def test_quota_keeps_each_jurisdiction_represented(self) -> None:
        ordered = [
            ("a1", {"jurisdiction": "CN", "score": 0.9}),
            ("a2", {"jurisdiction": "CN", "score": 0.8}),
            ("a3", {"jurisdiction": "CN", "score": 0.7}),
            ("b1", {"jurisdiction": "US", "score": 0.6}),
        ]
        stub = type(
            "Stub",
            (),
            {"chunk_by_id": {cid: {"jurisdiction": hit["jurisdiction"]} for cid, hit in ordered}},
        )()
        picked = _apply_jurisdiction_quota(ordered, 2, ["CN", "US"], 1, stub)
        self.assertEqual([cid for cid, _ in picked], ["a1", "b1"])
        # No quota means plain top_k.
        self.assertEqual(
            [cid for cid, _ in _apply_jurisdiction_quota(ordered, 2, ["CN", "US"], 0, stub)],
            ["a1", "a2"],
        )
        # Soft quota: a jurisdiction far below the global top-1 gives up its slot.
        soft = _apply_jurisdiction_quota(ordered, 2, ["CN", "US"], 1, stub, 0.9)
        self.assertEqual([cid for cid, _ in soft], ["a1", "a2"])
        soft_ok = _apply_jurisdiction_quota(ordered, 2, ["CN", "US"], 1, stub, 0.5)
        self.assertEqual([cid for cid, _ in soft_ok], ["a1", "b1"])
        # A single requested jurisdiction never reserves slots.
        self.assertEqual(_quota_for(["CN"], 5, None), 0)
        self.assertEqual(_quota_for(["CN", "US"], 4, None), 2)
        self.assertEqual(_quota_for(["CN", "US", "JP"], 3, None), 1)
        self.assertEqual(_quota_for(["CN", "US"], 3, 2), 2)

    def test_query_cache_is_isolated_from_caller_mutation(self) -> None:
        first = self.retriever.retrieve(["关税"], top_k=2)
        self.assertTrue(first["chunks"])
        original = first["chunks"][0]["text"]
        first["chunks"][0]["text"] = "mutated"
        second = self.retriever.retrieve(["关税"], top_k=2)
        self.assertEqual(second["chunks"][0]["text"], original)

    def test_rerank_failure_keeps_first_stage_order(self) -> None:
        def broken_rerank(query: str, docs: list[str], top_n: int):
            raise RuntimeError("reranker down")

        retriever = Retriever(
            self.retriever.index, self.cfg, embed_fn=fake_embed, rerank_fn=broken_rerank
        )
        result = retriever.retrieve(["关税"], top_k=2)
        self.assertFalse(result["diagnostics"]["rerank"]["applied"])
        self.assertIn("reranker down", result["diagnostics"]["rerank"]["error"])
        self.assertTrue(result["chunks"])
        self.assertIsNone(result["chunks"][0]["score_parts"]["rerank"])


class ArticleAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        corpus = root / "corpus"
        (corpus / "cn").mkdir(parents=True)
        (corpus / "cn" / "long.md").write_text(
            "<!-- document: 测试长法条 -->\n"
            "## 第一章 总则\n"
            "第一条 " + "甲" * 200 + "。\n\n"
            "第二条 乙。\n\n"
            "第三条 丙。\n",
            encoding="utf-8",
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": "test",
                    "docs": [
                        {
                            "doc_id": "cn-long-law",
                            "path": "cn/long.md",
                            "source_doc": "测试长法条",
                            "jurisdiction": "CN",
                            "language": "zh",
                            "precision": "exact",
                            "applies_to": ["customs"],
                            "effective_from": "2024-01-01",
                            "effective_to": None,
                            "snapshot_date": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.cfg = AppConfig(
            corpus_dir=str(corpus),
            index_dir=str(root / "index"),
            manifest_path=str(manifest_path),
            documents_dir=str(root / "documents"),
        )
        self.cfg.embedding.dim = 3
        self.cfg.chunking.max_chars = 120
        self.cfg.chunking.overlap = 20
        build_index(
            load_manifest(manifest_path),
            self.cfg,
            self.cfg.resolved_index_dir(),
            embed_fn=fake_embed,
        )
        self.index = open_current(self.cfg.resolved_index_dir())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parts_are_stitched_without_overlap(self) -> None:
        payload = self.index.article_payload("cn-long-law:1")
        self.assertEqual(len(payload["chunks"]), 2)
        self.assertEqual(payload["text"].count("甲"), 200)
        self.assertTrue(payload["text"].startswith("第一条"))
        self.assertEqual(len({c["chunk_id"] for c in payload["chunks"]}), 2)

    def test_neighbors_follow_source_order(self) -> None:
        payload = self.index.article_payload("cn-long-law:2", neighbors=1)
        self.assertEqual(
            [n["article_id"] for n in payload["neighbors"]["previous"]], ["cn-long-law:1"]
        )
        self.assertEqual(
            [n["article_id"] for n in payload["neighbors"]["next"]], ["cn-long-law:3"]
        )
        previous = payload["neighbors"]["previous"][0]
        self.assertEqual(previous["text"].count("甲"), 200)
        self.assertEqual(
            previous["chunk_ids"],
            [c["chunk_id"] for c in self.index.article_chunks("cn-long-law:1")],
        )
        self.assertIsNone(self.index.article_payload("cn-long-law:42"))


class ContextHeaderTests(unittest.TestCase):
    def _build(self, root: Path, header: bool):
        corpus = root / "corpus"
        (corpus / "cn").mkdir(parents=True)
        (corpus / "cn" / "chapter.md").write_text(
            "<!-- document: 测试关税法 -->\n"
            "## 第五章 税率\n"
            "第二条 进口货物的税率由国务院规定。\n",
            encoding="utf-8",
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": "test",
                    "docs": [
                        {
                            "doc_id": "cn-chapter-law",
                            "path": "cn/chapter.md",
                            "source_doc": "测试关税法",
                            "jurisdiction": "CN",
                            "language": "zh",
                            "precision": "exact",
                            "applies_to": ["customs"],
                            "effective_from": "2024-01-01",
                            "effective_to": None,
                            "snapshot_date": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        cfg = AppConfig(
            corpus_dir=str(corpus),
            index_dir=str(root / "index"),
            manifest_path=str(manifest_path),
            documents_dir=str(root / "documents"),
        )
        cfg.embedding.dim = 3
        cfg.chunking.index_context_header = header
        build_index(
            load_manifest(manifest_path), cfg, cfg.resolved_index_dir(), embed_fn=fake_embed
        )
        return open_current(cfg.resolved_index_dir())

    def test_header_keeps_chunk_id_and_text_but_enriches_index(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with_header = self._build(Path(a), True)
            without = self._build(Path(b), False)
            ids_on = [c["chunk_id"] for c in with_header.chunks]
            ids_off = [c["chunk_id"] for c in without.chunks]
            self.assertEqual(ids_on, ids_off)
            self.assertEqual(
                [c["text"] for c in with_header.chunks],
                [c["text"] for c in without.chunks],
            )
            self.assertEqual(with_header.chunks[0]["text"], "第二条 进口货物的税率由国务院规定。")
            terms = ["关税", "第五章"]
            candidates = set(with_header.chunk_ids)
            self.assertTrue(with_header.bm25_scores(candidates, terms, 1.5, 0.75))
            self.assertFalse(without.bm25_scores(candidates, terms, 1.5, 0.75))


class JurisdictionQuotaTests(unittest.TestCase):
    """The quota must be able to promote a jurisdiction outside the best top_k."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        corpus = root / "corpus"
        (corpus / "cn").mkdir(parents=True)
        (corpus / "us").mkdir(parents=True)
        (corpus / "cn" / "duty.md").write_text(
            "第一条 关税的税额。\n\n第二条 关税的减免。\n\n第三条 关税的征收。\n",
            encoding="utf-8",
        )
        (corpus / "us" / "entry.md").write_text(
            "## § 1.1 Scope.\n\nThis section covers entry procedures.\n",
            encoding="utf-8",
        )
        docs = [
            ("cn-duty-law", "cn/duty.md", "CN", "zh", "exact"),
            ("us-entry-1", "us/entry.md", "US", "en", "degraded"),
        ]
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": "test",
                    "docs": [
                        {
                            "doc_id": doc_id,
                            "path": path,
                            "source_doc": doc_id,
                            "jurisdiction": jur,
                            "language": lang,
                            "precision": precision,
                            "applies_to": ["customs"],
                            "effective_from": "2024-01-01",
                            "effective_to": None,
                            "snapshot_date": None,
                        }
                        for doc_id, path, jur, lang, precision in docs
                    ],
                }
            ),
            encoding="utf-8",
        )
        cfg = AppConfig(
            corpus_dir=str(corpus),
            index_dir=str(root / "index"),
            manifest_path=str(manifest_path),
            documents_dir=str(root / "documents"),
        )
        cfg.embedding.dim = 3
        build_index(
            load_manifest(manifest_path), cfg, cfg.resolved_index_dir(), embed_fn=fake_embed
        )
        self.retriever = Retriever(open_current(cfg.resolved_index_dir()), cfg, embed_fn=fake_embed)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_quota_promotes_second_jurisdiction(self) -> None:
        filters = {"jurisdictions": ["CN", "US"]}
        plain = self.retriever.retrieve(
            ["关税"], top_k=2, filters=filters, per_jurisdiction_k=0
        )
        self.assertEqual([c["jurisdictions"][0] for c in plain["chunks"]], ["CN", "CN"])
        self.assertIsNone(plain["diagnostics"]["per_jurisdiction_k"])

        quotad = self.retriever.retrieve(
            ["关税"], top_k=2, filters=filters, per_jurisdiction_k=1
        )
        self.assertEqual(quotad["diagnostics"]["per_jurisdiction_k"], 1)
        self.assertEqual(
            sorted(c["jurisdictions"][0] for c in quotad["chunks"]), ["CN", "US"]
        )
        # Two requested jurisdictions reserve a slot automatically.
        auto = self.retriever.retrieve(["关税"], top_k=2, filters=filters)
        self.assertEqual(auto["diagnostics"]["per_jurisdiction_k"], 1)
        self.assertEqual(sorted(c["jurisdictions"][0] for c in auto["chunks"]), ["CN", "US"])


if __name__ == "__main__":
    unittest.main()
