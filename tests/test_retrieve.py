from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mercatoria.config import AppConfig
from mercatoria.manifest import load_manifest
from mercatoria.retrieve import Retriever
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

    def test_rerank_reorders_the_first_stage(self) -> None:
        def fake_rerank(query: str, docs: list[str], top_n: int):
            pairs = []
            for i, doc in enumerate(docs):
                score = 10.0 if "原产地" in doc else (1.0 if "关税" in doc else 0.1)
                pairs.append((i, score))
            return pairs, 1.5

        retriever = Retriever(
            self.retriever.index, self.cfg, embed_fn=fake_embed, rerank_fn=fake_rerank
        )
        result = retriever.retrieve(["关税"], top_k=2)
        self.assertTrue(result["diagnostics"]["rerank"]["applied"])
        self.assertEqual(result["diagnostics"]["rerank"]["model"], self.cfg.rerank.model)
        self.assertIn("原产地", result["chunks"][0]["text"])
        self.assertIsNotNone(result["chunks"][0]["score_parts"]["rerank"])
        self.assertLessEqual(len(result["chunks"]), 2)

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


if __name__ == "__main__":
    unittest.main()
