"""Tests for the v2 additions: switches, caches, confidence, references, schema."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mercatoria.config import AppConfig
from mercatoria.manifest import load_manifest
from mercatoria.retrieve import Retriever
from mercatoria.store import (
    SchemaTooNew,
    build_index,
    open_current,
)

EMBED_DIM = 3


def fake_embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        lower = text.lower()
        if "关税" in text or "tariff" in lower:
            rows.append([1.0, 0.0, 0.0])
        elif "原产地" in text or "origin" in lower:
            rows.append([0.0, 1.0, 0.0])
        else:
            rows.append([0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def write_corpus(root: Path, files: dict[str, str], docs: list[dict]) -> tuple[Path, Path]:
    corpus = root / "corpus"
    for relative, text in files.items():
        path = corpus / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": "test", "docs": docs}, ensure_ascii=False),
        encoding="utf-8",
    )
    return corpus, manifest_path


def doc_entry(doc_id: str, path: str, **overrides) -> dict:
    entry = {
        "doc_id": doc_id,
        "path": path,
        "source_doc": overrides.pop("source_doc", doc_id),
        "jurisdiction": overrides.pop("jurisdiction", "CN"),
        "language": overrides.pop("language", "zh"),
        "precision": overrides.pop("precision", "exact"),
        "applies_to": overrides.pop("applies_to", ["customs"]),
        "effective_from": overrides.pop("effective_from", "2024-01-01"),
        "effective_to": overrides.pop("effective_to", None),
        "snapshot_date": None,
    }
    entry.update(overrides)
    return entry


def build_case_corpus(root: Path, files: dict[str, str], docs: list[dict]) -> AppConfig:
    corpus, manifest_path = write_corpus(root, files, docs)
    cfg = AppConfig(
        corpus_dir=str(corpus),
        index_dir=str(root / "index"),
        manifest_path=str(manifest_path),
    )
    cfg.embedding.dim = EMBED_DIM
    cfg.embedding.query_cache_size = 8
    build_index(load_manifest(manifest_path), cfg, cfg.resolved_index_dir(), embed_fn=fake_embed)
    return cfg


REFERENCE_FILES = {
    "cn/ref.md": (
        "第一条 本法所称货物，包括一切进口货物。\n\n"
        "第二条 进口货物的税率依照本法第一条确定。\n\n"
        "第三条 前条规定的税率适用于保税货物，另见 § 1.2。\n\n"
        "第四条 依照本法第二条至第三条办理报关手续。\n"
    ),
    "wto/ref.md": (
        "## Article 1\n\nThis Agreement applies to goods.\n\n"
        "## Article 2\n\nSee Article 1 and section 5 for details.\n"
    ),
}


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cfg = build_case_corpus(
            cls.root,
            REFERENCE_FILES,
            [
                doc_entry("cn-ref-law", "cn/ref.md"),
                doc_entry(
                    "wto-ref",
                    "wto/ref.md",
                    jurisdiction="WTO",
                    language="en",
                    precision="snapshot",
                ),
            ],
        )
        cls.index = open_current(cls.cfg.resolved_index_dir())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_self_reference_resolves(self) -> None:
        refs = self.index.article_references("cn-ref-law:2")
        targets = [ref["article_id"] for ref in refs["resolved"]]
        self.assertEqual(targets, ["cn-ref-law:1"])

    def test_relative_and_unresolved(self) -> None:
        refs = self.index.article_references("cn-ref-law:3")
        self.assertEqual([r["article_id"] for r in refs["resolved"]], ["cn-ref-law:2"])
        self.assertEqual([r["text"] for r in refs["unresolved"]], ["§ 1.2"])

    def test_range_expands(self) -> None:
        refs = self.index.article_references("cn-ref-law:4")
        self.assertEqual(
            [r["article_id"] for r in refs["resolved"]],
            ["cn-ref-law:2", "cn-ref-law:3"],
        )

    def test_english_reference(self) -> None:
        refs = self.index.article_references("wto-ref:art-2")
        self.assertEqual([r["article_id"] for r in refs["resolved"]], ["wto-ref:art-1"])
        self.assertEqual([r["text"] for r in refs["unresolved"]], ["section 5"])

    def test_follow_refs_returns_summaries(self) -> None:
        payload = self.index.article_payload("cn-ref-law:2", follow_refs=1)
        self.assertEqual([item["article_id"] for item in payload["followed"]], ["cn-ref-law:1"])
        self.assertIn("货物", payload["followed"][0]["text"])
        self.assertFalse(payload["truncated"])

    def test_references_do_not_change_chunk_text(self) -> None:
        payload = self.index.article_payload("cn-ref-law:2")
        self.assertTrue(payload["text"].startswith("第二条"))


class ConfidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cfg = build_case_corpus(
            cls.root,
            {"cn/a.md": "第一条 关税的征收适用本法。\n\n第二条 原产地规则另行规定。\n"},
            [doc_entry("cn-a", "cn/a.md")],
        )
        cls.retriever = Retriever(
            open_current(cls.cfg.resolved_index_dir()), cls.cfg, embed_fn=fake_embed
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_confidence_is_present_and_low_for_nonsense(self) -> None:
        hit = self.retriever.retrieve(["关税的征收"], top_k=2)
        self.assertIn(hit["diagnostics"]["confidence"], {"high", "medium"})
        miss = self.retriever.retrieve(["zzzz"], top_k=2)
        self.assertEqual(miss["diagnostics"]["confidence"], "low")

    def test_timings_block(self) -> None:
        result = self.retriever.retrieve(["关税"], top_k=2)
        timings = result["diagnostics"]["timings_ms"]
        for stage in ("filter", "embed", "bm25", "vector", "rank", "total"):
            self.assertIn(stage, timings)

    def test_superseded_by_is_returned_and_warned(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = build_case_corpus(
            root,
            {"cn/old.md": "第一条 旧法关税规定。\n", "cn/new.md": "第一条 新法关税规定。\n"},
            [
                doc_entry(
                    "cn-old",
                    "cn/old.md",
                    effective_from="1997-01-01",
                    effective_to="2004-03-31",
                    superseded_by="cn-new",
                ),
                doc_entry("cn-new", "cn/new.md", effective_from="2004-04-01"),
            ],
        )
        retriever = Retriever(
            open_current(cfg.resolved_index_dir()), cfg, embed_fn=fake_embed
        )
        result = retriever.retrieve(
            ["关税"], top_k=5, filters={"jurisdictions": ["CN"], "as_of": "2000-01-01"}
        )
        old = next(c for c in result["chunks"] if c["doc_id"] == "cn-old")
        self.assertEqual(old["superseded_by"], "cn-new")
        self.assertTrue(any("cn-old" in warning for warning in result["diagnostics"]["warnings"]))


class CacheAndSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = build_case_corpus(
            self.root,
            {"cn/a.md": "第一条 关税的征收适用本法。\n\n第二条 原产地规则另行规定。\n"},
            [doc_entry("cn-a", "cn/a.md")],
        )
        self.calls: list[list[str]] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_retriever(self, embed_fn=None) -> Retriever:
        def counting(texts: list[str]) -> np.ndarray:
            self.calls.append(list(texts))
            return fake_embed(texts)

        return Retriever(
            open_current(self.cfg.resolved_index_dir()),
            self.cfg,
            embed_fn=embed_fn or counting,
        )

    def test_query_vector_cache_and_dedupe(self) -> None:
        retriever = self.make_retriever()
        first = retriever.retrieve(["关税"], top_k=2)
        self.assertEqual(first["diagnostics"]["embedding_cache_hits"], 0)
        second = retriever.retrieve(["关税"], top_k=2, fields={"doc_ids": ["cn-a"]})
        self.assertEqual(second["diagnostics"]["embedding_cache_hits"], 1)
        # Within one request duplicate queries are embedded once.
        retriever.retrieve(["原产地", "原产地"], top_k=2)
        self.assertEqual(len(self.calls[-1]), 1)

    def test_clear_cache_clears_query_vectors(self) -> None:
        retriever = self.make_retriever()
        retriever.retrieve(["关税"], top_k=2)
        retriever.clear_cache()
        again = retriever.retrieve(["关税"], top_k=2)
        self.assertEqual(again["diagnostics"]["embedding_cache_hits"], 0)

    def test_bm25_saturation_can_be_disabled(self) -> None:
        def tiny_embed(texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), EMBED_DIM), dtype=np.float32)

        retriever = self.make_retriever(tiny_embed)
        saturated = retriever.retrieve(["关税"], top_k=2)
        self.assertLess(saturated["chunks"][0]["score_parts"]["keyword_norm"], 1.0)
        self.cfg.retrieval.bm25_saturation = 0.0
        retriever.clear_cache()
        plain = retriever.retrieve(["关税"], top_k=2)
        self.assertAlmostEqual(plain["chunks"][0]["score_parts"]["keyword_norm"], 1.0)

    def test_embedding_failure_can_degrade_to_bm25(self) -> None:
        def broken(texts: list[str]) -> np.ndarray:
            raise RuntimeError("gateway down")

        strict = self.make_retriever(broken)
        with self.assertRaises(RuntimeError):
            strict.retrieve(["关税"], top_k=2)

        self.cfg.retrieval.fallback_bm25_on_embed_error = True
        lenient = self.make_retriever(broken)
        result = lenient.retrieve(["关税"], top_k=2)
        self.assertTrue(result["chunks"])
        self.assertEqual(result["diagnostics"]["degraded"], ["embedding"])

    def test_cache_key_tracks_new_switches(self) -> None:
        retriever = self.make_retriever()
        base = retriever._cache_key(["q"], 5, {}, {}, False)
        for name, value in (
            ("query_stopwords", False),
            ("bm25_saturation", 0.0),
            ("quota_min_ratio", 0.6),
        ):
            old = getattr(self.cfg.retrieval, name)
            setattr(self.cfg.retrieval, name, value)
            try:
                self.assertNotEqual(base, retriever._cache_key(["q"], 5, {}, {}, False), name)
            finally:
                setattr(self.cfg.retrieval, name, old)


class ConcurrencyTests(unittest.TestCase):
    """The retriever is shared by request threads: caches must be locked."""

    def test_parallel_retrieval_is_consistent(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = build_case_corpus(
                root,
                {"cn/a.md": "第一条 关税的征收适用本法。\n\n第二条 原产地规则另行规定。\n"},
                [doc_entry("cn-a", "cn/a.md")],
            )
            retriever = Retriever(
                open_current(cfg.resolved_index_dir()), cfg, embed_fn=fake_embed
            )
            prompts = ["关税", "原产地", "关税的征收", "保税货物的原产地"] * 5

            def call(prompt: str) -> list[str]:
                return [chunk["article_ids"][0] for chunk in retriever.retrieve([prompt], top_k=2)["chunks"]]

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(call, prompts * 4))
            self.assertEqual(len(results), len(prompts) * 4)
            # Same prompt, same answer, regardless of which thread ran it.
            by_prompt: dict[str, set[tuple[str, ...]]] = {}
            for prompt, ids in zip(prompts * 4, results):
                by_prompt.setdefault(prompt, set()).add(tuple(ids))
            self.assertTrue(all(len(seen) == 1 for seen in by_prompt.values()))


class SchemaCompatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = build_case_corpus(
            self.root,
            {"cn/a.md": "第一条 关税的征收适用本法。\n"},
            [doc_entry("cn-a", "cn/a.md")],
        )
        self.db = self._current_db()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _current_db(self) -> Path:
        index_dir = self.cfg.resolved_index_dir()
        name = (index_dir / "current.txt").read_text(encoding="utf-8").strip()
        return index_dir / name

    def test_current_schema_is_reported(self) -> None:
        index = open_current(self.cfg.resolved_index_dir())
        self.assertEqual(index.schema_version, 2)
        self.assertFalse(index.schema_outdated)

    def test_old_index_falls_back(self) -> None:
        conn = sqlite3.connect(self.db)
        conn.execute("ALTER TABLE chunks DROP COLUMN index_token_count")
        conn.execute("ALTER TABLE docs DROP COLUMN superseded_by")
        conn.execute("DELETE FROM meta WHERE key IN ('schema_version','index_text_hash')")
        conn.commit()
        conn.close()
        index = open_current(self.cfg.resolved_index_dir())
        self.assertEqual(index.schema_version, 1)
        self.assertTrue(index.schema_outdated)
        self.assertEqual(index.chunks[0]["token_count"], index.chunks[0]["index_token_count"])

    def test_newer_schema_is_refused(self) -> None:
        conn = sqlite3.connect(self.db)
        conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        conn.commit()
        conn.close()
        with self.assertRaises(SchemaTooNew):
            open_current(self.cfg.resolved_index_dir())


if __name__ == "__main__":
    unittest.main()
