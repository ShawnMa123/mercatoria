from __future__ import annotations

import unittest

from mercatoria.chunker import (
    ChunkingConfig,
    chunk_document,
    make_chunk_id,
    normalize_for_id,
)


CN_DOC = {
    "doc_id": "cn-test-law",
    "jurisdiction": "CN",
    "language": "zh",
}


class ChunkIdTests(unittest.TestCase):
    def test_contract_example(self) -> None:
        cid = make_chunk_id(
            "cn-civil-code",
            "cn-civil-code:464",
            "第四百六十四条 合同是民事主体之间设立、变更、终止民事法律关系的协议。",
        )
        self.assertEqual(cid, "cb7a0919eb3459dd")

    def test_whitespace_and_nfc_are_normalized(self) -> None:
        a = make_chunk_id("d", "d:1", "第一条  内容")
        b = make_chunk_id("d", "d:1", "  第一条 内容\n")
        self.assertEqual(a, b)
        self.assertEqual(normalize_for_id("a  b"), "a b")

    def test_content_change_changes_id(self) -> None:
        a = make_chunk_id("d", "d:1", "第一条 内容")
        b = make_chunk_id("d", "d:1", "第一条 内容。")
        self.assertNotEqual(a, b)


class ChineseChunkTests(unittest.TestCase):
    def test_chapter_and_articles(self) -> None:
        raw = """<!-- document: 测试法 -->
## 第一章 总则

第一条 第一条内容。

第二条 第二条内容。

（一）第二条第（一）项。
"""
        recs = chunk_document(CN_DOC, raw, ChunkingConfig())
        self.assertEqual([r.article_id for r in recs], ["cn-test-law:1", "cn-test-law:2"])
        self.assertEqual(recs[0].article_label, "第一条")
        self.assertIn("第一条内容", recs[0].text)
        self.assertIn("第二条第（一）项", recs[1].text)
        self.assertNotIn("document:", recs[0].text)

    def test_chinese_suffix_article(self) -> None:
        raw = "第一条 甲。\n\n第一条之一 乙。\n\n第二条 丙。\n"
        recs = chunk_document(CN_DOC, raw, ChunkingConfig())
        self.assertEqual(
            [r.article_id for r in recs],
            ["cn-test-law:1", "cn-test-law:1.1", "cn-test-law:2"],
        )

    def test_long_article_is_split_with_same_article_id(self) -> None:
        raw = "第一条 " + "甲" * 120 + "。" + "乙" * 120 + "。"
        cfg = ChunkingConfig(max_chars=150, overlap=20)
        recs = chunk_document(CN_DOC, raw, cfg)
        self.assertGreater(len(recs), 1)
        self.assertEqual({r.article_id for r in recs}, {"cn-test-law:1"})
        self.assertEqual(len({r.chunk_id for r in recs}), len(recs))


class EnglishChunkTests(unittest.TestCase):
    def test_cisg_articles(self) -> None:
        doc = {"doc_id": "cisg", "jurisdiction": "INT", "language": "en"}
        raw = """## Article 1

(1) This Convention applies.

## Article 2

This Convention does not apply.
"""
        recs = chunk_document(doc, raw, ChunkingConfig())
        self.assertEqual([r.article_id for r in recs], ["cisg:art-1", "cisg:art-2"])
        self.assertIn("Article 1", recs[0].text)

    def test_ecfr_sections(self) -> None:
        doc = {"doc_id": "us-cfr-19-101", "jurisdiction": "US", "language": "en"}
        raw = "## § 101.0 Scope.\n\nText.\n\n## § 101.1 Definitions.\n\nMore.\n"
        recs = chunk_document(doc, raw, ChunkingConfig())
        self.assertEqual(
            [r.article_id for r in recs],
            ["us-cfr-19-101:101.0", "us-cfr-19-101:101.1"],
        )

    def test_inline_section_reference_is_not_an_article(self) -> None:
        doc = {"doc_id": "au-test", "jurisdiction": "AU", "language": "en"}
        raw = "## Section 1 — Short title\n\nThis Act may be cited. See section 50A for details.\n"
        recs = chunk_document(doc, raw, ChunkingConfig())
        self.assertEqual([r.article_id for r in recs], ["au-test:sec-1"])

    def test_australian_sections(self) -> None:
        doc = {"doc_id": "au-customs-act-1901", "jurisdiction": "AU", "language": "en"}
        raw = "## Section 1 — Short title\n\nThis Act.\n\n## Section 2 — Commencement\n\nStarts.\n"
        recs = chunk_document(doc, raw, ChunkingConfig())
        self.assertEqual(
            [r.article_id for r in recs],
            ["au-customs-act-1901:sec-1", "au-customs-act-1901:sec-2"],
        )


class DuplicateArticleTests(unittest.TestCase):
    def test_repeated_article_numbers_are_disambiguated(self) -> None:
        doc = {"doc_id": "jp-test", "jurisdiction": "JP", "language": "ja"}
        raw = "## 第一条\n\n甲。\n\n## 附則\n\n## 第一条\n\n乙。\n"
        recs = chunk_document(doc, raw, ChunkingConfig())
        self.assertEqual([r.article_id for r in recs], ["jp-test:1", "jp-test:1@2"])
        self.assertNotEqual(recs[0].chunk_id, recs[1].chunk_id)


class ChunkingOptionTests(unittest.TestCase):
    def test_include_heading_false(self) -> None:
        doc = {"doc_id": "cisg", "jurisdiction": "INT", "language": "en"}
        raw = "## Article 1\n\nText.\n"
        recs = chunk_document(doc, raw, ChunkingConfig(include_heading=False))
        self.assertEqual(len(recs), 1)
        self.assertFalse(recs[0].text.startswith("##"))
        self.assertIn("Article 1", recs[0].article_label)

    def test_max_chunks_per_article(self) -> None:
        raw = "第一条 " + "甲" * 500 + "。" + "乙" * 500 + "。"
        cfg = ChunkingConfig(max_chars=100, overlap=0, max_chunks_per_article=1)
        recs = chunk_document(CN_DOC, raw, cfg)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].article_id, "cn-test-law:1")

    def test_section_unit(self) -> None:
        raw = "## 第一章 总则\n\n第一条 甲。\n\n第二条 乙。\n"
        recs = chunk_document(CN_DOC, raw, ChunkingConfig(unit="section"))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].article_id, "cn-test-law:section-0")


class JapaneseChunkTests(unittest.TestCase):
    def test_section_headings_are_not_articles(self) -> None:
        doc = {"doc_id": "jp-kazei-hou", "jurisdiction": "JP", "language": "ja"}
        raw = "### Section 1 第一節 通則\n\n## 第一条\n\n本文。\n\n## 第二条の二\n\n本文。\n"
        recs = chunk_document(doc, raw, ChunkingConfig())
        self.assertEqual(
            [r.article_id for r in recs],
            ["jp-kazei-hou:1", "jp-kazei-hou:2.2"],
        )


if __name__ == "__main__":
    unittest.main()
