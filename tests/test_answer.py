from __future__ import annotations

import unittest

from mercatoria.answer import ChatClient, answer_question
from mercatoria.config import AnswerConfig


def chunk(cid: str = "a" * 16) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": "cn-test-law",
        "source_doc": "测试关税法",
        "text": "第一条 关税的征收适用本法。",
        "article_ids": ["cn-test-law:1"],
        "articles": ["第一条"],
    }


class FakeChat(ChatClient):
    def __init__(self, text: str):
        self._text = text
        self.cfg = AnswerConfig()

    def chat(self, messages):  # type: ignore[override]
        return self._text


class AnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = AnswerConfig(enabled=True)
        self.chunks = [chunk()]
        self.cid = self.chunks[0]["chunk_id"]

    def test_citation_is_parsed(self) -> None:
        result = answer_question(
            self.cfg,
            "关税怎么征收？",
            self.chunks,
            client=FakeChat(f"关税依法征收 [{self.cid}]。"),
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["citations"][0]["chunk_id"], self.cid)

    def test_refusal_without_citation_is_unavailable(self) -> None:
        result = answer_question(
            self.cfg, "未知问题", self.chunks, client=FakeChat("依据不足，语料没有相关规定。")
        )
        self.assertEqual(result["answer_status"], "unavailable")
        self.assertEqual(result["citations"], [])

    def test_no_chunks_is_unavailable(self) -> None:
        result = answer_question(self.cfg, "问题", [])
        self.assertEqual(result["answer_status"], "unavailable")

    def test_disabled_skips_llm(self) -> None:
        cfg = AnswerConfig(enabled=False)
        result = answer_question(cfg, "问题", self.chunks)
        self.assertEqual(result["answer_status"], "skipped")
        self.assertIsNone(result["answer"])

    def test_unknown_citation_is_not_reported(self) -> None:
        result = answer_question(
            self.cfg, "问题", self.chunks, client=FakeChat("答案 [0123456789abcdef]")
        )
        self.assertEqual(result["answer_status"], "unsupported")
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
