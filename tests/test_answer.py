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
        self.messages = None

    def chat(self, messages):  # type: ignore[override]
        self.messages = messages
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

    def test_references_carry_full_original_text(self) -> None:
        result = answer_question(
            self.cfg,
            "关税怎么征收？",
            self.chunks,
            client=FakeChat(f"关税依法征收 [{self.cid}]。"),
        )
        self.assertEqual(len(result["references"]), 1)
        reference = result["references"][0]
        self.assertEqual(reference["chunk_id"], self.cid)
        self.assertEqual(reference["text"], self.chunks[0]["text"])
        self.assertTrue(reference["cited"])
        self.assertEqual(result["citations"][0]["text"], self.chunks[0]["text"])

    def test_history_is_forwarded_to_the_model(self) -> None:
        fake = FakeChat(f"关税依法征收 [{self.cid}]。")
        history = [
            {"role": "user", "content": "关税是什么？"},
            {"role": "assistant", "content": "关税是进口环节的税。"},
        ]
        answer_question(self.cfg, "它什么时候缴纳？", self.chunks, client=fake, history=history)
        roles = [m["role"] for m in fake.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertIn("关税是什么", fake.messages[1]["content"])
        self.assertIn("它什么时候缴纳", fake.messages[-1]["content"])

    def test_references_exist_when_answer_is_disabled(self) -> None:
        cfg = AnswerConfig(enabled=False)
        result = answer_question(cfg, "问题", self.chunks)
        self.assertEqual(result["answer_status"], "skipped")
        self.assertEqual(result["references"][0]["text"], self.chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
