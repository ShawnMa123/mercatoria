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

    def test_wrapped_citation_form_is_accepted(self) -> None:
        result = answer_question(
            self.cfg,
            "关税怎么征收？",
            self.chunks,
            client=FakeChat(f"关税依法征收 [citation: {self.cid}]。"),
        )
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["citations"][0]["chunk_id"], self.cid)

    def test_wrapped_form_still_rejects_unknown_ids(self) -> None:
        result = answer_question(
            self.cfg, "问题", self.chunks, client=FakeChat("答案 [citation: 0123456789abcdef]")
        )
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


class RecordingHttp:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def post(self, path: str, body: dict, headers=None) -> dict:
        self.calls.append({"path": path, "body": body, "headers": headers})
        return {"choices": [{"message": {"content": "ok"}}]}


class ChatPayloadTests(unittest.TestCase):
    """The thinking switch must reach the gateway, not just the config."""

    def call(self, **kwargs) -> dict:
        client = ChatClient(AnswerConfig(**kwargs))
        http = RecordingHttp()
        client._http = http  # type: ignore[assignment]
        client.chat([{"role": "user", "content": "关税"}])
        return http.calls[0]["body"]

    def test_thinking_disabled_sends_the_flag(self) -> None:
        body = self.call(enable_thinking=False)
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_thinking_enabled_sends_no_flag(self) -> None:
        self.assertNotIn("thinking", self.call(enable_thinking=True))

    def test_other_request_fields_are_untouched(self) -> None:
        body = self.call(enable_thinking=False, model="local-wb/cn:deepseek-v4.1-flash")
        self.assertEqual(body["model"], "local-wb/cn:deepseek-v4.1-flash")
        self.assertEqual(body["messages"][-1]["content"], "关税")
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(body["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
