from __future__ import annotations

import unittest
from unittest import mock

from mercatoria.config import RerankConfig
from mercatoria.http_client import HttpError
from mercatoria.rerank import RerankClient, RerankError


class PayloadTests(unittest.TestCase):
    def test_dashscope_payload_uses_input_and_parameters(self) -> None:
        client = RerankClient(RerankConfig(style="dashscope", model="gte-rerank-v2"))
        body = client._payload("q", ["a", "b"], 2)
        self.assertEqual(body["input"], {"query": "q", "documents": ["a", "b"]})
        self.assertEqual(body["parameters"]["top_n"], 2)
        self.assertNotIn("query", body)

    def test_cohere_payload_is_flat(self) -> None:
        client = RerankClient(RerankConfig(style="cohere", model="rerank-1"))
        body = client._payload("q", ["a"], 1)
        self.assertEqual(body["query"], "q")
        self.assertEqual(body["documents"], ["a"])
        self.assertEqual(body["top_n"], 1)

    def test_parse_native_and_cohere_shapes(self) -> None:
        native = {"output": {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}}
        cohere = {"results": [{"index": 0, "relevance_score": 0.4}, {"index": 2, "score": 0.8}]}
        self.assertEqual(RerankClient._parse(native), [(1, 0.9), (0, 0.1)])
        self.assertEqual(RerankClient._parse(cohere), [(2, 0.8), (0, 0.4)])

    def test_parse_rejects_missing_results(self) -> None:
        with self.assertRaises(RerankError):
            RerankClient._parse({"output": {}})


class RequestTests(unittest.TestCase):
    def test_posts_bearer_and_returns_sorted_pairs(self) -> None:
        cfg = RerankConfig(
            style="dashscope",
            base="https://example.test/",
            path="/rerank",
            model="gte-rerank-v2",
            api_key="secret",
            retries=0,
        )
        client = RerankClient(cfg)
        payload = {"output": {"results": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.8}]}}
        with mock.patch.object(client._http, "post", return_value=payload) as posted:
            pairs, latency = client.rerank("q", ["a", "b"])
        self.assertEqual(pairs, [(1, 0.8), (0, 0.2)])
        self.assertGreaterEqual(latency, 0.0)
        path, body, headers = posted.call_args[0]
        self.assertEqual(path, "rerank")
        self.assertEqual(body["model"], "gte-rerank-v2")
        self.assertEqual(headers["Authorization"], "Bearer secret")

    def test_client_error_is_not_retried(self) -> None:
        cfg = RerankConfig(base="https://example.test", path="/rerank", retries=3)
        client = RerankClient(cfg)
        with mock.patch.object(
            client._http, "post", side_effect=HttpError(400, "invalid")
        ) as posted:
            with self.assertRaises(RerankError):
                client.rerank("q", ["a"])
        self.assertEqual(posted.call_count, 1)

    def test_empty_documents_skips_network(self) -> None:
        client = RerankClient(RerankConfig())
        with mock.patch.object(client._http, "post") as posted:
            pairs, latency = client.rerank("q", [])
        self.assertEqual(pairs, [])
        self.assertEqual(latency, 0.0)
        posted.assert_not_called()


if __name__ == "__main__":
    unittest.main()
