from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mercatoria.api import create_app
from mercatoria.config import load_config
from mercatoria.manifest import load_manifest
from mercatoria.store import build_index


def fake_embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        if "关税" in text or "tariff" in text.lower():
            rows.append([1.0, 0.0, 0.0])
        elif "原产地" in text or "origin" in text.lower():
            rows.append([0.0, 1.0, 0.0])
        else:
            rows.append([0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        (self.corpus / "cn").mkdir(parents=True)
        (self.corpus / "cn" / "test.md").write_text(
            "<!-- document: 测试关税法 -->\n第一条 关税的征收适用本法。\n",
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
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.config_path = self.root / "config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "corpus_dir": str(self.corpus),
                    "index_dir": str(self.root / "index"),
                    "manifest_path": str(self.manifest_path),
                    "documents_dir": str(self.root / "documents"),
                    "uploads_dir": str(self.root / "uploads"),
                    "embedding": {"dim": 3, "batch_size": 2, "api_key": "secret"},
                    "rerank": {"enabled": False, "api_key": "rerank-secret"},
                    "answer": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )
        cfg = load_config(self.config_path)
        build_index(load_manifest(self.manifest_path), cfg, cfg.resolved_index_dir(), embed_fn=fake_embed)
        self.app = create_app(str(self.config_path), embed_fn=fake_embed)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        # The service owns the control database, the registry and the data
        # directory lock; Windows cannot delete a temp directory with those
        # handles still open.
        self.app.state.service.close()
        self.tmp.cleanup()

    def test_health_and_retrieve(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ready")
        self.assertEqual(health.json()["config"]["embedding"]["api_key"], "***")
        self.assertEqual(health.json()["config"]["rerank"]["api_key"], "***")
        resp = self.client.post(
            "/v1/retrieve",
            json={"queries": ["关税"], "filters": {"jurisdictions": ["CN"]}},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["chunks"])
        self.assertEqual(body["chunks"][0]["article_ids"], ["cn-test-law:1"])
        self.assertEqual(body["field_matches"]["jurisdictions"]["requested"], ["CN"])
        alias = self.client.post("/retrieve", json={"queries": ["关税"]})
        self.assertEqual(alias.status_code, 200)
        self.assertTrue(alias.json()["chunks"])

    def test_ask_without_llm(self) -> None:
        resp = self.client.post("/v1/ask", json={"question": "关税怎么征收？"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["answer_status"], "skipped")
        self.assertTrue(body["chunks"])
        self.assertFalse(body["coverage"]["unavailable"])

    def test_chat_accepts_messages_and_returns_references(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            json={
                "messages": [{"role": "user", "content": "关税怎么征收？"}],
                "answer": {"enabled": False},
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["question"], "关税怎么征收？")
        self.assertEqual(body["answer_status"], "skipped")
        self.assertTrue(body["references"])
        self.assertEqual(body["references"][0]["text"], "第一条 关税的征收适用本法。")
        self.assertFalse(body["references"][0]["cited"])

    def test_chat_rejects_messages_without_user(self) -> None:
        resp = self.client.post(
            "/v1/chat", json={"messages": [{"role": "assistant", "content": "你好"}]}
        )
        self.assertEqual(resp.status_code, 400)

    def test_upload_runs_ingest_and_lists_chunks(self) -> None:
        resp = self.client.post(
            "/v1/documents",
            files=[("files", ("upload.md", b"## Test\n\nfirst article\n", "text/markdown"))],
            data={"jurisdiction": "CN", "applies_to": "customs"},
        )
        self.assertEqual(resp.status_code, 200)
        job_id = resp.json()["job_id"]
        job = self._wait_job(job_id)
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["indexed"], 1)

        listing = self.client.get("/v1/documents").json()
        self.assertEqual(listing["count"], 1)
        doc_id = listing["documents"][0]["doc_id"]
        self.assertEqual(listing["documents"][0]["status"], "ready")
        self.assertGreater(listing["documents"][0]["chunk_count"], 0)

        detail = self.client.get(f"/v1/documents/{doc_id}")
        self.assertEqual(detail.status_code, 200)
        chunks = self.client.get(f"/v1/documents/{doc_id}/chunks")
        self.assertEqual(chunks.status_code, 200)
        self.assertGreaterEqual(chunks.json()["total"], 1)
        self.assertIn("text", chunks.json()["chunks"][0])

        duplicate = self.client.post(
            "/v1/documents",
            files=[("files", ("upload.md", b"## Test\n\nfirst article\n", "text/markdown"))],
            data={"jurisdiction": "CN"},
        )
        self.assertEqual(duplicate.status_code, 200)
        dup_job = self._wait_job(duplicate.json()["job_id"])
        self.assertEqual(dup_job["result"]["indexed"], 0)
        self.assertTrue(dup_job["result"]["documents"][0]["duplicate"])

    def test_batch_size_limit(self) -> None:
        service = self.app.state.service
        old = service.cfg.max_batch_bytes
        service.cfg.max_batch_bytes = 10
        try:
            resp = self.client.post(
                "/v1/documents",
                files=[
                    ("files", ("a.md", b"0123456789", "text/markdown")),
                    ("files", ("b.md", b"0123456789", "text/markdown")),
                ],
                data={"jurisdiction": "CN"},
            )
            self.assertEqual(resp.status_code, 413)
        finally:
            service.cfg.max_batch_bytes = old

    def test_update_document_metadata_rebuilds(self) -> None:
        resp = self.client.post(
            "/v1/documents",
            files=[("files", ("meta.md", b"## Test\n\nfirst article\n", "text/markdown"))],
            data={"jurisdiction": "CN"},
        )
        job = self._wait_job(resp.json()["job_id"])
        doc_id = job["result"]["documents"][0]["doc_id"]
        updated = self.client.patch(
            f"/v1/documents/{doc_id}",
            json={"applies_to": "guarantee", "source_doc": "改名后的法规"},
        )
        self.assertEqual(updated.status_code, 200)
        update_job = self._wait_job(updated.json()["job_id"])
        self.assertEqual(update_job["status"], "done")
        detail = self.client.get(f"/v1/documents/{doc_id}").json()
        self.assertEqual(detail["applies_to"], ["guarantee"])
        self.assertEqual(detail["source_doc"], "改名后的法规")

    def test_delete_document_rebuilds_without_it(self) -> None:
        resp = self.client.post(
            "/v1/documents",
            files=[("files", ("delete-me.md", b"## Test\n\nfirst article\n", "text/markdown"))],
            data={"jurisdiction": "CN"},
        )
        job = self._wait_job(resp.json()["job_id"])
        self.assertEqual(job["status"], "done")
        doc_id = job["result"]["documents"][0]["doc_id"]
        deleted = self.client.delete(f"/v1/documents/{doc_id}")
        self.assertEqual(deleted.status_code, 200)
        delete_job = self._wait_job(deleted.json()["job_id"])
        self.assertEqual(delete_job["status"], "done")
        self.assertEqual(self.client.get("/v1/documents").json()["count"], 0)
        self.assertEqual(self.client.get(f"/v1/documents/{doc_id}").status_code, 404)

    def _wait_job(self, job_id: str, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.client.get(f"/v1/jobs/{job_id}").json()
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        self.fail(f"job did not finish: {job_id}")

    def test_config_pending_rebuild(self) -> None:
        resp = self.client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["pending_rebuild"])
        resp = self.client.put("/v1/config", json={"chunking": {"max_chars": 512}})
        self.assertTrue(resp.json()["pending_rebuild"])
        # A retrieval-only change must not clear a pending chunking change.
        resp = self.client.put("/v1/config", json={"retrieval": {"top_k": 4}})
        self.assertTrue(resp.json()["pending_rebuild"])

    def test_rebuild_endpoint(self) -> None:
        resp = self.client.post("/v1/admin/rebuild", json={"background": False})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "done")
        self.assertEqual(body["result"]["documents"], 1)
        indexes = self.client.get("/v1/admin/indexes").json()
        self.assertTrue(indexes["indexes"])
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_config_update_replaces_serving_snapshot(self) -> None:
        self.client.post("/v1/retrieve", json={"queries": ["关税"]})
        service = self.app.state.service
        before = service.require_retriever()
        before_snapshot = service.current().snapshot_version
        self.assertTrue(before._cache)
        resp = self.client.put("/v1/config", json={"retrieval": {"bm25_k1": 2.0}})
        self.assertEqual(resp.status_code, 200)
        # A config change publishes a new snapshot over the same index: the
        # old retriever is retired rather than mutated under a live request.
        after = service.require_retriever()
        self.assertIsNot(after, before)
        self.assertFalse(after._cache)
        self.assertEqual(after.cfg.retrieval.bm25_k1, 2.0)
        self.assertNotEqual(resp.json()["snapshot_version"], before_snapshot)
        self.assertEqual(resp.json()["active_index"], service.current().index_version)

    def test_config_update_republishes_rerank_flag(self) -> None:
        service = self.app.state.service
        self.assertFalse(service.require_retriever().rerank_enabled)
        self.client.put("/v1/config", json={"rerank": {"enabled": True}})
        self.assertTrue(service.require_retriever().rerank_enabled)
        self.client.put("/v1/config", json={"rerank": {"enabled": False}})
        self.assertFalse(service.require_retriever().rerank_enabled)

    def test_article_endpoint_returns_full_text(self) -> None:
        resp = self.client.get("/v1/articles/cn-test-law:1")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["article_id"], "cn-test-law:1")
        self.assertEqual(body["doc_id"], "cn-test-law")
        self.assertIn("关税的征收", body["text"])
        expected = self.app.state.service.index.articles["cn-test-law:1"][0]["chunk_id"]
        self.assertEqual([c["chunk_id"] for c in body["chunks"]], [expected])
        self.assertEqual(body["chunks"][0]["text"], "第一条 关税的征收适用本法。")
        self.assertEqual(body["neighbors"], {"previous": [], "next": []})

    def test_article_endpoint_404_and_batch(self) -> None:
        self.assertEqual(self.client.get("/v1/articles/nope:1").status_code, 404)
        resp = self.client.post(
            "/v1/articles/batch",
            json={"article_ids": ["cn-test-law:1", "nope:2"], "neighbors": 0},
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["missing"], ["nope:2"])
        self.assertEqual(body["articles"][0]["article_id"], "cn-test-law:1")
        bad = self.client.get("/v1/articles/cn-test-law:1?neighbors=9")
        self.assertEqual(bad.status_code, 400)
        too_many = self.client.post(
            "/v1/articles/batch", json={"article_ids": [f"x:{i}" for i in range(33)]}
        )
        self.assertEqual(too_many.status_code, 400)

    def test_chat_rewrite_query_can_be_disabled(self) -> None:
        resp = self.client.post(
            "/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": "关税怎么征收？"},
                    {"role": "assistant", "content": "按法定税率。"},
                    {"role": "user", "content": "它还有别的规定吗？"},
                ],
                "rewrite_query": False,
                "answer": {"enabled": False},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["retrieval_query"], "它还有别的规定吗？")
        rewritten = self.client.post(
            "/v1/chat",
            json={
                "messages": [
                    {"role": "user", "content": "关税怎么征收？"},
                    {"role": "user", "content": "它还有别的规定吗？"},
                ],
                "answer": {"enabled": False},
            },
        )
        self.assertIn("关税怎么征收", rewritten.json()["retrieval_query"])

    def test_retrieve_accepts_per_jurisdiction_k(self) -> None:
        resp = self.client.post(
            "/v1/retrieve",
            json={
                "queries": ["关税"],
                "top_k": 2,
                "per_jurisdiction_k": 1,
                "filters": {"jurisdictions": ["CN"]},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["diagnostics"]["per_jurisdiction_k"], 1)
        bad = self.client.post(
            "/v1/retrieve", json={"queries": ["关税"], "per_jurisdiction_k": 101}
        )
        self.assertEqual(bad.status_code, 400)

    def test_manifest_endpoint(self) -> None:
        resp = self.client.post("/v1/admin/manifest")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["documents"], 1)


if __name__ == "__main__":
    unittest.main()
