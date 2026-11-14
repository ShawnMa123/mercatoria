"""M0 acceptance: reliable publication, snapshots, migration and write ownership.

Every case here is a failure path from the design doc (A01-A07, A11, A14, A20,
A28). They use the real Service/ControlStore code paths with a fake embedder and
a temporary data directory.
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from mercatoria.api import Service, create_app
from mercatoria.config import ConfigError, load_config, merge_config_update
from mercatoria.control import ControlConflict, ControlStore
from mercatoria.documents import DocumentRegistry
from mercatoria.lock import LockBusy
from mercatoria.manifest import load_manifest
from mercatoria.publication import PublicationCoordinator, legacy_index_version
from mercatoria.snapshot import SnapshotError, SnapshotHolder, build_snapshot
from mercatoria.store import build_index

CORPUS = "<!-- document: 测试关税法 -->\n第一条 关税的征收适用本法。\n"


def fake_embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        if "关税" in text or "tariff" in text.lower():
            rows.append([1.0, 0.0, 0.0])
        else:
            rows.append([0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def build_case(root: Path) -> tuple[object, dict]:
    """Minimal corpus + config in a temp root, with one published index."""
    corpus = root / "corpus"
    (corpus / "cn").mkdir(parents=True)
    (corpus / "cn" / "test.md").write_text(CORPUS, encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
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
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "corpus_dir": str(corpus),
                "index_dir": str(root / "index"),
                "manifest_path": str(manifest_path),
                "documents_dir": str(root / "documents"),
                "uploads_dir": str(root / "uploads"),
                "embedding": {"dim": 3, "batch_size": 2, "cache_path": ""},
                "rerank": {"enabled": False},
                "answer": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    build_index(load_manifest(manifest_path), cfg, cfg.resolved_index_dir(), embed_fn=fake_embed)
    return cfg, load_manifest(manifest_path)


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg, self.manifest = build_case(self.root)
        self.services: list[Service] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.close()
        self.tmp.cleanup()

    def service(self, cfg=None) -> Service:
        service = Service(cfg or self.cfg, embed_fn=fake_embed)
        self.services.append(service)
        return service

    def head(self) -> dict | None:
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            state = control.head()
            return state.describe() if state else None
        finally:
            control.close()

    # -- stage vs activate -------------------------------------------------

    def test_staged_build_changes_nothing_until_activated(self) -> None:
        """A02: activate=false never moves the head, before or after restart."""
        before = self.head()
        staged = build_index(
            self.manifest,
            self.cfg,
            self.cfg.resolved_index_dir(),
            embed_fn=fake_embed,
            activate=False,
        )
        self.assertFalse(staged["activated"])
        self.assertIsNone(staged["generation"])
        self.assertTrue(Path(staged["path"]).is_file())
        self.assertEqual(self.head(), before)

        # Restart: the head, not the newest index file, decides what serves.
        service = self.service()
        self.assertEqual(service.current().index_version, before["index_version"])

    def test_activated_build_advances_generation(self) -> None:
        before = self.head()
        published = build_index(
            self.manifest, self.cfg, self.cfg.resolved_index_dir(), embed_fn=fake_embed
        )
        self.assertTrue(published["activated"])
        after = self.head()
        self.assertGreater(after["generation"], before["generation"])
        self.assertEqual(after["index_version"], published["version"])
        self.assertEqual(published["snapshot_version"], str(after["generation"]))

    def test_publish_rejects_stale_base_generation(self) -> None:
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            current = control.head_generation()
            with self.assertRaises(ControlConflict):
                control.publish(
                    index_version="nope",
                    manifest=self.manifest,
                    build_profile_id="x",
                    runtime_config_id="y",
                    base_generation=(current or 0) + 5,
                )
            self.assertEqual(control.head_generation(), current)
        finally:
            control.close()

    def test_restart_follows_the_committed_head(self) -> None:
        """A12: recovery is a function of the committed head, never of memory."""
        service = self.service()
        first = service.current().index_version
        published = build_index(
            self.manifest, self.cfg, self.cfg.resolved_index_dir(), embed_fn=fake_embed
        )
        # The process dies before (or right after) swapping its snapshot.
        service.close()
        self.services.remove(service)

        restarted = self.service()
        self.assertEqual(restarted.current().index_version, published["version"])
        self.assertNotEqual(restarted.current().index_version, first)
        self.assertEqual(self.head()["generation"], published["generation"])
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            self.assertEqual(len(control.states()), published["generation"])
        finally:
            control.close()

    # -- failed loads keep the old snapshot --------------------------------

    def test_failed_reload_keeps_serving_the_old_snapshot(self) -> None:
        """A01: a candidate that cannot load must not replace a working index."""
        service = self.service()
        old = service.current()
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            control.publish(
                index_version="missing-index",
                manifest=self.manifest,
                build_profile_id=self.cfg.chunking_hash(),
                runtime_config_id=self.cfg.runtime_fingerprint(),
            )
        finally:
            control.close()

        service.reload()
        self.assertIs(service.current(), old)
        self.assertIn("missing-index", service.error)
        result = service.current().retriever.retrieve(queries=["关税"])
        self.assertTrue(result["chunks"])

    def test_publish_refuses_foreign_embedding_space(self) -> None:
        """A07: same dimensions do not make it the same space."""
        other = copy.deepcopy(self.cfg)
        other.embedding.model = "someone/other-embedding"
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            before = control.head()
            coordinator = PublicationCoordinator(
                control, other, other.resolved_index_dir()
            )
            with self.assertRaises(SnapshotError):
                coordinator.publish(
                    before.index_version,
                    before.manifest,
                    other,
                    expected_generation=before.generation,
                )
            self.assertEqual(control.head_generation(), before.generation)
        finally:
            control.close()

    # -- configuration -----------------------------------------------------

    def test_nested_config_update_keeps_dataclass_types(self) -> None:
        """A05: a nested dict replaces fields, not the dataclass itself."""
        service = self.service()
        updated = merge_config_update(
            service.cfg, {"retrieval": {"confidence": {"high": 0.9}}}
        )
        self.assertEqual(updated.retrieval.confidence.high, 0.9)
        self.assertTrue(hasattr(updated.retrieval.confidence, "medium"))
        self.assertNotEqual(service.cfg.retrieval.confidence.high, 0.9)

    def test_config_endpoint_rejects_unknown_and_wrong_typed_fields(self) -> None:
        service = self.service()
        app = create_app(str(self.root / "config.json"), embed_fn=fake_embed, service=service)
        client = TestClient(app)
        before = service.cfg.retrieval.top_k

        unknown = client.put("/v1/config", json={"retrieval": {"nope": 1}})
        self.assertEqual(unknown.status_code, 422)
        wrong_type = client.put("/v1/config", json={"retrieval": {"top_k": "5"}})
        self.assertEqual(wrong_type.status_code, 422)
        bool_as_string = client.put("/v1/config", json={"rerank": {"enabled": "false"}})
        self.assertEqual(bool_as_string.status_code, 422)
        secret = client.put("/v1/config", json={"embedding": {"api_key": "sk-plaintext"}})
        self.assertEqual(secret.status_code, 422)
        self.assertNotEqual(service.cfg.embedding.api_key, "sk-plaintext")
        # A rejected update applies nothing at all.
        self.assertEqual(service.cfg.retrieval.top_k, before)
        self.assertFalse(service.cfg.rerank.enabled)

        ok = client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(service.cfg.retrieval.top_k, 3)

    def test_config_error_type(self) -> None:
        self.assertTrue(issubclass(ConfigError, ValueError))

    def test_timeout_change_rebuilds_the_embedding_client(self) -> None:
        """A06: a client-parameter change must reach the client, not just cfg."""
        # No injected embedder: the point of the test is the real client object.
        service = Service(self.cfg, embed_fn=None)
        self.services.append(service)
        before = service.require_retriever().embed.__self__._http
        app = create_app(str(self.root / "config.json"), embed_fn=None, service=service)
        resp = TestClient(app).put("/v1/config", json={"embedding": {"timeout": 5}})
        self.assertEqual(resp.status_code, 200)
        after = service.require_retriever().embed.__self__._http
        self.assertIsNot(after, before)
        self.assertEqual(after.timeout, 5)

    def test_embedding_endpoint_change_waits_for_a_rebuild(self) -> None:
        """R09: base is space identity, not a hot-swappable client setting."""
        service = Service(self.cfg, embed_fn=None)
        self.services.append(service)
        before = service.require_retriever().embed.__self__._http
        app = create_app(str(self.root / "config.json"), embed_fn=None, service=service)
        resp = TestClient(app).put(
            "/v1/config", json={"embedding": {"base": "http://127.0.0.1:9999/v1"}}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["pending_rebuild"])
        after = service.require_retriever().embed.__self__._http
        self.assertIs(after, before)
        self.assertEqual(service.cfg.embedding.base, "http://127.0.0.1:9999/v1")

    # -- one snapshot per request -----------------------------------------

    def test_lease_keeps_a_snapshot_alive_across_a_publish(self) -> None:
        """A11: an in-flight request never mixes two states."""
        service = self.service()
        holder: SnapshotHolder = service.snapshots
        first = holder.current()
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            state = control.head()
        finally:
            control.close()
        second = build_snapshot(state, self.cfg, self.cfg.resolved_index_dir(), fake_embed)
        with holder.lease() as leased:
            self.assertIs(leased, first)
            holder.install(second)
            # The old snapshot is retired but still usable by the in-flight
            # request that pinned it.
            self.assertFalse(first.closed)
            self.assertTrue(leased.retriever.retrieve(queries=["关税"])["chunks"])
        self.assertTrue(first.closed)

    # -- write ownership ---------------------------------------------------

    def test_second_instance_is_refused(self) -> None:
        """A14/R07: only the owner may run; no replica that never follows."""
        owner = self.service()
        self.assertIsNotNone(owner.current())
        with self.assertRaises(LockBusy):
            Service(self.cfg, embed_fn=fake_embed)

    def test_cli_rebuild_refuses_a_locked_directory(self) -> None:
        from mercatoria.cli import main

        self.service()
        code = main(["--config", str(self.root / "config.json"), "rebuild"])
        self.assertEqual(code, 1)

    # -- empty library -----------------------------------------------------

    def test_empty_library_is_published_and_ready(self) -> None:
        """A20: a published empty index differs from an unbuilt one."""
        empty = {"manifest_version": "test", "corpus_dir": ".", "documents": 0, "docs": []}
        other_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(other_tmp.cleanup)
        other_root = Path(other_tmp.name)
        cfg, _ = build_case(other_root)
        build_index(empty, cfg, cfg.resolved_index_dir(), embed_fn=fake_embed)

        service = Service(cfg, embed_fn=fake_embed)
        self.services.append(service)
        app = create_app(str(other_root / "config.json"), embed_fn=fake_embed, service=service)
        client = TestClient(app)

        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ready")
        self.assertEqual(health.json()["chunks"], 0)
        retrieved = client.post("/v1/retrieve", json={"queries": ["关税"]})
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.json()["chunks"], [])
        chat = client.post("/v1/chat", json={"question": "关税怎么征收？", "answer": {"enabled": False}})
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(chat.json()["answer_status"], "unavailable")

    def test_unbuilt_library_is_not_ready(self) -> None:
        empty_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(empty_tmp.cleanup)
        root = Path(empty_tmp.name)
        corpus = root / "corpus"
        corpus.mkdir()
        (root / "config.json").write_text(
            json.dumps(
                {
                    "corpus_dir": str(corpus),
                    "index_dir": str(root / "index"),
                    "manifest_path": str(root / "manifest.json"),
                    "documents_dir": str(root / "documents"),
                    "uploads_dir": str(root / "uploads"),
                    "embedding": {"dim": 3},
                }
            ),
            encoding="utf-8",
        )
        service = Service(load_config(root / "config.json"), embed_fn=fake_embed)
        self.services.append(service)
        app = create_app(str(root / "config.json"), embed_fn=fake_embed, service=service)
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 503)
        self.assertEqual(client.get("/health").json()["status"], "not_built")

    # -- migration ---------------------------------------------------------

    def test_legacy_index_is_imported_once(self) -> None:
        """A28: adopting an old deployment is idempotent and non-destructive."""
        version = legacy_index_version(self.cfg.resolved_index_dir())
        self.assertIsNotNone(version)
        self.cfg.resolved_control_path().unlink()  # pretend: pre-control install

        first = self.service()
        self.assertEqual(first.current().index_version, version)
        self.assertEqual(self.head()["generation"], 1)
        first.close()
        self.services.remove(first)

        second = self.service()
        self.assertEqual(self.head()["generation"], 1)
        self.assertEqual(second.current().index_version, version)

    def test_broken_legacy_index_is_not_adopted(self) -> None:
        """A28: fail closed instead of publishing an index that cannot load."""
        index_dir = self.cfg.resolved_index_dir()
        version = legacy_index_version(index_dir)
        (index_dir / f"index-{version}.db").write_bytes(b"not a database")
        self.cfg.resolved_control_path().unlink()

        service = self.service()
        self.assertIsNone(service.current())
        self.assertIsNone(self.head())
        self.assertTrue(service.error)


class DocumentLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg, _manifest = build_case(self.root)
        self.app = create_app(str(self.root / "config.json"), embed_fn=fake_embed)
        self.client = TestClient(self.app)
        self.service = self.app.state.service

    def tearDown(self) -> None:
        self.service.close()
        self.tmp.cleanup()

    def _wait_job(self, job_id: str, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.client.get(f"/v1/jobs/{job_id}").json()
            if job["status"] != "running":
                return job
            time.sleep(0.02)
        self.fail(f"job did not finish: {job_id}")

    def _upload(self) -> str:
        resp = self.client.post(
            "/v1/documents",
            files=[("files", ("life.md", b"## Test\n\nfirst article about tariffs\n", "text/markdown"))],
            data={"jurisdiction": "CN"},
        )
        job = self._wait_job(resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        return job["result"]["documents"][0]["doc_id"]

    def test_delete_is_logical_and_keeps_files(self) -> None:
        """A03: a publish-time delete must not destroy the source revision."""
        doc_id = self._upload()
        record = self.service.registry.get_raw(doc_id)
        original = Path(record["original_path"])
        parsed = Path(record["parsed_path"])
        self.assertTrue(original.is_file())

        resp = self.client.delete(f"/v1/documents/{doc_id}")
        job = self._wait_job(resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)

        self.assertTrue(original.is_file())
        self.assertTrue(parsed.is_file())
        self.assertEqual(self.service.registry.get_raw(doc_id)["status"], "deleted")
        self.assertNotIn(doc_id, self.service.current().index.docs)
        self.assertEqual(self.client.get("/v1/documents").json()["count"], 0)
        self.assertEqual(self.client.get(f"/v1/documents/{doc_id}").status_code, 404)

    def test_failed_delete_restores_the_document(self) -> None:
        """A03: a failed rebuild leaves the old document live and intact."""
        doc_id = self._upload()
        record = self.service.registry.get_raw(doc_id)
        before = self.service.current().snapshot_version
        with mock.patch("mercatoria.ingest.build_index", side_effect=RuntimeError("boom")):
            resp = self.client.delete(f"/v1/documents/{doc_id}")
            job = self._wait_job(resp.json()["job_id"])

        self.assertEqual(job["status"], "error")
        self.assertIn("boom", job["error"])
        self.assertTrue(Path(record["original_path"]).is_file())
        self.assertEqual(self.service.registry.get(doc_id)["status"], "ready")
        self.assertEqual(self.service.current().snapshot_version, before)
        self.assertEqual(self.client.get("/v1/documents").json()["count"], 1)

    def test_failed_metadata_update_restores_metadata(self) -> None:
        """A04: metadata and the served index must not disagree."""
        doc_id = self._upload()
        before = self.service.registry.get(doc_id)
        with mock.patch("mercatoria.ingest.build_index", side_effect=RuntimeError("boom")):
            resp = self.client.patch(
                f"/v1/documents/{doc_id}", json={"source_doc": "改名后的法规"}
            )
            job = self._wait_job(resp.json()["job_id"])

        self.assertEqual(job["status"], "error")
        after = self.service.registry.get(doc_id)
        self.assertEqual(after["source_doc"], before["source_doc"])
        self.assertEqual(after["status"], "ready")


if __name__ == "__main__":
    unittest.main()
