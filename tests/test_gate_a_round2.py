"""Gate A round 2: deployment config, embedding space, management, commit boundary.

Each test names the review item it closes (G01-G07) and uses only temporary
directories, a fake embedder and in-memory ASGI calls.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from test_publication import build_case, fake_embed

from mercatoria import config as config_module
from mercatoria.api import Service, create_app
from mercatoria.config import (
    EmbeddingConfig,
    deployment_config_payload,
    embedding_space_id,
    load_config,
    snapshot_config_payload,
)
from mercatoria.control import ControlStore
from mercatoria.embed import EmbeddingClient
from mercatoria.publication import prepare_snapshot
from mercatoria.snapshot import SnapshotError
from mercatoria.store import build_index, index_path, open_index

REPO = Path(__file__).resolve().parent.parent
SECRETS = ("sk-FAKE-deploy-11", "sk-FAKE-rerank-12", "sk-FAKE-answer-13")


class CountingEmbedder:
    """Deterministic fake embedder with a call counter."""

    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.calls = 0
        self.texts: list[str] = []

    def __call__(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        return np.asarray([self.vector for _ in texts], dtype=np.float32)


class GateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg, self.manifest = build_case(self.root)
        self.services: list[Service] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.close()

    def config_path(self) -> Path:
        return self.root / "config.json"

    def customize(self, **overrides) -> object:
        data = json.loads(self.config_path().read_text(encoding="utf-8"))
        data.update(overrides)
        self.config_path().write_text(json.dumps(data), encoding="utf-8")
        self.cfg = load_config(self.config_path())
        return self.cfg

    def app(self, cfg=None, embed_fn=fake_embed):
        service = Service(
            cfg or self.cfg, embed_fn=embed_fn, config_path=self.config_path()
        )
        self.services.append(service)
        app = create_app(str(self.config_path()), embed_fn=embed_fn, service=service)
        return TestClient(app), service

    def read_head(self, cfg=None) -> dict | None:
        store = ControlStore((cfg or self.cfg).resolved_control_path())
        try:
            state = store.head()
            return state.describe() if state else None
        finally:
            store.close()

    def head_state(self, cfg=None):
        store = ControlStore((cfg or self.cfg).resolved_control_path())
        try:
            return store.head()
        finally:
            store.close()

    def wait_job(self, client, job_id: str, timeout: float = 20.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = client.get(f"/v1/jobs/{job_id}").json()
            if job["status"] != "running":
                return job
            time.sleep(0.02)
        self.fail(f"job did not finish: {job_id}")

    def upload(self, client, name: str = "doc.md") -> str:
        resp = client.post(
            "/v1/documents",
            files=[("files", (name, b"## X\n\narticle body\n", "text/markdown"))],
            data={"jurisdiction": "CN"},
        )
        job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        return job["result"]["documents"][0]["doc_id"]


# --------------------------------------------------------------------- G01
class DeploymentConfigTests(GateTestCase):
    def _customized(self) -> object:
        manifest_copy = self.root / "manifest-custom.json"
        manifest_copy.write_text(
            self.cfg.resolved_manifest_path().read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        cfg = self.customize(
            corpus_dir=str(self.root / "corpus"),
            index_dir=str(self.root / "index-custom"),
            manifest_path=str(manifest_copy),
            uploads_dir=str(self.root / "uploads-custom"),
            documents_dir=str(self.root / "documents-custom"),
            max_upload_bytes=1_234_567,
            max_batch_bytes=7_654_321,
            ingest_workers=3,
        )
        # The fixture state must live in the customized directories.
        build_index(
            self.manifest, cfg, cfg.resolved_index_dir(), embed_fn=fake_embed
        )
        return cfg

    def test_save_preserves_deployment_paths_and_limits(self) -> None:
        """G01: saving must not rewrite the file into the snapshot sections."""
        cfg = self._customized()
        cfg.embedding.api_key = SECRETS[0]
        cfg.rerank.api_key = SECRETS[1]
        cfg.answer.api_key = SECRETS[2]
        client, service = self.app(cfg=cfg)
        resp = client.put("/v1/config", json={"retrieval": {"top_k": 3}, "save": True})
        self.assertEqual(resp.status_code, 200, resp.text)

        reloaded = load_config(self.config_path())
        for field in (
            "corpus_dir",
            "index_dir",
            "manifest_path",
            "uploads_dir",
            "documents_dir",
            "max_upload_bytes",
            "max_batch_bytes",
            "ingest_workers",
        ):
            self.assertEqual(
                getattr(reloaded, field), getattr(cfg, field), f"{field} was not saved"
            )
        self.assertEqual(reloaded.retrieval.top_k, 3)
        raw = self.config_path().read_text(encoding="utf-8")
        for secret in SECRETS:
            self.assertNotIn(secret, raw)

    def test_restart_after_save_uses_same_control_store(self) -> None:
        """G01: the saved file must point at the same control/index/registry."""
        cfg = self._customized()
        client, service = self.app(cfg=cfg)
        client.put("/v1/config", json={"retrieval": {"top_k": 3}, "save": True})
        version = service.current().snapshot_version
        service.close()
        self.services.remove(service)

        reloaded = load_config(self.config_path())
        self.assertEqual(reloaded.resolved_control_path(), cfg.resolved_control_path())
        restarted = Service(
            reloaded, embed_fn=fake_embed, config_path=self.config_path()
        )
        self.services.append(restarted)
        self.assertEqual(restarted.current().snapshot_version, version)
        self.assertEqual(restarted.current().config.retrieval.top_k, 3)
        self.assertEqual(restarted.control.head_generation(), int(version))

    def test_save_failure_keeps_old_file_and_reports_committed(self) -> None:
        """G01/G04: a failed file export is a warning, not an undone commit."""
        cfg = self._customized()
        client, service = self.app(cfg=cfg)
        before = self.config_path().read_bytes()
        head_before = self.read_head(cfg)
        with mock.patch.object(config_module.os, "replace", side_effect=OSError("disk full")):
            resp = client.put("/v1/config", json={"retrieval": {"top_k": 3}, "save": True})
        self.assertEqual(resp.status_code, 200, resp.text)
        warnings = resp.json()["warnings"]
        self.assertTrue(any("save failed" in w for w in warnings), warnings)
        self.assertEqual(self.config_path().read_bytes(), before)
        # The accepted change is committed and serving.
        head_after = self.read_head(cfg)
        self.assertGreater(head_after["generation"], head_before["generation"])
        self.assertEqual(service.current().config.retrieval.top_k, 3)
        store = ControlStore(cfg.resolved_control_path())
        try:
            committed = store.head()
        finally:
            store.close()
        self.assertEqual(committed.config["retrieval"]["top_k"], 3)
        # And a restart still serves it, because the control state carries it.
        service.close()
        self.services.remove(service)
        restarted = Service(
            load_config(self.config_path()), embed_fn=fake_embed, config_path=self.config_path()
        )
        self.services.append(restarted)
        self.assertEqual(restarted.current().config.retrieval.top_k, 3)

    def test_deployment_payload_is_complete_and_secret_free(self) -> None:
        cfg = self._customized()
        cfg.embedding.api_key = SECRETS[0]
        payload = deployment_config_payload(cfg)
        for field in config_module.DEPLOYMENT_FIELDS:
            self.assertIn(field, payload)
        for section in config_module.PERSISTED_SECTIONS:
            self.assertNotIn("api_key", payload[section])
        snapshot = snapshot_config_payload(cfg)
        self.assertNotIn("corpus_dir", snapshot)


# --------------------------------------------------------------------- G02
class EmbeddingSpaceTests(GateTestCase):
    def build(self, cfg, embedder) -> dict:
        return build_index(
            self.manifest,
            cfg,
            cfg.resolved_index_dir(),
            embed_fn=embedder,
            activate=False,
        )

    def test_endpoint_rebuild_embeds_unchanged_documents(self) -> None:
        """G02: a new endpoint must not inherit vectors from the old space."""
        new_cfg = load_config(self.config_path())
        new_cfg.embedding.base = "http://127.0.0.1:9001/v2"
        embedder = CountingEmbedder([0.0, 1.0, 0.0])
        result = self.build(new_cfg, embedder)
        self.assertGreater(embedder.calls, 0, "new endpoint was never called")

        index = open_index(index_path(new_cfg.resolved_index_dir(), result["version"]))
        try:
            self.assertTrue(np.allclose(index.vectors, np.array([0.0, 1.0, 0.0])))
            self.assertEqual(
                index.embedding_space_id, embedding_space_id(new_cfg.embedding)
            )
        finally:
            index.close()

    def test_same_space_reuses_vectors(self) -> None:
        embedder = CountingEmbedder([1.0, 0.0, 0.0])
        self.build(self.cfg, embedder)
        self.assertEqual(embedder.calls, 0)

    def test_legacy_index_without_space_id_is_rebuilt(self) -> None:
        """G02: an index without a space id has no trustworthy identity."""
        path = index_path(self.cfg.resolved_index_dir(), self.read_head()["index_version"])
        conn = sqlite3.connect(path)
        try:
            conn.execute("DELETE FROM meta WHERE key='embedding_space_id'")
            conn.commit()
        finally:
            conn.close()
        embedder = CountingEmbedder([1.0, 0.0, 0.0])
        self.build(self.cfg, embedder)
        self.assertGreater(embedder.calls, 0)

    def test_persistent_cache_isolated_by_embedding_space(self) -> None:
        """G02: the durable cache is keyed by space, not by model name."""
        cache = self.root / "embedding-cache.db"
        first = EmbeddingConfig(
            base="http://a.example/v1", model="m", dim=3, cache_path=str(cache)
        )
        second = copy.deepcopy(first)
        second.base = "http://b.example/v1"
        client_a = EmbeddingClient(first)
        client_b = EmbeddingClient(second)
        calls: list[str] = []

        def fake_http(vector):
            class Fake:
                def post(self, path, body, headers=None):
                    calls.append(path)
                    return {
                        "data": [{"embedding": vector} for _ in body["input"]]
                    }

            return Fake()

        client_a._http = fake_http([1.0, 0.0, 0.0])
        client_b._http = fake_http([0.0, 1.0, 0.0])
        vector_a = client_a.embed(["same text"])[0]
        vector_b = client_b.embed(["same text"])[0]
        self.assertEqual(len(calls), 2, "second space reused the first cache entry")
        np.testing.assert_allclose(vector_a, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(vector_b, [0.0, 1.0, 0.0])

        # Same space, same text: a fresh client must hit the cache.
        again = EmbeddingClient(copy.deepcopy(first))
        again._http = fake_http([9.0, 9.0, 9.0])
        np.testing.assert_allclose(again.embed(["same text"])[0], [1.0, 0.0, 0.0])
        self.assertEqual(len(calls), 2)
        for client in (client_a, client_b, again):
            client._cache.close()

    def test_timeout_and_credentials_do_not_change_the_space(self) -> None:
        """G02: client parameters must not force re-embedding."""
        relaxed = load_config(self.config_path())
        relaxed.embedding.timeout = 5
        relaxed.embedding.retries = 1
        relaxed.embedding.api_key = SECRETS[0]
        self.assertEqual(
            embedding_space_id(relaxed.embedding),
            embedding_space_id(self.cfg.embedding),
        )
        embedder = CountingEmbedder([1.0, 0.0, 0.0])
        self.build(relaxed, embedder)
        self.assertEqual(embedder.calls, 0, "timeout/credential change re-embedded")

    def test_explicit_revision_and_space_version_change_the_space(self) -> None:
        base = embedding_space_id(self.cfg.embedding)
        revision = copy.deepcopy(self.cfg.embedding)
        revision.model_revision = "2026-01"
        version = copy.deepcopy(self.cfg.embedding)
        version.space_version = "operator-bump"
        endpoint = copy.deepcopy(self.cfg.embedding)
        endpoint.base = "http://127.0.0.1:9002/v1"
        ids = {base, embedding_space_id(revision), embedding_space_id(version), embedding_space_id(endpoint)}
        self.assertEqual(len(ids), 4, ids)

    def test_index_from_another_space_is_not_accepted(self) -> None:
        """G02: serving an index under a different space must fail closed."""
        head = self.read_head()
        other = load_config(self.config_path())
        other.embedding.base = "http://127.0.0.1:9003/v1"
        with self.assertRaises(SnapshotError) as raised:
            prepare_snapshot(
                head["index_version"],
                self.manifest,
                other,
                other.resolved_index_dir(),
                fake_embed,
            )
        self.assertIn("space", str(raised.exception))


# --------------------------------------------------------------------- G03
class ManagementOwnershipTests(GateTestCase):
    def _pausing_build(self):
        """A build_index wrapper that parks the worker after admission."""
        import mercatoria.ingest as ingest_module

        real = ingest_module.build_index
        paused = threading.Event()
        resume = threading.Event()

        def slow(*args, **kwargs):
            paused.set()
            resume.wait(20)
            return real(*args, **kwargs)

        return slow, paused, resume

    def test_upload_and_config_update_do_not_interleave(self) -> None:
        """G03: a job holding the inputs must make a config write 409, not race."""
        client, service = self.app()
        slow, paused, resume = self._pausing_build()
        with mock.patch("mercatoria.ingest.build_index", side_effect=slow):
            resp = client.post(
                "/v1/documents",
                files=[("files", ("g03.md", b"## X\n\narticle g03\n", "text/markdown"))],
                data={"jurisdiction": "CN"},
            )
            self.assertTrue(paused.wait(20), "upload never reached the build")
            head_before = self.read_head()
            busy = client.put("/v1/config", json={"retrieval": {"top_k": 3}})
            self.assertEqual(busy.status_code, 409, busy.text)
            self.assertEqual(busy.json()["error"]["code"], "management_busy")
            self.assertEqual(busy.json()["error"]["details"]["current"], "ingest")
            # Refused before changing anything.
            self.assertEqual(service.cfg.retrieval.top_k, 5)
            self.assertEqual(self.read_head(), head_before)
            resume.set()
            job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        self.assertIsNone(service.management.owner)

        # The upload published on the active config; the update now applies on
        # top of it without reverting the document set.
        doc_id = job["result"]["documents"][0]["doc_id"]
        applied = client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        self.assertEqual(applied.status_code, 200, applied.text)
        live = service.current()
        self.assertEqual(live.config.retrieval.top_k, 3)
        self.assertIn(doc_id, live.index.docs)
        head = self.head_state()
        self.assertEqual(head.config["retrieval"]["top_k"], 3)
        self.assertIn(doc_id, [doc["doc_id"] for doc in head.manifest["docs"]])

    def test_busy_rebuild_with_config_body_changes_nothing(self) -> None:
        """G03: a refused rebuild must not have applied its config body."""
        client, service = self.app()
        slow, paused, resume = self._pausing_build()
        with mock.patch("mercatoria.ingest.build_index", side_effect=slow):
            first = client.post("/v1/admin/rebuild", json={"background": True})
            self.assertTrue(paused.wait(20), "rebuild never reached the build")
            head_before = self.read_head()
            busy = client.post(
                "/v1/admin/rebuild",
                json={"background": False, "chunking": {"max_chars": 512}},
            )
            self.assertEqual(busy.status_code, 409, busy.text)
            self.assertEqual(service.cfg.chunking.max_chars, 2048)
            self.assertEqual(self.read_head(), head_before)
            resume.set()
            job = self.wait_job(client, first.json()["job_id"])
        self.assertEqual(job["status"], "done", job)

    def test_two_config_updates_are_serialized(self) -> None:
        """G03: concurrent config writes cannot lose one field silently."""
        client, service = self.app()
        entered = threading.Event()
        resume = threading.Event()
        real = service.coordinator.publish_runtime

        def slow(*args, **kwargs):
            entered.set()
            resume.wait(20)
            return real(*args, **kwargs)

        results: list[int] = []

        def first_update() -> None:
            results.append(
                client.put("/v1/config", json={"retrieval": {"top_k": 3}}).status_code
            )

        with mock.patch.object(service.coordinator, "publish_runtime", side_effect=slow):
            worker = threading.Thread(target=first_update)
            worker.start()
            self.assertTrue(entered.wait(20), "first update never paused")
            second = client.put("/v1/config", json={"answer": {"max_tokens": 4096}})
            self.assertEqual(second.status_code, 409, second.text)
            resume.set()
            worker.join(20)
        self.assertEqual(results, [200])

        retried = client.put("/v1/config", json={"answer": {"max_tokens": 4096}})
        self.assertEqual(retried.status_code, 200, retried.text)
        live = service.current()
        self.assertEqual(live.config.retrieval.top_k, 3)
        self.assertEqual(live.config.answer.max_tokens, 4096)
        self.assertEqual(service.cfg.retrieval.top_k, 3)
        self.assertEqual(service.cfg.answer.max_tokens, 4096)

    def test_thread_start_failure_releases_ownership(self) -> None:
        """G03: a worker that never starts must not leave the gate claimed."""
        client, service = self.app()

        class NoThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("no threads")

        with mock.patch("mercatoria.api.threading.Thread", NoThread):
            with self.assertRaises(RuntimeError):
                client.post(
                    "/v1/documents",
                    files=[("files", ("x.md", b"## X\n\nbody\n", "text/markdown"))],
                    data={"jurisdiction": "CN"},
                )
        self.assertIsNone(service.management.owner)
        ok = client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_precommit_failure_releases_ownership(self) -> None:
        client, service = self.app()
        doc_id = self.upload(client)
        with mock.patch("mercatoria.ingest.build_index", side_effect=RuntimeError("boom")):
            resp = client.delete(f"/v1/documents/{doc_id}")
            job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "error")
        self.assertIsNone(service.management.owner)
        self.assertEqual(service.registry.get_raw(doc_id)["status"], "ready")

    def test_reads_are_not_blocked_by_management(self) -> None:
        client, service = self.app()
        slow, paused, resume = self._pausing_build()
        with mock.patch("mercatoria.ingest.build_index", side_effect=slow):
            resp = client.post(
                "/v1/documents",
                files=[("files", ("r.md", b"## X\n\nreadable\n", "text/markdown"))],
                data={"jurisdiction": "CN"},
            )
            self.assertTrue(paused.wait(20))
            retrieved = client.post("/v1/retrieve", json={"queries": ["关税"]})
            self.assertEqual(retrieved.status_code, 200)
            # Readiness recovery is a committed-state read: it must not need the
            # management token the upload holds.
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertIsNotNone(service.management.owner)
            resume.set()
            self.wait_job(client, resp.json()["job_id"])


# --------------------------------------------------------------------- G04
class PostCommitBoundaryTests(GateTestCase):
    def test_final_progress_failure_does_not_undo_a_committed_delete(self) -> None:
        """G04: progress after the commit is a warning, not a pre-commit failure."""
        client, service = self.app()
        doc_id = self.upload(client)
        fired = {"n": 0}

        def progress(stage, done, total):
            fired["n"] += 1
            if stage == "done":
                raise OSError("progress sink broken")

        with mock.patch.object(service, "_progress", side_effect=lambda job: progress):
            resp = client.delete(f"/v1/documents/{doc_id}")
            job = self.wait_job(client, resp.json()["job_id"])

        self.assertEqual(job["status"], "done", job)
        self.assertTrue(any("progress" in w for w in job["result"]["warnings"]))
        self.assertEqual(service.registry.get_raw(doc_id)["status"], "deleted")
        head = self.head_state()
        self.assertNotIn(doc_id, [doc["doc_id"] for doc in head.manifest["docs"]])
        self.assertEqual(service.current().generation, head.generation)
        self.assertNotIn(doc_id, service.current().index.docs)

    def test_chunk_count_failure_after_commit_keeps_the_delete(self) -> None:
        client, service = self.app()
        doc_id = self.upload(client)
        with mock.patch(
            "mercatoria.ingest._chunk_counts", side_effect=RuntimeError("counts broken")
        ):
            resp = client.delete(f"/v1/documents/{doc_id}")
            job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        self.assertTrue(any("counts" in w for w in job["result"]["warnings"]))
        self.assertEqual(service.registry.get_raw(doc_id)["status"], "deleted")
        self.assertNotIn(doc_id, service.current().index.docs)

        # A restart rebuilds from the committed state, not from the failed
        # projection: the deleted document must not come back.
        service.close()
        self.services.remove(service)
        restarted = Service(
            load_config(self.config_path()), embed_fn=fake_embed, config_path=self.config_path()
        )
        self.services.append(restarted)
        self.assertNotIn(doc_id, restarted.current().index.docs)
        self.assertEqual(restarted.registry.get_raw(doc_id)["status"], "deleted")

    def test_registry_projection_failure_after_commit_is_a_warning(self) -> None:
        client, service = self.app()
        real_update = service.registry.update

        def failing_update(doc_id, **fields):
            if fields.get("status") == "ready":
                raise RuntimeError("registry write broken")
            return real_update(doc_id, **fields)

        with mock.patch.object(service.registry, "update", side_effect=failing_update):
            resp = client.post(
                "/v1/documents",
                files=[("files", ("p.md", b"## X\n\nprojected\n", "text/markdown"))],
                data={"jurisdiction": "CN"},
            )
            job = self.wait_job(client, resp.json()["job_id"])

        self.assertEqual(job["status"], "done", job)
        self.assertTrue(any("projection" in w for w in job["result"]["warnings"]))
        doc_id = job["result"]["documents"][0]["doc_id"]
        self.assertIn(doc_id, service.current().index.docs)
        self.assertEqual(service.current().generation, self.head_state().generation)

    def test_metadata_update_survives_a_post_commit_count_failure(self) -> None:
        client, service = self.app()
        doc_id = self.upload(client)
        with mock.patch(
            "mercatoria.ingest._chunk_counts", side_effect=RuntimeError("counts broken")
        ):
            resp = client.patch(f"/v1/documents/{doc_id}", json={"source_doc": "新名字"})
            job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(service.registry.get(doc_id)["source_doc"], "新名字")
        self.assertEqual(service.current().index.docs[doc_id]["source_doc"], "新名字")

    def test_reindex_survives_a_post_commit_count_failure(self) -> None:
        client, service = self.app()
        doc_id = self.upload(client)
        with mock.patch(
            "mercatoria.ingest._chunk_counts", side_effect=RuntimeError("counts broken")
        ):
            resp = client.post(f"/v1/documents/{doc_id}/reindex")
            job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        self.assertTrue(any("counts" in w for w in job["result"]["warnings"]))
        self.assertIn(doc_id, service.current().index.docs)

    def test_precommit_failure_restores_pre_delete_state(self) -> None:
        client, service = self.app()
        doc_id = self.upload(client)
        head_before = self.read_head()
        with mock.patch("mercatoria.ingest.build_index", side_effect=RuntimeError("boom")):
            resp = client.delete(f"/v1/documents/{doc_id}")
            job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "error")
        self.assertEqual(service.registry.get_raw(doc_id)["status"], "ready")
        self.assertEqual(self.read_head(), head_before)
        self.assertIn(doc_id, service.current().index.docs)


# --------------------------------------------------------------------- G06
class ConfigAcceptanceTests(GateTestCase):
    def test_desired_write_failure_aborts_the_whole_update(self) -> None:
        """G06: the runtime publish and the desired row are one transaction."""
        client, service = self.app()
        head_before = self.read_head()
        with mock.patch.object(
            ControlStore, "_write_desired", side_effect=OSError("db busy")
        ):
            with self.assertRaises(OSError):
                service.apply_config({"retrieval": {"top_k": 3}})
        self.assertEqual(self.read_head(), head_before)
        self.assertEqual(service.current().config.retrieval.top_k, 5)
        self.assertEqual(service.cfg.retrieval.top_k, 5)
        store = ControlStore(self.cfg.resolved_control_path())
        try:
            self.assertIsNone(store.desired_config())
            self.assertEqual(store.head_generation(), head_before["generation"])
        finally:
            store.close()

    def test_pending_only_failure_leaves_the_intent_unchanged(self) -> None:
        client, service = self.app()
        with mock.patch.object(
            ControlStore, "set_desired_config", side_effect=OSError("db busy")
        ):
            with self.assertRaises(OSError):
                service.apply_config({"chunking": {"max_chars": 512}})
        self.assertEqual(service.cfg.chunking.max_chars, 2048)
        self.assertFalse(service.pending_rebuild)

    def test_crash_after_acceptance_keeps_active_and_desired_together(self) -> None:
        """G06: a hard exit between commit and intent must not diverge."""
        script = textwrap.dedent(
            """
            import json
            import os
            import sys

            import numpy as np

            from mercatoria.api import Service
            from mercatoria.config import load_config

            def fake(texts):
                return np.ones((len(texts), 3), dtype=np.float32)

            cfg = load_config(sys.argv[1])
            service = Service(cfg, embed_fn=fake, config_path=sys.argv[1])
            result = service.apply_config({"retrieval": {"top_k": 3}})
            print(json.dumps({"snapshot_version": result["snapshot_version"]}), flush=True)
            os._exit(0)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        done = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.config_path())],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        published = json.loads(done.stdout.strip().splitlines()[-1])

        restarted = Service(
            load_config(self.config_path()), embed_fn=fake_embed, config_path=self.config_path()
        )
        self.services.append(restarted)
        self.assertEqual(restarted.current().snapshot_version, published["snapshot_version"])
        self.assertEqual(restarted.current().config.retrieval.top_k, 3)
        store = ControlStore(self.cfg.resolved_control_path())
        try:
            desired = store.desired_config()
        finally:
            store.close()
        self.assertEqual(desired["retrieval"]["top_k"], 3)

        # A later, unrelated update must not merge from a stale intent.
        client = TestClient(
            create_app(str(self.config_path()), embed_fn=fake_embed, service=restarted)
        )
        applied = client.put("/v1/config", json={"answer": {"max_tokens": 4096}})
        self.assertEqual(applied.status_code, 200, applied.text)
        self.assertEqual(restarted.current().config.retrieval.top_k, 3)
        self.assertEqual(restarted.current().config.answer.max_tokens, 4096)

    def test_pending_build_and_runtime_intent_survive_restart(self) -> None:
        client, service = self.app()
        pending = client.put("/v1/config", json={"chunking": {"max_chars": 512}})
        self.assertTrue(pending.json()["pending_rebuild"])
        runtime = client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        self.assertEqual(runtime.status_code, 200, runtime.text)
        self.assertTrue(runtime.json()["pending_rebuild"])
        service.close()
        self.services.remove(service)

        restarted = Service(
            load_config(self.config_path()), embed_fn=fake_embed, config_path=self.config_path()
        )
        self.services.append(restarted)
        live = restarted.current()
        self.assertEqual(live.config.retrieval.top_k, 3)
        self.assertEqual(live.config.chunking.max_chars, 2048)
        self.assertTrue(restarted.pending_rebuild)
        self.assertEqual(restarted.cfg.chunking.max_chars, 512)
        store = ControlStore(self.cfg.resolved_control_path())
        try:
            desired = store.desired_config()
        finally:
            store.close()
        self.assertEqual(desired["chunking"]["max_chars"], 512)
        self.assertEqual(desired["retrieval"]["top_k"], 3)


# --------------------------------------------------------------------- G05
class ServingLeaseTests(GateTestCase):
    def test_install_failure_blocks_data_routes_without_health(self) -> None:
        """G05: every evidence route checks the committed head, not just /health."""
        client, service = self.app()
        doc_id = self.upload(client)
        head_before = self.read_head()
        with mock.patch.object(
            service.snapshots, "install", side_effect=SnapshotError("install boom")
        ):
            rebuilt = client.post("/v1/admin/rebuild", json={"background": False})
            self.assertEqual(rebuilt.json()["status"], "error", rebuilt.json())
            self.assertTrue(rebuilt.json()["result"]["committed"])
            responses = {
                "retrieve": client.post("/v1/retrieve", json={"queries": ["关税"]}),
                "chat": client.post(
                    "/v1/chat", json={"question": "关税", "answer": {"enabled": False}}
                ),
                "article": client.get("/v1/articles/cn-test-law:1"),
                "articles": client.post(
                    "/v1/articles/batch", json={"article_ids": ["cn-test-law:1"]}
                ),
                "chunks": client.get(f"/v1/documents/{doc_id}/chunks"),
            }
        for name, response in responses.items():
            self.assertEqual(response.status_code, 503, f"{name}: {response.text}")
        self.assertGreater(self.read_head()["generation"], head_before["generation"])

        # With the fault gone, the next request follows the committed head.
        healed = client.post("/v1/retrieve", json={"queries": ["关税"]})
        self.assertEqual(healed.status_code, 200, healed.text)
        self.assertEqual(
            healed.json()["snapshot_version"], str(self.read_head()["generation"])
        )

    def test_committed_delete_is_not_served_from_the_old_snapshot(self) -> None:
        """G05: after a committed delete, no new request reads the old evidence."""
        client, service = self.app()
        doc_id = self.upload(client)
        with mock.patch.object(
            service.snapshots, "install", side_effect=SnapshotError("install boom")
        ):
            resp = client.delete(f"/v1/documents/{doc_id}")
            job = self.wait_job(client, resp.json()["job_id"])
            self.assertEqual(job["status"], "error", job)
            self.assertTrue(job["result"]["committed"])
            retrieve = client.post("/v1/retrieve", json={"queries": ["article body"]})
            article = client.get("/v1/articles/cn-test-law:1")
            chunks = client.get(f"/v1/documents/{doc_id}/chunks")
        self.assertEqual(retrieve.status_code, 503, retrieve.text)
        self.assertEqual(article.status_code, 503, article.text)
        self.assertIn(chunks.status_code, (404, 503), chunks.text)

        # After recovery the document is gone from the served evidence.
        recovered = client.post("/v1/retrieve", json={"queries": ["article body"]})
        self.assertEqual(recovered.status_code, 200)
        self.assertNotIn(doc_id, service.current().index.docs)

    def test_old_lease_completes_and_new_lease_follows(self) -> None:
        client, service = self.app()
        published = {}
        with service.serving_lease() as old:
            self.assertIsNotNone(old)
            old_generation = old.generation
            cfg = copy.deepcopy(service.current().config)
            cfg.retrieval.top_k = 3
            published = service.coordinator.publish_runtime(cfg)
            self.assertGreater(published.generation, old_generation)
            # The in-flight request keeps its (now retired) snapshot usable.
            self.assertFalse(old.closed)
            self.assertTrue(old.retriever.retrieve(queries=["关税"])["chunks"])
        with service.serving_lease() as fresh:
            self.assertEqual(fresh.generation, published.generation)
        self.assertTrue(old.closed)

    def test_head_check_and_pin_are_atomic(self) -> None:
        """G05: no publish can install between reading the head and pinning it."""
        client, service = self.app()
        paused = threading.Event()
        resume = threading.Event()
        acquired = threading.Event()
        release_use = threading.Event()
        held: dict = {}
        real_acquire = service.snapshots.acquire

        def slow_acquire(generation):
            paused.set()
            resume.wait(20)
            return real_acquire(generation)

        def use() -> None:
            with service.serving_lease() as snapshot:
                held["snapshot"] = snapshot
                held["generation"] = snapshot.generation if snapshot else None
                acquired.set()
                release_use.wait(20)
                held["usable"] = bool(
                    snapshot.retriever.retrieve(queries=["关税"])["chunks"]
                )

        with mock.patch.object(service.snapshots, "acquire", side_effect=slow_acquire):
            worker = threading.Thread(target=use)
            worker.start()
            self.assertTrue(paused.wait(20), "acquire never paused")
            gate_free = service.coordinator.gate.acquire(blocking=False)
            if gate_free:
                service.coordinator.gate.release()
            self.assertFalse(gate_free, "check and pin did not hold the publish gate")
            self.assertEqual(service.control.head_generation(), 1)
            self.assertEqual(service.current().generation, 1)
            resume.set()
            self.assertTrue(acquired.wait(20), "lease was never acquired")

        cfg = copy.deepcopy(service.current().config)
        cfg.retrieval.top_k = 3
        published = service.coordinator.publish_runtime(cfg)
        with service.serving_lease() as fresh:
            self.assertEqual(fresh.generation, published.generation)
        self.assertEqual(held["generation"], 1)
        self.assertFalse(held["snapshot"].closed)
        release_use.set()
        worker.join(20)
        self.assertFalse(worker.is_alive())
        self.assertTrue(held["usable"])
        self.assertTrue(held["snapshot"].closed)

    def test_retrieval_does_not_hold_the_publication_gate(self) -> None:
        """G05: only the check+pin is under the gate, not the request work."""
        from mercatoria.retrieve import Retriever

        client, service = self.app()
        entered = threading.Event()
        resume = threading.Event()
        real = Retriever.retrieve

        def slow(self_, *args, **kwargs):
            entered.set()
            resume.wait(20)
            return real(self_, *args, **kwargs)

        responses: dict = {}

        def request() -> None:
            responses["retrieve"] = client.post("/v1/retrieve", json={"queries": ["关税"]})

        published: dict = {}

        def publish() -> None:
            cfg = copy.deepcopy(service.current().config)
            cfg.retrieval.top_k = 3
            published["result"] = service.coordinator.publish_runtime(cfg)

        with mock.patch.object(Retriever, "retrieve", slow):
            worker = threading.Thread(target=request)
            worker.start()
            self.assertTrue(entered.wait(20), "retrieval never paused")
            publisher = threading.Thread(target=publish)
            publisher.start()
            publisher.join(10)
            self.assertFalse(
                publisher.is_alive(), "publication blocked by an in-flight retrieval"
            )
            resume.set()
            worker.join(20)
        self.assertEqual(responses["retrieve"].status_code, 200)
        self.assertEqual(
            responses["retrieve"].json()["snapshot_version"], "1"
        )
        self.assertEqual(published["result"].generation, 2)

    def test_not_ready_does_not_serve_an_unknown_head(self) -> None:
        """G05: a head that cannot be loaded yields 503, not the previous snapshot."""
        client, service = self.app()
        with mock.patch.object(service.control, "head_generation", return_value=999):
            response = client.post("/v1/retrieve", json={"queries": ["关税"]})
        self.assertEqual(response.status_code, 503, response.text)


if __name__ == "__main__":
    unittest.main()
