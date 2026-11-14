"""Recovery of the published state: config, active/desired, secrets (R05, R09, R10).

Everything runs against the real Service/ControlStore with a fake embedder and
a temporary data directory. No gateway is contacted and no port is opened.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from test_publication import build_case, fake_embed

from mercatoria.api import Service, create_app
from mercatoria.config import export_public_config, load_config, needs_index_rebuild
from mercatoria.control import ControlState, ControlStore
from mercatoria.snapshot import SnapshotError, build_snapshot

SECRETS = ("sk-FAKE-embedding-0001", "sk-FAKE-rerank-0002", "sk-FAKE-answer-0003")


class RecoveryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg, self.manifest = build_case(self.root)
        self.services: list[Service] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.close()

    def app(self, cfg=None, embed_fn=fake_embed):
        service = Service(cfg or self.cfg, embed_fn=embed_fn, config_path=self.root / "config.json")
        self.services.append(service)
        app = create_app(str(self.root / "config.json"), embed_fn=embed_fn, service=service)
        from fastapi.testclient import TestClient

        return TestClient(app), service

    def restart(self) -> Service:
        """A fresh process-equivalent: deployment config only, state from the DB."""
        service = Service(
            load_config(self.root / "config.json"),
            embed_fn=fake_embed,
            config_path=self.root / "config.json",
        )
        self.services.append(service)
        return service

    def wait_job(self, client, job_id: str, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = client.get(f"/v1/jobs/{job_id}").json()
            if job["status"] != "running":
                return job
            time.sleep(0.02)
        self.fail(f"job did not finish: {job_id}")


class SnapshotConfigTests(RecoveryTestCase):
    def test_runtime_update_keeps_the_same_config_after_restart(self) -> None:
        """R05: a snapshot_version must bind one config, before and after restart."""
        client, service = self.app()
        resp = client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        self.assertEqual(resp.status_code, 200)
        published = service.current()
        version = published.snapshot_version
        recorded_fingerprint = published.runtime_config_id
        self.assertEqual(published.config.retrieval.top_k, 3)

        service.close()
        self.services.remove(service)

        restarted = self.restart()
        restored = restarted.current()
        self.assertEqual(restored.snapshot_version, version)
        self.assertEqual(restored.config.retrieval.top_k, 3)
        self.assertEqual(restored.runtime_config_id, recorded_fingerprint)
        self.assertEqual(restored.config.runtime_fingerprint(), recorded_fingerprint)
        self.assertEqual(restarted.cfg.retrieval.top_k, 3)

    def test_deployment_secrets_are_resolved_again_on_restart(self) -> None:
        """Config rows carry no credentials; the environment supplies them."""
        client, service = self.app()
        client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        service.close()
        self.services.remove(service)
        with mock.patch.dict(os.environ, {"EMBEDDING_API_KEY": SECRETS[0]}):
            restarted = self.restart()
        self.assertEqual(restarted.current().config.embedding.api_key, SECRETS[0])

    def test_legacy_row_without_config_fails_closed_on_mismatch(self) -> None:
        """R05: a row that only has a fingerprint is not guessed."""
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            state = control.head()
            control.conn.execute(
                "UPDATE control_states SET config=NULL, runtime_config_id='bogus' "
                "WHERE generation=?",
                (state.generation,),
            )
            legacy = control.head()
        finally:
            control.close()
        self.assertIsNone(legacy.config)
        with self.assertRaises(SnapshotError) as raised:
            build_snapshot(legacy, self.cfg, self.cfg.resolved_index_dir(), fake_embed)
        self.assertIn("predates config persistence", str(raised.exception))

    def test_legacy_row_with_matching_fingerprint_is_recovered(self) -> None:
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            state = control.head()
            control.conn.execute(
                "UPDATE control_states SET config=NULL WHERE generation=?",
                (state.generation,),
            )
            legacy = control.head()
        finally:
            control.close()
        snapshot = build_snapshot(legacy, self.cfg, self.cfg.resolved_index_dir(), fake_embed)
        self.assertEqual(snapshot.config.runtime_fingerprint(), self.cfg.runtime_fingerprint())
        snapshot.close()


class ActiveDesiredTests(RecoveryTestCase):
    def test_pending_build_does_not_swallow_runtime_updates(self) -> None:
        """R09: a pending chunking change must not block top_k from applying."""
        client, service = self.app()
        active = service.current()
        pending = client.put("/v1/config", json={"chunking": {"max_chars": 512}})
        self.assertTrue(pending.json()["pending_rebuild"])
        self.assertEqual(service.current().snapshot_version, active.snapshot_version)

        applied = client.put("/v1/config", json={"retrieval": {"top_k": 3}})
        self.assertEqual(applied.status_code, 200)
        self.assertTrue(applied.json()["pending_rebuild"])
        live = service.current()
        self.assertEqual(live.config.retrieval.top_k, 3)
        self.assertEqual(live.config.chunking.max_chars, active.config.chunking.max_chars)
        self.assertEqual(live.build_profile_id, active.build_profile_id)
        self.assertNotEqual(live.snapshot_version, active.snapshot_version)

    def test_ordinary_upload_uses_the_active_build(self) -> None:
        """R09: uploads must not silently activate a pending space change."""
        client, service = self.app()
        active = service.current().config
        client.put("/v1/config", json={"embedding": {"model": "someone/other-embedding"}})

        resp = client.post(
            "/v1/documents",
            files=[("files", ("pending.md", b"## X\n\nsecond article\n", "text/markdown"))],
            data={"jurisdiction": "CN"},
        )
        job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        live = service.current()
        self.assertEqual(live.config.embedding.model, active.embedding.model)
        self.assertEqual(live.config.chunking.max_chars, active.chunking.max_chars)
        self.assertTrue(service.pending_rebuild)
        self.assertEqual(service.cfg.embedding.model, "someone/other-embedding")
        # The served index really was embedded in the active space.
        self.assertEqual(live.index.embedding_model, active.embedding.model)

    def test_explicit_rebuild_activates_the_desired_build(self) -> None:
        client, service = self.app()
        client.put("/v1/config", json={"chunking": {"max_chars": 512}})
        self.assertTrue(service.pending_rebuild)

        resp = client.post("/v1/admin/rebuild", json={"background": False})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["status"], "done", resp.json())
        live = service.current()
        self.assertEqual(live.config.chunking.max_chars, 512)
        self.assertFalse(service.pending_rebuild)

    def test_pending_desired_state_survives_restart(self) -> None:
        client, service = self.app()
        client.put("/v1/config", json={"chunking": {"max_chars": 512}})
        active = service.current()
        service.close()
        self.services.remove(service)

        restarted = self.restart()
        self.assertTrue(restarted.pending_rebuild)
        self.assertEqual(restarted.cfg.chunking.max_chars, 512)
        self.assertEqual(restarted.current().config.chunking.max_chars, active.config.chunking.max_chars)
        self.assertEqual(restarted.current().snapshot_version, active.snapshot_version)

    def test_runtime_only_change_is_not_a_rebuild(self) -> None:
        before = self.cfg
        after = load_config(self.root / "config.json")
        after.retrieval.top_k = 7
        self.assertFalse(needs_index_rebuild(before, after))
        after.embedding.base = "http://127.0.0.1:1/v1"
        self.assertTrue(needs_index_rebuild(before, after))


class SecretPersistenceTests(RecoveryTestCase):
    def _cfg_with_secrets(self):
        cfg = load_config(self.root / "config.json")
        cfg.embedding.api_key = SECRETS[0]
        cfg.rerank.api_key = SECRETS[1]
        cfg.answer.api_key = SECRETS[2]
        return cfg

    def _assert_no_secret(self, path: Path) -> None:
        if not path.is_file():
            return
        raw = path.read_bytes()
        for secret in SECRETS:
            self.assertNotIn(secret.encode(), raw, f"{secret} leaked into {path.name}")

    def test_save_never_writes_credentials(self) -> None:
        """R10: save persists the non-sensitive config only."""
        cfg = self._cfg_with_secrets()
        client, service = self.app(cfg=cfg)
        resp = client.put("/v1/config", json={"retrieval": {"top_k": 4}, "save": True})
        self.assertEqual(resp.status_code, 200)

        saved = self.root / "config.json"
        self._assert_no_secret(saved)
        payload = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(payload["retrieval"]["top_k"], 4)
        for section in ("embedding", "rerank", "answer"):
            self.assertNotIn("api_key", payload[section])

    def test_published_artifacts_carry_no_credentials(self) -> None:
        cfg = self._cfg_with_secrets()
        client, service = self.app(cfg=cfg)
        resp = client.post("/v1/admin/rebuild", json={"background": False})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["status"], "done", resp.json())

        documents = cfg.resolved_documents_dir()
        for name in ("control.db", "control.db-wal", "registry.db", "registry.db-wal"):
            self._assert_no_secret(documents / name)
        for index_file in cfg.resolved_index_dir().glob("index-*.db"):
            self._assert_no_secret(index_file)
        # Jobs are in-memory; their payloads must not echo secrets either.
        for job in service.jobs.values():
            self._assert_no_secret_text(json.dumps(job, ensure_ascii=False))

    def _assert_no_secret_text(self, text: str) -> None:
        for secret in SECRETS:
            self.assertNotIn(secret, text)

    def test_export_public_config_omits_secret_fields(self) -> None:
        exported = export_public_config(self._cfg_with_secrets())
        self.assertNotIn("api_key", exported["embedding"])
        self.assertNotIn("api_key", exported["rerank"])
        self.assertNotIn("api_key", exported["answer"])
        self.assertEqual(exported["retrieval"]["top_k"], 5)


if __name__ == "__main__":
    unittest.main()
