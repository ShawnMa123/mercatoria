"""Gate A round 2: deployment config, embedding space, management, commit boundary.

Each test names the review item it closes (G01-G07) and uses only temporary
directories, a fake embedder and in-memory ASGI calls.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import struct
import subprocess
import sys
import tempfile
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
from mercatoria.publication import PostCommitError, prepare_snapshot
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


if __name__ == "__main__":
    unittest.main()
