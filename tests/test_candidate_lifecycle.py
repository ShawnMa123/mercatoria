"""Candidate lifecycle: staged builds, activation and vector alignment (R06, R08, R11).

A staged build is a file, not a state. Only an explicit activation publishes it,
using the manifest/build config/base generation the build itself recorded, and
only if the vectors line up with the chunks one by one.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from test_publication import build_case, fake_embed

from mercatoria.api import Service, create_app
from mercatoria.config import load_config
from mercatoria.control import ControlStore
from mercatoria.publication import legacy_index_version
from mercatoria.snapshot import SnapshotError
from mercatoria.store import build_index, index_path, open_index


class CandidateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg, self.manifest = build_case(self.root)
        self.services: list[Service] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.close()

    def config_path(self) -> str:
        return str(self.root / "config.json")

    def service(self, cfg=None) -> Service:
        service = Service(
            cfg or self.cfg, embed_fn=fake_embed, config_path=self.config_path()
        )
        self.services.append(service)
        return service

    def client(self, service: Service) -> TestClient:
        return TestClient(create_app(self.config_path(), embed_fn=fake_embed, service=service))

    def head(self, cfg=None) -> dict | None:
        control = ControlStore((cfg or self.cfg).resolved_control_path())
        try:
            state = control.head()
            return state.describe() if state else None
        finally:
            control.close()

    def forget_control(self, cfg=None) -> None:
        """Pretend this deployment predates the control head."""
        target = cfg or self.cfg
        for path in target.resolved_documents_dir().glob("control.db*"):
            path.unlink()
        (target.resolved_index_dir() / "current.txt").unlink(missing_ok=True)

    def staged(self, manifest=None, cfg=None) -> dict:
        target = cfg or self.cfg
        return build_index(
            manifest or self.manifest,
            target,
            target.resolved_index_dir(),
            embed_fn=fake_embed,
            activate=False,
        )

    # -- index tampering helpers ------------------------------------------

    def copy_version(self, index_version: str) -> Path:
        current = self.head()
        source = index_path(self.cfg.resolved_index_dir(), current["index_version"])
        target = index_path(self.cfg.resolved_index_dir(), index_version)
        shutil.copyfile(source, target)
        return target

    def sql(self, path: Path, statement: str, params: tuple = ()) -> None:
        conn = sqlite3.connect(path)
        try:
            conn.execute(statement, params)
            conn.commit()
        finally:
            conn.close()


class MigrationTests(CandidateTestCase):
    def test_first_staged_build_is_not_activated_by_startup(self) -> None:
        """R06: a staged file is not evidence of an active index."""
        self.forget_control()
        staged = self.staged()
        self.assertFalse(staged["activated"])

        service = self.service()
        self.assertIsNone(service.current())
        self.assertIsNone(self.head())
        health = self.client(service).get("/health")
        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["status"], "not_built")
        self.assertTrue(Path(staged["path"]).is_file())

        service.close()
        self.services.remove(service)
        self.assertIsNone(self.service().current())
        self.assertIsNone(self.head())

    def test_multiple_staged_builds_are_not_guessed(self) -> None:
        self.forget_control()
        empty = {"manifest_version": "test", "corpus_dir": ".", "documents": 0, "docs": []}
        self.staged()
        self.staged(empty)
        service = self.service()
        self.assertIsNone(service.current())
        self.assertIsNone(self.head())

    def test_valid_legacy_pointer_is_imported_once(self) -> None:
        self.forget_control()
        staged = self.staged()
        (self.cfg.resolved_index_dir() / "current.txt").write_text(
            f"index-{staged['version']}.db", encoding="utf-8"
        )
        self.assertEqual(legacy_index_version(self.cfg.resolved_index_dir()), staged["version"])

        first = self.service()
        self.assertEqual(first.current().index_version, staged["version"])
        self.assertEqual(self.head()["generation"], 1)
        first.close()
        self.services.remove(first)

        second = self.service()
        self.assertEqual(self.head()["generation"], 1)
        self.assertEqual(second.current().index_version, staged["version"])

    def test_corrupt_pointer_fails_closed(self) -> None:
        """R06: a bad pointer must not fall back to "some other index"."""
        self.forget_control()
        self.staged()
        pointer = self.cfg.resolved_index_dir() / "current.txt"

        pointer.write_text("index-missing-file.db", encoding="utf-8")
        service = self.service()
        self.assertIsNone(service.current())
        self.assertIsNone(self.head())
        self.assertIn("missing index", service.error)
        service.close()
        self.services.remove(service)

        pointer.write_text("not-an-index-name", encoding="utf-8")
        with self.assertRaises(SnapshotError):
            legacy_index_version(self.cfg.resolved_index_dir())
        broken = self.service()
        self.assertIsNone(broken.current())
        self.assertIsNone(self.head())
        self.assertIn("current.txt", broken.error)

    def test_existing_head_ignores_current_txt(self) -> None:
        """R06: the control head is the only authority once it exists."""
        first_version = self.head()["index_version"]
        published = build_index(
            self.manifest, self.cfg, self.cfg.resolved_index_dir(), embed_fn=fake_embed
        )
        self.assertGreater(published["generation"], 1)
        (self.cfg.resolved_index_dir() / "current.txt").write_text(
            f"index-{first_version}.db", encoding="utf-8"
        )

        service = self.service()
        self.assertEqual(service.current().index_version, published["version"])
        self.assertNotEqual(service.current().index_version, first_version)


class StagedActivationTests(CandidateTestCase):
    def test_empty_candidate_activates_its_own_manifest(self) -> None:
        """R08: the candidate's document set is published, not the current one."""
        client = self.client(self.service())
        before = self.head()
        self.assertEqual(len(before["manifest_id"]), 16)

        empty = {"manifest_version": "test", "corpus_dir": ".", "documents": 0, "docs": []}
        staged = self.staged(empty)
        resp = client.post(f"/v1/admin/indexes/{staged['version']}/activate")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertTrue(body["committed"])
        self.assertEqual(body["base_generation"], before["generation"])

        state = self.head()
        self.assertEqual(state["documents"], 0)
        control = ControlStore(self.cfg.resolved_control_path())
        try:
            head = control.head()
        finally:
            control.close()
        self.assertEqual(head.manifest["docs"], [])
        service = self.app_service(client)
        self.assertEqual(service.current().index.docs, {})
        self.assertEqual(service.current().manifest["docs"], [])

    def app_service(self, client: TestClient) -> Service:
        return client.app.state.service

    def test_metadata_change_comes_from_the_candidate(self) -> None:
        """R08: equal doc_ids are not enough; the candidate manifest wins."""
        client = self.client(self.service())
        changed = json.loads(json.dumps(self.manifest))
        changed["docs"][0]["source_doc"] = "候选法名"
        staged = self.staged(changed)
        resp = client.post(f"/v1/admin/indexes/{staged['version']}/activate")
        self.assertEqual(resp.status_code, 200, resp.text)
        snapshot = self.app_service(client).current()
        self.assertEqual(snapshot.manifest["docs"][0]["source_doc"], "候选法名")
        self.assertEqual(snapshot.index.docs["cn-test-law"]["source_doc"], "候选法名")

    def test_candidate_with_stale_base_is_rejected(self) -> None:
        """R08: a candidate that lost its base cannot overwrite newer work."""
        service = self.service()
        client = self.client(service)
        staged = self.staged()  # base = current head
        rebuilt = client.post("/v1/admin/rebuild", json={"background": False})
        self.assertEqual(rebuilt.status_code, 200, rebuilt.text)
        self.assertEqual(rebuilt.json()["status"], "done", rebuilt.json())
        head_after = self.head()
        online_after = service.current().snapshot_version

        resp = client.post(f"/v1/admin/indexes/{staged['version']}/activate")
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("head moved", resp.text)
        self.assertEqual(self.head(), head_after)
        self.assertEqual(service.current().snapshot_version, online_after)

    def test_two_first_candidates_cannot_both_publish(self) -> None:
        """R08: explicit expected-empty CAS, not "whatever the head is"."""
        self.forget_control()
        empty = {"manifest_version": "test", "corpus_dir": ".", "documents": 0, "docs": []}
        first = self.staged(empty, self.cfg)
        other_cfg = load_config(self.root / "config.json")
        second = build_index(
            empty, other_cfg, other_cfg.resolved_index_dir(), embed_fn=fake_embed, activate=False
        )

        service = self.service()
        client = self.client(service)
        ok = client.post(f"/v1/admin/indexes/{first['version']}/activate")
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(self.head()["generation"], 1)

        conflict = client.post(f"/v1/admin/indexes/{second['version']}/activate")
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(self.head()["generation"], 1)
        self.assertEqual(service.current().index_version, first["version"])

    def test_candidate_without_record_uses_current_base(self) -> None:
        """Compatibility: a candidate built before candidate records existed."""
        path = self.copy_version("legacy-candidate")
        self.sql(path, "DELETE FROM meta WHERE key='candidate'")
        service = self.service()
        client = self.client(service)
        resp = client.post("/v1/admin/indexes/legacy-candidate/activate")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self.head()["index_version"], "legacy-candidate")


class VectorAlignmentTests(CandidateTestCase):
    def activate(self, version: str, client: TestClient) -> any:
        return client.post(f"/v1/admin/indexes/{version}/activate")

    def test_wrong_vector_id_with_equal_count_is_rejected(self) -> None:
        """R11: matching counts proved nothing; ids must match one by one."""
        path = self.copy_version("wrong-id")
        first = self.sql(path, "SELECT chunk_id FROM vectors ORDER BY chunk_id")
        conn = sqlite3.connect(path)
        try:
            chunk_id = conn.execute("SELECT chunk_id FROM vectors ORDER BY chunk_id").fetchone()[0]
            conn.execute(
                "UPDATE vectors SET chunk_id=? WHERE chunk_id=?", ("ghost-chunk", chunk_id)
            )
            conn.commit()
        finally:
            conn.close()

        service = self.service()
        head_before = self.head()
        online_before = service.current().snapshot_version
        resp = self.activate("wrong-id", self.client(service))
        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertIn("vector ids", resp.text)
        self.assertEqual(self.head(), head_before)
        self.assertEqual(service.current().snapshot_version, online_before)
        path.unlink()  # no leaked handle after the rejection

    def test_missing_and_extra_vectors_are_rejected(self) -> None:
        for label, statement in (
            ("missing", "DELETE FROM vectors WHERE chunk_id=(SELECT chunk_id FROM vectors LIMIT 1)"),
            (
                "extra",
                "INSERT INTO vectors(chunk_id, dim, vec) VALUES ('ghost', 3, zeroblob(12))",
            ),
        ):
            with self.subTest(case=label):
                path = self.copy_version(f"vectors-{label}")
                self.sql(path, statement)
                service = self.service()
                head_before = self.head()
                resp = self.activate(f"vectors-{label}", self.client(service))
                self.assertEqual(resp.status_code, 409, resp.text)
                self.assertEqual(self.head(), head_before)
                service.close()
                self.services.remove(service)
                path.unlink()

    def test_nan_and_dimension_errors_are_rejected(self) -> None:
        cases = {
            "nan": struct.pack("<3f", float("nan"), 0.0, 0.0),
            "short": struct.pack("<2f", 1.0, 0.0),
        }
        for label, blob in cases.items():
            with self.subTest(case=label):
                path = self.copy_version(f"vectors-{label}")
                chunk_id = sqlite3.connect(path).execute(
                    "SELECT chunk_id FROM vectors LIMIT 1"
                ).fetchone()[0]
                conn = sqlite3.connect(path)
                try:
                    conn.execute(
                        "UPDATE vectors SET vec=? WHERE chunk_id=?",
                        (sqlite3.Binary(blob), chunk_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
                service = self.service()
                head_before = self.head()
                resp = self.activate(f"vectors-{label}", self.client(service))
                self.assertEqual(resp.status_code, 409, resp.text)
                self.assertEqual(self.head(), head_before)
                service.close()
                self.services.remove(service)
                path.unlink()

    def test_vectors_are_exactly_aligned_with_chunks(self) -> None:
        """R11: row i belongs to chunk_ids[i], not just "same count"."""
        index = open_index(
            index_path(self.cfg.resolved_index_dir(), self.head()["index_version"])
        )
        try:
            self.assertEqual(index.vector_chunk_ids, index.chunk_ids)
            self.assertEqual(index.vectors.shape[0], len(index.chunk_ids))
            expected = fake_embed([chunk["text"] for chunk in index.chunks])
            for row, chunk in enumerate(index.chunks):
                position = index.vector_chunk_ids.index(chunk["chunk_id"])
                np.testing.assert_allclose(
                    index.vectors[position], expected[row], atol=1e-6,
                    err_msg=f"vector for {chunk['chunk_id']} is not its embedding",
                )
            tariff = next(c for c in index.chunks if "关税" in c["text"])
            position = index.vector_chunk_ids.index(tariff["chunk_id"])
            np.testing.assert_allclose(index.vectors[position], [1.0, 0.0, 0.0], atol=1e-6)
        finally:
            index.close()


if __name__ == "__main__":
    unittest.main()
