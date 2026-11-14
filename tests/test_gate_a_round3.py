"""Gate A round 3: control transaction isolation, post-commit boundary, manifest writes.

Each test names the review item it closes (H01-H03). Everything runs on
temporary directories with a fake embedder; the CLI case starts a second
process that opens no socket and only touches the temp data directory.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from test_publication import build_case, fake_embed

from mercatoria.api import Service, create_app
from mercatoria.config import load_config, snapshot_config_payload
from mercatoria.control import ControlStore
from mercatoria.lock import DirectoryLock
from mercatoria.manifest import write_manifest

REPO = Path(__file__).resolve().parent.parent


class Round3Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg, self.manifest = build_case(self.root)
        self.services: list[Service] = []

    def tearDown(self) -> None:
        for service in self.services:
            try:
                service.close()
            except Exception:
                pass

    def config_path(self) -> Path:
        return self.root / "config.json"

    def service(self, cfg=None, embed_fn=fake_embed) -> Service:
        service = Service(
            cfg or self.cfg, embed_fn=embed_fn, config_path=self.config_path()
        )
        self.services.append(service)
        return service

    def app(self, cfg=None, embed_fn=fake_embed):
        service = self.service(cfg=cfg, embed_fn=embed_fn)
        app = create_app(str(self.config_path()), embed_fn=embed_fn, service=service)
        return TestClient(app), service

    def close_service(self, service: Service) -> None:
        if service in self.services:
            self.services.remove(service)
        service.close()

    def head_state(self, cfg=None):
        store = ControlStore((cfg or self.cfg).resolved_control_path())
        try:
            return store.head()
        finally:
            store.close()

    def wait_job(self, client, job_id: str, timeout: float = 30.0) -> dict:
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
        self.assertEqual(resp.status_code, 200, resp.text)
        job = self.wait_job(client, resp.json()["job_id"])
        self.assertEqual(job["status"], "done", job)
        return job["result"]["documents"][0]["doc_id"]


# --------------------------------------------------------------------- H01
class ControlTransactionIsolationTests(Round3Case):
    """Public control reads and served leases never expose uncommitted state."""

    def _paused_publish(self, service: Service, cfg) -> tuple[threading.Event, callable]:
        """Publish that pauses after writing head+desired, before COMMIT."""

        reached = threading.Event()
        release = threading.Event()
        original = ControlStore._write_desired
        outcome: dict = {}

        def paused(conn, config, at=None):
            original(conn, config, at)
            reached.set()
            if not release.wait(30):
                raise TimeoutError("the test never released the writer")
            raise OSError("rollback forced")

        def writer() -> None:
            try:
                service.coordinator.publish_runtime(
                    cfg, desired_config=snapshot_config_payload(cfg)
                )
            except BaseException as exc:
                outcome["error"] = exc

        patcher = mock.patch.object(
            ControlStore, "_write_desired", staticmethod(paused)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        thread = threading.Thread(target=writer, daemon=True)
        thread.start()

        def finish() -> dict:
            release.set()
            thread.join(30)
            patcher.stop()
            self.assertFalse(thread.is_alive(), "the paused writer never finished")
            return outcome

        return reached, finish

    def _runtime_cfg(self, service: Service):
        cfg = copy.deepcopy(service.current().config)
        cfg.retrieval.top_k = 3
        return cfg

    def test_request_never_leases_an_uncommitted_generation(self) -> None:
        _client, service = self.app()
        self.assertEqual(service.control.head_generation(), 1)
        reached, finish = self._paused_publish(service, self._runtime_cfg(service))

        leased: list = []
        attempted = threading.Event()
        leased_event = threading.Event()

        def request() -> None:
            attempted.set()
            with service.serving_lease() as snapshot:
                leased.append(snapshot.generation if snapshot is not None else None)
                leased_event.set()

        reader = threading.Thread(target=request, daemon=True)
        self.assertTrue(reached.wait(30), "the writer never paused inside its transaction")
        reader.start()
        self.assertTrue(attempted.wait(30), "the request thread never started")
        # A bounded wait for something that must not happen: while the write
        # transaction is open, no lease may complete.
        self.assertFalse(
            leased_event.wait(2.0),
            "a lease completed while the control transaction was uncommitted",
        )
        outcome = finish()
        reader.join(30)
        self.assertFalse(reader.is_alive(), "the request thread never finished")

        self.assertIsInstance(outcome.get("error"), OSError)
        self.assertEqual(leased, [1], "a request leased an uncommitted generation")
        self.assertEqual(service.control.head_generation(), 1)

    def test_rollback_never_leaves_online_ahead_of_control(self) -> None:
        client, service = self.app()
        reached, finish = self._paused_publish(service, self._runtime_cfg(service))

        leased: list = []
        attempted = threading.Event()
        leased_event = threading.Event()

        def request() -> None:
            attempted.set()
            with service.serving_lease() as snapshot:
                leased.append(snapshot.generation if snapshot is not None else None)
                leased_event.set()

        reader = threading.Thread(target=request, daemon=True)
        self.assertTrue(reached.wait(30), "the writer never paused inside its transaction")
        reader.start()
        self.assertTrue(attempted.wait(30), "the request thread never started")
        self.assertFalse(
            leased_event.wait(2.0),
            "a lease completed while the control transaction was uncommitted",
        )
        outcome = finish()
        reader.join(30)
        self.assertFalse(reader.is_alive(), "the request thread never finished")

        self.assertIsInstance(outcome.get("error"), OSError)
        self.assertEqual(leased, [1])
        self.assertEqual(service.control.head_generation(), 1)
        self.assertIsNotNone(service.current())
        self.assertEqual(service.current().generation, 1)
        self.assertIsNone(service.error)
        self.assertEqual(client.get("/health/ready").status_code, 200)
        retrieved = client.post("/v1/retrieve", json={"queries": ["关税"]})
        self.assertEqual(retrieved.status_code, 200, retrieved.text)
        self.assertEqual(retrieved.json()["snapshot_version"], "1")

    def test_control_reads_do_not_expose_uncommitted_state(self) -> None:
        client, service = self.app()
        reached, finish = self._paused_publish(service, self._runtime_cfg(service))
        self.assertTrue(reached.wait(30), "the writer never paused inside its transaction")
        try:
            self.assertEqual(service.control.head_generation(), 1)
            head = service.control.head()
            self.assertIsNotNone(head)
            self.assertEqual(head.generation, 1)
            self.assertEqual(
                [state.generation for state in service.control.states()], [1]
            )
            self.assertIsNone(service.control.desired_config())
            snapshots = client.get("/v1/admin/snapshots").json()
            self.assertEqual(snapshots["current"]["generation"], 1)
            self.assertEqual(
                [state["generation"] for state in snapshots["snapshots"]], [1]
            )
        finally:
            finish()
        self.assertEqual(service.control.head_generation(), 1)
        self.assertIsNone(service.control.desired_config())

    def test_commit_then_lease_has_a_single_version_boundary(self) -> None:
        _client, service = self.app()
        cfg = self._runtime_cfg(service)
        with service.serving_lease() as old:
            self.assertIsNotNone(old)
            self.assertEqual(old.generation, 1)
            published = service.coordinator.publish_runtime(cfg)
            self.assertEqual(published.generation, 2)
            self.assertTrue(published.installed)
            # The in-flight request keeps its committed snapshot...
            self.assertFalse(old.closed)
            self.assertTrue(old.retriever.retrieve(queries=["关税"])["chunks"])
            # ...and every request after the commit sees exactly the new one.
            with service.serving_lease() as fresh:
                self.assertIsNotNone(fresh)
                self.assertEqual(fresh.generation, 2)
        self.assertTrue(old.closed)
        self.assertEqual(service.control.head_generation(), 2)
        self.assertEqual(service.current().generation, 2)

    def test_concurrent_leases_never_miss_a_committed_generation(self) -> None:
        _client, service = self.app()
        cfg = self._runtime_cfg(service)
        seen: list = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                with service.serving_lease() as snapshot:
                    seen.append(snapshot.generation if snapshot is not None else None)

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
        for thread in readers:
            thread.start()
        published = service.coordinator.publish_runtime(cfg)
        stop.set()
        for thread in readers:
            thread.join(30)
            self.assertFalse(thread.is_alive())

        self.assertEqual(published.generation, 2)
        self.assertTrue(seen)
        self.assertTrue(
            set(seen) <= {1, 2}, f"a lease missed the committed generation: {set(seen)}"
        )
        self.assertEqual(service.current().generation, 2)


# --------------------------------------------------------------------- H02
class _ExplodingIterable:
    """Stands in for a listing whose iteration fails, not just its call."""

    def __iter__(self):
        raise OSError("registry list iteration failed")


class PostCommitProjectionTests(Round3Case):
    """Anything after a commit is a warning, never a pre-commit failure."""

    def _ready_document(self, client) -> tuple[str, int]:
        doc_id = self.upload(client)
        state = self.head_state()
        self.assertIn(doc_id, [doc["doc_id"] for doc in state.manifest["docs"]])
        return doc_id, state.generation

    def _assert_committed(self, job: dict) -> list:
        result = job.get("result") or {}
        self.assertIs(result.get("committed"), True, job)
        return list(result.get("warnings") or [])

    def test_registry_enumeration_failure_after_commit_does_not_undo_delete(self) -> None:
        client, service = self.app()
        doc_id, generation = self._ready_document(client)
        with mock.patch.object(
            service.registry, "list", side_effect=OSError("projection list failed")
        ):
            resp = client.delete(f"/v1/documents/{doc_id}")
            self.assertEqual(resp.status_code, 200, resp.text)
            job = self.wait_job(client, resp.json()["job_id"])

        warnings = self._assert_committed(job)
        self.assertTrue(
            any("list" in warning or "enumeration" in warning for warning in warnings),
            warnings,
        )
        self.assertEqual(service.registry.get_raw(doc_id)["status"], "deleted")
        self.assertIsNone(service.registry.get(doc_id))
        head = self.head_state()
        self.assertGreater(head.generation, generation)
        self.assertNotIn(doc_id, [doc["doc_id"] for doc in head.manifest["docs"]])
        self.assertNotIn(doc_id, service.current().index.docs)
        self.assertEqual(client.get("/health/ready").status_code, 200)

    def test_registry_iteration_failure_after_commit_does_not_undo_delete(self) -> None:
        client, service = self.app()
        doc_id, _generation = self._ready_document(client)
        with mock.patch.object(
            service.registry, "list", return_value=_ExplodingIterable()
        ):
            resp = client.delete(f"/v1/documents/{doc_id}")
            self.assertEqual(resp.status_code, 200, resp.text)
            job = self.wait_job(client, resp.json()["job_id"])

        self._assert_committed(job)
        self.assertEqual(service.registry.get_raw(doc_id)["status"], "deleted")
        self.assertNotIn(doc_id, service.current().index.docs)

    def test_metadata_update_survives_registry_enumeration_failure(self) -> None:
        client, service = self.app()
        doc_id, _generation = self._ready_document(client)
        with mock.patch.object(
            service.registry, "list", side_effect=OSError("projection list failed")
        ):
            resp = client.patch(
                f"/v1/documents/{doc_id}", json={"jurisdiction": "US"}
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            job = self.wait_job(client, resp.json()["job_id"])

        self._assert_committed(job)
        self.assertEqual(service.registry.get_raw(doc_id)["jurisdiction"], "US")
        head = self.head_state()
        documented = next(
            doc for doc in head.manifest["docs"] if doc["doc_id"] == doc_id
        )
        self.assertEqual(documented["jurisdiction"], "US")

    def test_precommit_publication_failure_still_reverts_registry_state(self) -> None:
        client, service = self.app()
        doc_id, generation = self._ready_document(client)
        with mock.patch.object(
            service.coordinator,
            "publish_built",
            side_effect=RuntimeError("build failed before the commit"),
        ):
            resp = client.delete(f"/v1/documents/{doc_id}")
            self.assertEqual(resp.status_code, 200, resp.text)
            job = self.wait_job(client, resp.json()["job_id"])

        self.assertEqual(job["status"], "error", job)
        self.assertFalse((job.get("result") or {}).get("committed"))
        self.assertEqual(service.registry.get_raw(doc_id)["status"], "ready")
        self.assertEqual(service.control.head_generation(), generation)
        self.assertIn(doc_id, service.current().index.docs)

    def test_restart_does_not_revive_deleted_document_after_projection_failure(self) -> None:
        client, service = self.app()
        doc_id, _generation = self._ready_document(client)
        with mock.patch.object(
            service.registry, "list", side_effect=OSError("projection list failed")
        ):
            resp = client.delete(f"/v1/documents/{doc_id}")
            self.wait_job(client, resp.json()["job_id"])
        self.close_service(service)

        restarted = self.service()
        self.assertIsNone(restarted.registry.get(doc_id))
        self.assertEqual(restarted.registry.get_raw(doc_id)["status"], "deleted")
        head = self.head_state()
        self.assertNotIn(doc_id, [doc["doc_id"] for doc in head.manifest["docs"]])
        self.assertNotIn(doc_id, restarted.current().index.docs)
        self.assertEqual(restarted.current().generation, head.generation)


# --------------------------------------------------------------------- H03
class ManifestWriteOwnershipTests(Round3Case):
    """The manifest is a build input: writing it is a management operation."""

    def run_cli_manifest(self) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(REPO)
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "mercatoria",
                "--config",
                str(self.config_path()),
                "manifest",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=120,
        )

    def test_manifest_api_is_rejected_while_management_is_busy(self) -> None:
        client, service = self.app()
        path = self.cfg.resolved_manifest_path()
        before = path.read_bytes()
        before_mtime = path.stat().st_mtime_ns
        token = service.management.claim("rebuild")
        try:
            resp = client.post("/v1/admin/manifest")
            self.assertEqual(service.management.owner, "rebuild")
        finally:
            service.management.release(token)

        self.assertEqual(resp.status_code, 409, resp.text)
        self.assertEqual(resp.json()["error"]["code"], "management_busy")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, before_mtime)
        self.assertIsNone(service.management.owner)

    def test_manifest_api_releases_management_on_failure(self) -> None:
        client, service = self.app()
        path = self.cfg.resolved_manifest_path()
        before = path.read_bytes()
        with mock.patch(
            "mercatoria.api.write_manifest", side_effect=OSError("disk is full")
        ):
            with self.assertRaises(OSError):
                client.post("/v1/admin/manifest")
        self.assertIsNone(service.management.owner)
        self.assertEqual(path.read_bytes(), before)

        resp = client.post("/v1/admin/manifest")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIsNone(service.management.owner)
        self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["docs"])

    def test_cli_manifest_refuses_a_locked_data_directory(self) -> None:
        path = self.cfg.resolved_manifest_path()
        before = path.read_bytes()
        lock = DirectoryLock(self.cfg.resolved_write_lock_path())
        lock.acquire()
        try:
            refused = self.run_cli_manifest()
        finally:
            lock.release()

        self.assertNotEqual(refused.returncode, 0, refused.stdout)
        self.assertIn("in use", refused.stderr)
        self.assertEqual(path.read_bytes(), before)

        allowed = self.run_cli_manifest()
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["docs"])

    def test_manifest_write_is_atomic_on_failure(self) -> None:
        path = self.cfg.resolved_manifest_path()
        before = path.read_bytes()
        with mock.patch(
            "mercatoria.manifest.os.replace", side_effect=OSError("replace failed")
        ):
            with self.assertRaises(OSError):
                write_manifest(self.cfg.resolved_corpus_dir(), path)

        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8"))["docs"][0]["doc_id"],
            "cn-test-law",
        )
        leftovers = [
            item.name
            for item in path.parent.iterdir()
            if item.name.endswith(".tmp") and item.name.startswith(".")
        ]
        self.assertEqual(leftovers, [])

        manifest = write_manifest(self.cfg.resolved_corpus_dir(), path)
        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["documents"], manifest["documents"])


if __name__ == "__main__":
    unittest.main()
