"""The default Uvicorn entrypoint and single-writer ownership (R01, R07).

These tests never bind a port: uvicorn only *loads* the app to inspect its
interface, and the HTTP/lifespan exchanges are driven in memory. The lock cases
run a real second process, because in-process objects cannot prove an OS-level
exclusive lock.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from test_publication import build_case, fake_embed

from mercatoria import api as api_module
from mercatoria.api import Service, create_app
from mercatoria.lock import DirectoryLock, LockBusy

REPO = Path(__file__).resolve().parent.parent

HOLD_LOCK = textwrap.dedent(
    """
    import sys
    from mercatoria.lock import DirectoryLock

    lock = DirectoryLock(sys.argv[1])
    lock.acquire()
    print("locked", flush=True)
    sys.stdin.read()
    lock.release()
    print("released", flush=True)
    """
)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class DefaultEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _mini_app(self):
        """One callable for both scopes, so the proxy's forwarding is exercised."""
        events: list[str] = []

        async def mini(scope, receive, send):
            events.append(scope["type"])
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    events.append(message["type"])
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            body = b'{"ok": true}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})

        return mini, events

    def test_default_app_is_detected_as_asgi3(self) -> None:
        """R01: the lazy proxy must not be classified as ASGI2 by Uvicorn."""
        import uvicorn

        config = uvicorn.Config(api_module.app, lifespan="off")
        config.load()
        self.assertEqual(config.interface, "asgi3")

    def test_lazy_app_serves_http_and_lifespan_in_memory(self) -> None:
        mini, events = self._mini_app()
        with mock.patch("mercatoria.api.create_app", side_effect=lambda *a, **k: mini):
            proxy = api_module._LazyApp()

            async def drive_http() -> list[dict]:
                sent: list[dict] = []

                async def receive() -> dict:
                    return {"type": "http.request", "body": b"", "more_body": False}

                async def send(message: dict) -> None:
                    sent.append(message)

                await proxy(
                    {
                        "type": "http",
                        "asgi": {"version": "3.0", "spec_version": "2.3"},
                        "http_version": "1.1",
                        "method": "GET",
                        "scheme": "http",
                        "path": "/",
                        "raw_path": b"/",
                        "query_string": b"",
                        "root_path": "",
                        "headers": [],
                        "client": ("127.0.0.1", 1234),
                        "server": ("127.0.0.1", 80),
                    },
                    receive,
                    send,
                )
                return sent

            async def drive_lifespan() -> list[dict]:
                sent: list[dict] = []
                pending = ["lifespan.startup", "lifespan.shutdown"]

                async def receive() -> dict:
                    return {"type": pending.pop(0)} if pending else {"type": "lifespan.shutdown"}

                async def send(message: dict) -> None:
                    sent.append(message)

                await proxy(
                    {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.3"}},
                    receive,
                    send,
                )
                return sent

            import asyncio

            http_messages = asyncio.run(drive_http())
            lifespan_messages = asyncio.run(drive_lifespan())

        self.assertEqual(http_messages[0]["status"], 200)
        self.assertEqual(http_messages[1]["body"], b'{"ok": true}')
        self.assertEqual(
            [m["type"] for m in lifespan_messages],
            ["lifespan.startup.complete", "lifespan.shutdown.complete"],
        )
        self.assertEqual(events[:1], ["http"])
        self.assertIn("lifespan.startup", events)

    def test_import_does_not_open_the_data_directory(self) -> None:
        """R01: importing the module must stay side-effect free."""
        documents = self.root / "documents"
        index_dir = self.root / "index"
        env = _env()
        env.update(
            {
                "MERCATORIA_DOCUMENTS_DIR": str(documents),
                "MERCATORIA_INDEX_DIR": str(index_dir),
                "MERCATORIA_UPLOADS_DIR": str(self.root / "uploads"),
                "MERCATORIA_CORPUS_DIR": str(self.root / "corpus"),
            }
        )
        code = (
            "import mercatoria.api as m\n"
            "assert m.app.__dict__['_app'] is None, 'app was built at import time'\n"
            "print('imported')\n"
        )
        done = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("imported", done.stdout)
        self.assertFalse((documents / "control.db").exists())
        self.assertFalse((documents / "registry.db").exists())
        self.assertFalse(index_dir.exists())


class SingleWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg, _manifest = build_case(self.root)
        self.services: list[Service] = []

    def tearDown(self) -> None:
        for service in self.services:
            service.close()

    def service(self) -> Service:
        service = Service(self.cfg, embed_fn=fake_embed)
        self.services.append(service)
        return service

    def _hold_lock_in_subprocess(self) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-B", "-c", HOLD_LOCK, str(self.cfg.resolved_write_lock_path())],
            cwd=str(REPO),
            env=_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def cleanup() -> None:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()

        self.addCleanup(cleanup)
        line = proc.stdout.readline().strip()
        self.assertEqual(line, "locked", proc.stderr.read() if proc.poll() else line)
        return proc

    def _release(self, holder: subprocess.Popen) -> None:
        assert holder.stdin is not None
        holder.stdin.close()
        holder.wait(timeout=30)

    def test_second_process_cannot_own_the_directory(self) -> None:
        """R07: no read-only replica that silently serves a stale head."""
        holder = self._hold_lock_in_subprocess()
        try:
            with self.assertRaises(LockBusy):
                Service(self.cfg, embed_fn=fake_embed)
        finally:
            self._release(holder)

        # Crash/exit releases the OS lock, so a fresh process can take over.
        service = self.service()
        self.assertIsNotNone(service.current())

    def test_second_in_process_service_is_refused(self) -> None:
        owner = self.service()
        self.assertIsNotNone(owner.current())
        with self.assertRaises(LockBusy):
            Service(self.cfg, embed_fn=fake_embed)
        # The failed instance must not have published or served anything.
        self.assertEqual(owner.control.head_generation(), owner.current().generation)

    def test_failed_initialization_releases_lock_and_handles(self) -> None:
        with mock.patch("mercatoria.api.ControlStore", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                Service(self.cfg, embed_fn=fake_embed)
        lock = DirectoryLock(self.cfg.resolved_write_lock_path())
        lock.acquire()
        lock.release()
        # No leaked registry/index handle: the directory is removable again.
        shutil.rmtree(self.cfg.resolved_documents_dir())
        self.assertFalse(self.cfg.resolved_documents_dir().exists())

    def test_create_app_propagates_a_locked_directory(self) -> None:
        holder = self._hold_lock_in_subprocess()
        try:
            with self.assertRaises(LockBusy):
                create_app(str(self.root / "config.json"), embed_fn=fake_embed)
        finally:
            self._release(holder)


if __name__ == "__main__":
    unittest.main()
