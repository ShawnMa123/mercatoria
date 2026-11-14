"""Cross-process lock over the data directory.

One writer per data directory: the API process owns it for its lifetime, so an
offline CLI rebuild cannot race a running service. Readers are unaffected — a
service that cannot take the lock still answers reads and refuses management
writes with an explicit error instead of writing behind the owner's back.

The implementation is OS-level (``fcntl.flock`` / ``msvcrt.locking``), not a
PID file, so a crashed process releases the lock automatically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WINDOWS = sys.platform == "win32"


class LockBusy(RuntimeError):
    """Another process owns the data directory."""


class DirectoryLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        """Take the lock or raise :class:`LockBusy`. Idempotent per instance."""
        if self._fd is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if WINDOWS:
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise LockBusy(
                f"data directory is in use by another process: {self.path.parent}"
            ) from exc
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if WINDOWS:
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "DirectoryLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


__all__ = ["DirectoryLock", "LockBusy"]
