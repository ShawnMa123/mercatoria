"""The single publication path: prepare, commit, install, export.

Every state change goes through :class:`PublicationCoordinator`, so there is
exactly one place where the control head moves and one definition of what
"published" means.

The order is deliberate and not interchangeable:

1. **prepare** — open the candidate index, validate it against the config and
   manifest it will serve with, and construct the real clients. Nothing has
   changed yet; a failure here leaves head, config and the live snapshot alone.
2. **commit** — append the control generation and move the head inside one
   transaction, compare-and-swapping the expected base generation.
3. **export** — update ``current.txt`` for legacy tooling. This is an export,
   not an authority: a failure is a warning that can be retried, never a reason
   to pretend the commit did not happen.
4. **install** — swap the process-local snapshot. If this fails *after* the
   commit, the coordinator raises :class:`PostCommitError`: the caller must not
   run "undo" compensations, and readiness must report not-ready until the
   committed state is loaded again.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import AppConfig, ConfigError, config_from_sections, snapshot_config_payload
from .control import ControlState, ControlStore, manifest_id
from .snapshot import (
    ServingSnapshot,
    SnapshotError,
    SnapshotHolder,
    build_snapshot,
    manifest_from_index,
    validate_index,
)
from .store import export_current, index_path, open_index
from .retrieve import Retriever


@dataclass(frozen=True)
class PublicationResult:
    """What actually happened. ``committed`` is the state change, not a guess."""

    committed: bool
    index_version: str
    generation: int | None = None
    snapshot_version: str | None = None
    warnings: tuple[str, ...] = ()
    installed: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "committed": self.committed,
            "generation": self.generation,
            "snapshot_version": self.snapshot_version,
            "index_version": self.index_version,
            "installed": self.installed,
            "warnings": list(self.warnings),
        }


class PostCommitError(RuntimeError):
    """The state change committed, but the process could not follow it.

    Callers must not undo business state that was decided before the commit, and
    readiness must not claim the old snapshot is current.
    """

    committed = True

    def __init__(self, message: str, result: PublicationResult):
        super().__init__(message)
        self.result = result


def committed_failure(exc: BaseException) -> PublicationResult | None:
    """The publication result if ``exc`` happened *after* a commit.

    This is the explicit record callers branch on, not an exception type name:
    every post-commit path either appends a warning to the result or raises an
    error carrying ``committed=True`` and the result.
    """
    result = getattr(exc, "result", None)
    if getattr(exc, "committed", False) and isinstance(result, PublicationResult):
        return result
    return None


def prepare_snapshot(
    index_version: str,
    manifest: dict[str, Any],
    cfg: AppConfig,
    index_dir: str | Path,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    build_profile_id: str = "",
) -> ServingSnapshot:
    """Prove a candidate can serve, before anything is committed.

    Opens the index, validates chunk/vector alignment, dimensions, finiteness,
    embedding space, chunking profile and the manifest document set, then builds
    the retriever (which constructs the embedding/rerank clients). Any failure
    closes the index and raises :class:`SnapshotError`.
    """
    path = index_path(index_dir, index_version)
    if not path.is_file():
        raise SnapshotError(f"index version not found: {index_version} ({path})")
    try:
        index = open_index(path)
    except (ValueError, RuntimeError) as exc:
        raise SnapshotError(f"index {index_version} cannot be opened: {exc}") from exc
    try:
        problems = validate_index(index, cfg, manifest)
        if problems:
            raise SnapshotError("; ".join(problems))
        frozen = copy.deepcopy(cfg)
        retriever = (
            Retriever(index, frozen, embed_fn=embed_fn) if index.total_chunks else None
        )
    except BaseException:
        index.close()
        raise
    return ServingSnapshot(
        snapshot_version="",
        index_version=index_version,
        generation=0,
        manifest=manifest,
        manifest_id=manifest_id(manifest),
        build_profile_id=build_profile_id or index.chunking_hash or frozen.chunking_hash(),
        runtime_config_id=frozen.runtime_fingerprint(),
        config=frozen,
        index=index,
        retriever=retriever,
    )


class PublicationCoordinator:
    """One publication gate for config, candidates, rollback and reload."""

    def __init__(
        self,
        control: ControlStore,
        deployment_cfg: AppConfig,
        index_dir: str | Path,
        *,
        holder: SnapshotHolder | None = None,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        gate: threading.RLock | None = None,
    ):
        self.control = control
        self.deployment_cfg = deployment_cfg
        self.index_dir = Path(index_dir)
        self.holder = holder
        self.embed_fn = embed_fn
        # Loading may run outside the gate; installing never does. The gate is
        # what makes "read the head, then install" atomic against publishes.
        self.gate = gate or threading.RLock()

    # -- preparation -------------------------------------------------------

    def prepare(
        self,
        index_version: str,
        manifest: dict[str, Any],
        cfg: AppConfig,
    ) -> ServingSnapshot:
        return prepare_snapshot(
            index_version, manifest, cfg, self.index_dir, self.embed_fn
        )

    def load_state(self, state: ControlState) -> ServingSnapshot:
        return build_snapshot(state, self.deployment_cfg, self.index_dir, self.embed_fn)

    def load_committed(self) -> ServingSnapshot | None:
        """Load the committed head without installing it."""
        state = self.control.head()
        return None if state is None else self.load_state(state)

    # -- commit ------------------------------------------------------------

    def _commit(self, prepared: ServingSnapshot, *, expected_generation: int | None,
                created_by_job_id: str | None,
                desired_config: dict[str, Any] | None = None) -> ControlState:
        expect_empty = expected_generation is None
        state = self.control.publish(
            index_version=prepared.index_version,
            manifest=prepared.manifest,
            build_profile_id=prepared.build_profile_id,
            runtime_config_id=prepared.runtime_config_id,
            base_generation=expected_generation,
            expect_empty=expect_empty,
            created_by_job_id=created_by_job_id,
            config=snapshot_config_payload(prepared.config),
            desired_config=desired_config,
        )
        prepared.generation = state.generation
        prepared.snapshot_version = state.snapshot_version
        return state

    def publish(
        self,
        index_version: str,
        manifest: dict[str, Any],
        cfg: AppConfig,
        *,
        expected_generation: int | None,
        created_by_job_id: str | None = None,
        install: bool = True,
        desired_config: dict[str, Any] | None = None,
    ) -> PublicationResult:
        """Prepare, commit, export, then (optionally) install one new state.

        ``expected_generation=None`` means "the head must still be empty": a
        first publication cannot quietly overwrite a state that appeared while
        it was being built. ``desired_config`` is committed in the same control
        transaction as the generation.
        """
        prepared = self.prepare(index_version, manifest, cfg)
        try:
            self._commit(
                prepared,
                expected_generation=expected_generation,
                created_by_job_id=created_by_job_id,
                desired_config=desired_config,
            )
        except BaseException:
            prepared.close()
            raise
        result = PublicationResult(
            committed=True,
            index_version=index_version,
            generation=prepared.generation,
            snapshot_version=prepared.snapshot_version,
        )
        result = self._export(result)
        if not install:
            prepared.close()
            return result
        return self._install_or_fail(prepared, result)

    def publish_runtime(
        self, cfg: AppConfig, *, desired_config: dict[str, Any] | None = None
    ) -> PublicationResult:
        """Publish a config-only state over the current index."""
        state = self.control.head()
        if state is None:
            raise SnapshotError("nothing is published yet; nothing to update")
        return self.publish(
            state.index_version,
            state.manifest,
            cfg,
            expected_generation=state.generation,
            desired_config=desired_config,
        )

    def publish_built(
        self,
        index_version: str,
        manifest: dict[str, Any],
        cfg: AppConfig,
        *,
        base_generation: int | None,
        created_by_job_id: str | None = None,
    ) -> PublicationResult:
        """Publish a freshly built index; ``base_generation=None`` = first build."""
        return self.publish(
            index_version,
            manifest,
            cfg,
            expected_generation=base_generation,
            created_by_job_id=created_by_job_id,
        )

    def rollback(self, target: ControlState) -> PublicationResult:
        """Re-publish a retained generation as a new generation."""
        try:
            cfg = config_from_sections(self.deployment_cfg, target.config)
        except ConfigError as exc:
            raise SnapshotError(
                f"snapshot {target.snapshot_version} cannot be restored: {exc}"
            ) from exc
        head = self.control.head_generation()
        return self.publish(
            target.index_version,
            target.manifest,
            cfg,
            expected_generation=head,
            created_by_job_id=target.created_by_job_id,
        )

    # -- export ------------------------------------------------------------

    def _export(self, result: PublicationResult) -> PublicationResult:
        try:
            export_current(self.index_dir, result.index_version)
        except OSError as exc:
            # The control head is authoritative; the export is a convenience
            # that the next successful publication rewrites.
            return replace(
                result,
                warnings=result.warnings
                + (f"export_current failed ({exc}); the control head is authoritative",),
            )
        return result

    # -- install -----------------------------------------------------------

    def install(self, snapshot: ServingSnapshot) -> None:
        """Install a prepared snapshot if it is still the committed head."""
        with self.gate:
            head = self.control.head_generation()
            if head is None or int(snapshot.generation) != head:
                snapshot.close()
                raise SnapshotError(
                    f"refusing to install generation {snapshot.generation}: "
                    f"the committed head is {head}"
                )
            self._install_locked(snapshot)

    def _install_locked(self, snapshot: ServingSnapshot) -> None:
        if self.holder is None:
            snapshot.close()
            return
        try:
            self.holder.install(snapshot)
        except BaseException:
            snapshot.close()
            raise

    def _install_or_fail(
        self, prepared: ServingSnapshot, result: PublicationResult
    ) -> PublicationResult:
        try:
            self.install(prepared)
        except BaseException as exc:
            raise PostCommitError(
                f"generation {result.generation} committed but could not be "
                f"installed: {exc}",
                replace(result, installed=False),
            ) from exc
        return replace(result, installed=True)

    # -- reload ------------------------------------------------------------

    def reload(self, attempts: int = 3) -> ServingSnapshot | None:
        """Load the committed head and install it, discarding stale loads.

        The head is read before loading and checked again under the gate: a
        snapshot prepared before another publication committed is closed, not
        installed, so a slow reload can never move the process back a version.
        """
        last: ServingSnapshot | None = None
        for _ in range(max(1, attempts)):
            state = self.control.head()
            if state is None:
                return None
            snapshot = self.load_state(state)
            with self.gate:
                head = self.control.head_generation()
                if head != state.generation:
                    snapshot.close()
                    last = None
                    continue
                self._install_locked(snapshot)
                return snapshot
        raise SnapshotError(
            f"control head kept moving while loading (last attempt: {last}, "
            f"head is {self.control.head_generation()})"
        )


__all__ = [
    "PostCommitError",
    "PublicationCoordinator",
    "PublicationResult",
    "committed_failure",
    "prepare_snapshot",
    "legacy_index_version",
    "import_legacy",
]


def legacy_index_version(index_dir: str | Path) -> str | None:
    """Index version named by the pre-control ``current.txt`` pointer.

    Only an explicit pointer counts as evidence of an active index: a staged
    candidate is also an ``index-*.db`` file, and guessing "newest" would
    activate a build that was never published.
    """
    directory = Path(index_dir)
    current = directory / "current.txt"
    if not current.is_file():
        return None
    try:
        name = current.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SnapshotError(f"legacy current.txt cannot be read: {exc}") from exc
    if not name.startswith("index-") or not name.endswith(".db"):
        raise SnapshotError(
            f"legacy current.txt does not name an index file: {name!r}"
        )
    if not (directory / name).is_file():
        raise SnapshotError(
            f"legacy current.txt names a missing index: {name}"
        )
    return name[len("index-") : -len(".db")]


def import_legacy(control: ControlStore, cfg: AppConfig) -> ControlState | None:
    """Adopt the index named by a pre-control pointer as generation 1.

    Migration bootstrap only: runs while the control head is empty, so repeated
    startups are no-ops, and a pointer that cannot be loaded fails closed rather
    than falling back to some other file. An index that does not match the
    configured space or chunking profile is refused with a rebuild hint.
    """
    if control.head() is not None:
        return None
    version = legacy_index_version(cfg.resolved_index_dir())
    if version is None:
        return None
    path = index_path(cfg.resolved_index_dir(), version)
    try:
        index = open_index(path)
    except (ValueError, RuntimeError) as exc:
        raise SnapshotError(f"existing index {version} cannot be opened: {exc}") from exc
    try:
        manifest = manifest_from_index(index)
        problems = validate_index(index, cfg, manifest)
    finally:
        index.close()
    if problems:
        raise SnapshotError(
            "existing index cannot be adopted: " + "; ".join(problems)
        )
    return control.publish(
        index_version=version,
        manifest=manifest,
        build_profile_id=cfg.chunking_hash(),
        runtime_config_id=cfg.runtime_fingerprint(),
        expect_empty=True,
        config=snapshot_config_payload(cfg),
    )
