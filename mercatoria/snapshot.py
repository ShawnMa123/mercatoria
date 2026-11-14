"""Immutable serving snapshots and the swap that publishes them.

A request takes one snapshot and uses it end to end: filter rules, index
version, embedding space and every retrieval/answer parameter come from the
same object, so a publish in the middle of a request cannot mix two states.

The snapshot is replaced wholesale. An old snapshot is closed only after the
last in-flight request released its lease.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .config import AppConfig, ConfigError, config_from_sections, embedding_space_id
from .control import ControlState
from .retrieve import Retriever
from .store import Index, index_path, open_index


class SnapshotError(RuntimeError):
    """A candidate state cannot serve: missing, inconsistent or foreign index."""


def manifest_from_index(index: Index) -> dict[str, Any]:
    """Rebuild a manifest from an index's own document table (migration)."""
    docs = []
    for doc_id in sorted(index.docs):
        doc = index.docs[doc_id]
        docs.append(
            {
                "doc_id": doc_id,
                "path": doc.get("path") or "",
                "source_doc": doc.get("source_doc") or "",
                "jurisdiction": doc.get("jurisdiction") or "",
                "language": doc.get("language") or "",
                "precision": doc.get("precision") or "",
                "applies_to": list(doc.get("applies_to") or []),
                "effective_from": doc.get("effective_from"),
                "effective_to": doc.get("effective_to"),
                "snapshot_date": doc.get("snapshot_date"),
            }
        )
    return {
        "manifest_version": "control-import",
        "corpus_dir": ".",
        "documents": len(docs),
        "docs": docs,
    }


def resolve_state_config(state: ControlState, deployment_cfg: AppConfig) -> AppConfig:
    """The config a control state served with, over the deployment environment.

    Deployment paths, limits and credentials always come from the environment;
    everything a request can observe comes from the state. A row written before
    config persistence cannot be reconstructed: it is accepted only while the
    deployment config still has the recorded fingerprint, otherwise the state
    fails closed instead of serving different settings under the same
    snapshot_version.
    """
    if state.config is not None:
        try:
            return config_from_sections(deployment_cfg, state.config)
        except ConfigError as exc:
            raise SnapshotError(
                f"control state {state.generation} has an unusable persisted config: {exc}"
            ) from exc
    if state.runtime_config_id and deployment_cfg.runtime_fingerprint() != state.runtime_config_id:
        raise SnapshotError(
            f"control state {state.generation} predates config persistence and the "
            f"deployment config no longer matches its recorded runtime fingerprint "
            f"({state.runtime_config_id}); rebuild or roll back explicitly"
        )
    return copy.deepcopy(deployment_cfg)


def validate_index(
    index: Index,
    cfg: AppConfig,
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Return every reason this index must not be published. Empty = usable.

    The rules are identical for a candidate and for a committed load: one
    definition of "can serve", so a state that passed validation at publish
    time cannot fail it on restart.
    """
    problems: list[str] = []
    if len(index.chunk_ids) != len(index.vectors):
        problems.append(
            f"chunk/vector mismatch: {len(index.chunk_ids)} chunks, {len(index.vectors)} vectors"
        )
    if index.total_chunks:
        width = int(index.vectors.shape[1])
        if cfg.embedding.dim and width != cfg.embedding.dim:
            problems.append(
                f"embedding dim mismatch: index={width}, configured={cfg.embedding.dim}"
            )
        if not np.isfinite(index.vectors).all():
            problems.append("embedding matrix contains NaN/Inf")
        if bool(np.any(~np.any(index.vectors, axis=1))):
            problems.append("embedding matrix contains an all-zero vector")
        if cfg.embedding.model and index.embedding_model != cfg.embedding.model:
            problems.append(
                f"embedding space mismatch: index model={index.embedding_model}, "
                f"configured={cfg.embedding.model}"
            )
        recorded_space = index.embedding_space_id
        if recorded_space:
            current_space = embedding_space_id(cfg.embedding)
            if recorded_space != current_space:
                problems.append(
                    f"embedding space mismatch: index space={recorded_space}, "
                    f"configured={current_space} "
                    f"(endpoint/model/revision/dim differ; rebuild required)"
                )
        recorded_base = ""
        if isinstance(index.config, dict):
            recorded_base = str((index.config.get("embedding") or {}).get("base") or "")
        if recorded_base and cfg.embedding.base and recorded_base != cfg.embedding.base:
            problems.append(
                f"embedding endpoint mismatch: index base={recorded_base}, "
                f"configured={cfg.embedding.base}"
            )
        if index.chunking_hash != cfg.chunking_hash():
            problems.append(
                "chunking/index-text profile mismatch: "
                f"index={index.chunking_hash}, configured={cfg.chunking_hash()}"
            )
    if index.schema_outdated:
        problems.append(
            f"index schema {index.schema_version} is older than this build; rebuild required"
        )
    if manifest is not None:
        indexed = set(index.docs)
        expected = {str(doc.get("doc_id")) for doc in manifest.get("docs") or []}
        if indexed != expected:
            missing = sorted(expected - indexed)[:5]
            extra = sorted(indexed - expected)[:5]
            problems.append(
                f"manifest/index document set differs (missing={missing}, extra={extra})"
            )
    return problems


@dataclass
class ServingSnapshot:
    snapshot_version: str
    index_version: str
    generation: int
    manifest: dict[str, Any]
    manifest_id: str
    build_profile_id: str
    runtime_config_id: str
    config: AppConfig
    index: Index
    retriever: Retriever | None = field(default=None, repr=False)
    refs: int = field(default=0, repr=False)
    retired: bool = field(default=False, repr=False)
    closed: bool = field(default=False, repr=False)

    @property
    def empty(self) -> bool:
        return self.index.total_chunks == 0

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.index.close()

    def describe(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "generation": self.generation,
            "index_version": self.index_version,
            "manifest_id": self.manifest_id,
            "build_profile_id": self.build_profile_id,
            "runtime_config_id": self.runtime_config_id,
            "documents": self.index.documents,
            "chunks": self.index.total_chunks,
        }


def build_snapshot(
    state: ControlState,
    deployment_cfg: AppConfig,
    index_dir: str | Path,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> ServingSnapshot:
    """Load and fully validate one control state. Raises SnapshotError."""
    cfg = resolve_state_config(state, deployment_cfg)
    path = index_path(index_dir, state.index_version)
    if not path.is_file():
        raise SnapshotError(f"index version not found: {state.index_version} ({path})")
    try:
        index = open_index(path)
    except (ValueError, RuntimeError) as exc:
        raise SnapshotError(f"index {state.index_version} cannot be opened: {exc}") from exc
    try:
        problems = validate_index(index, cfg, state.manifest)
        if problems:
            raise SnapshotError("; ".join(problems))
        frozen = copy.deepcopy(cfg)
        retriever = Retriever(index, frozen, embed_fn=embed_fn) if index.total_chunks else None
    except BaseException:
        index.close()
        raise
    return ServingSnapshot(
        snapshot_version=state.snapshot_version,
        index_version=state.index_version,
        generation=state.generation,
        manifest=state.manifest,
        manifest_id=state.manifest_id,
        build_profile_id=state.build_profile_id,
        runtime_config_id=state.runtime_config_id,
        config=frozen,
        index=index,
        retriever=retriever,
    )


class SnapshotHolder:
    """The process-local current snapshot, with reference-counted replacement."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: ServingSnapshot | None = None

    def current(self) -> ServingSnapshot | None:
        with self._lock:
            return self._current

    def install(self, snapshot: ServingSnapshot) -> ServingSnapshot | None:
        """Swap the live snapshot; the old one closes after its last lease.

        An older generation is refused: publication only ever appends, so a
        stale load completing late must not move the process backwards. The
        caller keeps ownership of a rejected snapshot.
        """
        with self._lock:
            previous = self._current
            if previous is not None and snapshot.generation < previous.generation:
                raise SnapshotError(
                    f"refusing to install generation {snapshot.generation}: "
                    f"generation {previous.generation} is live"
                )
            self._current = snapshot
            if previous is not None and previous is not snapshot:
                previous.retired = True
                self._maybe_close(previous)
            return previous

    @contextmanager
    def lease(self) -> Iterator[ServingSnapshot | None]:
        """Pin the current snapshot for one request.

        Yields ``None`` when nothing is published; the caller turns that into a
        503 instead of guessing.
        """
        with self._lock:
            snapshot = self._current
            if snapshot is not None:
                snapshot.refs += 1
        try:
            yield snapshot
        finally:
            if snapshot is not None:
                with self._lock:
                    snapshot.refs -= 1
                    self._maybe_close(snapshot)

    def _maybe_close(self, snapshot: ServingSnapshot) -> None:
        if snapshot.retired and snapshot.refs <= 0:
            snapshot.close()

    def close(self) -> None:
        with self._lock:
            snapshot, self._current = self._current, None
            if snapshot is not None:
                snapshot.retired = True
                self._maybe_close(snapshot)


__all__ = [
    "ServingSnapshot",
    "SnapshotError",
    "SnapshotHolder",
    "build_snapshot",
    "manifest_from_index",
    "validate_index",
]
