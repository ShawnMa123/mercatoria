"""The single publication path: validate a candidate, append, export.

Everything that changes the live state goes through :func:`publish_index`, so
there is exactly one place where the control head moves. A candidate that does
not fully load is rejected *before* the head moves: the process never claims a
state it cannot serve, and a crash mid-publish leaves the previous generation
in force.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig, export_public_config
from .control import ControlState, ControlStore
from .snapshot import SnapshotError, manifest_from_index, validate_index
from .store import export_current, index_path, open_index


def validate_candidate(
    cfg: AppConfig,
    index_version: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Open a built index and prove it can serve before publishing it.

    Returns the index's own build facts so the control state records what was
    actually built rather than what the config claims.
    """
    path = index_path(cfg.resolved_index_dir(), index_version)
    if not path.is_file():
        raise SnapshotError(f"index version not found: {index_version} ({path})")
    try:
        index = open_index(path)
    except (ValueError, RuntimeError) as exc:
        raise SnapshotError(f"index {index_version} cannot be opened: {exc}") from exc
    try:
        problems = validate_index(index, cfg, manifest)
        facts = {
            "chunking_hash": index.chunking_hash,
            "embedding_model": index.embedding_model,
            "chunks": index.total_chunks,
        }
    finally:
        index.close()
    if problems:
        raise SnapshotError("; ".join(problems))
    return facts


def publish_index(
    control: ControlStore,
    cfg: AppConfig,
    index_version: str,
    manifest: dict[str, Any],
    *,
    base_generation: int | None = None,
    created_by_job_id: str | None = None,
) -> ControlState:
    facts = validate_candidate(cfg, index_version, manifest)
    state = control.publish(
        index_version=index_version,
        manifest=manifest,
        build_profile_id=str(facts["chunking_hash"]),
        runtime_config_id=cfg.runtime_fingerprint(),
        base_generation=base_generation,
        created_by_job_id=created_by_job_id,
        config=export_public_config(cfg),
    )
    export_current(cfg.resolved_index_dir(), index_version)
    return state


def legacy_index_version(index_dir: str | Path) -> str | None:
    """Index version recorded by pre-control deployments, if any."""
    directory = Path(index_dir)
    current = directory / "current.txt"
    if current.is_file():
        try:
            name = current.read_text(encoding="utf-8").strip()
        except OSError:
            name = ""
        if name.startswith("index-") and name.endswith(".db"):
            return name[len("index-") : -len(".db")]
    candidates = sorted(directory.glob("index-*.db"))
    if not candidates:
        return None
    return candidates[-1].name[len("index-") : -len(".db")]


def import_legacy(control: ControlStore, cfg: AppConfig) -> ControlState | None:
    """Adopt an existing index as generation 1. Idempotent, fail-closed.

    Only runs while the control head is empty, so repeated startups are no-ops.
    An index that does not match the configured embedding space or chunking
    profile is refused with a rebuild hint instead of being silently served.
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
        base_generation=None,
        config=export_public_config(cfg),
    )


__all__ = [
    "import_legacy",
    "legacy_index_version",
    "publish_index",
    "validate_candidate",
]
