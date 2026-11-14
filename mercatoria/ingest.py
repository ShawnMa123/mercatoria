"""Upload -> parse -> normalize -> chunk -> embed -> index.

One batch is parsed in parallel and then triggers a single index rebuild.
Embedding is cached by ``(model, text)``, so re-running a batch mostly costs
chunking and SQLite writes.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .documents import DocumentRegistry, combined_manifest
from .manifest import build_manifest, load_manifest
from .parse import ParseError, parse_document
from .publication import PostCommitError
from .store import BASE_UNSET, build_index, index_path

Progress = Callable[[str, int, int], None]


class IngestError(RuntimeError):
    pass


@dataclass
class UploadItem:
    filename: str
    data: bytes
    jurisdiction: str = "CN"
    language: str = "zh"
    precision: str = "exact"
    applies_to: list[str] | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    source_doc: str | None = None
    doc_id: str | None = None


def _safe_name(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", "_", name)
    return name or "upload"


def _source_doc(item: UploadItem) -> str:
    if item.source_doc:
        return item.source_doc.strip()
    return Path(item.filename).stem


def _doc_id(item: UploadItem, digest: str) -> str:
    if item.doc_id:
        candidate = re.sub(r"[^a-z0-9._-]+", "-", item.doc_id.lower()).strip("-")
        if candidate:
            return candidate
    return f"{item.jurisdiction.lower()}-upload-{digest[:12]}"


def _base_manifest(cfg: AppConfig) -> dict[str, Any]:
    path = cfg.resolved_manifest_path()
    if path.is_file():
        try:
            return load_manifest(path)
        except (OSError, ValueError):
            pass
    return build_manifest(cfg.resolved_corpus_dir())


def ingest_batch(
    cfg: AppConfig,
    registry: DocumentRegistry,
    items: list[UploadItem],
    *,
    embed_fn: Callable[[list[str]], Any] | None = None,
    progress: Progress | None = None,
    publish: Any = None,
    base_generation: Any = BASE_UNSET,
) -> dict[str, Any]:
    if not items:
        raise IngestError("no files")
    if len(items) > 200:
        raise IngestError("at most 200 files per batch")

    uploads_dir = cfg.resolved_uploads_dir()
    documents_dir = cfg.resolved_documents_dir()
    uploads_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[tuple[UploadItem, dict[str, Any], str]] = []
    skipped: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        if len(item.data) > cfg.max_upload_bytes:
            raise IngestError(
                f"{item.filename}: file exceeds {cfg.max_upload_bytes // (1024 * 1024)} MB"
            )
        digest = hashlib.sha256(item.data).hexdigest()
        existing = registry.get_by_sha(digest)
        if existing and existing["status"] == "deleted":
            # A logical delete keeps the row (and its unique sha256) so the
            # content can still be traced; re-uploading the same bytes is a new
            # intent. Same doc_id replaces the row, a different one (other
            # jurisdiction) drops the stale identity first.
            if _doc_id(item, digest) != existing["doc_id"]:
                registry.delete(existing["doc_id"])
            existing = None
        if existing and existing["status"] == "failed":
            # A failed row still holds the unique sha256. Clear its files and
            # row so the same content can be retried with a new document id.
            if existing.get("parsed_path"):
                Path(existing["parsed_path"]).unlink(missing_ok=True)
            shutil.rmtree(Path(existing["original_path"]).parent, ignore_errors=True)
            registry.delete(existing["doc_id"])
            existing = None
        if existing and existing["status"] in {"queued", "parsing", "parsed", "indexing", "ready"}:
            skipped.append(
                {
                    "doc_id": existing["doc_id"],
                    "filename": existing["filename"],
                    "status": existing["status"],
                    "duplicate": True,
                }
            )
            if progress:
                progress("upload", index, len(items))
            continue

        doc_id = _doc_id(item, digest)
        filename = _safe_name(item.filename)
        original = uploads_dir / doc_id / filename
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(item.data)
        record = {
            "doc_id": doc_id,
            "filename": filename,
            "original_path": str(original),
            "sha256": digest,
            "status": "queued",
            "jurisdiction": item.jurisdiction.upper(),
            "language": item.language,
            "precision": item.precision,
            "applies_to": item.applies_to or [],
            "effective_from": item.effective_from,
            "effective_to": item.effective_to,
            "source_doc": _source_doc(item),
            "size": len(item.data),
        }
        registry.insert(record)
        accepted.append((item, record, digest))
        if progress:
            progress("upload", index, len(items))

    if not accepted:
        return {
            "documents": skipped,
            "indexed": 0,
            "failed": [],
            "version": None,
        }

    parsed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    def parse_one(entry: tuple[UploadItem, dict[str, Any], str]) -> dict[str, Any]:
        item, record, _digest = entry
        doc_id = record["doc_id"]
        registry.update(doc_id, status="parsing", error=None)
        try:
            result = parse_document(item.filename, item.data)
            parsed_path = documents_dir / f"{doc_id}.md"
            parsed_path.write_text(result.text, encoding="utf-8")
            registry.update(doc_id, status="parsed", parsed_path=str(parsed_path), error=None)
            return {"doc_id": doc_id, "ok": True, "format": result.format, "pages": result.pages}
        except (ParseError, OSError, ValueError) as exc:
            registry.update(doc_id, status="failed", error=str(exc)[:500])
            return {"doc_id": doc_id, "ok": False, "error": str(exc)[:500]}

    workers = max(1, min(int(cfg.ingest_workers or 8), len(accepted)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(parse_one, entry) for entry in accepted]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            (parsed if result["ok"] else failed).append(result)
            if progress:
                progress("parse", done, len(futures))

    if not parsed:
        return {
            "documents": skipped + failed,
            "indexed": 0,
            "failed": failed,
            "version": None,
        }

    for doc in parsed:
        registry.update(doc["doc_id"], status="indexing")

    base = _base_manifest(cfg)
    manifest = combined_manifest(base, registry)
    if progress:
        progress("index", 0, len(manifest["docs"]))

    def index_progress(stage: str, done: int, total: int) -> None:
        if progress:
            progress("index", done, total)

    try:
        result = build_index(
            manifest,
            cfg,
            cfg.resolved_index_dir(),
            embed_fn=embed_fn,
            activate=True,
            progress=index_progress,
            publish=publish,
            base_generation=base_generation,
        )
    except PostCommitError as exc:
        # The publication committed: the documents are in the live document set
        # even though this process could not install the snapshot. Marking them
        # failed would contradict the committed head.
        _project_committed(registry, parsed, cfg, exc.result.index_version)
        raise
    except Exception as exc:
        for doc in parsed:
            registry.update(
                doc["doc_id"], status="failed", error=f"index rebuild failed: {exc}"[:500]
            )
        raise

    warnings: list[str] = result.setdefault("warnings", [])
    # Post-commit from here on: progress, counts and the registry projection are
    # derived from an already published state.
    if progress:
        try:
            progress("index", len(manifest["docs"]), len(manifest["docs"]))
        except Exception as exc:
            warnings.append(f"progress report failed after commit: {exc}")
    counts: dict[str, int] = {}
    try:
        counts = _chunk_counts(result["path"])
    except Exception as exc:
        warnings.append(f"chunk counts unavailable: {exc}")
    failures = 0
    last_error = None
    for doc in parsed:
        try:
            registry.update(
                doc["doc_id"], status="ready", chunk_count=counts.get(doc["doc_id"], 0)
            )
        except Exception as exc:
            failures += 1
            last_error = exc
    if failures:
        warnings.append(
            f"registry projection failed for {failures} document(s): {last_error}"
        )

    return {
        "documents": skipped + parsed + failed,
        "indexed": len(parsed),
        "failed": failed,
        "version": result["version"],
        "chunks": result["chunks"],
        "documents_total": result["documents"],
        "committed": result.get("committed", False),
        "installed": result.get("installed", False),
        "snapshot_version": result.get("snapshot_version"),
        "warnings": warnings,
    }


def _project_committed(
    registry: DocumentRegistry,
    parsed: list[dict[str, Any]],
    cfg: AppConfig,
    index_version: str,
) -> None:
    """Best-effort registry projection for a committed-but-not-installed build."""
    counts: dict[str, int] = {}
    try:
        counts = _chunk_counts(str(index_path(cfg.resolved_index_dir(), index_version)))
    except Exception:
        pass
    for doc in parsed:
        try:
            registry.update(
                doc["doc_id"], status="ready", chunk_count=counts.get(doc["doc_id"], 0)
            )
        except Exception:
            pass


def _chunk_counts(index_path: str) -> dict[str, int]:
    conn = sqlite3.connect(index_path)
    try:
        rows = conn.execute("SELECT doc_id, COUNT(*) FROM chunks GROUP BY doc_id").fetchall()
        return {str(doc_id): int(count) for doc_id, count in rows}
    finally:
        conn.close()


def rebuild_with_uploads(
    cfg: AppConfig,
    registry: DocumentRegistry,
    *,
    embed_fn: Callable[[list[str]], Any] | None = None,
    progress: Progress | None = None,
    activate: bool = True,
    publish: Any = None,
    base_generation: Any = BASE_UNSET,
) -> dict[str, Any]:
    """Rebuild using the static corpus plus every parsed upload.

    ``activate=False`` leaves a candidate index unpublished; registry statuses
    only move to "ready" when the build actually became the served state.

    Every step after the commit (chunk counts, registry projection) is *post
    commit*: a failure there is recorded as a warning and must never travel back
    to the caller as "the rebuild failed", because that would trigger a
    pre-commit compensation and leave the registry contradicting the published
    manifest.
    """
    base = _base_manifest(cfg)
    manifest = combined_manifest(base, registry)
    result = build_index(
        manifest,
        cfg,
        cfg.resolved_index_dir(),
        embed_fn=embed_fn,
        activate=activate,
        progress=progress,
        publish=publish,
        base_generation=base_generation,
    )
    if not activate or not result.get("committed"):
        return result
    warnings = result.setdefault("warnings", [])
    counts: dict[str, int] = {}
    try:
        counts = _chunk_counts(result["path"])
    except Exception as exc:  # projection is derived data, not the commit
        warnings.append(f"chunk counts unavailable: {exc}")
    failures = 0
    last_error = None
    for record in registry.list():
        if record["status"] not in {"parsed", "indexing", "ready"}:
            continue
        try:
            registry.update(
                record["doc_id"],
                status="ready",
                chunk_count=counts.get(record["doc_id"], 0),
            )
        except Exception as exc:  # keep the committed state authoritative
            failures += 1
            last_error = exc
    if failures:
        warnings.append(
            f"registry projection failed for {failures} document(s): {last_error}"
        )
    return result


def reindex_document(
    cfg: AppConfig,
    registry: DocumentRegistry,
    doc_id: str,
    *,
    embed_fn: Callable[[list[str]], Any] | None = None,
    progress: Progress | None = None,
    publish: Any = None,
    base_generation: Any = BASE_UNSET,
) -> dict[str, Any]:
    record = registry.get(doc_id)
    if not record:
        raise IngestError(f"document not found: {doc_id}")
    original = Path(record["original_path"])
    if not original.is_file():
        raise IngestError(f"original file missing: {original}")
    item = UploadItem(
        filename=record["filename"],
        data=original.read_bytes(),
        jurisdiction=record["jurisdiction"],
        language=record["language"],
        precision=record["precision"],
        applies_to=record.get("applies_to") or [],
        effective_from=record.get("effective_from"),
        effective_to=record.get("effective_to"),
        source_doc=record["source_doc"],
        doc_id=doc_id,
    )
    registry.update(doc_id, status="queued", error=None)
    # ingest_batch would dedupe by sha256; reindex must parse the same content.
    parsed = parse_document(item.filename, item.data)
    parsed_path = cfg.resolved_documents_dir() / f"{doc_id}.md"
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text(parsed.text, encoding="utf-8")
    registry.update(doc_id, status="parsed", parsed_path=str(parsed_path), error=None)
    return rebuild_with_uploads(
        cfg,
        registry,
        embed_fn=embed_fn,
        progress=progress,
        publish=publish,
        base_generation=base_generation,
    )


__all__ = [
    "IngestError",
    "UploadItem",
    "ingest_batch",
    "rebuild_with_uploads",
    "reindex_document",
]
