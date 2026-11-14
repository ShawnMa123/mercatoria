"""HTTP surface: retrieval, optional answer generation, config and rebuild."""

from __future__ import annotations

import copy
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .answer import ChatError, answer_question
from .config import (
    REPO_ROOT,
    AppConfig,
    ConfigError,
    config_from_sections,
    export_public_config,
    load_config,
    merge_config_update,
    needs_index_rebuild,
    save_config,
    with_active_build,
)
from .control import ControlConflict, ControlStore
from .documents import DocumentRegistry
from .embed import EmbeddingError
from .ingest import (
    IngestError,
    UploadItem,
    ingest_batch,
    rebuild_with_uploads,
    reindex_document,
)
from .lock import DirectoryLock
from .manifest import write_manifest
from .publication import PostCommitError, PublicationCoordinator, import_legacy
from .retrieve import Retriever
from .snapshot import (
    ServingSnapshot,
    SnapshotError,
    SnapshotHolder,
    manifest_from_index,
)
from .store import (
    CorpusUnavailable,
    Index,
    IndexNotFound,
    SchemaTooNew,
    index_path,
    open_index,
)

CONFIG_SECTIONS = ("chunking", "embedding", "retrieval", "rerank", "answer")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_conversation(
    body: dict[str, Any]
) -> tuple[str, list[dict[str, str]], int]:
    """Return (last user question, prior turns, index of that question)."""
    raw = body.get("messages")
    if isinstance(raw, list) and raw:
        turns: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                turns.append({"role": role, "content": content})
        last_user = max((i for i, turn in enumerate(turns) if turn["role"] == "user"), default=-1)
        if last_user < 0:
            raise HTTPException(status_code=400, detail="messages must contain a user message")
        return turns[last_user]["content"], turns[:last_user], last_user
    question = str(body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question or messages is required")
    return question, [], 0


def _retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """Follow-up questions need the previous turn to retrieve anything useful."""
    previous = [turn["content"] for turn in history if turn["role"] == "user"]
    if not previous:
        return question
    first = question.lower().split()[0] if question.split() else ""
    pronoun = any(question.startswith(p) for p in ("它", "这", "该", "此", "上述", "那", "其"))
    pronoun = pronoun or first in {"it", "this", "that", "these", "those", "they", "them"}
    if len(question) > 24 and not pronoun:
        return question
    return (previous[-1] + " " + question)[:400]


def _tag_snapshot(result: dict[str, Any], snapshot: ServingSnapshot) -> dict[str, Any]:
    """Every response names the state it was computed from."""
    result["snapshot_version"] = snapshot.snapshot_version
    result["index_version"] = snapshot.index_version
    index_block = result.setdefault("index", {})
    if isinstance(index_block, dict):
        index_block.setdefault("version", snapshot.index_version)
        index_block["snapshot_version"] = snapshot.snapshot_version
    return result


def _empty_retrieval(snapshot: ServingSnapshot) -> dict[str, Any]:
    """A published but empty library: ready, retrievable, nothing to return."""
    return _tag_snapshot(
        {
            "schema_version": "2",
            "chunks": [],
            "corpus": snapshot.index.corpus_block(),
            "temporal": {},
            "diagnostics": {"zero_reason": "empty index", "funnel": []},
        },
        snapshot,
    )


class Service:
    def __init__(
        self,
        cfg: AppConfig,
        embed_fn=None,
        config_path: str | Path | None = None,
    ):
        # The deployment config supplies paths, limits and credentials. What a
        # request observes comes from the control state; what the operator
        # wants next lives in ``self.cfg`` (desired).
        self.deployment_cfg = copy.deepcopy(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.embed_fn = embed_fn
        self.config_path = Path(config_path) if config_path else REPO_ROOT / "config.json"
        self.error: str | None = None
        self.desired_error: str | None = None
        self.pending_rebuild = False
        self.jobs: dict[str, dict[str, Any]] = {}
        self.job_lock = threading.Lock()
        self.snapshots = SnapshotHolder()
        # Ownership comes before anything else: a process that cannot take the
        # directory lock must not open (or migrate) the writable control and
        # registry stores, and it must not serve a head it will never follow.
        self.write_lock = DirectoryLock(cfg.resolved_write_lock_path())
        self.write_lock.acquire()
        registry: DocumentRegistry | None = None
        control: ControlStore | None = None
        try:
            registry = DocumentRegistry(cfg.resolved_documents_dir() / "registry.db")
            control = ControlStore(cfg.resolved_control_path())
        except BaseException:
            for handle in (registry, control):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
            self.write_lock.release()
            raise
        self.registry = registry
        self.control = control
        self.coordinator = PublicationCoordinator(
            self.control,
            self.deployment_cfg,
            self.deployment_cfg.resolved_index_dir(),
            holder=self.snapshots,
            embed_fn=embed_fn,
        )
        self.cfg = self._load_desired_config()
        self.reload()

    def _load_desired_config(self) -> AppConfig:
        """Restore the operator's desired config, over the deployment one."""
        sections = self.control.desired_config()
        if sections is None:
            return copy.deepcopy(self.deployment_cfg)
        try:
            return config_from_sections(self.deployment_cfg, sections)
        except ConfigError as exc:
            self.desired_error = f"desired config is unusable: {exc}"
            return copy.deepcopy(self.deployment_cfg)

    # -- serving state -----------------------------------------------------

    def current(self) -> ServingSnapshot | None:
        return self.snapshots.current()

    @property
    def index(self) -> Index | None:
        snapshot = self.snapshots.current()
        return snapshot.index if snapshot is not None else None

    @property
    def retriever(self) -> Retriever | None:
        snapshot = self.snapshots.current()
        return snapshot.retriever if snapshot is not None else None

    def reload(self) -> bool:
        """Load the committed head and install it. Returns success.

        Loading is never done in place: the previous snapshot is closed only
        after its last request released its lease, so a failed reload cannot
        leave the process with neither the old nor the new index. A load that
        finishes after a newer generation committed is discarded by the
        coordinator instead of being installed.
        """
        try:
            if self.control.head() is None:
                # Migration bootstrap: only an explicit legacy pointer is adopted.
                import_legacy(self.control, self.deployment_cfg)
            snapshot = self.coordinator.reload()
        except (IndexNotFound, SnapshotError, SchemaTooNew, CorpusUnavailable) as exc:
            self.error = str(exc)
            return False
        except Exception as exc:  # unexpected loader failure must not kill serving
            self.error = f"{type(exc).__name__}: {exc}"
            return False
        if snapshot is None:
            self.error = "index is not built"
            return False
        self._sync_state(snapshot)
        return True

    def _sync_state(self, snapshot: ServingSnapshot) -> None:
        self.error = None if not snapshot.empty else "index is empty"
        self.pending_rebuild = needs_index_rebuild(snapshot.config, self.cfg)

    def _sync_after_publish(self, result: dict[str, Any]) -> None:
        """Adopt the served state after a publication installed it."""
        snapshot = self.snapshots.current()
        if snapshot is not None and result.get("installed"):
            self._sync_state(snapshot)

    def active_build_cfg(self) -> AppConfig:
        """Config an ordinary document operation builds with.

        A pending build change belongs to an explicit rebuild; an upload, delete
        or metadata update must leave an index the active build can serve.
        """
        snapshot = self.snapshots.current()
        return copy.deepcopy(snapshot.config if snapshot is not None else self.cfg)

    def require_retriever(self) -> Retriever:
        snapshot = self.snapshots.current()
        if snapshot is None:
            raise IndexNotFound(self.error or "index is not built")
        if snapshot.retriever is None:
            raise IndexNotFound("index is empty")
        return snapshot.retriever

    # -- configuration -----------------------------------------------------

    def apply_config(self, body: dict[str, Any]) -> dict[str, Any]:
        """Persist the desired config and publish its runtime part immediately.

        A build change (chunking or embedding space) is desired state only: it
        needs an explicit rebuild before it can serve. A runtime change is
        published on the *active* build identity, so a pending rebuild never
        leaks into the running state. Everything accepted here is durable
        (control DB), because a temporary, restart-losing setting meant a
        snapshot_version could serve a different config after a restart.
        """
        updated = merge_config_update(self.cfg, body)
        snapshot = self.snapshots.current()
        active = snapshot.config if snapshot is not None else self.deployment_cfg
        serving = with_active_build(updated, active)
        runtime_changed = (
            snapshot is not None
            and serving.runtime_fingerprint() != snapshot.runtime_config_id
        )
        try:
            if runtime_changed:
                self.coordinator.publish_runtime(serving)
        except PostCommitError as exc:
            # The new state is committed. Adopt the accepted config, but do not
            # pretend the old snapshot is current: readiness reports not-ready.
            self.control.set_desired_config(export_public_config(updated))
            self.cfg = updated
            self.error = str(exc)
            raise
        self.control.set_desired_config(export_public_config(updated))
        self.cfg = updated
        if body.get("save"):
            # Deprecated no-op for compatibility: acceptance already persists.
            save_config(updated, self.config_path)
        if runtime_changed:
            self._sync_state(self.snapshots.current())
        else:
            self.pending_rebuild = needs_index_rebuild(active, updated)
        return self.config_payload()

    def config_payload(self) -> dict[str, Any]:
        snapshot = self.snapshots.current()
        active = snapshot.config if snapshot is not None else self.cfg
        return {
            "config": active.public_dict(),
            "config_version": active.runtime_fingerprint(),
            "desired_config": self.cfg.public_dict() if self.pending_rebuild else None,
            "desired_config_version": self.cfg.runtime_fingerprint(),
            "active_index": snapshot.index_version if snapshot else None,
            "snapshot_version": snapshot.snapshot_version if snapshot else None,
            "pending_rebuild": self.pending_rebuild,
            "deployment_error": self.desired_error,
        }

    # -- jobs --------------------------------------------------------------

    def _new_job(self, kind: str) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        job = {
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "result": None,
            "error": None,
            "progress": {"stage": "starting", "done": 0, "total": 0},
        }
        self.jobs[job_id] = job
        return job

    def _claim_job(self) -> None:
        if not self.job_lock.acquire(blocking=False):
            raise IngestError("another ingest/rebuild job is running")

    def _progress(self, job: dict[str, Any]):
        def progress(stage: str, done: int, total: int) -> None:
            job["progress"] = {"stage": stage, "done": done, "total": total}

        return progress

    def _finish_job(self, job: dict[str, Any], work) -> None:
        try:
            job["result"] = work()
            job["status"] = "done"
        except PostCommitError as exc:
            # The publication committed; the job failed to *follow* it. Report
            # both facts instead of pretending nothing happened.
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["result"] = exc.result.describe()
        except Exception as exc:  # background jobs must report, not crash
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["finished_at"] = _now()
            self.job_lock.release()

    def _run_or_queue(self, job: dict[str, Any], work, background: bool) -> dict[str, Any]:
        if background:
            threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
            return {"job_id": job["job_id"], "status": "running"}
        self._finish_job(job, work)
        return {"job_id": job["job_id"], **job}

    # -- management operations --------------------------------------------

    def start_rebuild(self, body: dict[str, Any]) -> dict[str, Any]:
        config_body = {
            key: body[key] for key in CONFIG_SECTIONS if isinstance(body.get(key), dict)
        }
        if body.get("save"):
            config_body["save"] = True
        if config_body:
            self.apply_config(config_body)
        self._claim_job()
        job = self._new_job("rebuild")
        cfg = copy.deepcopy(self.cfg)
        activate = bool(body.get("activate", True))

        def work() -> dict[str, Any]:
            result = rebuild_with_uploads(
                cfg,
                self.registry,
                embed_fn=self.embed_fn,
                progress=self._progress(job),
                activate=activate,
                publish=self.coordinator,
            )
            self._sync_after_publish(result)
            return result

        return self._run_or_queue(job, work, bool(body.get("background", True)))

    def start_ingest(self, items: list[UploadItem]) -> dict[str, Any]:
        self._claim_job()
        job = self._new_job("ingest")
        cfg = self.active_build_cfg()

        def work() -> dict[str, Any]:
            result = ingest_batch(
                cfg,
                self.registry,
                items,
                embed_fn=self.embed_fn,
                progress=self._progress(job),
                publish=self.coordinator,
            )
            self._sync_after_publish(result)
            return result

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {
            "job_id": job["job_id"],
            "status": "running",
            "documents": [item.filename for item in items],
        }

    def start_delete(self, doc_id: str) -> dict[str, Any]:
        record = self.registry.get_raw(doc_id)
        if record is None or record["status"] == "deleted":
            raise IngestError(f"document not found: {doc_id}")
        self._claim_job()
        job = self._new_job("delete")
        cfg = self.active_build_cfg()
        previous = record["status"]

        def work() -> dict[str, Any]:
            # Logical delete: the record, the original file and the parsed
            # revision stay on disk. The document only leaves the manifest
            # once the new index is published, so a failed rebuild leaves the
            # old state completely intact.
            self.registry.update(doc_id, status="deleted")
            try:
                result = rebuild_with_uploads(
                    cfg,
                    self.registry,
                    embed_fn=self.embed_fn,
                    progress=self._progress(job),
                    publish=self.coordinator,
                )
            except PostCommitError:
                # Committed: restoring the old status would disagree with the
                # published manifest. Readiness reports the install failure.
                raise
            except Exception:
                self.registry.update(doc_id, status=previous)
                raise
            self._sync_after_publish(result)
            return {"deleted": doc_id, **result}

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {"job_id": job["job_id"], "status": "running", "doc_id": doc_id}

    def start_reindex(self, doc_id: str) -> dict[str, Any]:
        if self.registry.get(doc_id) is None:
            raise IngestError(f"document not found: {doc_id}")
        self._claim_job()
        job = self._new_job("reindex")
        cfg = self.active_build_cfg()

        def work() -> dict[str, Any]:
            result = reindex_document(
                cfg,
                self.registry,
                doc_id,
                embed_fn=self.embed_fn,
                progress=self._progress(job),
                publish=self.coordinator,
            )
            self._sync_after_publish(result)
            return result

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {"job_id": job["job_id"], "status": "running", "doc_id": doc_id}

    def start_update(self, doc_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        record = self.registry.get(doc_id)
        if record is None:
            raise IngestError(f"document not found: {doc_id}")
        values: dict[str, Any] = {}
        for key in (
            "jurisdiction",
            "language",
            "precision",
            "source_doc",
            "effective_from",
            "effective_to",
        ):
            if key in fields:
                values[key] = fields[key]
        if "jurisdiction" in values:
            values["jurisdiction"] = str(values["jurisdiction"]).strip().upper()
        if "source_doc" in values:
            source = str(values["source_doc"]).strip()
            if not source:
                raise IngestError("source_doc cannot be empty")
            values["source_doc"] = source
        if "applies_to" in fields:
            raw = fields["applies_to"]
            if isinstance(raw, str):
                raw = raw.replace(";", ",").split(",")
            if not isinstance(raw, list):
                raise IngestError("applies_to must be a list or comma-separated string")
            values["applies_to"] = [
                str(item).strip().lower() for item in raw if str(item).strip()
            ]
        if not values:
            raise IngestError("no updatable fields")
        self._claim_job()
        job = self._new_job("update")
        cfg = self.active_build_cfg()
        previous = {key: record.get(key) for key in values}

        def work() -> dict[str, Any]:
            self.registry.update(doc_id, **values)
            try:
                result = rebuild_with_uploads(
                    cfg,
                    self.registry,
                    embed_fn=self.embed_fn,
                    progress=self._progress(job),
                    publish=self.coordinator,
                )
            except PostCommitError:
                # Committed: the published manifest already carries the new
                # metadata, so restoring the old values would contradict it.
                raise
            except Exception:
                # A failed rebuild must not leave metadata that disagrees
                # with the index that is still serving.
                self.registry.update(doc_id, **previous)
                raise
            self._sync_after_publish(result)
            return {"updated": doc_id, "fields": values, **result}

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {"job_id": job["job_id"], "status": "running", "doc_id": doc_id}

    def publish_staged(self, index_version: str) -> dict[str, Any]:
        """Publish a candidate using the description its own build recorded.

        The candidate carries its manifest, build config and base generation.
        Borrowing the *current* manifest here published a state whose document
        set did not match the built index; borrowing the current head as the
        base let a stale candidate overwrite newer work.
        """
        path = index_path(self.cfg.resolved_index_dir(), index_version)
        if not path.is_file():
            raise FileNotFoundError(f"index version not found: {index_version}")
        try:
            index = open_index(path)
        except (ValueError, RuntimeError) as exc:
            raise SnapshotError(
                f"candidate {index_version} cannot be opened: {exc}"
            ) from exc
        try:
            record = index.candidate if isinstance(index.candidate, dict) else None
            manifest = record.get("manifest") if record else None
            if not isinstance(manifest, dict) or "docs" not in manifest:
                manifest = manifest_from_index(index)
            if record is not None:
                expected = record.get("base_generation")
            else:
                expected = self.control.head_generation()
            candidate_config = index.config if isinstance(index.config, dict) else {}
        finally:
            index.close()
        if candidate_config:
            try:
                cfg = config_from_sections(self.deployment_cfg, candidate_config)
            except ConfigError as exc:
                raise SnapshotError(
                    f"candidate {index_version} has an unusable config: {exc}"
                ) from exc
        else:
            snapshot = self.snapshots.current()
            cfg = snapshot.config if snapshot is not None else self.cfg
        result = self.coordinator.publish(
            index_version, manifest, cfg, expected_generation=expected
        )
        self._sync_state(self.snapshots.current())
        return {**result.describe(), "base_generation": expected}

    def rollback(self, snapshot_version: str) -> dict[str, Any]:
        """Re-publish a retained generation (index + manifest + config)."""
        if not snapshot_version.isdigit():
            raise KeyError(
                f"unknown snapshot_version: {snapshot_version} "
                "(index versions use POST /v1/admin/indexes/{index_version}/activate)"
            )
        target = self.control.get(int(snapshot_version))
        if target is None:
            raise KeyError(f"unknown snapshot_version: {snapshot_version}")
        result = self.coordinator.rollback(target)
        self._sync_state(self.snapshots.current())
        return {**result.describe(), "source_snapshot_version": target.snapshot_version}

    def close(self) -> None:
        self.snapshots.close()
        self.registry.close()
        self.control.close()
        self.write_lock.release()


def create_app(
    config_path: str | None = None,
    embed_fn=None,
    service: Service | None = None,
) -> FastAPI:
    cfg = load_config(config_path)
    service = service or Service(cfg, embed_fn=embed_fn, config_path=config_path)
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        service.close()

    app = FastAPI(title="mercatoria", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    def unavailable() -> JSONResponse:
        snapshot = service.current()
        return JSONResponse(
            status_code=503,
            content={
                "error": service.error or "index is not built",
                "snapshot_version": snapshot.snapshot_version if snapshot else None,
                "corpus": snapshot.index.corpus_block() if snapshot is not None else None,
            },
        )

    def write_error(exc: Exception, status: int = 409) -> HTTPException:
        return HTTPException(status_code=status, detail=str(exc))

    @app.get("/health")
    @app.get("/health/ready")
    def health() -> Any:
        head = service.control.head_generation()
        snapshot = service.current()
        # A committed head that this process is not serving is not ready. Try a
        # deterministic recovery first (reload the authoritative state); if that
        # fails, say so instead of reporting the stale snapshot as current.
        if head is not None and (snapshot is None or snapshot.generation != head):
            service.reload()
            snapshot = service.current()
        common = {"pending_rebuild": service.pending_rebuild}
        online = snapshot.generation if snapshot is not None else None
        if snapshot is None or online != head:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready" if head is not None else "not_built",
                    "error": service.error,
                    "head_generation": head,
                    "online_generation": online,
                    **common,
                },
            )
        return {
            "status": "ready",
            "snapshot_version": snapshot.snapshot_version,
            "index_version": snapshot.index_version,
            "chunks": snapshot.index.total_chunks,
            "documents": snapshot.index.documents,
            "head_generation": head,
            "online_generation": online,
            "error": service.error,
            **common,
            "index": {
                "version": service.index.version if service.index else None,
                "chunks": snapshot.index.total_chunks,
                "documents": snapshot.index.documents,
                "embedding_model": snapshot.index.embedding_model,
                "chunking_hash": snapshot.index.chunking_hash,
                "built_at": snapshot.index.built_at,
                "schema_version": snapshot.index.schema_version,
                "index_schema_outdated": snapshot.index.schema_outdated,
            },
            "config": snapshot.config.public_dict(),
        }

    @app.get("/health/live")
    def live() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/v1/config")
    def get_config() -> dict[str, Any]:
        return service.config_payload()

    @app.put("/v1/config")
    def put_config(body: dict[str, Any]) -> Any:
        try:
            return service.apply_config(body)
        except ConfigError as exc:
            raise write_error(exc, 422) from exc
        except PostCommitError as exc:
            raise write_error(exc, 503) from exc
        except (SnapshotError, ControlConflict) as exc:
            raise write_error(exc, 409) from exc

    def _retrieve(body: dict[str, Any], snapshot: ServingSnapshot) -> dict[str, Any]:
        if snapshot.retriever is None:
            return _empty_retrieval(snapshot)
        return _tag_snapshot(
            snapshot.retriever.retrieve(
                queries=body.get("queries") or body.get("question") or [],
                top_k=body.get("top_k"),
                filters=body.get("filters") or {},
                fields=body.get("fields") or {},
                use_rerank=body.get("rerank"),
                per_jurisdiction_k=body.get("per_jurisdiction_k"),
            ),
            snapshot,
        )

    @app.post("/retrieve")
    @app.post("/v1/retrieve")
    def retrieve(body: dict[str, Any]) -> Any:
        snapshot = service.current()
        if snapshot is None:
            return unavailable()
        try:
            with service.snapshots.lease() as leased:
                if leased is None:
                    return unavailable()
                return _retrieve(body, leased)
        except (IndexNotFound, CorpusUnavailable) as exc:
            raise write_error(exc, 503) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _neighbors(value: Any) -> int:
        try:
            count = int(value or 0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="neighbors must be an integer") from exc
        if not 0 <= count <= 5:
            raise HTTPException(status_code=400, detail="neighbors must be in [0, 5]")
        return count

    def _follow(value: Any) -> int:
        try:
            depth = int(value or 0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="follow_refs must be an integer") from exc
        if not 0 <= depth <= 2:
            raise HTTPException(status_code=400, detail="follow_refs must be in [0, 2]")
        return depth

    def _with_index(call) -> Any:
        with service.snapshots.lease() as snapshot:
            if snapshot is None:
                raise HTTPException(status_code=503, detail=service.error or "index is not built")
            return call(snapshot)

    @app.get("/v1/articles/{article_id:path}")
    def get_article(article_id: str, neighbors: int = 0, follow_refs: int = 0) -> dict[str, Any]:
        count = _neighbors(neighbors)
        follow = _follow(follow_refs)

        def payload(snapshot: ServingSnapshot) -> dict[str, Any]:
            body = snapshot.index.article_payload(article_id, count, follow)
            if body is None:
                raise HTTPException(status_code=404, detail=f"article not found: {article_id}")
            body["snapshot_version"] = snapshot.snapshot_version
            body["index_version"] = snapshot.index_version
            return body

        return _with_index(payload)

    @app.post("/v1/articles/batch")
    def get_articles(body: dict[str, Any]) -> dict[str, Any]:
        raw = body.get("article_ids")
        if not isinstance(raw, list) or not raw:
            raise HTTPException(status_code=400, detail="article_ids must be a non-empty list")
        if len(raw) > 32:
            raise HTTPException(status_code=400, detail="at most 32 article_ids per request")
        neighbors = _neighbors(body.get("neighbors"))
        follow = _follow(body.get("follow_refs"))

        def payload(snapshot: ServingSnapshot) -> dict[str, Any]:
            articles: list[dict[str, Any]] = []
            missing: list[str] = []
            for value in raw:
                article_id = str(value)
                article = snapshot.index.article_payload(article_id, neighbors, follow)
                if article is None:
                    missing.append(article_id)
                else:
                    articles.append(article)
            return {
                "articles": articles,
                "missing": missing,
                "count": len(articles),
                "snapshot_version": snapshot.snapshot_version,
                "index_version": snapshot.index_version,
            }

        return _with_index(payload)

    @app.post("/chat")
    @app.post("/v1/chat")
    @app.post("/ask")
    @app.post("/v1/ask")
    def chat_answer(body: dict[str, Any]) -> Any:
        question, history, _ = _parse_conversation(body)
        rewrite = body.get("rewrite_query", True)
        query = str(body.get("retrieval_query") or "").strip()
        if not query:
            # An agent that already rewrote the question must be able to opt
            # out of the history-concatenation heuristic.
            query = _retrieval_query(question, history) if rewrite else question
        snapshot = service.current()
        if snapshot is None:
            return unavailable()
        try:
            with service.snapshots.lease() as leased:
                if leased is None:
                    return unavailable()
                result = _retrieve({**body, "queries": body.get("queries") or [query]}, leased)
                answer_opt = body.get("answer")
                answer_cfg = leased.config.answer
                if isinstance(answer_opt, dict) and answer_opt.get("enabled") is False:
                    answer_cfg = copy.deepcopy(answer_cfg)
                    answer_cfg.enabled = False
                if leased.retriever is None:
                    result.update(
                        {
                            "answer": None,
                            "answer_status": "unavailable",
                            "citations": [],
                            "references": [],
                        }
                    )
                else:
                    try:
                        result.update(
                            answer_question(
                                answer_cfg, question, result["chunks"], history=history
                            )
                        )
                    except ChatError as exc:
                        result.update(
                            {
                                "answer": None,
                                "answer_status": "error",
                                "citations": [],
                                "references": [],
                                "error": str(exc),
                            }
                        )
        except (IndexNotFound, CorpusUnavailable) as exc:
            raise write_error(exc, 503) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        result["question"] = question
        result["retrieval_query"] = query
        result["messages"] = list(history)
        result["messages"].append({"role": "user", "content": question})
        if result.get("answer"):
            result["messages"].append({"role": "assistant", "content": result["answer"]})
        notes: list[str] = []
        reason = result.get("diagnostics", {}).get("zero_reason")
        if reason:
            notes.append(f"zero_reason={reason}")
        uncovered = result.get("diagnostics", {}).get("requested_uncovered") or {}
        for key, values in uncovered.items():
            if values:
                notes.append(f"uncovered {key}: {', '.join(values)}")
        result["coverage"] = {
            "unavailable": not result.get("chunks"),
            "notes": notes,
        }
        return result

    @app.post("/v1/documents")
    async def upload_documents(
        files: list[UploadFile] = File(...),
        jurisdiction: str = Form("CN"),
        language: str = Form(""),
        precision: str = Form(""),
        applies_to: str = Form(""),
        effective_from: str | None = Form(None),
        effective_to: str | None = Form(None),
        source_doc: str | None = Form(None),
    ) -> Any:
        if not files:
            raise HTTPException(status_code=400, detail="at least one file is required")
        if len(files) > 200:
            raise HTTPException(status_code=400, detail="at most 200 files per batch")
        jur = jurisdiction.strip().upper() or "CN"
        lang = language.strip() or ("zh" if jur == "CN" else "ja" if jur == "JP" else "en")
        prec = precision.strip() or ("exact" if jur == "CN" else "degraded")
        applies = [
            part.strip().lower()
            for part in applies_to.replace(";", ",").split(",")
            if part.strip()
        ]
        items: list[UploadItem] = []
        total = 0
        for upload in files:
            data = await upload.read()
            total += len(data)
            if total > service.cfg.max_batch_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "batch exceeds "
                        f"{service.cfg.max_batch_bytes // (1024 * 1024)} MB"
                    ),
                )
            if len(data) > service.cfg.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{upload.filename}: file exceeds "
                        f"{service.cfg.max_upload_bytes // (1024 * 1024)} MB"
                    ),
                )
            items.append(
                UploadItem(
                    filename=upload.filename or "upload.md",
                    data=data,
                    jurisdiction=jur,
                    language=lang,
                    precision=prec,
                    applies_to=applies,
                    effective_from=effective_from,
                    effective_to=effective_to,
                    source_doc=source_doc,
                )
            )
        try:
            return service.start_ingest(items)
        except IngestError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/documents")
    def list_documents(status: str | None = None) -> dict[str, Any]:
        docs = service.registry.list()
        if status:
            docs = [doc for doc in docs if doc.get("status") == status]
        return {"documents": docs, "count": len(docs)}

    @app.patch("/v1/documents/{doc_id}")
    def update_document(doc_id: str, body: dict[str, Any]) -> Any:
        if service.registry.get(doc_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
        try:
            return service.start_update(doc_id, body)
        except IngestError as exc:
            message = str(exc)
            code = 409 if "another" in message else 400
            raise HTTPException(status_code=code, detail=message) from exc

    @app.get("/v1/documents/{doc_id}")
    def get_document(doc_id: str) -> Any:
        record = service.registry.get(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail="document not found")
        snapshot = service.current()
        active = snapshot.index.docs if snapshot is not None else {}
        record["snapshot_version"] = snapshot.snapshot_version if snapshot else None
        record["active"] = doc_id in active
        return record

    @app.delete("/v1/documents/{doc_id}")
    def delete_document(doc_id: str) -> Any:
        if service.registry.get(doc_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
        try:
            return service.start_delete(doc_id)
        except IngestError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/documents/{doc_id}/reindex")
    def reindex_document_endpoint(doc_id: str) -> Any:
        if service.registry.get(doc_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
        try:
            return service.start_reindex(doc_id)
        except IngestError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/documents/{doc_id}/chunks")
    def document_chunks(doc_id: str, limit: int = 100, offset: int = 0) -> Any:
        if service.registry.get(doc_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))

        def payload(snapshot: ServingSnapshot) -> dict[str, Any]:
            rows = [chunk for chunk in snapshot.index.chunks if chunk["doc_id"] == doc_id]
            page = rows[offset : offset + limit]
            return {
                "doc_id": doc_id,
                "total": len(rows),
                "offset": offset,
                "limit": limit,
                "snapshot_version": snapshot.snapshot_version,
                "chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "article_id": chunk["article_id"],
                        "article_label": chunk["article_label"],
                        "section": chunk["section"],
                        "part": chunk["part"],
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "effective_from": chunk["effective_from"],
                        "effective_to": chunk["effective_to"],
                    }
                    for chunk in page
                ],
            }

        return _with_index(payload)

    @app.post("/v1/admin/manifest")
    def rebuild_manifest() -> dict[str, Any]:
        manifest = write_manifest(
            service.cfg.resolved_corpus_dir(), service.cfg.resolved_manifest_path()
        )
        return {"documents": manifest["documents"], "path": str(service.cfg.resolved_manifest_path())}

    @app.post("/v1/admin/rebuild")
    def rebuild(body: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return service.start_rebuild(body or {})
        except PostCommitError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (SnapshotError, ControlConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IngestError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    @app.get("/v1/admin/rebuild/{job_id}")
    def rebuild_status(job_id: str) -> dict[str, Any]:
        job = service.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/v1/admin/activate/{snapshot_version}")
    def activate_snapshot(snapshot_version: str) -> dict[str, Any]:
        try:
            return service.rollback(snapshot_version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        except PostCommitError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ControlConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (SnapshotError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/admin/indexes/{index_version}/activate")
    def activate_index_version(index_version: str) -> dict[str, Any]:
        try:
            return service.publish_staged(index_version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PostCommitError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ControlConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SnapshotError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/admin/snapshots")
    def snapshots() -> dict[str, Any]:
        head = service.control.head()
        return {
            "current": head.describe() if head else None,
            "snapshots": [state.describe() for state in service.control.states()],
        }

    @app.get("/v1/admin/indexes")
    def indexes() -> dict[str, Any]:
        d = service.cfg.resolved_index_dir()
        files = sorted(p.name for p in d.glob("index-*.db")) if d.is_dir() else []
        head = service.control.head()
        return {
            "indexes": files,
            "current": index_path(d, head.index_version).name if head else None,
            "snapshot_version": head.snapshot_version if head else None,
            "staged": sorted(
                name
                for name in files
                if head is None or name != index_path(d, head.index_version).name
            ),
        }

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "mercatoria",
            "endpoints": [
                "/health",
                "/health/ready",
                "/v1/retrieve",
                "/v1/articles/{article_id}",
                "/v1/articles/batch",
                "/v1/chat",
                "/v1/ask",
                "/v1/documents",
                "/v1/jobs/{job_id}",
                "/v1/config",
                "/v1/admin/rebuild",
                "/v1/admin/snapshots",
            ],
        }

    return app


class _LazyApp:
    """Build the default app on first use.

    Importing ``mercatoria.api`` must not open (or migrate) the real data
    directory: tests and tools import this module, and a side effect there is
    a side effect on the deployment. ``uvicorn mercatoria.api:app`` still works
    because the proxy forwards the ASGI call and every attribute.

    ``__call__`` is a coroutine function on purpose: Uvicorn decides between
    ASGI2 and ASGI3 by inspecting the callable, so a synchronous proxy makes
    the documented entrypoint answer with a TypeError instead of serving
    requests.
    """

    def __init__(self) -> None:
        self.__dict__["_app"] = None

    def _get(self) -> FastAPI:
        app = self.__dict__["_app"]
        if app is None:
            app = create_app()
            self.__dict__["_app"] = app
        return app

    async def __call__(self, scope, receive, send):
        await self._get()(scope, receive, send)

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


app = _LazyApp()
