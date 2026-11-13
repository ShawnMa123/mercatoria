"""HTTP surface: retrieval, optional answer generation, config and rebuild."""

from __future__ import annotations

import copy
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .answer import ChatError, answer_question
from .config import AppConfig, load_config, save_config
from .documents import DocumentRegistry
from .embed import EmbeddingError
from .ingest import (
    IngestError,
    UploadItem,
    ingest_batch,
    rebuild_with_uploads,
    reindex_document,
)
from .manifest import write_manifest
from .retrieve import Retriever
from .store import (
    CorpusUnavailable,
    Index,
    IndexNotFound,
    activate_index,
    open_current,
)


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


class Service:
    def __init__(self, cfg: AppConfig, embed_fn=None):
        self.cfg = cfg
        self.embed_fn = embed_fn
        self.index: Index | None = None
        self.retriever: Retriever | None = None
        self.error: str | None = None
        self.pending_rebuild = False
        self.jobs: dict[str, dict[str, Any]] = {}
        self.registry = DocumentRegistry(cfg.resolved_documents_dir() / "registry.db")
        self.job_lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        if self.index is not None:
            self.index.close()
            self.index = None
            self.retriever = None
        try:
            self.index = open_current(self.cfg.resolved_index_dir())
            if self.index.total_chunks:
                self.retriever = Retriever(self.index, self.cfg, embed_fn=self.embed_fn)
                self.error = None
            else:
                self.retriever = None
                self.error = "index is empty"
            self.pending_rebuild = False
        except (IndexNotFound, CorpusUnavailable) as exc:
            self.index = None
            self.retriever = None
            self.error = str(exc)

    def require_retriever(self) -> Retriever:
        if self.retriever is None:
            raise IndexNotFound(self.error or "index is not built")
        return self.retriever

    def _apply_updates(self, body: dict[str, Any]) -> bool:
        """Apply config sections in place. Return True if a rebuild is needed."""
        pending = False
        old_hash = self.cfg.chunking_hash()
        for section in ("chunking", "embedding", "retrieval", "rerank", "answer"):
            values = body.get(section)
            if not isinstance(values, dict):
                continue
            target = getattr(self.cfg, section)
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
        if self.cfg.chunking_hash() != old_hash:
            pending = True
            self.pending_rebuild = True
        if self.retriever is not None:
            # The retriever holds cfg by reference, but its cache and the
            # rerank flag are snapshots of the old settings.
            self.retriever.rerank_enabled = self.retriever._rerank_forced or bool(
                self.cfg.rerank.enabled
            )
            self.retriever.clear_cache()
        if body.get("save"):
            save_config(self.cfg, Path("config.json"))
        return pending

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
        except Exception as exc:  # background jobs must report, not crash
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["finished_at"] = _now()
            self.job_lock.release()

    def start_rebuild(self, body: dict[str, Any]) -> dict[str, Any]:
        pending = self._apply_updates(body)
        self._claim_job()
        job = self._new_job("rebuild")
        cfg = copy.deepcopy(self.cfg)
        activate = bool(body.get("activate", True))

        def work() -> dict[str, Any]:
            result = rebuild_with_uploads(
                cfg, self.registry, embed_fn=self.embed_fn, progress=self._progress(job)
            )
            if activate:
                self.reload()
            return result

        if body.get("background", True):
            threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
            return {"job_id": job["job_id"], "status": "running", "pending_rebuild": pending}
        self._finish_job(job, work)
        return {"job_id": job["job_id"], **job, "pending_rebuild": pending}

    def start_ingest(self, items: list[UploadItem]) -> dict[str, Any]:
        self._claim_job()
        job = self._new_job("ingest")
        cfg = copy.deepcopy(self.cfg)

        def work() -> dict[str, Any]:
            result = ingest_batch(
                cfg,
                self.registry,
                items,
                embed_fn=self.embed_fn,
                progress=self._progress(job),
            )
            if result.get("indexed"):
                self.reload()
            return result

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {
            "job_id": job["job_id"],
            "status": "running",
            "documents": [item.filename for item in items],
        }

    def start_delete(self, doc_id: str) -> dict[str, Any]:
        record = self.registry.get(doc_id)
        if record is None:
            raise IngestError(f"document not found: {doc_id}")
        self._claim_job()
        job = self._new_job("delete")
        cfg = copy.deepcopy(self.cfg)

        def work() -> dict[str, Any]:
            parsed = record.get("parsed_path")
            if parsed:
                Path(parsed).unlink(missing_ok=True)
            original = Path(record["original_path"])
            shutil.rmtree(original.parent, ignore_errors=True)
            self.registry.delete(doc_id)
            result = rebuild_with_uploads(
                cfg, self.registry, embed_fn=self.embed_fn, progress=self._progress(job)
            )
            self.reload()
            return {"deleted": doc_id, **result}

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {"job_id": job["job_id"], "status": "running", "doc_id": doc_id}

    def start_reindex(self, doc_id: str) -> dict[str, Any]:
        if self.registry.get(doc_id) is None:
            raise IngestError(f"document not found: {doc_id}")
        self._claim_job()
        job = self._new_job("reindex")
        cfg = copy.deepcopy(self.cfg)

        def work() -> dict[str, Any]:
            result = reindex_document(
                cfg,
                self.registry,
                doc_id,
                embed_fn=self.embed_fn,
                progress=self._progress(job),
            )
            self.reload()
            return result

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {"job_id": job["job_id"], "status": "running", "doc_id": doc_id}

    def start_update(self, doc_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        if self.registry.get(doc_id) is None:
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
        cfg = copy.deepcopy(self.cfg)

        def work() -> dict[str, Any]:
            self.registry.update(doc_id, **values)
            result = rebuild_with_uploads(
                cfg, self.registry, embed_fn=self.embed_fn, progress=self._progress(job)
            )
            self.reload()
            return {"updated": doc_id, "fields": values, **result}

        threading.Thread(target=lambda: self._finish_job(job, work), daemon=True).start()
        return {"job_id": job["job_id"], "status": "running", "doc_id": doc_id}


def create_app(config_path: str | None = None, embed_fn=None) -> FastAPI:
    cfg = load_config(config_path)
    service = Service(cfg, embed_fn=embed_fn)
    app = FastAPI(title="mercatoria", version="0.1.0")
    app.state.service = service

    def unavailable(exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "corpus": service.index.corpus_block() if service.index else None,
            },
        )

    @app.get("/health")
    def health() -> Any:
        if service.index is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_built", "error": service.error},
            )
        if not service.index.total_chunks:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "empty",
                    "error": service.error,
                    "corpus": service.index.corpus_block(),
                },
            )
        return {
            "status": "ready",
            "pending_rebuild": service.pending_rebuild,
            "index": {
                "version": service.index.version,
                "chunks": service.index.total_chunks,
                "documents": service.index.documents,
                "embedding_model": service.index.embedding_model,
            },
            "config": service.cfg.public_dict(),
        }

    @app.get("/v1/config")
    def get_config() -> dict[str, Any]:
        return {
            "config": service.cfg.public_dict(),
            "active_index": service.index.version if service.index else None,
            "pending_rebuild": service.pending_rebuild,
        }

    @app.put("/v1/config")
    def put_config(body: dict[str, Any]) -> dict[str, Any]:
        pending = service._apply_updates(body)
        return {
            "config": service.cfg.public_dict(),
            "active_index": service.index.version if service.index else None,
            "pending_rebuild": service.pending_rebuild,
        }

    @app.post("/retrieve")
    @app.post("/v1/retrieve")
    def retrieve(body: dict[str, Any]) -> Any:
        try:
            retriever = service.require_retriever()
            return retriever.retrieve(
                queries=body.get("queries") or body.get("question") or [],
                top_k=body.get("top_k"),
                filters=body.get("filters") or {},
                fields=body.get("fields") or {},
                use_rerank=body.get("rerank"),
            )
        except (IndexNotFound, CorpusUnavailable) as exc:
            return unavailable(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/chat")
    @app.post("/v1/chat")
    @app.post("/ask")
    @app.post("/v1/ask")
    def chat_answer(body: dict[str, Any]) -> Any:
        question, history, _ = _parse_conversation(body)
        query = str(body.get("retrieval_query") or "").strip() or _retrieval_query(
            question, history
        )
        try:
            retriever = service.require_retriever()
            result = retriever.retrieve(
                queries=body.get("queries") or [query],
                top_k=body.get("top_k"),
                filters=body.get("filters") or {},
                fields=body.get("fields") or {},
                use_rerank=body.get("rerank"),
            )
        except (IndexNotFound, CorpusUnavailable) as exc:
            return unavailable(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        answer_opt = body.get("answer")
        if isinstance(answer_opt, dict) and answer_opt.get("enabled") is False:
            answer_cfg = copy.deepcopy(service.cfg.answer)
            answer_cfg.enabled = False
        else:
            answer_cfg = service.cfg.answer
        try:
            result.update(
                answer_question(answer_cfg, question, result["chunks"], history=history)
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
        result["question"] = question
        result["retrieval_query"] = query
        result["messages"] = history
        result["messages"].append({"role": "user", "content": question})
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
        if service.index is None:
            raise HTTPException(status_code=503, detail="index is not built")
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        rows = [chunk for chunk in service.index.chunks if chunk["doc_id"] == doc_id]
        page = rows[offset : offset + limit]
        return {
            "doc_id": doc_id,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
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
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    @app.get("/v1/admin/rebuild/{job_id}")
    def rebuild_status(job_id: str) -> dict[str, Any]:
        job = service.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/v1/admin/activate/{version}")
    def activate(version: str) -> dict[str, Any]:
        try:
            activate_index(service.cfg.resolved_index_dir(), version)
            service.reload()
            return {"active": service.index.version if service.index else None}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/admin/indexes")
    def indexes() -> dict[str, Any]:
        d = service.cfg.resolved_index_dir()
        files = sorted(p.name for p in d.glob("index-*.db")) if d.is_dir() else []
        current = None
        cur = d / "current.txt"
        if cur.is_file():
            current = cur.read_text(encoding="utf-8").strip()
        return {"indexes": files, "current": current}

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "mercatoria",
            "endpoints": [
                "/health",
                "/v1/retrieve",
                "/v1/chat",
                "/v1/ask",
                "/v1/documents",
                "/v1/jobs/{job_id}",
                "/v1/config",
                "/v1/admin/rebuild",
            ],
        }

    return app


app = create_app()
