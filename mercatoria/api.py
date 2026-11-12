"""HTTP surface: retrieval, optional answer generation, config and rebuild."""

from __future__ import annotations

import copy
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .answer import ChatError, answer_question
from .config import AppConfig, load_config, save_config
from .embed import EmbeddingError
from .manifest import build_manifest, load_manifest, write_manifest
from .retrieve import Retriever
from .store import (
    CorpusUnavailable,
    Index,
    IndexNotFound,
    activate_index,
    build_index,
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
        if body.get("save"):
            save_config(self.cfg, Path("config.json"))
        return pending

    def start_rebuild(self, body: dict[str, Any]) -> dict[str, Any]:
        pending = self._apply_updates(body)
        job_id = uuid.uuid4().hex[:10]
        job = {
            "job_id": job_id,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "result": None,
            "error": None,
            "progress": {"stage": "starting", "done": 0, "total": 0},
        }
        self.jobs[job_id] = job
        cfg = copy.deepcopy(self.cfg)
        background = body.get("background", True)

        def worker() -> None:
            try:
                manifest_path = cfg.resolved_manifest_path()
                if manifest_path.is_file():
                    manifest = load_manifest(manifest_path)
                else:
                    manifest = write_manifest(cfg.resolved_corpus_dir(), manifest_path)

                def progress(stage: str, done: int, total: int) -> None:
                    job["progress"] = {"stage": stage, "done": done, "total": total}

                result = build_index(
                    manifest,
                    cfg,
                    cfg.resolved_index_dir(),
                    embed_fn=self.embed_fn,
                    activate=bool(body.get("activate", True)),
                    progress=progress,
                )
                job["result"] = result
                job["status"] = "done"
                job["finished_at"] = _now()
                if body.get("activate", True):
                    self.reload()
            except Exception as exc:  # background jobs must report, not crash
                job["status"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["finished_at"] = _now()

        if background:
            threading.Thread(target=worker, daemon=True).start()
            return {"job_id": job_id, "status": "running", "pending_rebuild": self.pending_rebuild}
        worker()
        return {"job_id": job_id, **job, "pending_rebuild": self.pending_rebuild}


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
                "/v1/config",
                "/v1/admin/rebuild",
            ],
        }

    return app


app = create_app()
