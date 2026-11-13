"""Configuration for the lightweight mercatoria retrieval/answer service.

Everything is plain dataclasses + JSON so the service can be embedded into an
agent without a framework-specific config layer. Environment variables override
only secrets/endpoints; chunking and retrieval parameters live in JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    """Load repo-root .env without overriding real environment variables."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass
class ChunkingConfig:
    # "article" splits at legal article boundaries; "section" keeps each
    # markdown heading together; "fixed" is a plain character window.
    unit: str = "article"
    max_chars: int = 2048
    min_chars: int = 0
    overlap: int = 120
    include_heading: bool = True
    strip_comments: bool = True
    drop_preamble: bool = True
    drop_sections: bool = True
    # 0 = unlimited. A legal article must never be silently truncated.
    max_chunks_per_article: int = 0


@dataclass
class EmbeddingConfig:
    base: str = "http://localhost:3425/v1"
    model: str = "my-team-2k/qwen3.7-text-embedding"
    api_key: str = ""
    batch_size: int = 16
    max_batch: int = 20
    concurrency: int = 4
    timeout: int = 120
    dim: int = 0
    retries: int = 3
    cache_path: str = "data/embedding-cache.db"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    vector_weight: float = 0.8
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    similarity_threshold: float = 0.0
    # Multi-query fusion: a chunk hit by several sub-queries is more likely to
    # answer the whole question, so it earns a small bounded bonus.
    multi_query_bonus: float = 0.03
    # Append English equivalents of known Chinese legal terms so a Chinese
    # question can reach WTO/CISG English text without a translation call.
    expand_cross_language: bool = True


@dataclass
class RerankConfig:
    # Cross-encoder reranking of the first-stage candidate set. Disabled by
    # default so offline tests and installations without a reranker keep the
    # previous behaviour; enable it in config.json or RERANK_ENABLED=1.
    enabled: bool = False
    # "dashscope" is the native gte-rerank-v2 contract
    # (input.query/input.documents); "cohere" is the /v1/rerank contract
    # served by Jina, Cohere, TEI and magpie.
    style: str = "dashscope"
    base: str = "https://llm-tlndh47wd9ustbss.cn-beijing.maas.aliyuncs.com"
    path: str = "/api/v1/services/rerank/text-rerank/text-rerank"
    model: str = "gte-rerank-v2"
    api_key: str = ""
    # First-stage hits handed to the reranker. The final top_k is taken from
    # this list, so candidates must be >= top_k.
    candidates: int = 10
    # 0 = score every candidate the client sends.
    top_n: int = 0
    # Rerank contribution in max fusion:
    #   final = max(first_stage_norm, weight * rerank_norm)
    # Pure rerank overrides exact legal matches; max fusion keeps them.
    weight: float = 0.5
    # "max" | "weighted" | "rerank" -- how first-stage and rerank scores fuse.
    # The 1000-question benchmark picked weighted 0.5 for MRR.
    fusion: str = "weighted"
    min_score: float = 0.0
    timeout: float = 30.0
    retries: int = 2


@dataclass
class AnswerConfig:
    enabled: bool = True
    base: str = "http://localhost:3425/v1"
    model: str = "local-wb/cn:deepseek-v4.1-flash"
    api_key: str = ""
    max_tokens: int = 8192
    temperature: float = 0.0
    timeout: int = 180
    max_context_chunks: int = 12


@dataclass
class AppConfig:
    corpus_dir: str = "corpus"
    index_dir: str = "data/index"
    manifest_path: str = "corpus/manifest.json"
    # Uploaded documents: originals, parsed Markdown and the registry.
    uploads_dir: str = "data/uploads"
    documents_dir: str = "data/documents"
    max_upload_bytes: int = 20 * 1024 * 1024
    max_batch_bytes: int = 200 * 1024 * 1024
    ingest_workers: int = 8
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)

    def resolved_corpus_dir(self) -> Path:
        return _resolve(self.corpus_dir)

    def resolved_index_dir(self) -> Path:
        return _resolve(self.index_dir)

    def resolved_manifest_path(self) -> Path:
        return _resolve(self.manifest_path)

    def resolved_uploads_dir(self) -> Path:
        return _resolve(self.uploads_dir)

    def resolved_documents_dir(self) -> Path:
        return _resolve(self.documents_dir)

    def chunking_hash(self) -> str:
        payload = {
            "chunking": asdict(self.chunking),
            "embedding_model": self.embedding.model,
            "embedding_dim": self.embedding.dim,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public_dict(self) -> dict[str, Any]:
        """Config safe to return over HTTP: redact write-only secrets."""
        data = self.to_dict()
        for section in ("embedding", "rerank", "answer"):
            if data.get(section, {}).get("api_key"):
                data[section]["api_key"] = "***"
        return data


def _merge_dataclass(cls, data: dict[str, Any] | None):
    obj = cls()
    for key, value in (data or {}).items():
        if hasattr(obj, key):
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load JSON config, then apply endpoint/secret environment overrides."""
    _load_env_file()
    data: dict[str, Any] = {}
    if path:
        p = Path(path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
    elif (REPO_ROOT / "config.json").is_file():
        data = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))

    cfg = AppConfig(
        corpus_dir=data.get("corpus_dir", AppConfig.corpus_dir),
        index_dir=data.get("index_dir", AppConfig.index_dir),
        manifest_path=data.get("manifest_path", AppConfig.manifest_path),
        uploads_dir=data.get("uploads_dir", AppConfig.uploads_dir),
        documents_dir=data.get("documents_dir", AppConfig.documents_dir),
        max_upload_bytes=int(data.get("max_upload_bytes", AppConfig.max_upload_bytes)),
        max_batch_bytes=int(data.get("max_batch_bytes", AppConfig.max_batch_bytes)),
        ingest_workers=int(data.get("ingest_workers", AppConfig.ingest_workers)),
        chunking=_merge_dataclass(ChunkingConfig, data.get("chunking")),
        embedding=_merge_dataclass(EmbeddingConfig, data.get("embedding")),
        retrieval=_merge_dataclass(RetrievalConfig, data.get("retrieval")),
        rerank=_merge_dataclass(RerankConfig, data.get("rerank")),
        answer=_merge_dataclass(AnswerConfig, data.get("answer")),
    )

    # Endpoints/secrets are environment-first so a deployed agent keeps them
    # out of the checked-in config file.
    cfg.corpus_dir = _env("MERCATORIA_CORPUS_DIR", cfg.corpus_dir)
    cfg.index_dir = _env("MERCATORIA_INDEX_DIR", cfg.index_dir)
    cfg.manifest_path = _env("MERCATORIA_MANIFEST", cfg.manifest_path)
    cfg.uploads_dir = _env("MERCATORIA_UPLOADS_DIR", cfg.uploads_dir)
    cfg.documents_dir = _env("MERCATORIA_DOCUMENTS_DIR", cfg.documents_dir)
    if _env("MERCATORIA_MAX_UPLOAD_MB"):
        cfg.max_upload_bytes = int(_env("MERCATORIA_MAX_UPLOAD_MB")) * 1024 * 1024
    if _env("MERCATORIA_MAX_BATCH_MB"):
        cfg.max_batch_bytes = int(_env("MERCATORIA_MAX_BATCH_MB")) * 1024 * 1024
    if _env("MERCATORIA_INGEST_WORKERS"):
        cfg.ingest_workers = int(_env("MERCATORIA_INGEST_WORKERS"))
    cfg.embedding.base = _env("EMBEDDING_BASE", cfg.embedding.base)
    cfg.embedding.model = _env("EMBEDDING_MODEL", cfg.embedding.model)
    cfg.embedding.api_key = _env("EMBEDDING_API_KEY", cfg.embedding.api_key)
    if _env("RERANK_ENABLED"):
        cfg.rerank.enabled = _env("RERANK_ENABLED").lower() in {"1", "true", "yes"}
    cfg.rerank.style = _env("RERANK_STYLE", cfg.rerank.style)
    cfg.rerank.base = _env("RERANK_BASE", cfg.rerank.base)
    cfg.rerank.path = _env("RERANK_PATH", cfg.rerank.path)
    cfg.rerank.model = _env("RERANK_MODEL", cfg.rerank.model)
    cfg.rerank.api_key = _env("RERANK_API_KEY", cfg.rerank.api_key)
    if _env("RERANK_CANDIDATES"):
        cfg.rerank.candidates = int(_env("RERANK_CANDIDATES"))
    if _env("RERANK_WEIGHT"):
        cfg.rerank.weight = float(_env("RERANK_WEIGHT"))
    if _env("RERANK_TOP_N"):
        cfg.rerank.top_n = int(_env("RERANK_TOP_N"))
    if _env("RERANK_TIMEOUT"):
        cfg.rerank.timeout = float(_env("RERANK_TIMEOUT"))
    cfg.answer.base = _env("LLM_BASE", cfg.answer.base)
    cfg.answer.model = _env("LLM_MODEL", cfg.answer.model)
    cfg.answer.api_key = _env("LLM_API_KEY", cfg.answer.api_key)
    if _env("MERCATORIA_ANSWER_ENABLED"):
        cfg.answer.enabled = _env("MERCATORIA_ANSWER_ENABLED").lower() in {"1", "true", "yes"}
    return cfg


def save_config(cfg: AppConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
