"""Configuration for the lightweight mercatoria retrieval/answer service.

Everything is plain dataclasses + JSON so the service can be embedded into an
agent without a framework-specific config layer. Environment variables override
only secrets/endpoints; chunking and retrieval parameters live in JSON.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, is_dataclass
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
    # Prefix the *indexed* text (embedding + BM25) with
    # "[法典名 | 章节 | 条号]" so long-article part>=1 chunks and
    # chapter-dependent articles keep their retrieval signal. ChunkRecord.text
    # and chunk_id stay derived from the bare article text.
    index_context_header: bool = True


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
    # LRU cache for query vectors: (model, normalized query) -> vector.
    # 0 disables it. Shared by every request of a Retriever.
    query_cache_size: int = 1024


@dataclass
class ConfidenceConfig:
    """Thresholds for the retrieval confidence label (diagnostics only).

    Calibrated on the dev trade-law set: score = (1-w)*idf_coverage +
    w*top1_cosine, where idf_coverage is the IDF-weighted share of query
    terms that literally occur in the top-1 chunk. 0.64 marks the boundary
    between "the corpus talks about this" and "it does not".
    """

    # score at or above which a hit may be labelled "high".
    high: float = 0.80
    # below this score the answer is "low".
    medium: float = 0.64
    # minimum top1-top2 gap for "high".
    min_gap: float = 0.02
    # weight of the raw cosine in the score (1-w goes to term coverage).
    vector_weight: float = 0.8


@dataclass
class RetrievalConfig:
    top_k: int = 5
    vector_weight: float = 0.8
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    similarity_threshold: float = 0.0
    # Multi-query fusion bonus. The v2 ablation (A6) measured -0.37pp Hit@5 /
    # -0.85pp MRR, so it defaults to off; the knob stays for future tuning.
    multi_query_bonus: float = 0.0
    # Append English equivalents of known Chinese legal terms so a Chinese
    # question can reach WTO/CISG English text without a translation call.
    expand_cross_language: bool = True
    # Drop regulatory boilerplate ("本法", "shall", ...) from query terms.
    query_stopwords: bool = True
    # Scale BM25 normalisation by magnitude, saturating at this raw score.
    # 0 restores the plain _minmax behaviour (kept for ablation runs).
    bm25_saturation: float = 8.0
    # Soft quota (P1-7): a jurisdiction keeps its reserved slot only when its
    # best score is at least this fraction of the global top-1. 0 = hard quota.
    quota_min_ratio: float = 0.0
    # Degrade to BM25-only when the embedding gateway fails.
    fallback_bm25_on_embed_error: bool = False
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)


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

    def resolved_control_path(self) -> Path:
        """Publication authority: the control head lives with the registry."""
        return self.resolved_documents_dir() / "control.db"

    def resolved_write_lock_path(self) -> Path:
        """Cross-process writer lock, shared by the API and offline CLI."""
        return self.resolved_documents_dir() / "write.lock"

    def runtime_fingerprint(self) -> str:
        """Id of everything a request can observe except the built index.

        Chunking and the embedding space are part of the index, not of this
        fingerprint: changing them is a rebuild, not a new runtime snapshot.
        """
        payload = {
            "retrieval": asdict(self.retrieval),
            "rerank": {k: v for k, v in asdict(self.rerank).items() if k != "api_key"},
            "answer": {k: v for k, v in asdict(self.answer).items() if k != "api_key"},
            "embedding": {
                "base": self.embedding.base,
                "model": self.embedding.model,
                "dim": self.embedding.dim,
                "timeout": self.embedding.timeout,
                "retries": self.embedding.retries,
            },
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

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


# Sections a published snapshot records. Deployment paths and limits stay in
# the deployment config (config.json/env) and are never copied into state.
PERSISTED_SECTIONS = ("chunking", "embedding", "retrieval", "rerank", "answer")

_SECRET_FIELDS = {
    ("embedding", "api_key"),
    ("rerank", "api_key"),
    ("answer", "api_key"),
}


def export_public_config(cfg: AppConfig) -> dict[str, Any]:
    """Non-sensitive configuration for durable artifacts.

    Used for control-state rows, index metadata and saved config files. Secrets
    are *absent*, not masked: an artifact that has no field cannot leak one.
    """
    data = copy.deepcopy(asdict(cfg))
    for section, key in _SECRET_FIELDS:
        data.get(section, {}).pop(key, None)
    return {section: data[section] for section in PERSISTED_SECTIONS}


def config_from_sections(base: AppConfig, sections: dict[str, Any] | None) -> AppConfig:
    """Resolve a persisted snapshot's config over a deployment config.

    Paths, limits and credentials come from ``base`` (deployment config), the
    published non-sensitive values from ``sections``. Validation is the same one
    a live config update goes through, so a corrupt or stale row fails closed
    instead of quietly serving different settings.
    """
    if sections is None:
        raise ConfigError("control state has no persisted config")
    updated = copy.deepcopy(base)
    for section in PERSISTED_SECTIONS:
        if section not in sections:
            continue
        setattr(updated, section, _merge_section((section,), getattr(updated, section), sections[section]))
    if updated.chunking.overlap >= updated.chunking.max_chars:
        raise ConfigError("chunking.overlap must be smaller than chunking.max_chars")
    return updated


def build_identity(cfg: AppConfig) -> dict[str, Any]:
    """What an index is built from: chunking, embedding space and endpoint.

    Changing any of these requires a rebuild; everything else is a runtime
    parameter that a new snapshot can carry without touching the vectors.
    """
    return {
        "chunking": asdict(cfg.chunking),
        "embedding_model": cfg.embedding.model,
        "embedding_dim": cfg.embedding.dim,
        "embedding_base": cfg.embedding.base,
    }


def with_active_build(desired: AppConfig, active: AppConfig) -> AppConfig:
    """``desired`` runtime settings on ``active``'s build identity.

    A runtime update must never drag a pending chunking/embedding-space change
    into the serving state: the served index still belongs to the active build.
    """
    serving = copy.deepcopy(desired)
    serving.chunking = copy.deepcopy(active.chunking)
    serving.embedding.base = active.embedding.base
    serving.embedding.model = active.embedding.model
    serving.embedding.dim = active.embedding.dim
    return serving


def _merge_dataclass(cls, data: dict[str, Any] | None):
    obj = cls()
    for key, value in (data or {}).items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, key, _merge_dataclass(type(current), value))
        else:
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
    """Write a durable config file without credentials.

    The file is an operator convenience (deployment paths and secrets still come
    from the environment), so it uses the same non-sensitive export as the
    control state. ``to_dict()`` must never be written to disk.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(export_public_config(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class ConfigError(ValueError):
    """Unknown field, wrong type or out-of-range value in a config update."""


# Bounds for knobs where a wrong value silently degrades every request. A field
# without an entry is still type-checked; ranges are a review aid, not policy.
_RANGES: dict[tuple[str, ...], tuple[float, float]] = {
    ("chunking", "max_chars"): (1, 1_000_000),
    ("chunking", "min_chars"): (0, 1_000_000),
    ("chunking", "overlap"): (0, 1_000_000),
    ("embedding", "batch_size"): (1, 1000),
    ("embedding", "max_batch"): (1, 1000),
    ("embedding", "concurrency"): (1, 1024),
    ("embedding", "timeout"): (1, 3600),
    ("embedding", "retries"): (1, 20),
    ("embedding", "query_cache_size"): (0, 1_000_000),
    ("retrieval", "top_k"): (1, 100),
    ("retrieval", "vector_weight"): (0.0, 1.0),
    ("retrieval", "bm25_k1"): (0.0, 100.0),
    ("retrieval", "bm25_b"): (0.0, 1.0),
    ("retrieval", "similarity_threshold"): (0.0, 1.0),
    ("retrieval", "multi_query_bonus"): (0.0, 10.0),
    ("retrieval", "bm25_saturation"): (0.0, 1000.0),
    ("retrieval", "quota_min_ratio"): (0.0, 1.0),
    ("rerank", "candidates"): (1, 1000),
    ("rerank", "top_n"): (0, 1000),
    ("rerank", "weight"): (0.0, 1.0),
    ("rerank", "min_score"): (0.0, 1000.0),
    ("rerank", "timeout"): (0.1, 3600),
    ("rerank", "retries"): (1, 20),
    ("answer", "max_tokens"): (1, 1_000_000),
    ("answer", "temperature"): (0.0, 2.0),
    ("answer", "timeout"): (1, 3600),
    ("answer", "max_context_chunks"): (1, 1000),
}

_CHOICES: dict[tuple[str, ...], set[str]] = {
    ("chunking", "unit"): {"article", "section", "fixed"},
    ("rerank", "style"): {"dashscope", "cohere"},
    ("rerank", "fusion"): {"max", "weighted", "rerank"},
}

# Endpoints a request actually dials. An invalid value here used to be accepted
# and only fail when a client was constructed *after* the state was published.
_URL_FIELDS = {
    ("embedding", "base"),
    ("rerank", "base"),
    ("answer", "base"),
}

# Top-level keys a config update may touch. Deployment paths and limits are
# environment/deployment concerns and are not remotely editable.
_EDITABLE_TOP = {"chunking", "embedding", "retrieval", "rerank", "answer"}


def _coerce(path: tuple[str, ...], value: Any, current: Any) -> Any:
    name = ".".join(path)
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{name} must be a boolean")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{name} must be an integer")
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name} must be a number")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(f"{name} must be a string")
        return value
    if isinstance(current, list):
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ConfigError(f"{name} must be a list of strings")
        return list(value)
    if current is None:
        if value is not None and not isinstance(value, str):
            raise ConfigError(f"{name} must be a string or null")
        return value
    if isinstance(value, type(current)):
        return value
    raise ConfigError(f"{name} has an unsupported type")


def _check_range(path: tuple[str, ...], value: Any) -> None:
    bounds = _RANGES.get(path)
    if bounds is not None and isinstance(value, (int, float)):
        low, high = bounds
        if not low <= float(value) <= high:
            raise ConfigError(f"{'.'.join(path)} must be in [{low}, {high}]")
    choices = _CHOICES.get(path)
    if choices is not None and value not in choices:
        raise ConfigError(f"{'.'.join(path)} must be one of {sorted(choices)}")
    if path in _URL_FIELDS:
        _check_url(path, value)


def _check_url(path: tuple[str, ...], value: Any) -> None:
    """Reject an endpoint that no client could ever dial."""
    from urllib.parse import urlsplit

    name = ".".join(path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be an http(s) URL")
    parts = urlsplit(value.strip().rstrip("/"))
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ConfigError(f"{name} must be an http(s) URL, got {value!r}")
    try:
        parts.port
    except ValueError as exc:
        raise ConfigError(f"{name} has an invalid port: {value!r}") from exc


def _merge_section(path: tuple[str, ...], current: Any, value: Any) -> Any:
    if not isinstance(value, dict):
        raise ConfigError(f"{'.'.join(path)} must be an object")
    fields = type(current).__dataclass_fields__
    for key, item in value.items():
        field_path = path + (key,)
        if key not in fields:
            raise ConfigError(f"unknown field: {'.'.join(field_path)}")
        if field_path in _SECRET_FIELDS:
            raise ConfigError(
                f"{'.'.join(field_path)} is environment-only; set the corresponding "
                "environment variable instead"
            )
        child = getattr(current, key)
        if is_dataclass(child):
            setattr(current, key, _merge_section(field_path, child, item))
        else:
            coerced = _coerce(field_path, item, child)
            _check_range(field_path, coerced)
            setattr(current, key, coerced)
    return current


def merge_config_update(cfg: AppConfig, body: dict[str, Any] | None) -> AppConfig:
    """Apply a validated deep merge and return a *new* config.

    Nothing is mutated in place: the caller swaps in the result only after the
    whole update validated, so a rejected field never leaves a half-applied
    state behind, and a live snapshot keeps the config it was built with.
    """
    if body is None:
        return copy.deepcopy(cfg)
    if not isinstance(body, dict):
        raise ConfigError("config update must be a JSON object")
    updated = copy.deepcopy(cfg)
    for section, value in body.items():
        if section == "save":
            continue
        if section not in _EDITABLE_TOP:
            raise ConfigError(f"unknown or non-editable field: {section}")
        setattr(updated, section, _merge_section((section,), getattr(updated, section), value))
    if updated.chunking.overlap >= updated.chunking.max_chars:
        raise ConfigError("chunking.overlap must be smaller than chunking.max_chars")
    if updated.ingest_workers < 1:
        raise ConfigError("ingest_workers must be positive")
    return updated


def needs_index_rebuild(before: AppConfig, after: AppConfig) -> bool:
    """True when the change alters the indexed text or the embedding space."""
    return build_identity(before) != build_identity(after)
