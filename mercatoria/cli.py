"""Command line entry points: manifest, rebuild, serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .documents import DocumentRegistry, combined_manifest
from .lock import DirectoryLock, LockBusy
from .manifest import load_manifest, write_manifest
from .store import build_index


def _cmd_manifest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = Path(args.out or cfg.resolved_manifest_path())
    manifest = write_manifest(args.corpus or cfg.resolved_corpus_dir(), out)
    print(json.dumps({"documents": manifest["documents"], "path": str(out)}, ensure_ascii=False))
    return 0


def _cmd_rebuild(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    # Same lock as the service: an offline rebuild must not move the control
    # head of a data directory a live process is serving.
    lock = DirectoryLock(cfg.resolved_write_lock_path())
    try:
        lock.acquire()
    except LockBusy as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "stop the service first, or use POST /v1/admin/rebuild with the same --config",
            file=sys.stderr,
        )
        return 1
    try:
        manifest_path = Path(args.manifest or cfg.resolved_manifest_path())
        if manifest_path.is_file():
            manifest = load_manifest(manifest_path)
        else:
            manifest = write_manifest(cfg.resolved_corpus_dir(), manifest_path)
        registry = DocumentRegistry(cfg.resolved_documents_dir() / "registry.db")
        try:
            merged = combined_manifest(manifest, registry)
        finally:
            registry.close()
        if args.limit:
            docs = merged["docs"][: args.limit]
            merged = {**merged, "docs": docs, "documents": len(docs)}

        def progress(stage: str, done: int, total: int) -> None:
            if stage in {"chunk", "embed", "done"}:
                print(f"{stage}: {done}/{total}", flush=True)

        result = build_index(
            merged,
            cfg,
            args.out or cfg.resolved_index_dir(),
            activate=not args.no_activate,
            progress=progress,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        lock.release()


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    try:
        app = create_app(args.config)
    except LockBusy as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mercatoria")
    ap.add_argument("--config", default=None, help="JSON config path")
    sub = ap.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="generate corpus/manifest.json")
    p_manifest.add_argument("--corpus", default=None)
    p_manifest.add_argument("--out", default=None)
    p_manifest.set_defaults(func=_cmd_manifest)

    p_rebuild = sub.add_parser("rebuild", help="chunk corpus and rebuild the SQLite index")
    p_rebuild.add_argument("--manifest", default=None)
    p_rebuild.add_argument("--out", default=None, help="index directory or .db path")
    p_rebuild.add_argument("--limit", type=int, default=0, help="only first N docs (smoke)")
    p_rebuild.add_argument(
        "--no-activate",
        action="store_true",
        help="build a candidate only; the served state is unchanged",
    )
    p_rebuild.set_defaults(func=_cmd_rebuild)

    p_serve = sub.add_parser("serve", help="run the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--log-level", default="info")
    p_serve.set_defaults(func=_cmd_serve)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
