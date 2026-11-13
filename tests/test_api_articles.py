"""Article routes accept ids containing '/', '§' and '.' (P0-5-3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mercatoria.api import create_app
from mercatoria.config import load_config
from mercatoria.manifest import load_manifest
from mercatoria.store import build_index


def fake_embed(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 3), dtype=np.float32)


class ArticleRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        corpus = root / "corpus"
        (corpus / "cn").mkdir(parents=True)
        (corpus / "cn" / "a.md").write_text(
            "第一条 关税的征收适用本法。\n\n第二条 原产地规则另行规定。\n",
            encoding="utf-8",
        )
        manifest = {
            "manifest_version": "test",
            "docs": [
                {
                    "doc_id": "cn/law/x",
                    "path": "cn/a.md",
                    "source_doc": "斜杠文档",
                    "jurisdiction": "CN",
                    "language": "zh",
                    "precision": "exact",
                    "applies_to": ["customs"],
                    "effective_from": "2024-01-01",
                    "effective_to": None,
                    "snapshot_date": None,
                }
            ],
        }
        self.manifest_path = root / "manifest.json"
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.config_path = root / "config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "corpus_dir": str(corpus),
                    "index_dir": str(root / "index"),
                    "manifest_path": str(self.manifest_path),
                    "documents_dir": str(root / "documents"),
                    "uploads_dir": str(root / "uploads"),
                    "embedding": {"dim": 3},
                    "answer": {"enabled": False},
                }
            ),
            encoding="utf-8",
        )
        cfg = load_config(self.config_path)
        build_index(
            load_manifest(self.manifest_path), cfg, cfg.resolved_index_dir(), embed_fn=fake_embed
        )
        self.app = create_app(str(self.config_path), embed_fn=fake_embed)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        service = self.app.state.service
        if service.index is not None:
            service.index.close()
        service.registry.close()
        self.tmp.cleanup()

    def test_slash_in_article_id_is_routed(self) -> None:
        resp = self.client.get("/v1/articles/cn/law/x:1")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["article_id"], "cn/law/x:1")

    def test_unknown_id_is_404_not_405(self) -> None:
        resp = self.client.get("/v1/articles/%C2%A7%20101.1")
        self.assertEqual(resp.status_code, 404)

    def test_batch_accepts_slashed_ids(self) -> None:
        resp = self.client.post(
            "/v1/articles/batch", json={"article_ids": ["cn/law/x:2"], "follow_refs": 1}
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["articles"][0]["article_id"], "cn/law/x:2")


if __name__ == "__main__":
    unittest.main()
