from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mercatoria.config import AppConfig
from mercatoria.documents import DocumentRegistry, combined_manifest
from mercatoria.ingest import UploadItem, ingest_batch, reindex_document
from mercatoria.store import open_current


def fake_embed(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        if "关税" in text:
            rows.append([1.0, 0.0, 0.0])
        elif "原产地" in text:
            rows.append([0.0, 1.0, 0.0])
        else:
            rows.append([0.0, 0.0, 1.0])
    return np.asarray(rows, dtype=np.float32)


def make_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading("测试上传法", level=1)
    doc.add_paragraph("第一章 总则")
    doc.add_paragraph("第一条 上传文件的关税规定。")
    doc.add_paragraph("第二条 原产地规则另行规定。")
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        (self.corpus / "cn").mkdir(parents=True)
        (self.corpus / "cn" / "base.md").write_text(
            "<!-- document: 基础关税法 -->\n第一条 基础关税条文。\n", encoding="utf-8"
        )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "manifest_version": "test",
                    "docs": [
                        {
                            "doc_id": "cn-base",
                            "path": "cn/base.md",
                            "source_doc": "基础关税法",
                            "jurisdiction": "CN",
                            "language": "zh",
                            "precision": "exact",
                            "applies_to": ["customs"],
                            "effective_from": None,
                            "effective_to": None,
                            "snapshot_date": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.cfg = AppConfig(
            corpus_dir=str(self.corpus),
            index_dir=str(self.root / "index"),
            manifest_path=str(self.manifest),
            documents_dir=str(self.root / "documents"),
            uploads_dir=str(self.root / "uploads"),
        )
        self.cfg.embedding.dim = 3
        self.cfg.embedding.batch_size = 2
        self.registry = DocumentRegistry(self.cfg.resolved_documents_dir() / "registry.db")

    def tearDown(self) -> None:
        self.registry.close()
        self.tmp.cleanup()

    def test_upload_markdown_and_docx_is_indexed(self) -> None:
        items = [
            UploadItem(
                filename="upload.md",
                data="# 第一章 总则\n第一条 上传关税。\n".encode(),
                applies_to=["customs"],
            ),
            UploadItem(
                filename="upload.docx",
                data=make_docx(),
                applies_to=["customs"],
            ),
        ]
        result = ingest_batch(self.cfg, self.registry, items, embed_fn=fake_embed)
        self.assertEqual(result["indexed"], 2)
        self.assertEqual(len(result["failed"]), 0)
        docs = {doc["doc_id"]: doc for doc in self.registry.list()}
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(doc["status"] == "ready" for doc in docs.values()))
        self.assertTrue(all(doc["chunk_count"] > 0 for doc in docs.values()))

        index = open_current(self.cfg.resolved_index_dir())
        try:
            indexed_docs = {chunk["doc_id"] for chunk in index.chunks}
            self.assertTrue(set(docs).issubset(indexed_docs))
            self.assertIn("cn-base", indexed_docs)
        finally:
            index.close()

    def test_duplicate_upload_is_idempotent(self) -> None:
        item = UploadItem(filename="again.md", data=b"## Test\n\nhello\n")
        first = ingest_batch(self.cfg, self.registry, [item], embed_fn=fake_embed)
        second = ingest_batch(self.cfg, self.registry, [item], embed_fn=fake_embed)
        self.assertEqual(first["indexed"], 1)
        self.assertEqual(second["indexed"], 0)
        self.assertTrue(second["documents"][0]["duplicate"])
        self.assertEqual(len(self.registry.list()), 1)

    def test_parse_failure_marks_document_failed(self) -> None:
        result = ingest_batch(
            self.cfg,
            self.registry,
            [UploadItem(filename="bad.pdf", data=b"not a pdf")],
            embed_fn=fake_embed,
        )
        self.assertEqual(result["indexed"], 0)
        self.assertEqual(len(result["failed"]), 1)
        failed = self.registry.list()[0]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("PDF", failed["error"])

    def test_reindex_reuses_original(self) -> None:
        item = UploadItem(filename="re.md", data=b"## Test\n\nfirst\n")
        result = ingest_batch(self.cfg, self.registry, [item], embed_fn=fake_embed)
        doc_id = result["documents"][0]["doc_id"]
        reindexed = reindex_document(self.cfg, self.registry, doc_id, embed_fn=fake_embed)
        self.assertEqual(reindexed["documents"], 2)
        self.assertEqual(self.registry.get(doc_id)["status"], "ready")

    def test_combined_manifest_includes_uploads(self) -> None:
        ingest_batch(
            self.cfg,
            self.registry,
            [UploadItem(filename="one.md", data=b"## Test\n\nhello\n")],
            embed_fn=fake_embed,
        )
        base = json.loads(self.manifest.read_text(encoding="utf-8"))
        merged = combined_manifest(base, self.registry)
        self.assertEqual(merged["documents"], 2)
        self.assertTrue(any(doc["path"].endswith(".md") for doc in merged["docs"]))


if __name__ == "__main__":
    unittest.main()
