from __future__ import annotations

import io
import unittest

from mercatoria.parse import ParseError, normalize_markdown, parse_document


class MarkdownTests(unittest.TestCase):
    def test_chapter_gets_heading_and_article_stays(self) -> None:
        text = "第一章 总则\n第一条 关税的征收适用本法。\n"
        out = normalize_markdown(text)
        self.assertIn("## 第一章 总则", out)
        self.assertIn("第一条 关税的征收适用本法。", out)

    def test_markdown_passthrough(self) -> None:
        parsed = parse_document("a.md", "## § 1.1 Scope\n\nText.".encode())
        self.assertEqual(parsed.format, "md")
        self.assertIn("## § 1.1 Scope", parsed.text)

    def test_txt_decodes_gb18030(self) -> None:
        parsed = parse_document("a.txt", "第一条 中文。".encode("gb18030"))
        self.assertEqual(parsed.format, "txt")
        self.assertIn("第一条", parsed.text)


class DocxTests(unittest.TestCase):
    def test_paragraphs_headings_and_tables(self) -> None:
        from docx import Document

        doc = Document()
        doc.add_heading("测试法", level=1)
        doc.add_paragraph("第一章 总则")
        doc.add_paragraph("第一条 这是测试。")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "项目"
        table.cell(0, 1).text = "税率"
        table.cell(1, 0).text = "A"
        table.cell(1, 1).text = "5%"
        buffer = io.BytesIO()
        doc.save(buffer)

        parsed = parse_document("b.docx", buffer.getvalue())
        self.assertEqual(parsed.format, "docx")
        self.assertIn("## 测试法", parsed.text)
        self.assertIn("## 第一章 总则", parsed.text)
        self.assertIn("第一条 这是测试。", parsed.text)
        self.assertIn("| 项目 | 税率 |", parsed.text)

    def test_rejects_non_zip(self) -> None:
        with self.assertRaises(ParseError):
            parse_document("b.docx", b"not a docx")


class PdfTests(unittest.TestCase):
    def test_text_layer(self) -> None:
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Article 1\nThis is a test with enough text.", fontsize=12)
        data = doc.tobytes()
        doc.close()

        parsed = parse_document("c.pdf", data)
        self.assertEqual(parsed.format, "pdf")
        self.assertEqual(parsed.pages, 1)
        self.assertIn("## Article 1", parsed.text)

    def test_pdf_without_text_layer_fails(self) -> None:
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()
        data = doc.tobytes()
        doc.close()
        with self.assertRaises(ParseError) as ctx:
            parse_document("empty.pdf", data)
        self.assertIn("no text layer", str(ctx.exception))

    def test_rejects_non_pdf(self) -> None:
        with self.assertRaises(ParseError):
            parse_document("c.pdf", b"not a pdf")


class LegacyDocTests(unittest.TestCase):
    def test_doc_is_rejected_with_guidance(self) -> None:
        with self.assertRaises(ParseError) as ctx:
            parse_document("old.doc", b"\xd0\xcf\x11\xe0")
        self.assertIn("convert it to .docx", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
