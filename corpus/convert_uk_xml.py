"""Convert a locally downloaded legislation.gov.uk ``data.xml`` to markdown.

``legislation.gov.uk`` answers scripted clients with an AWS WAF challenge
(HTTP 202, empty body), so the fetch cannot be automated from every network.
Download ``https://www.legislation.gov.uk/ukpga/2018/22/data.xml`` in a browser,
then run:

    python corpus/convert_uk_xml.py --input data.xml \
        --slug uk-cross-border-trade-2018 \
        --title "Taxation (Cross-border Trade) Act 2018"

It writes ``corpus/uk/<slug>.md`` and updates ``corpus/uk/index.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "uk"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_eu_uk import uk_to_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="local data.xml")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", default=str(CORPUS))
    args = ap.parse_args()

    raw = Path(args.input).read_bytes()
    md, sections = uk_to_markdown(raw, args.title)
    if sections == 0:
        print("未识别到 Section，未写入", file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.slug}.md"
    (out_dir / filename).write_text(md, encoding="utf-8")

    index_path = out_dir / "index.json"
    index = {"source": "legislation.gov.uk (manual download)", "fetched": [], "missing": []}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    index.setdefault("fetched", [])
    index["fetched"] = [item for item in index["fetched"] if item.get("slug") != args.slug]
    index["fetched"].append(
        {
            "slug": args.slug,
            "file": filename,
            "title": args.title,
            "sections": sections,
            "chars": len(md),
            "source": "manual download (WAF blocks scripted clients)",
        }
    )
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{filename}: {sections} sections, {len(md):,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
