"""Check that the local corpus contains the expected documents.

The shell entry point (``scripts/setup-corpus.sh --check``) delegates here so
the expected counts and the optional-source rules live in one place and are
unit-testable without spawning a shell.

Exit code: 0 when every required source is present, 1 otherwise. Optional
sources (``legislation.gov.uk`` blocks non-browser clients, so uk/ is often
empty) never fail the check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (relative dir, expected *.md count, optional)
SPECS: list[tuple[str, int, bool]] = [
    ("ecfr/title-19", 51, False),
    ("ecfr/title-15", 16, False),
    ("ecfr/title-31", 6, False),
    ("wto", 14, False),
    ("au", 5, False),
    ("jp", 9, False),
    ("eu", 2, False),
    ("cn", 39, False),
    ("uncitral", 1, False),
    ("uk", 1, True),
]


def check(corpus_dir: str | Path) -> tuple[bool, list[str]]:
    """Return (ok, report lines)."""
    root = Path(corpus_dir)
    lines: list[str] = ["检查语料完整性…"]
    ok = True
    for relative, want, optional in SPECS:
        found = len(list((root / relative).glob("*.md"))) if (root / relative).is_dir() else 0
        if found >= want:
            status = "OK"
        elif optional:
            status = "⚠ optional（抓不到不影响结果）"
        else:
            status = f"缺 {want - found} 个"
            ok = False
        lines.append(f"  {relative:<16} {found:3d}/{want:<3d} {status}")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    args = ap.parse_args(argv)
    ok, lines = check(args.corpus)
    for line in lines:
        print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
