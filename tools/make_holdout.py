"""Freeze a stratified hold-out set from the dev evaluation cases.

The dev set (``corpus/eval/test_cases.json``) is what tuning looks at. This
script samples 100 cases, stratified by category, with a fixed seed, and writes
them to ``tests/eval/holdout_cases.json``. ``tools/eval_local.py`` excludes
those ids unless ``--holdout`` is passed.

Usage:
    python tools/make_holdout.py [--size 100] [--seed 20261113]
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "corpus" / "eval" / "test_cases.json"
OUT = ROOT / "tests" / "eval" / "holdout_cases.json"


def sample(cases: list[dict], size: int, seed: int) -> list[dict]:
    by_category: dict[str, list[dict]] = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)
    rng = random.Random(seed)
    picked: list[dict] = []
    # Largest categories first so the remainder is spread evenly.
    for category in sorted(by_category):
        items = by_category[category]
        rng.shuffle(items)
        share = max(1, round(size * len(items) / len(cases)))
        picked.extend(items[:share])
        by_category[category] = items[share:]
    # Top up from whatever is left, then trim deterministically.
    leftovers = [case for items in by_category.values() for case in items]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: max(0, size - len(picked))])
    picked = picked[:size]
    picked.sort(key=lambda case: case["id"])
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    cases = [case for case in cases if case.get("expect") != "refuse"]
    picked = sample(cases, args.size, args.seed)
    payload = {
        "_meta": {
            "generated_at": date.today().isoformat(),
            "script": "tools/make_holdout.py",
            "source": "corpus/eval/test_cases.json",
            "seed": args.seed,
            "size": len(picked),
            "rule": "stratified by category, sorted by id; excluded from dev runs unless --holdout",
        },
        "cases": picked,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(picked)} cases (seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
