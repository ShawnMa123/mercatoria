"""Build the P1-7 quota fixture: multi-jurisdiction filters, one relevant law.

Each case is a real dev question about one legal system, retrieved with a
filter that requests that system plus one distractor jurisdiction. That is the
regime where a hard quota can waste a slot on an irrelevant law.

Usage:
    python tools/make_quota_cases.py [--seed 20261113]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "corpus" / "eval" / "test_cases.json"
OUT = ROOT / "tests" / "eval" / "quota_cases.json"

SOURCE_TO_JURISDICTION = {
    "cn-laws": "CN",
    "ecfr": "US",
    "wto": "WTO",
    "uncitral": "INT",
    "jp": "JP",
    "au": "AU",
}
DISTRACTORS = ["CN", "US", "WTO", "JP", "AU", "INT"]


def build(cases: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    singles = [
        case
        for case in cases
        if len(case.get("jurisdiction") or []) == 1 and case.get("must_mention")
    ]
    rng.shuffle(singles)
    picked: list[dict] = []
    for case in singles:
        if len(picked) >= count:
            break
        relevant = SOURCE_TO_JURISDICTION.get((case.get("jurisdiction") or [""])[0])
        if not relevant:
            continue
        distractor = next(j for j in DISTRACTORS if j != relevant)
        picked.append(
            {
                "id": f"Q-{len(picked) + 1:02d}",
                "source_id": case["id"],
                "category": case["category"],
                "question": case["q_zh"],
                "must_mention": case["must_mention"],
                "relevant_jurisdictions": [relevant],
                "requested_jurisdictions": [relevant, distractor],
            }
        )
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    cases = json.loads(SOURCE.read_text(encoding="utf-8"))
    picked = build(cases, args.count, args.seed)
    payload = {
        "_meta": {
            "purpose": "P1-7: multi-jurisdiction filters where only one jurisdiction is relevant",
            "generated_by": "tools/make_quota_cases.py",
            "source": "corpus/eval/test_cases.json",
            "seed": args.seed,
            "rule": "single-jurisdiction dev case + one rotating distractor jurisdiction",
        },
        "cases": picked,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(picked)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
