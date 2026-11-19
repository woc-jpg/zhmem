"""Make an EvalResult reproducible from the CLI: ``python -m zhmem.eval``.

Exits non-zero when the delivery thresholds are missed, so CI actually catches
a ranking regression instead of merely printing a table nobody reads.
"""

from __future__ import annotations

import sys

from .corpus import build
from .metrics import evaluate, format_report
from ..store import MemoryStore

# Same gates as tests/test_eval.py: a release must clear all of them.
_GATES = {"recall_at5": 0.95, "mrr": 0.90, "fp_rate": 0.15}


def main() -> int:
    store = MemoryStore(":memory:")
    try:
        mapping = build(store)
        result = evaluate(store, mapping)
    finally:
        store.close()
    print(format_report(result, label="zhmem (memory store)"))

    failed: list[str] = []
    if result.recall_at5 < _GATES["recall_at5"]:
        failed.append(f"R@5 {result.recall_at5:.3f} < {_GATES['recall_at5']}")
    if result.mrr < _GATES["mrr"]:
        failed.append(f"MRR {result.mrr:.3f} < {_GATES['mrr']}")
    if result.fp_rate > _GATES["fp_rate"]:
        failed.append(f"FPR {result.fp_rate:.3f} > {_GATES['fp_rate']}")
    for name, c in result.by_category.items():
        if name == "negative":
            # Negatives are supposed to return nothing: hit_rate should be 0 and
            # their quality is already gated by fp_rate above.
            continue
        if c["hit_rate"] < 1.0:
            failed.append(f"{name} hit {c['hit_rate']:.3f} < 1.000")
    if failed:
        print("FAIL: " + "; ".join(failed), file=sys.stderr)
        return 1
    print("PASS: all delivery thresholds met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
