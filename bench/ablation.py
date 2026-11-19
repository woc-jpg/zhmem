"""Ablation: turn the design decisions into numbers.

Each row disables one decision and reruns the same 68 queries over the same 60
memories. The point is not to show the default winning everything — it is to show
*how* each choice earns its place, and to leave the numbers where a reviewer can
reproduce them with `python -m bench.ablation`.

Run from the project root:  python -m bench.ablation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from zhmem import text as T
from zhmem.store import MemoryStore
from zhmem.eval.corpus import QUERIES, build

# evaluations that return a hit for a negative query are counted; Recall/MRR use
# positives only, exactly as in zhmem.eval.metrics.


def measure(**store_kwargs) -> dict[str, float]:
    store = MemoryStore(":memory:", **store_kwargs)
    mapping = build(store)
    pos = [q for q in QUERIES if not q.is_negative]
    neg = [q for q in QUERIES if q.is_negative]
    r1 = r5 = 0
    rr = 0.0
    for q in pos:
        hits = store.recall(q.namespace, q.query, limit=5, recent=q.recent)
        gold = {mapping[q.namespace][k] for k in q.gold}
        ids = [h.id for h in hits]
        if ids and ids[0] in gold:
            r1 += 1
            rr += 1.0
        elif any(i in gold for i in ids):
            rank = next(i for i, x in enumerate(ids, 1) if x in gold)
            rr += 1.0 / rank
        if any(i in gold for i in ids):
            r5 += 1
    fp = sum(1 for q in neg if store.recall(q.namespace, q.query, limit=1))
    n = len(pos)
    return {
        "R@1": round(r1 / n, 3),
        "R@5": round(r5 / n, 3),
        "MRR": round(rr / n, 3),
        "FPR": round(fp / len(neg), 3),
    }


_orig_terms = T.query_terms
_orig_resolve = T.resolve_query


def allchar_terms(query):
    """Variant that gives every content character unigram weight — the config
    that reached R@5=0.978 with FPR=0.750 before isolation was introduced."""
    weights: dict[str, float] = {}
    for kind, run in T._content_segments(T.strip_query_noise(query)):
        if kind == "latin":
            weights[run] = 1.0
            continue
        for ch in run:
            weights.setdefault(ch, 0.2)
        for i in range(len(run) - 1):
            weights[run[i:i + 2]] = 1.0
    return weights


def main() -> None:
    rows: dict[str, dict[str, float]] = {}

    rows["default (coverage + fold both + isolated char + ctx max)"] = measure()

    # 1. bm25 instead of coverage.
    rows["scorer=bm25"] = measure(scorer="bm25")

    # 2. trigram tokeniser (cannot represent a two-character word).
    rows["tokenizer=trigram"] = measure(tokenizer="trigram")

    # 3. synonym folding on one side only, or neither.
    rows["fold: query side only"] = measure(normalize=None)
    rows["fold: index side only"] = measure(query_normalize=None)
    rows["fold: neither side"] = measure(normalize=None, query_normalize=None)

    # 4. every content char gets unigram weight (isolated-char trade).
    T.query_terms = allchar_terms
    rows["all-char unigram weight"] = measure()
    T.query_terms = _orig_terms

    # 5. char tier off — only word matches count.
    rows["char_tier=off"] = measure(char_tier=False)

    # 6. context ignored (no previous turn).
    rows["no previous turn"] = measure(context_weight=0.0)

    # 7. old concatenation instead of separate-query max. Merge the turns into
    #    one string AND clear `recent` so the ctx-max path cannot also fire —
    #    otherwise this measures concatenation + max, not concatenation alone.
    def concat_recall(self, namespace, query, *, limit=8, candidate_limit=200, kinds=None, recent=()):
        merged = T.resolve_query(query, recent)
        return _orig_recall(self, namespace, merged, limit=limit,
                            candidate_limit=candidate_limit, kinds=kinds, recent=())  # no recent: merged already carries it
    _orig_recall = MemoryStore.recall
    MemoryStore.recall = concat_recall
    rows["concatenate previous turn"] = measure()
    MemoryStore.recall = _orig_recall

    # NOTE: min-max normalisation is deliberately absent from this table. Its
    # failure only appears when *every* candidate scores zero (min==max), which
    # cannot be reproduced by patching the per-row `coverage()` — it needs the
    # whole candidate set. The measured case is documented in README; shipping
    # a fake stand-in here would imply the effect is small when it is not.

    print(f"{'variant':<58} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'FPR':>6}")
    print("-" * 84)
    for name, m in rows.items():
        print(f"{name:<58} {m['R@1']:>6} {m['R@5']:>6} {m['MRR']:>6} {m['FPR']:>6}")

    out = Path(__file__).with_name("ablation_results.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
