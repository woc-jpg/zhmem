"""Metrics for the graded query set.

The three reported numbers answer different questions, so all three are printed:

* **Recall@k** — did the right memory appear in the top k at all? For a personal
  memory assistant this is the one that matters: the prompt must contain the fact.
* **MRR** — how high was the first correct one? Ranking it first means the model
  sees it first and the prompt can be shorter.
* **nDCG@k** — graded credit that decays with position, computed from our binary
  gold. Reported because it moves when Recall@k is already saturated.

Negatives are scored separately as **false-positive rate**: the fraction that
returned anything at all. Mixing them into an aggregate Recall number is how a
ranker that returns everything scores 1.0 and looks perfect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .corpus import CATEGORIES, QUERIES, Query

__all__ = ["EvalResult", "evaluate", "format_report", "query_ids"]


def query_ids(store, mapping: dict[str, dict[str, int]], q: Query) -> set[int]:
    return {mapping[q.namespace][key] for key in q.gold if key in mapping.get(q.namespace, {})}


@dataclass
class EvalResult:
    recall: dict[int, float]
    mrr: float
    ndcg: dict[int, float]
    fp_rate: float
    by_category: dict[str, dict[str, float]]
    detail: list[tuple[Query, list[int], set[int]]]

    @property
    def recall_at5(self) -> float:
        return self.recall.get(5, 0.0)


def _dcg(ranks: list[int], k: int) -> float:
    return sum(1.0 / math.log2(r + 2) for r in ranks if r < k)


def _category_name(q: Query) -> str:
    return q.category


def evaluate(
    store,
    mapping: dict[str, dict[str, int]],
    queries: list[Query] | None = None,
    *,
    k_values=(1, 3, 5),
) -> EvalResult:
    """Run retrieval over the graded queries and compute all reported metrics."""
    queries = queries if queries is not None else QUERIES
    recall = {k: 0.0 for k in k_values}
    ndcg = {k: 0.0 for k in k_values}
    mrr_total = 0.0
    neg_total = 0
    neg_wrong = 0
    cats: dict[str, dict[str, float]] = {
        c: {"n": 0, "hit": 0, "mrr": 0.0, "fp": 0} for c in CATEGORIES
    }
    detail: list[tuple[Query, list[int], set[int]]] = []
    counted = 0

    for q in queries:
        gold = query_ids(store, mapping, q)
        # The context is the previous user turns; `recent` is what the caller
        # would pass as the fallback for pronoun resolution.
        hits = store.recall(
            q.namespace,
            q.query,
            limit=max(k_values),
            recent=q.recent,
        )
        got = [int(h.id) for h in hits]
        detail.append((q, got, gold))
        cat = cats.setdefault(_category_name(q), {"n": 0, "hit": 0, "mrr": 0.0, "fp": 0})
        cat["n"] += 1

        if q.is_negative:
            neg_total += 1
            if got:
                neg_wrong += 1
                cat["fp"] += 1
            continue

        counted += 1
        ranks = [got.index(g) for g in gold if g in got]
        best = min(ranks) if ranks else None
        if best is not None:
            mrr_total += 1.0 / (best + 1)
            cat["mrr"] += 1.0 / (best + 1)
            cat["hit"] += 1
        ideal = _dcg(list(range(len(gold))), max(k_values))
        for k in k_values:
            if best is not None and best < k:
                recall[k] += 1.0
            if ideal > 0:
                ndcg[k] += _dcg(ranks, k) / ideal
            else:
                ndcg[k] += 1.0 if not got else 0.0

    for k in k_values:
        if counted:
            recall[k] /= counted
            ndcg[k] /= counted
    summary = {
        "mrr": mrr_total / counted if counted else 0.0,
        "fp_rate": neg_wrong / neg_total if neg_total else 0.0,
    }
    for cat in cats.values():
        n = cat["n"]
        cat["hit_rate"] = cat["hit"] / n if n else 0.0
        cat["mrr"] = cat["mrr"] / n if n else 0.0
        cat["fp_rate"] = cat["fp"] / n if n else 0.0
    return EvalResult(
        recall=recall,
        mrr=summary["mrr"],
        ndcg=ndcg,
        fp_rate=summary["fp_rate"],
        by_category=cats,
        detail=detail,
    )


def format_report(result: EvalResult, *, label: str = "zhmem", show_details: bool = False) -> str:
    lines = [f"== {label} =="]
    lines.append("Recall@" + "  Recall@".join(str(k) for k in sorted(result.recall)) + ":  " +
                 "  ".join(f"R@{k}={v:.3f}" for k, v in sorted(result.recall.items())))
    lines.append("  ".join(f"nDCG@{k}={v:.3f}" for k, v in sorted(result.ndcg.items())))
    lines.append(f"MRR={result.mrr:.3f}   false-positive-rate={result.fp_rate:.3f}")
    lines.append("-- by category --")
    for name, c in result.by_category.items():
        lines.append(
            f"  {name:<11} n={int(c['n']):2d}  hit={c['hit_rate']:.3f}"
            f"  MRR={c['mrr']:.3f}  FPR={c['fp_rate']:.3f}"
        )
    if show_details:
        lines.append("-- per query --")
        for q, ranks, gold in result.detail:
            lines.append(
                f"  [{q.category:<10}] {q.query!r}  gold={sorted(gold)}  top={ranks[:5]}"
            )
    return "\n".join(lines)
