"""Eval harness integrity: gold labels are reachable and metrics are honest."""

from zhmem.eval.corpus import CATEGORIES, PERSONAS, QUERIES, build
from zhmem.eval.metrics import evaluate, format_report, query_ids
from zhmem.store import MemoryStore


def test_corpus_shape():
    assert len(PERSONAS) == 6
    assert len(QUERIES) >= 50
    assert set(CATEGORIES) <= {q.category for q in QUERIES}
    assert all(q.gold for q in QUERIES if not q.is_negative)
    assert all(not q.gold for q in QUERIES if q.is_negative)


def test_context_queries_carry_previous_turns():
    ctx = [q for q in QUERIES if q.category == "context"]
    assert ctx and all(q.recent for q in ctx)


def test_switch_queries_carry_unrelated_previous_turns():
    sw = [q for q in QUERIES if q.category == "switch"]
    assert sw and all(q.recent for q in sw)


def test_gold_ids_resolve():
    s = MemoryStore(":memory:")
    mapping = build(s)
    for q in QUERIES:
        ids = query_ids(s, mapping, q)
        assert len(ids) == len(q.gold)
    s.close()


def test_eval_meets_delivery_thresholds():
    s = MemoryStore(":memory:")
    mapping = build(s)
    result = evaluate(s, mapping)
    report = format_report(result)
    s.close()
    assert result.recall_at5 >= 0.95
    assert result.mrr >= 0.90
    assert result.fp_rate <= 0.15
    assert "false-positive-rate" in report
    # Every positive family at full hit: the aggregate must not hide a broken one.
    for name, c in result.by_category.items():
        if name == "negative":
            continue
        assert c["hit_rate"] == 1.0, f"{name} hit rate {c['hit_rate']}"
