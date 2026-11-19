"""Store behaviour: FTS5 candidate selection, evidence floor, context, lifecycle."""

import pytest

from zhmem.store import MemoryStore


@pytest.fixture
def store():
    s = MemoryStore(":memory:", clock=lambda: 1_800_000_000.0)
    yield s
    s.close()


def test_add_and_get_round_trip(store):
    i = store.add("mum", "膝盖不太好", kind="fact", salience=0.9, meta={"src": "chat"})
    m = store.get(i)
    assert m.body == "膝盖不太好"
    assert m.kind == "fact"
    assert m.salience == 0.9
    assert m.meta == {"src": "chat"}


def test_empty_body_rejected(store):
    with pytest.raises(ValueError):
        store.add("mum", "   ")


def test_namespace_isolation(store):
    store.add("mum", "膝盖疼")
    store.add("dad", "膝盖疼")
    hits = store.recall("mum", "膝盖疼")
    assert len(hits) == 1
    assert hits[0].namespace == "mum"


def test_evidence_floor_blocks_recency_only_matches(store):
    # A fresh, high-salience memory with no lexical overlap must not outrank
    # nothing. This is the exact failure bm25 had on a small corpus.
    store.add("mum", "膝盖不太好", salience=0.9)
    assert store.recall("mum", "量子力学") == []


def test_negative_queries_return_nothing(store):
    store.add("kid", "下周三有数学期中考试")
    # The gold says "她数学考了多少分" should return nothing; a shared word 数学
    # is a near-miss that the eval set tolerates only via the FPR metric.
    # Here: an unrelated query must stay empty.
    assert store.recall("kid", "量子力学") == []


def test_bigram_tokeniser_matches_two_character_word(store):
    store.add("mum", "喜欢喝普洱")
    hits = store.recall("mum", "普洱")
    assert hits and hits[0].body == "喜欢喝普洱"


def test_trigram_tokeniser_cannot_match_two_character_word():
    s = MemoryStore(":memory:", tokenizer="trigram")
    s.add("mum", "喜欢喝普洱")
    assert s.recall("mum", "普洱") == []  # a trigram cannot represent a 2-char word
    s.close()


def test_bm25_scorer_is_selectable():
    s = MemoryStore(":memory:", scorer="bm25")
    s.add("mum", "膝盖不太好")
    hits = s.recall("mum", "膝盖")
    # Not merely a list: a single relevant row must actually surface. The old
    # single-candidate branch dropped it and this test passed on an empty list.
    assert hits and hits[0].body == "膝盖不太好"
    s.close()


def test_unknown_scorer_and_tokenizer_rejected():
    with pytest.raises(ValueError):
        MemoryStore(":memory:", scorer="nope")
    with pytest.raises(ValueError):
        MemoryStore(":memory:", tokenizer="nope")


def test_synonym_folding_on_both_sides(store):
    store.add("mum", "最近睡不着")
    assert store.recall("mum", "老人家失眠")  # query folded to 失眠, index has 失眠


def test_fold_query_only_breaks_recall():
    s = MemoryStore(":memory:", normalize=None)  # index keeps 睡不着
    s.add("mum", "最近睡不着")
    assert s.recall("mum", "老人家失眠") == []  # query folded to 失眠, no match
    s.close()


def test_previous_turn_lifts_pronoun_followup(store):
    store.add("mum", "老花镜放在客厅电视柜第二个抽屉")
    hits = store.recall("mum", "放在哪儿了", recent=("老花镜找不到了",))
    assert hits and hits[0].body.startswith("老花镜")


def test_topic_switch_does_not_drift_to_previous_turn(store):
    store.add("mum", "喜欢喝普洱，下午三点")
    store.add("mum", "膝盖不太好，阴雨天会疼")
    hits = store.recall("mum", "膝盖疼", recent=("今天下午喝什么好",))
    assert hits and hits[0].body.startswith("膝盖")


def test_kinds_filter(store):
    store.add("mum", "膝盖不太好", kind="fact")
    store.add("mum", "喜欢喝普洱", kind="preference")
    hits = store.recall("mum", "膝盖", kinds=["preference"])
    assert all(h.kind == "preference" for h in hits)


def test_forget_and_clear(store):
    i = store.add("mum", "膝盖不太好")
    assert store.forget(i)
    assert not store.forget(i)
    store.add("mum", "喜欢喝普洱")
    assert store.clear("mum") == 1
    assert store.count("mum") == 0


def test_reindex_is_idempotent(store):
    store.add("mum", "膝盖不太好")
    first = store.reindex()
    assert first == 0  # nothing changed with the same tokenizer/normalizer


def test_render_caps_items_and_chars(store):
    hits = []
    from zhmem.store import Recalled
    for i in range(10):
        m = store.get(store.add("m", f"第{i}条" + "很长" * 200))
        hits.append(Recalled(memory=m, score=1.0, parts={}))
    out = store.render(hits, max_chars=50, max_items=2)
    assert out.count("\n") <= 1
    assert len(out) <= 50


# --- regressions from the review ------------------------------------------

class _ConstantEmbedder:
    """Every string maps to the same vector, so any pair has cosine 1.0."""
    def embed(self, text):
        return [1.0, 0.0]


def test_pure_vector_evidence_is_not_swallowed(tmp_path):
    # A query with no FTS overlap must still reach the vector block. The old
    # `if not by_id: return []` returned early and made pure-vector recall empty.
    s = MemoryStore(str(tmp_path / "v.db"), embedder=_ConstantEmbedder(), normalize=None)
    try:
        s.add("m", "苹果")
        s.add("m", "橘子")
        hits = s.recall("m", "香蕉")
        assert len(hits) == 2  # only vector similarity supports these rows
        assert all(h.parts["semantic"] > 0 for h in hits)
    finally:
        s.close()


def test_reindex_preserves_custom_search_text():
    # reindex used to rebuild from subject+body and drop folded-in tags, so a
    # knowledge-base entry lost its tag-only matches after a tokeniser change.
    s = MemoryStore(":memory:", subject_bonus=1.0)
    try:
        i = s.add("ns", "每天摄入足够水分", subject="喝水", search_text="喝水 每日 饮水 量化 标签")
        row = s._db.execute("SELECT search_text, search_source FROM items WHERE id=?", (i,)).fetchone()
        assert row["search_source"] == "喝水 每日 饮水 量化 标签"
        s.reindex()
        after = s._db.execute("SELECT search_text, search_source FROM items WHERE id=?", (i,)).fetchone()
        assert after["search_text"] == row["search_text"]
        assert after["search_source"] == row["search_source"]
    finally:
        s.close()


def test_idf_cache_invalidated_on_write():
    # A stale IDF cache weights shared characters by a document frequency that
    # no longer matches the corpus after add/forget/clear.
    s = MemoryStore(":memory:", subject_bonus=1.0)
    try:
        s.add("ns", "盐摄入量", subject="每天盐摄入量")
        before = s._char_idf("ns")
        s.add("ns", "盐吃多了", subject="盐太多")
        assert s._char_idf("ns") != before  # cache must be invalidated by add()
    finally:
        s.close()


def test_bm25_ranks_relevant_first():
    # bm25 is negative and lower-is-better. The old normalisation inverted it,
    # putting the best match last; a two-row corpus makes the direction testable.
    s = MemoryStore(":memory:", scorer="bm25")
    try:
        for _ in range(4):
            s.add("m", "无关的记忆")
        s.add("m", "膝盖")
        hits = s.recall("m", "膝盖")
        assert hits and hits[0].body == "膝盖"
    finally:
        s.close()


def test_reindex_is_idempotent_after_source_is_stored(tmp_path):
    s = MemoryStore(str(tmp_path / "r.db"))
    try:
        s.add("m", "膝盖不太好", search_text="膝盖 关节 疼痛")
        assert s.reindex() == 0  # stored source keeps the custom search_text stable
    finally:
        s.close()
