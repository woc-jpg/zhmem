"""Built-in knowledge base and the unified Retriever facade."""

from __future__ import annotations

import pytest

from zhmem.images import ImageResult
from zhmem.kb import KnowledgeBase, install, load_entries
from zhmem.retriever import Retriever
from zhmem.web import WebResult

ENTRY_COUNT = 84


def test_kb_entries_are_complete_and_unique():
    entries = load_entries()
    assert len(entries) == ENTRY_COUNT
    assert len({(e.namespace, e.id) for e in entries}) == ENTRY_COUNT
    assert {e.namespace for e in entries} == {"daily", "travel", "fitness"}
    for e in entries:
        assert e.title and e.body and e.image_query
        assert 0.0 <= e.salience <= 1.0


@pytest.mark.parametrize("query,namespace,wanted", [
    ("敦煌莫高窟怎么预约", "travel", "敦煌旅游攻略"),
    ("地震来了怎么办", "daily", "地震避险"),
    ("深蹲动作要领", "fitness", "深蹲正确动作"),
    ("盐吃多了会怎样", "daily", "每天盐摄入量"),
    ("被烫伤了怎么处理", "daily", "烫伤烧伤急救"),
])
def test_knowledge_base_finds_relevant_entry(query, namespace, wanted):
    with KnowledgeBase() as kb:
        hits = kb.store.recall(namespace, query, kinds=("kb",), limit=1)
        # The assertion is intentionally on subject equality for exact matches.
        assert hits and hits[0].memory.subject == wanted, (query, [h.memory.subject for h in hits])


def test_install_reloads_entries_into_a_fresh_store():
    # `install` is intentionally not deduplicating: deciding that two rows are
    # the same is a caller policy. Install into a fresh store and check each
    # namespace gets exactly its entry count (not that a second install is a
    # no-op, which the old misleading test name claimed).
    from zhmem.store import MemoryStore
    s = MemoryStore(":memory:")
    try:
        added = install(s)
        assert added == {"daily": 36, "travel": 24, "fitness": 24}
        assert s.count("daily") == 36
    finally:
        s.close()


class FakeWeb:
    def search(self, query, limit=5):
        return [WebResult(title="网络结果", url="https://example.com", snippet="补充说明")]


class FakeImages:
    def search(self, query, limit=3, **kw):
        return [ImageResult(title="图", url="https://e.com/a.jpg", thumb="https://e.com/t.jpg", source="fake")]


def test_retriever_uses_local_first_and_attaches_citations():
    r = Retriever(web=FakeWeb(), images=FakeImages())
    try:
        ans = r.ask("深蹲怎么做", web=False)
        assert ans.local and ans.local[0].memory.subject == "深蹲正确动作"
        assert ans.web == []  # strong local hit suppresses web
        text = ans.render()
        assert "深蹲正确动作" in text
    finally:
        r.close()


def test_retriever_falls_back_to_web_when_local_weak():
    r = Retriever(web=FakeWeb(), images=FakeImages())
    try:
        ans = r.ask("量子计算机原理", images=True)
        assert ans.web and ans.web[0].url.startswith("https://")
        text = ans.render()
        assert "网络资料" in text and "相关图片" in text
    finally:
        r.close()


def test_weak_local_hit_does_not_block_web_fallback():
    # The old web check used the final score, which includes recency/salience.
    # A zero-evidence travel entry fresh and salient enough scored 0.64 and
    # suppressed the web for an unrelated question ("python怎么安装").
    r = Retriever(web=FakeWeb(), images=FakeImages())
    try:
        ans = r.ask("python怎么安装", images=False)
        assert ans.web  # must fall back to the web, not a travel entry
    finally:
        r.close()


def test_offline_retriever_is_deterministic():
    r = Retriever.offline()
    try:
        ans = r.ask("北京的景点", images=True)
        assert ans.web == [] and ans.images == []
        assert "故宫" in ans.render() or "北京" in ans.render()
    finally:
        r.close()
