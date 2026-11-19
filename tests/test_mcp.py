"""Tests for the MCP bridge. Skipped when the optional mcp extra isn't installed."""

import asyncio

import pytest

pytest.importorskip("mcp")

import zhmem.mcp_server as server  # noqa: E402
from zhmem.mcp_server import mcp  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_memory_db(tmp_path, monkeypatch):
    """Every MCP test uses a throwaway DB — never the real ~/.workbuddy one.

    Without this, zhmem_stats opens the user's persistent memory file (which may
    not exist or be writable in CI) and leaks user data into test assertions.
    """
    monkeypatch.setenv("ZHMEM_DB", str(tmp_path / "mcp-isolated.db"))
    server._close()
    yield
    server._close()


def call(name: str, **arguments) -> str:
    result = asyncio.run(mcp.call_tool(name, arguments))
    return result.content[0].text


def test_all_tools_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "zhmem_ask", "zhmem_web_search", "zhmem_image_search", "zhmem_remember",
        "zhmem_recall", "zhmem_forget", "zhmem_list_memories", "zhmem_stats",
    }


def test_stats_reports_all_three_namespaces():
    import json
    payload = json.loads(call("zhmem_stats"))
    assert payload["knowledge_base"] == {"daily": 36, "travel": 24, "fitness": 24}
    assert payload["version"] == "0.2.0"


def test_ask_uses_local_kb_offline():
    answer = call("zhmem_ask", query="深蹲怎么做才标准", web_mode="never", images=False)
    assert "深蹲正确动作" in answer
    assert "本地知识库" in answer
    assert "网络资料" not in answer


def test_lifecycle_remember_recall_forget():
    saved = call("zhmem_remember", text="测试记忆：小明喜欢打篮球", namespace="mcp-test", subject="小明")
    assert "saved id=" in saved
    item_id = int(saved.split("id=")[1].split()[0])

    recalled = call("zhmem_recall", query="小明喜欢什么", namespace="mcp-test")
    assert "篮球" in recalled

    listed = call("zhmem_list_memories", namespace="mcp-test")
    assert "小明" in listed

    assert call("zhmem_forget", item_id=item_id) == "deleted"
    assert "没有匹配" in call("zhmem_recall", query="小明", namespace="mcp-test")
