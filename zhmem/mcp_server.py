"""MCP server: expose zhmem to WorkBuddy / CodeBuddy over stdio.

Registered as `zhmem` in ~/.codebuddy/.mcp.json, tools appear as
`mcp__zhmem__zhmem_ask` etc.  Two independent stores are used:

* a read-only built-in KB (84 life / travel / fitness entries, rebuilt in
  memory each start) for `zhmem_ask`, `zhmem_web_search`, `zhmem_image_search`;
* a persistent personal-memory sqlite file for `zhmem_remember` / `zhmem_recall`
  / `zhmem_forget` / `zhmem_list_memories`.  Location: `$ZHMEM_DB` or
  `~/.workbuddy/zhmem-memory.db`.

stdout is reserved for the MCP protocol — never print anything here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from .images import ImageSearch
from .retriever import Retriever
from .store import MemoryStore
from .web import WebSearch

mcp = MCPServer("zhmem")

_retriever: Retriever | None = None
_memory: MemoryStore | None = None


def default_db_path() -> str:
    override = os.environ.get("ZHMEM_DB")
    if override:
        # Expand ~ so a config like ZHMEM_DB=~/mem.db does not create a literal
        # `~` directory under the current working directory.
        return str(Path(override).expanduser())
    base = Path.home() / ".workbuddy"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "zhmem-memory.db")


def _mem() -> MemoryStore:
    global _memory
    if _memory is None:
        _memory = MemoryStore(default_db_path())
    return _memory


def _ret() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _close() -> None:
    global _memory, _retriever
    if _retriever is not None:
        _retriever.close()
    if _memory is not None:
        _memory.close()
    _retriever = None
    _memory = None


@mcp.tool()
def zhmem_ask(query: str, recent: list[str] = [], web_mode: str = "auto", images: bool = True) -> str:
    """Answer a question: built-in Chinese knowledge base first, web and pictures when needed.

    web_mode: auto (only when local evidence is weak), always, or never.
    recent: preceding user turns, used as discounted context — never concatenated.
    """
    # Validate explicitly: the old truthiness mapping silently turned any
    # unrecognised value ("Auto", "off", "") into `never`, i.e. it disabled the
    # web without telling the caller.
    modes = {"auto": None, "always": True, "never": False}
    if web_mode not in modes:
        raise ValueError(f"web_mode must be one of {sorted(modes)}, got {web_mode!r}")
    web = modes[web_mode]
    answer = _ret().ask(query, recent=tuple(recent), web=web, images=images)
    return answer.render(images=images)


@mcp.tool()
def zhmem_web_search(query: str, limit: int = 5) -> str:
    """Search the live web (Bing) and return titled citations with URLs."""
    rows = WebSearch().search(query, max(1, min(int(limit), 10)))
    if not rows:
        return "（没有搜到结果，或当前离线）"
    return "\n".join(f"- [{r.title}]({r.url})\n  {r.snippet}" for r in rows)


@mcp.tool()
def zhmem_image_search(query: str, limit: int = 4) -> str:
    """Find pictures for a topic; returns markdown images with source links."""
    rows = ImageSearch().search(query, max(1, min(int(limit), 8)))
    if not rows:
        return "（没有搜到图片，或当前离线）"
    return "\n".join(f"- {r.title}\n  {r.markdown()}" for r in rows)


@mcp.tool()
def zhmem_remember(
    text: str, namespace: str = "default", subject: str = "",
    kind: str = "note", salience: float = 0.5,
) -> str:
    """Persist a fact or conversation note into long-term memory (sqlite)."""
    item_id = _mem().add(namespace, text, kind=kind, subject=subject, salience=salience)
    return f"saved id={item_id} namespace={namespace}"


@mcp.tool()
def zhmem_recall(query: str, namespace: str = "default", limit: int = 5, recent: list[str] = []) -> str:
    """Recall long-term memories. Recent turns are scored separately, never merged."""
    hits = _mem().recall(namespace, query, limit=max(1, min(int(limit), 20)), recent=tuple(recent))
    if not hits:
        return "（没有匹配的记忆）"
    return "\n\n".join(
        f"- [{h.memory.id}] {h.memory.subject or h.memory.body[:24]}（{h.score:.2f}）：{h.memory.body}"
        for h in hits
    )


@mcp.tool()
def zhmem_forget(item_id: int) -> str:
    """Delete one long-term memory by id."""
    return "deleted" if _mem().forget(int(item_id)) else "not found"


@mcp.tool()
def zhmem_list_memories(namespace: str | None = None, limit: int = 20) -> str:
    """List stored long-term memories, newest last. Omit namespace for all."""
    rows = _mem().all(namespace)[-max(1, min(int(limit), 100)):]
    if not rows:
        return "（还没有保存任何记忆）"
    return "\n".join(
        f"- [{m.id}] {m.namespace}/{m.kind} {m.subject or ''}: {m.body[:120]}" for m in rows
    )


@mcp.tool()
def zhmem_stats() -> str:
    """Report KB counts, personal-memory count and the sqlite path in use."""
    kb = _ret().kb
    payload = {
        "knowledge_base": kb.counts,
        "personal_memories": _mem().count(),
        "memory_db": default_db_path(),
        "version": "0.2.0",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    try:
        mcp.run(transport="stdio")
    finally:
        _close()


if __name__ == "__main__":
    main()
