"""Built-in Chinese knowledge base: daily life, travel, fitness.

The knowledge base is plain JSON shipped as package data, so it can be read,
diffed on GitHub, and extended without touching code. Each entry becomes one
row in a `MemoryStore`, which means the same retrieval machinery that was tuned
for personal memories also powers general information lookup.

An entry has:

    id           stable key (namespace + id is unique)
    title        short label, used as the memory `subject`
    body         the retrievable text
    tags         extra keywords folded into the indexed text
    image_query  what to search when pictures are wanted (English works best
                 for the Commons fallback; Chinese for Bing Images)
    salience     how canonical/important this entry is, 0..1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..store import MemoryStore

__all__ = ["KBEntry", "KnowledgeBase", "load_entries", "install"]

_DATA_DIR = Path(__file__).with_name("data")
_NAMESPACES = ("daily", "travel", "fitness")


@dataclass(frozen=True)
class KBEntry:
    id: str
    namespace: str
    title: str
    body: str
    tags: tuple[str, ...] = ()
    image_query: str = ""
    salience: float = 0.5

    @property
    def search_text(self) -> str:
        return " ".join([self.title, self.body, *self.tags])


def _read(ns: str) -> list[dict]:
    path = _DATA_DIR / f"{ns}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("entries", [])


def load_entries(namespace: str | None = None) -> list[KBEntry]:
    """Load and validate entries from the packaged JSON files."""
    out: list[KBEntry] = []
    seen: set[tuple[str, str]] = set()
    for ns in ((namespace,) if namespace else _NAMESPACES):
        for raw in _read(ns):
            eid = str(raw["id"])
            key = (ns, eid)
            if key in seen:
                raise ValueError(f"duplicate knowledge-base id: {ns}/{eid}")
            seen.add(key)
            out.append(KBEntry(
                id=eid,
                namespace=ns,
                title=str(raw.get("title", "")),
                body=str(raw.get("body", "")),
                tags=tuple(str(t) for t in raw.get("tags", [])),
                image_query=str(raw.get("image_query", "")),
                salience=float(raw.get("salience", 0.5)),
            ))
    return out


def install(store: MemoryStore, namespace: str | None = None) -> dict[str, int]:
    """Index the built-in knowledge base into `store`; returns counts per namespace."""
    counts: dict[str, int] = {}
    for e in load_entries(namespace):
        store.add(
            e.namespace,
            e.body,
            kind="kb",
            subject=e.title,
            salience=e.salience,
            meta={"id": e.id, "image_query": e.image_query},
            search_text=e.search_text,
        )
        counts[e.namespace] = counts.get(e.namespace, 0) + 1
    return counts


class KnowledgeBase:
    """Convenience wrapper: a store with the built-in knowledge installed."""

    def __init__(self, path: str = ":memory:", *, namespace: str | None = None, **store_kwargs) -> None:
        # A KB entry's title is a hand-written label for the row; matching it is
        # as meaningful as matching the body, so let title hits count as evidence.
        store_kwargs.setdefault("subject_bonus", 1.0)
        self.store = MemoryStore(path, **store_kwargs)
        self.counts = install(self.store, namespace)
        self._by_id = {e.id: e for e in load_entries(namespace)}

    def entry(self, entry_id: str) -> KBEntry | None:
        return self._by_id.get(entry_id)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
