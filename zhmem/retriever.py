"""Unified information retrieval: built-in knowledge + web + images.

`Retriever.ask()` is the single entry point most applications want:

    r = Retriever()
    answer = r.ask("敦煌莫高窟怎么预约")
    print(answer.render())

Layer order is deliberate:

1. **Local knowledge base first.** It is free, fast, deterministic and always
   available. Web calls happen only when local evidence is weak, so offline
   installs work and online installs don't waste requests.
2. **Web second.** Results become citations with URLs, never unsourced prose.
3. **Images last.** Attached to the answer as markdown, using the matched
   entry's `image_query` when available, otherwise the user's question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .images import ImageResult, ImageSearch
from .kb import KBEntry, KnowledgeBase, load_entries
from .store import MemoryStore, Recalled
from .web import WebResult, WebSearch

__all__ = ["Answer", "Retriever"]

# Weak local evidence triggers a web lookup. This compares *evidence only*
# (lexical / semantic / title), never the final score: recency and salience are
# tie-breakers worth up to 0.25 on their own and made a zero-evidence row clear
# the threshold ("python怎么安装" matched a travel entry at 0.64). Measured on
# the 84-entry KB: genuine topic hits carry evidence >= 0.45, unrelated
# questions <= 0.41.
_WEB_THRESHOLD = 0.45


@dataclass
class Answer:
    query: str
    local: list[Recalled]
    web: list[WebResult]
    images: list[ImageResult]
    _entries: dict[str, KBEntry] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(r.memory.body for r in self.local)

    def local_citations(self) -> list[str]:
        out = []
        for r in self.local:
            eid = r.memory.meta.get("id") if r.memory.meta else None
            out.append(f"[{r.memory.subject}（本地知识库）]({eid or r.memory.id})")
        return out

    def render(self, *, images: bool = True, limit: int = 5) -> str:
        """Markdown answer: local knowledge, web citations, optional pictures."""
        lines: list[str] = [f"## {self.query}", ""]
        if self.local:
            lines.append("### 本地知识库")
            for r in self.local[:limit]:
                lines.append(f"- **{r.memory.subject}**（相关度 {r.score:.2f}）：{r.memory.body}")
            lines.append("")
        if self.web:
            lines.append("### 网络资料")
            lines.extend(w.render() for w in self.web[:limit])
            lines.append("")
        if not self.local and not self.web:
            lines.append("暂时没有找到可靠资料，可以换个说法再问。")
            lines.append("")
        if images and self.images:
            lines.append("### 相关图片")
            for img in self.images[:4]:
                credit = f"（{img.license}）" if img.license else ""
                lines.append(f"- {img.title}{credit}\n  {img.markdown()}")
            lines.append("")
        return "\n".join(lines).rstrip()


class Retriever:
    """Compose local KB retrieval, web search and image search."""

    def __init__(
        self,
        *,
        kb: KnowledgeBase | None = None,
        web: WebSearch | None = None,
        images: ImageSearch | None = None,
        offline: bool = False,
        web_threshold: float = _WEB_THRESHOLD,
        local_limit: int = 3,
        web_limit: int = 5,
        images_limit: int = 3,
    ) -> None:
        self.kb = kb if kb is not None else KnowledgeBase()
        self.web = web or WebSearch(offline=offline)
        self.images = images or ImageSearch(offline=offline)
        self.web_threshold = web_threshold
        self.local_limit = local_limit
        self.web_limit = web_limit
        self.images_limit = images_limit
        self._entries = {e.id: e for e in load_entries()}

    @classmethod
    def offline(cls, **kwargs) -> "Retriever":
        """Deterministic, no-network retriever — handy for tests and demos."""
        return cls(web=WebSearch(offline=True), images=ImageSearch(offline=True), **kwargs)

    def close(self) -> None:
        self.kb.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _local(self, query: str, recent: Sequence[str] = (), namespaces: Sequence[str] | None = None):
        # `namespaces` lets a caller narrow the search (the desktop app's
        # 日常/旅游/健身 filters). None means "all three".
        hits: list[Recalled] = []
        for ns in (namespaces if namespaces is not None else ("daily", "travel", "fitness")):
            hits.extend(self.kb.store.recall(ns, query, kinds=("kb",), limit=self.local_limit, recent=recent))
        return hits

    def ask(self, query: str, *, recent: Sequence[str] = (), web: bool | None = None,
            images: bool = True, limit: int = 5,
            namespaces: Sequence[str] | None = None) -> Answer:
        """Retrieve an answer with local evidence, citations and pictures."""
        hits = sorted(self._local(query, recent, namespaces), key=lambda h: h.score, reverse=True)[:self.local_limit]
        # Deduplicate the same entry recalled from overlapping namespaces.
        seen: set[str] = set()
        uniq: list[Recalled] = []
        for h in hits:
            key = h.memory.id
            if key not in seen:
                seen.add(key)
                uniq.append(h)

        # Compare evidence, not the final score. recency/salience are always
        # present (a fresh, high-salience row scores ~0.25 with zero evidence),
        # so using h.score here sent "python怎么安装" to a travel entry at 0.64
        # and skipped the web entirely. `parts` carries the raw signals.
        best_evidence = 0.0
        if uniq:
            p = uniq[0].parts
            best_evidence = max(p.get("lexical", 0.0), p.get("semantic", 0.0), p.get("subject", 0.0))
        use_web = (
            self.web_threshold is not None and (not uniq or best_evidence < self.web_threshold)
        ) if web is None else web
        web_hits = self.web.search(query, self.web_limit) if use_web else []
        if images:
            img_query = self._image_query(uniq) or query
            imgs = self.images.search(img_query, self.images_limit)
        else:
            imgs = []
        return Answer(query=query, local=uniq, web=web_hits, images=imgs, _entries=self._entries)

    def _image_query(self, hits: Iterable[Recalled]) -> str:
        for h in hits:
            eid = h.memory.meta.get("id") if h.memory.meta else None
            e = self._entries.get(eid or "")
            if e and e.image_query:
                return e.image_query
        return ""
