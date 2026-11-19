"""Web information retrieval with zero dependencies and graceful degradation.

The provider is an object with ``search(query, limit) -> list[WebResult]``. The
default provider scrapes Bing China's HTML with the standard library, because
the environments this library targets often have no API keys but do have
outbound HTTP. Any provider can be injected — including a fake one in tests —
so no network is ever required to run the suite.

Design rules learned the hard way:

* Every provider failure is contained. Search must be a total function: a
  timeout returns an empty list, never an exception that kills the caller.
* HTML parsing uses narrow, documented patterns. Bing's result blocks are
  ``<li class="b_algo">`` with the title in ``<h2><a>`` and snippet in ``<p>``.
  A provider that silently returns 10 results of navigation chrome is worse
  than one that returns nothing.
* ``fetch`` is injectable at every level: the same parsing code is tested with
  saved HTML and used with urllib in production.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

__all__ = ["BingProvider", "NullProvider", "WebResult", "WebSearch", "Fetcher"]

Fetcher = Callable[[str, float], str]


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str
    source: str = "web"

    def render(self) -> str:
        tail = f" — {self.snippet}" if self.snippet else ""
        return f"- {self.title} ({self.url}){tail}"


def default_fetch(url: str, timeout: float = 10.0) -> str:
    """Fetch a URL as UTF-8 text with a browser-ish User-Agent."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; zhmem/0.2; research)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", "replace")


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _text(fragment: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", fragment))).strip()


class BingProvider:
    """Bing China HTML search. Works without an API key from mainland China.

    Parsing is deliberately conservative: a result is kept only if it has both
    a title link and a non-empty snippet. Those are the fields an LLM prompt
    actually needs; a bare link is a bad citation.
    """

    name = "bing"

    def __init__(self, fetch: Fetcher | None = None, base: str = "https://cn.bing.com/search") -> None:
        self.fetch = fetch or default_fetch
        self.base = base

    def search(self, query: str, limit: int = 5, *, timeout: float = 10.0) -> list[WebResult]:
        url = f"{self.base}?q={urllib.parse.quote(query)}"
        page = self.fetch(url, timeout)
        out: list[WebResult] = []
        # Each organic result is an <li class="b_algo"> ... </li>. Using the
        # closing tag as terminator keeps nested divs inside the same block.
        for block in re.findall(r'<li class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>', page, re.S):
            m = re.search(r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, re.S)
            s = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
            if not m:
                continue
            title = _text(m.group(2))
            snippet = _text(s.group(1)) if s else ""
            if not title or not snippet:
                continue
            out.append(WebResult(title=title, url=html.unescape(m.group(1)),
                                 snippet=snippet, source=self.name))
            if len(out) >= limit:
                break
        return out


class NullProvider:
    """Offline provider: returns nothing. Used by tests and air-gapped installs."""

    name = "none"

    def search(self, query: str, limit: int = 5, **kwargs) -> list[WebResult]:
        return []


class WebSearch:
    """Provider multiplexer with containment and optional strict mode.

    >>> ws = WebSearch()                       # online, Bing
    >>> ws = WebSearch(offline=True)           # deterministic no-op
    >>> ws = WebSearch(provider=MyProvider())  # custom / test provider
    """

    def __init__(
        self,
        provider=None,
        *,
        offline: bool = False,
        timeout: float = 10.0,
        strict: bool = False,
    ) -> None:
        self.provider = provider if provider is not None else (NullProvider() if offline else BingProvider())
        self.timeout = timeout
        self.strict = strict

    def search(self, query: str, limit: int = 5) -> list[WebResult]:
        query = (query or "").strip()
        if not query:
            return []
        try:
            return self.provider.search(query, limit, timeout=self.timeout)
        except Exception as exc:  # network, parse, provider bug — all contained
            if self.strict:
                raise RuntimeError(f"web provider {self.provider!r} failed") from exc
            return []
