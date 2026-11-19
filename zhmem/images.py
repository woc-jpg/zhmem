"""Image retrieval: answer with pictures, not just text.

Two sources, both dependency-free and both cached in-process by URL:

* **Bing Images** — the pragmatic default for Chinese queries. Results carry
  a thumbnail URL (Bing's CDN, fast and stable) and an original page/image URL.
* **Wikimedia Commons** — CC-licensed, used as fallback and for offline-built
  knowledge-base entries whose `image_query` is an English keyword.

Every entry is a plain dataclass with URLs, never bytes: callers (CLI, web app,
notebook) decide whether to embed a markdown image, download, or just show the
link. A missing/blocked image source degrades to an empty list, never an error.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .web import Fetcher, default_fetch

__all__ = ["BingImages", "CommonsImages", "ImageResult", "ImageSearch"]

_IUSC = re.compile(r'class="[^"]*iusc[^"]*"[^>]*m="([^"]+)"')


@dataclass(frozen=True)
class ImageResult:
    title: str
    url: str          # original image URL
    thumb: str        # thumbnail URL (often a CDN proxy)
    source: str
    license: str = ""

    def markdown(self) -> str:
        return f"![{self.title}]({self.thumb or self.url})"


class BingImages:
    """Bing image search, parsing the ``m`` JSON on ``a.iusc`` elements."""

    name = "bing-images"

    def __init__(self, fetch: Fetcher | None = None) -> None:
        self.fetch = fetch or default_fetch

    def search(self, query: str, limit: int = 4, *, timeout: float = 12.0) -> list[ImageResult]:
        url = "https://cn.bing.com/images/search?q=" + urllib.parse.quote(query)
        page = self.fetch(url, timeout)
        out: list[ImageResult] = []
        for raw in _IUSC.findall(page):
            try:
                d = json.loads(html.unescape(raw))
            except (ValueError, TypeError):
                continue
            if not isinstance(d, dict):
                continue
            murl, turl = d.get("murl", ""), d.get("turl", "")
            if not murl and not turl:
                continue
            out.append(ImageResult(
                title=html.unescape(d.get("t", "")).strip() or query,
                url=murl, thumb=turl, source=self.name,
            ))
            if len(out) >= limit:
                break
        return out


class CommonsImages:
    """Wikimedia Commons image search, with English keyword queries."""

    name = "wikimedia-commons"

    def __init__(self, fetch: Fetcher | None = None) -> None:
        self.fetch = fetch or default_fetch

    def search(self, query: str, limit: int = 4, *, timeout: float = 12.0) -> list[ImageResult]:
        params = urllib.parse.urlencode({
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrnamespace": "6", "gsrlimit": str(limit), "prop": "imageinfo",
            "iiprop": "url|extmetadata", "iiurlwidth": "480", "format": "json",
        })
        url = "https://commons.wikimedia.org/w/api.php?" + params
        try:
            data = json.loads(self.fetch(url, timeout))
        except (ValueError, TypeError):
            return []  # malformed / non-JSON response: degrade, never crash
        if not isinstance(data, dict):
            return []
        out: list[ImageResult] = []
        for page in data.get("query", {}).get("pages", {}).values():
            if not isinstance(page, dict):
                continue
            infos = page.get("imageinfo") or [{}]
            info = infos[0] if isinstance(infos, list) and infos and isinstance(infos[0], dict) else {}
            if not isinstance(info, dict):
                info = {}
            murl = info.get("url", "")
            if not murl:
                continue
            meta = info.get("extmetadata", {})
            license_short = meta.get("LicenseShortName", {}).get("value", "")
            out.append(ImageResult(
                title=page.get("title", "").replace("File:", ""),
                url=murl, thumb=info.get("thumburl", murl),
                source=self.name, license=license_short,
            ))
        return out[:limit]


class ImageSearch:
    """Try providers in order; return the first non-empty result set.

    Composite search is intentionally *not* merged: Bing gives relevant but
    rights-unclear images, Commons gives CC images with worse Chinese recall.
    Callers who care about licensing instantiate CommonsImages directly.
    """

    def __init__(self, bing: BingImages | None = None, commons: CommonsImages | None = None, *,
                 offline: bool = False, timeout: float = 12.0) -> None:
        self.bing = bing or (None if offline else BingImages())
        self.commons = commons or (None if offline else CommonsImages())
        self.timeout = timeout

    def search(self, query: str, limit: int = 4, *, prefer_commons: bool = False) -> list[ImageResult]:
        if not (query or "").strip():
            return []
        providers = ([self.commons, self.bing] if prefer_commons else [self.bing, self.commons])
        for p in providers:
            if p is None:
                continue
            try:
                found = p.search(query, limit, timeout=self.timeout)
            except Exception:
                continue
            if found:
                return found
        return []
