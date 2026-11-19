"""Web/image providers are tested with saved HTML fixtures — no network."""

from __future__ import annotations

import html
import json

from zhmem.images import BingImages, CommonsImages, ImageResult, ImageSearch
from zhmem.web import BingProvider, NullProvider, WebSearch

_BING_HTML = """
<html><body>
<li class="b_algo"><h2><a href="https://example.com/a">敦煌旅游攻略</a></h2>
<p>莫高窟需要提前预约，旺季一票难求。</p></li>
<li class="b_algo"><h2><a href="https://example.com/b">鸣沙山月牙泉</a></h2>
<p>傍晚登山看日落，注意防沙。</p></li>
<li class="b_algo"><h2><a href="https://example.com/nav">首页</a></h2></li>
</body></html>
"""

_IUSC = json.dumps({
    "t": "莫高窟外观", "murl": "https://img.example.com/mogao.jpg",
    "turl": "https://thumb.example.com/mogao.jpg",
}, ensure_ascii=False)
_BING_IMG_HTML = f'<a class="iusc" m="{html.escape(_IUSC, quote=True)}"></a>'

_COMMONS_JSON = json.dumps({
    "query": {"pages": {"123": {
        "title": "File:Dunhuang Mogao Caves.jpg",
        "imageinfo": [{
            "url": "https://upload.wikimedia.org/mogao.jpg",
            "thumburl": "https://thumb.wikimedia.org/mogao.jpg",
            "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}},
        }],
    }}}
}, ensure_ascii=False)


def _fetch(html_text):
    def f(url, timeout=10):
        if "images/search" in url:
            return _BING_IMG_HTML
        if "commons.wikimedia.org" in url:
            return _COMMONS_JSON
        return html_text
    return f


def test_bing_parses_only_results_with_title_and_snippet():
    got = BingProvider(fetch=_fetch(_BING_HTML)).search("敦煌", limit=10)
    assert len(got) == 2  # the bare "首页" link is skipped
    assert got[0].title == "敦煌旅游攻略"
    assert "预约" in got[0].snippet
    assert got[0].url == "https://example.com/a"


def test_bing_honours_limit_and_escapes_entities():
    page = _BING_HTML.replace("敦煌旅游攻略", "A &amp; B")
    got = BingProvider(fetch=_fetch(page)).search("q", limit=1)
    assert len(got) == 1
    assert "A & B" in got[0].title


def test_web_search_contains_provider_failures():
    def boom(q, limit=5, **kw):
        raise TimeoutError("offline")
    ws = WebSearch(provider=type("P", (), {"search": staticmethod(boom), "name": "bad"})())
    assert ws.search("q") == []
    try:
        WebSearch(provider=type("P", (), {"search": staticmethod(boom), "name": "bad"})(), strict=True).search("q")
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict mode should raise")


def test_null_provider_is_deterministic():
    assert NullProvider().search("anything") == []
    assert WebSearch(offline=True).search("anything") == []


def test_empty_query_short_circuits():
    assert WebSearch().search("  ") == []


def test_bing_images_parses_json_metadata():
    got = BingImages(fetch=_fetch("")).search("莫高窟", limit=4)
    assert len(got) == 1
    assert got[0].title == "莫高窟外观"
    assert got[0].thumb.startswith("https://thumb.example.com/")
    assert "![莫高窟外观]" in got[0].markdown()


def test_commons_images_reads_license():
    got = CommonsImages(fetch=_fetch("")).search("Dunhuang", limit=4)
    assert len(got) == 1
    assert got[0].source == "wikimedia-commons"
    assert got[0].license == "CC BY-SA 4.0"
    assert got[0].url.endswith("mogao.jpg")


def test_image_search_falls_back_to_second_provider():
    bing = BingImages(fetch=lambda u, t=10: (_ for _ in ()).throw(TimeoutError()))
    commons = CommonsImages(fetch=_fetch(""))
    got = ImageSearch(bing=bing, commons=commons).search("q", limit=2)
    assert got and got[0].source == "wikimedia-commons"
    assert ImageSearch(offline=True).search("q") == []
