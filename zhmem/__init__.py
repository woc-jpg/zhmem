"""zhmem — Chinese-first hybrid information retrieval.

Zero dependencies for the core; the vector path is opt-in (numpy). Three layers:

    from zhmem import MemoryStore          # 1. personal-memory recall
    from zhmem.kb import KnowledgeBase     # 2. built-in life/travel/fitness KB
    from zhmem import Retriever            # 3. KB + web + images in one call

    r = Retriever()
    print(r.ask("敦煌莫高窟怎么预约").render())

Why this exists rather than "just use BM25": on a small Chinese corpus BM25 is
degenerate (IDF hits zero), and the common tokenisers either lose two-character
words (`trigram`) or need a dictionary (jieba). `zhmem.text` documents what was
measured; `bench/ablation_results.json` has the numbers.
"""

from .store import Memory, MemoryStore, Recalled, VectorIndex, utcnow
from .synonyms import CANONICAL, fold
from .text import (
    GENERIC_CHARS,
    coverage,
    has_topic,
    index_text,
    query_terms,
    resolve_query,
    strip_query_noise,
    terms,
)
from .web import BingProvider, NullProvider, WebResult, WebSearch
from .images import BingImages, CommonsImages, ImageResult, ImageSearch
from .retriever import Answer, Retriever

__version__ = "0.2.0"

__all__ = [
    "Answer",
    "BingImages",
    "BingProvider",
    "CANONICAL",
    "CommonsImages",
    "GENERIC_CHARS",
    "ImageResult",
    "ImageSearch",
    "Memory",
    "MemoryStore",
    "NullProvider",
    "Recalled",
    "Retriever",
    "VectorIndex",
    "WebResult",
    "WebSearch",
    "coverage",
    "fold",
    "has_topic",
    "index_text",
    "query_terms",
    "resolve_query",
    "strip_query_noise",
    "terms",
    "utcnow",
    "__version__",
]
