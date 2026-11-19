"""Tokenisation and evidence scoring for Chinese retrieval over short memories.

Every rule here came out of a measured failure, not a guess:

* SQLite's `trigram` tokeniser cannot index a two-character word — a trigram is
  three characters, so 「普洱」 produces no token the index holds. Indexing CJK
  text as space-separated unigrams plus bigrams under `unicode61` fixes that.
* BM25 is degenerate on small corpora: a term in 2 of 4 rows has IDF 0 and
  `bm25()` returns `-0.0000` for every row, so recency and salience decide the
  ranking. `coverage()` is absolute and interpretable instead.
* Generic characters must be removed as *phrases* where possible. Stripping the
  single character 日 breaks 「日本」 into 「本」, and stripping 上/下 breaks
  「上周」「下午」 — each time a real topic word quietly disappears.

Stop characters act as *segment walls* as well as being dropped: a bigram spanning
one (我膝, 盖疼, 家失) is not a word anyone would type, so it can never match and
only sits in the denominator charging a tax on genuine matches.
"""

from __future__ import annotations

import re

__all__ = [
    "GENERIC_CHARS",
    "coverage",
    "has_topic",
    "index_text",
    "is_cjk",
    "query_terms",
    "resolve_query",
    "strip_query_noise",
    "terms",
]

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))

_WORD = 1.0
_CHAR = 0.2

# Pure function words and symptom words only. Content-bearing characters stay,
# even when they double as a date word: 上 builds 上周, 下 builds 下午, 日 builds
# 日本. Removing them is a regression that shows up as a lower score on queries
# that plainly match.
GENERIC_CHARS = frozenset(
    "的了是在和就都也还有没不很太会能要想说给对从咱这哪个些什"
    "么怎样吧呢吗啊呀嘛哦"
    "疼痛病累药医院"
)

# Question scaffolding and temporal deictics, removed as whole phrases. A single
# character stripped here would take a topic with it (日→日本, 次→上次).
_JUNK_PHRASES = (
    "记不记得", "还记得", "是不是", "有没有", "好不好", "行不行", "要不要",
    "怎么办", "怎么样", "怎样", "咋办", "咋样", "如何", "为什么", "是什么",
    "什么", "哪里", "哪个", "哪些", "哪儿", "多少", "几点", "多大",
    "今天", "昨天", "明天", "那天", "上次", "那次", "这次", "每次", "下次",
    "刚才", "以后", "以前",
)

_PUNCT = re.compile(r"[?？!！。,.，、;；:：\"'“”‘’()（）\[\]【】<>《》\s]+")


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _content_segments(text: str):
    """Maximal runs of CJK content chars, plus whole Latin/number words.

    Generic characters and punctuation are walls, so no generated bigram spans
    one. Mixed text (「用 sqlite3 存的」) is naturally kept apart.
    """
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    kind: str | None = None

    def flush() -> None:
        nonlocal buf, kind
        if buf and kind:
            out.append((kind, "".join(buf)))
        buf = []

    for ch in text:
        if is_cjk(ch) and ch not in GENERIC_CHARS:
            this = "cjk"
        elif ch.isascii() and ch.isalnum():
            this = "latin"
        else:
            this = None
        if this != kind:
            flush()
            kind = this
        if this:
            buf.append(ch.lower() if this == "latin" else ch)
    flush()
    return out


def terms(text: str) -> list[str]:
    """Unigrams, bigrams and Latin words, in order of first appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for kind, run in _content_segments(text):
        candidates = [run] if kind == "latin" else (
            list(run) + [run[i : i + 2] for i in range(len(run) - 1)]
        )
        for tok in candidates:
            if tok and tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


def index_text(text: str) -> str:
    """Space-separated terms — `unicode61` would treat Han text as one token."""
    return " ".join(terms(text))


def strip_query_noise(query: str) -> str:
    """Remove interrogative scaffolding and deictics, keeping topic words."""
    out = query
    for phrase in _JUNK_PHRASES:
        out = out.replace(phrase, " ")
    out = _PUNCT.sub(" ", out)
    out = " ".join(out.split())
    return out or query


def has_topic(text: str) -> bool:
    """Whether an utterance names something, after removing question scaffolding.

    「那次怎么样」 and 「她叫什么」 have no topic on their own: only the previous
    turn says what 那次 / 她 refers to. A two-character-or-longer content segment
    is the minimum that can be searched without guessing.
    """
    for kind, run in _content_segments(strip_query_noise(text)):
        if kind == "latin" and len(run) >= 2:
            return True
        if len(run) >= 2:
            return True
    return False


def resolve_query(text: str, recent=()) -> str:
    """Concatenate the previous turns with the current one.

    Kept as a standalone helper for callers that want one string to search, but
    `MemoryStore.recall` does NOT use it: joining turns makes each earlier topic
    vote on the answer, so the third question in a series drifts back to the first
    one's memory. The store instead scores each turn separately and takes a
    weighted max, which lifts a pronoun follow-up without contaminating a genuine
    topic switch. See `store.MemoryStore.recall`.
    """
    context = " ".join(recent)
    return f"{context} {text}".strip() if context else text


def query_terms(query: str) -> dict[str, float]:
    """Weighted terms of a query: bigrams and Latin words 1.0, unigrams 0.2.

    Only *isolated* characters — a content character with no content neighbour —
    are given unigram weight. A character inside a word (学 in 力学, 家 in 老家)
    is not evidence on its own: 「量子力学」 and 「小学二年级」 share 学, and at
    0.2 it beat genuine matches on nine negative queries. A character that stands
    alone (喝 in 咖啡怎么喝) is different — it is the whole topic word.
    """
    weights: dict[str, float] = {}
    for kind, run in _content_segments(strip_query_noise(query)):
        if kind == "latin":
            weights[run] = _WORD
            continue
        for i, ch in enumerate(run):
            before = run[i - 1] if i > 0 else ""
            after = run[i + 1] if i + 1 < len(run) else ""
            if not before and not after:
                weights.setdefault(ch, _WORD)
            bigram = run[i : i + 2]
            if len(bigram) == 2:
                weights[bigram] = _WORD
    return weights


def coverage(want: dict[str, float], have, *, char_tier: bool = True) -> float:
    """Fraction of the query's evidence that `have` contains, 0..1.

    `have` is a token set (bigram index) or a raw string (trigram index, where a
    match can only mean substring).

    Multi-character matches first; unigrams are ignored entirely when any word
    matched, because counting 膝 and 痛 on top of 膝盖 inflates weak overlap.

    The character tier applies only when no word matched *this candidate*. A
    query like 「咖啡怎么喝」 shares no bigram with 「喝美式不加糖」 but must still
    find it on the character 喝. Absolute, never min-max: min-max promotes the
    least-bad candidate in a set of bad ones to a perfect 1.0.
    """
    words = {t: w for t, w in want.items() if len(t) > 1}
    if not char_tier:
        # With the weak character tier disabled, an *isolated* single character
        # is still primary evidence — query_terms already gave it full _WORD
        # weight ("喝" in 咖啡怎么喝). Only word-internal characters get no weight
        # and can never reach `want`. Treat those isolated chars as words here;
        # otherwise a one-character topic query scores 0 for every candidate.
        words.update({t: w for t, w in want.items() if len(t) == 1})
    if words:
        got = sum(w for t, w in words.items() if t in have)
        if got:
            return got / sum(words.values())

    if not char_tier or not want:
        return 0.0
    got = sum(w for t, w in want.items() if t in have)
    return min(0.25, 0.5 * got / sum(want.values())) if got else 0.0
