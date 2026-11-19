"""A small hybrid retrieval store for Chinese personal memories.

Every design decision here exists because the obvious alternative failed on a
corpus of this shape — a few hundred short memories per person rather than
millions of documents — and each failure was measured, not assumed. The knobs
(`tokenizer`, `scorer`, `normalize`, `query_normalize`, `char_tier`) are exposed
precisely so those alternatives can be run and compared: see `bench/ablation.py`.

*FTS5 picks candidates; it does not score them.* `bm25()` is fine on thousands of
documents. On four memories it is not merely imprecise, it is degenerate: a term
appearing in two of four rows has an inverse document frequency of exactly zero,
so `bm25()` returns `-0.0000` for every row. Ranking then falls to recency and
salience, which is how a question about a stomach ache once returned a memory
about a knee. `text.coverage` supplies an absolute, interpretable score instead.

*Synonyms fold on both sides.* `normalize` is applied to the stored text and to
the query. Folding only the query makes recall quietly worse than doing nothing:
the index keeps 睡不着, the query becomes 失眠, and the row becomes unreachable
while nothing raises an error.

*A candidate needs positive lexical or vector evidence to be ranked at all.*
Recency and salience are tie-breakers, not evidence. Without that floor a
freshness-plus-importance score of 0.25 beats a real match on a small corpus and
the answer looks confident and wrong.

The vector index is optional. Without an embedder the store is lexical only.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import text as T
from .synonyms import fold

__all__ = [
    "Memory",
    "MemoryStore",
    "Recalled",
    "VectorIndex",
    "utcnow",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace   TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    kind        TEXT    NOT NULL DEFAULT 'note',
    subject     TEXT    NOT NULL DEFAULT '',
    salience    REAL    NOT NULL DEFAULT 0.5,
    search_text TEXT    NOT NULL DEFAULT '',
    search_source TEXT,
    created_at  REAL    NOT NULL,
    last_hit_at REAL,
    hit_count   INTEGER NOT NULL DEFAULT 0,
    meta        TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS items_ns ON items(namespace, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    search_text,
    content='items',
    content_rowid='id',
    tokenize=__TOKENIZER__
);

CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, search_text)
    VALUES ('delete', old.id, old.search_text);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, search_text)
    VALUES ('delete', old.id, old.search_text);
    INSERT INTO items_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
"""

# The two tokeniser configurations worth comparing. `unicode61` + pre-tokenised
# unigrams/bigrams is the default; `trigram` is what a first attempt reaches for,
# and it cannot represent a two-character word at all.
_TOKENIZERS = {
    # FTS5 needs the tokenizer *and its arguments* quoted as a single string.
    "bigram": "'unicode61 remove_diacritics 2'",
    "trigram": "'trigram'",
}

_TOKENIZER_NAMES = tuple(_TOKENIZERS)


def utcnow() -> float:
    """Wall clock as a float. Injectable in tests via `MemoryStore(clock=...)`."""
    return time.time()


@dataclass
class Memory:
    id: int
    namespace: str
    body: str
    kind: str
    subject: str
    salience: float
    created_at: float
    last_hit_at: float | None = None
    hit_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Recalled:
    """A hit plus the reason it is here.

    The breakdown is returned because tuning a hybrid ranker without seeing which
    signal did the work is guesswork: the bug that motivated this whole project
    showed a knee memory scoring 0.69 with `lexical 0.03` — obvious in the parts,
    invisible in the total.
    """

    memory: Memory
    score: float
    parts: dict[str, float]

    def __getattr__(self, name: str) -> Any:
        # Convenience so callers can treat a hit as the memory itself.
        return getattr(self.memory, name)


class VectorIndex:
    """Optional embedding side-index: id → vector with a dot-product scan.

    At a few thousand memories a brute-force scan is microseconds, and it avoids
    making a vector database a dependency for what is a rounding error in the
    ranking.
    """

    def __init__(self) -> None:
        self._vec: dict[int, list[float]] = {}

    def __len__(self) -> int:
        return len(self._vec)

    def put(self, item_id: int, vector: Sequence[float]) -> None:
        self._vec[item_id] = list(vector)

    def drop(self, item_id: int) -> None:
        self._vec.pop(item_id, None)

    def search(self, query: Sequence[float], candidates: Iterable[int] | None = None) -> dict[int, float]:
        """Cosine similarity against the stored vectors; missing vectors are skipped."""
        q = list(query)
        qn = math.sqrt(sum(x * x for x in q))
        if not qn:
            return {}
        pool = candidates if candidates is not None else list(self._vec)
        out: dict[int, float] = {}
        for item_id in pool:
            vec = self._vec.get(item_id)
            if not vec or len(vec) != len(q):
                continue
            vn = math.sqrt(sum(x * x for x in vec))
            if not vn:
                continue
            out[item_id] = sum(a * b for a, b in zip(q, vec)) / (qn * vn)
        return out


class MemoryStore:
    """SQLite-backed memory store with hybrid lexical + optional vector recall.

    Parameters
    ----------
    path:
        Database file, or ``":memory:"``. Each store owns its connection.
    normalize:
        Folding applied to stored text. Defaults to the built-in synonym table;
        ``None`` disables it (a real embedding model already knows synonyms, and
        a hand-written dictionary on top of one is a downgrade).
    query_normalize:
        Folding applied to queries. Defaults to `normalize`. Setting this to
        ``None`` while keeping `normalize` reproduces the "fold the index only"
        and "fold the query only" ablations — both of which are worse than either
        folding both sides or neither.
    scorer:
        ``"coverage"`` (default) or ``"bm25"``. The latter exists to be measured
        against, not recommended.
    tokenizer:
        ``"bigram"`` (default) or ``"trigram"``.
    weights:
        Ranking weights; every signal is in 0..1 before weighting.
    half_life_days:
        Recency decay: a memory this old scores 0.5 on recency.
    semantic_floor:
        Vector similarity below this is noise and does not admit a candidate.
    char_tier:
        Whether a single shared character may count as weak evidence when no word
        matches. Turning it off makes recall stricter and emptier.
    context_weight:
        How much the previous user turn may contribute when it is searched as a
        separate query. `recall` scores the current and previous turns separately
        and takes a weighted max, never one concatenated string: joining them lets
        every earlier topic vote on the answer, but a max leaves a strong current
        match untouched while still lifting a bare pronoun follow-up.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        normalize=fold,
        query_normalize="same",
        scorer: str = "coverage",
        tokenizer: str = "bigram",
        weights: dict[str, float] | None = None,
        half_life_days: float = 30.0,
        semantic_floor: float = 0.20,
        char_tier: bool = True,
        context_weight: float = 0.6,
        subject_bonus: float = 0.0,
        embedder=None,
        clock=utcnow,
    ) -> None:
        if scorer not in ("coverage", "bm25"):
            raise ValueError(f"unknown scorer: {scorer!r}")
        if tokenizer not in _TOKENIZERS:
            raise ValueError(f"unknown tokenizer: {tokenizer!r} (choose from {_TOKENIZER_NAMES})")

        self.path = str(path)
        self.normalize = normalize or (lambda s: s)
        self.query_normalize = (
            self.normalize if query_normalize == "same" else (query_normalize or (lambda s: s))
        )
        self.scorer = scorer
        self.tokenizer = tokenizer
        self.char_tier = char_tier
        self.context_weight = context_weight
        self.subject_bonus = subject_bonus
        self.weights = {
            "lexical": 0.45,
            "semantic": 0.30,
            "recency": 0.10,
            "salience": 0.15,
            # How much a query word matching the *title* contributes. Zero for
            # personal memories (their subjects are often empty); the built-in
            # knowledge base raises it because a hand-written title is a genuine
            # summary of the row.
            "subject": subject_bonus,
            **(weights or {}),
        }
        self.half_life_days = half_life_days
        self.semantic_floor = semantic_floor
        self.embedder = embedder
        self.clock = clock
        self.vectors = VectorIndex()
        # Character document frequencies, built lazily per namespace and only
        # when title matching is enabled. Cached because a KB has 24-36 entries
        # and recall is called on every question.
        self._idf_cache: dict[str, tuple[dict[str, float], float]] = {}

        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=15000")
        self._db.executescript(_SCHEMA.replace("__TOKENIZER__", _TOKENIZERS[tokenizer]))
        # Forward migration for databases created before search_source existed.
        # Without it an old persistent store would fail on the next add().
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(items)").fetchall()}
        if "search_source" not in cols:
            self._db.execute("ALTER TABLE items ADD COLUMN search_source TEXT")
        self._db.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- index representation ---------------------------------------------

    def _index_string(self, subject: str, body: str) -> str:
        """What goes into the FTS column.

        For the bigram tokeniser: folded, then space-separated unigrams and
        bigrams, which is the only way `unicode61` can match a two-character word
        (`普洱` → `普 洱 普洱`).

        For the trigram tokeniser: the folded text itself. Folding still applies —
        that ablation is about tokenisation, not about synonyms.
        """
        return self._index_from_source(f"{subject} {body}".strip())

    def _index_from_source(self, source: str) -> str:
        """Index a pre-concatenated source string with the current settings.

        `reindex` uses this with a row's stored `search_source`, so a custom
        `search_text` (knowledge-base tags) survives a tokeniser/synonym change
        instead of being silently rebuilt from `body`.
        """
        raw = self.normalize(source)
        if self.tokenizer == "trigram":
            return raw
        return T.index_text(raw)

    def _have(self, row: sqlite3.Row) -> set[str] | str:
        """The representation `coverage` compares against.

        A token set for bigram, the raw string for trigram (where `in` means
        substring, which is exactly what that tokeniser can offer).
        """
        stored = row["search_text"] or ""
        return stored if self.tokenizer == "trigram" else set(stored.split())

    def _vector_of(self, subject: str, body: str) -> list[float] | None:
        if self.embedder is None:
            return None
        return list(self.embedder.embed(self.normalize(f"{subject} {body}".strip())))

    # -- writing -----------------------------------------------------------

    def add(
        self,
        namespace: str,
        body: str,
        *,
        kind: str = "note",
        subject: str = "",
        salience: float = 0.5,
        meta: dict[str, Any] | None = None,
        search_text: str | None = None,
    ) -> int:
        """Insert a memory and return its id.

        No dedupe on purpose: deciding that two sentences mean the same thing is a
        policy call belonging to the caller, and a silent merge hides the fact
        that the caller is producing near-duplicates.

        `search_text` overrides the indexed text while leaving `body` clean for
        display. The built-in knowledge base uses this to fold its tags into the
        index without printing them as part of the answer.
        """
        body = body.strip()
        if not body:
            raise ValueError("body must not be empty")
        # Persist the exact source that produced search_text so reindex() can
        # rebuild tokens after a tokeniser/synonym change without losing a custom
        # search_text (e.g. knowledge-base tags folded into the index).
        raw_source = search_text if search_text is not None else f"{subject} {body}".strip()
        cur = self._db.execute(
            "INSERT INTO items(namespace, body, kind, subject, salience, search_text, search_source, created_at, meta)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                namespace,
                body,
                kind,
                subject,
                max(0.0, min(1.0, float(salience))),
                self._index_from_source(raw_source),
                raw_source,
                self.clock(),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        self._db.commit()
        item_id = int(cur.lastrowid)
        vec = self._vector_of(subject, body)
        if vec is not None:
            self.vectors.put(item_id, vec)
        self._idf_cache.pop(namespace, None)
        return item_id

    def forget(self, item_id: int) -> bool:
        row = self._db.execute(
            "SELECT namespace FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        cur = self._db.execute("DELETE FROM items WHERE id = ?", (item_id,))
        self._db.commit()
        self.vectors.drop(item_id)
        if row is not None:
            self._idf_cache.pop(row["namespace"], None)
        return cur.rowcount > 0

    def clear(self, namespace: str | None = None) -> int:
        if namespace is None:
            ids = [r["id"] for r in self._db.execute("SELECT id FROM items")]
            self._db.execute("DELETE FROM items")
        else:
            ids = [r["id"] for r in self._db.execute("SELECT id FROM items WHERE namespace = ?", (namespace,))]
            self._db.execute("DELETE FROM items WHERE namespace = ?", (namespace,))
        self._db.commit()
        for item_id in ids:
            self.vectors.drop(int(item_id))
        if namespace is None:
            self._idf_cache.clear()
        else:
            self._idf_cache.pop(namespace, None)
        return len(ids)

    def reindex(self) -> int:
        """Recompute `search_text` and vectors; returns how many rows changed.

        Needed whenever `normalize` or `tokenizer` changes: rows indexed before a
        synonym table grew keep their old tokens and become unreachable by the new
        words, silently. Idempotent, so a second run returns 0.
        """
        changed = 0
        for row in self._db.execute(
            "SELECT id, subject, body, search_text, search_source FROM items"
        ).fetchall():
            # Old rows have no search_source; fall back to subject+body for them.
            source = row["search_source"]
            if source is None:
                source = f"{row['subject']} {row['body']}".strip()
            want = self._index_from_source(source)
            if want != row["search_text"] or source != row["search_source"]:
                self._db.execute(
                    "UPDATE items SET search_text = ?, search_source = ? WHERE id = ?",
                    (want, source, row["id"]),
                )
                changed += 1
            vec = self._vector_of(row["subject"], row["body"])
            if vec is not None:
                self.vectors.put(int(row["id"]), vec)
        self._db.commit()
        self._idf_cache.clear()
        return changed

    # -- reading -----------------------------------------------------------

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=int(row["id"]),
            namespace=row["namespace"],
            body=row["body"],
            kind=row["kind"],
            subject=row["subject"],
            salience=float(row["salience"]),
            created_at=float(row["created_at"]),
            last_hit_at=row["last_hit_at"],
            hit_count=int(row["hit_count"]),
            meta=json.loads(row["meta"] or "{}"),
        )

    def get(self, item_id: int) -> Memory | None:
        row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def all(self, namespace: str | None = None) -> list[Memory]:
        if namespace is None:
            rows = self._db.execute("SELECT * FROM items ORDER BY id").fetchall()
        else:
            rows = self._db.execute("SELECT * FROM items WHERE namespace = ? ORDER BY id", (namespace,)).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            return int(self._db.execute("SELECT COUNT(*) FROM items").fetchone()[0])
        return int(self._db.execute("SELECT COUNT(*) FROM items WHERE namespace = ?", (namespace,)).fetchone()[0])

    # -- candidate selection ----------------------------------------------

    def _match_expression(self, query: str) -> str:
        """An FTS5 MATCH expression: every term OR'd and quoted.

        OR rather than AND because retrieval is recall-first here — `coverage`
        supplies precision afterwards, and an AND over the unigrams *and* bigrams
        of a long sentence matches nothing at all.
        """
        if self.tokenizer == "trigram":
            # The trigram tokeniser only understands phrases of three or more
            # characters, so the query is decomposed into its 3-char spans. This
            # is also why a two-character word cannot be searched: 「普洱」 has no
            # trigram to offer.
            clean = "".join(query.split())
            toks = [clean[i : i + 3] for i in range(len(clean) - 2)]
        else:
            toks = T.terms(query)
        seen: set[str] = set()
        out: list[str] = []
        for tok in toks:
            if tok and tok not in seen:
                seen.add(tok)
                out.append('"' + tok.replace('"', '""') + '"')
        return " OR ".join(out)

    def _candidates(self, namespace: str, match: str, limit: int) -> list[sqlite3.Row]:
        if not match:
            return []
        try:
            return self._db.execute(
                "SELECT i.*, bm25(items_fts) AS rank FROM items_fts f JOIN items i ON i.id = f.rowid"
                " WHERE items_fts MATCH ? AND i.namespace = ? ORDER BY rank LIMIT ?",
                (match, namespace, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # A malformed MATCH is a caller bug, not a crash: recall stays a total
            # function, and the vector side may still find something.
            return []

    # -- recall ------------------------------------------------------------

    def _char_idf(self, namespace: str) -> tuple[dict[str, float], float]:
        """Character document frequency in one namespace, and its mean IDF.

        Used only for title matching. A character like 盐 appears in one of 36
        daily entries while 多 appears in nine; treating them as equal evidence
        let a question about salt return the water entry because both shared
        exactly one of three topic characters and recency broke the tie.
        """
        cached = self._idf_cache.get(namespace)
        if cached is not None:
            return cached
        docs = [
            {
                ch
                for kind, run in T._content_segments(self.normalize(str(row[0] or "")))
                for ch in run
            }
            for row in self._db.execute(
                "SELECT search_text FROM items WHERE namespace = ?", (namespace,)
            )
        ]
        n = len(docs)
        df: dict[str, int] = {}
        for chars in docs:
            for ch in chars:
                df[ch] = df.get(ch, 0) + 1
        idf = {ch: math.log((n + 1) / (freq + 1)) + 1.0 for ch, freq in df.items()}
        mean = sum(idf.values()) / len(idf) if idf else 1.0
        self._idf_cache[namespace] = (idf, mean)
        return idf, mean

    def recall(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 8,
        candidate_limit: int = 200,
        kinds: Sequence[str] | None = None,
        recent: Sequence[str] = (),
    ) -> list[Recalled]:
        """Rank memories for `query`, best first.

        Each signal is in 0..1 before weighting, so the total is comparable across
        queries. A candidate must show lexical, vector, or (for the knowledge base)
        title evidence to be ranked at all — recency and salience only break ties
        between real matches.

        `recent` are the preceding user turns. Each is searched as its *own* query
        and the best lexical score is taken with a discount (`context_weight`),
        never concatenated into one string. Concatenation makes every earlier topic
        vote on the answer — ask about the knee after two other subjects and the
        third answer drifts back to the first one's memory. A discounted max keeps
        a strong current match intact while still letting a bare follow-up
        (「那次疼了几天」, 「她叫什么」) find the thing it refers to.
        """
        clean = self.query_normalize(T.strip_query_noise(query or "")).strip()
        ctx_clean = ""
        if recent:
            c = self.query_normalize(T.strip_query_noise(recent[-1])).strip()
            if c:
                ctx_clean = c
        if not clean and not ctx_clean:
            return []

        # Collect candidates from the current turn and (if present) the previous
        # one, in one pass. A row is scored once, with the max of both queries.
        by_id: dict[int, sqlite3.Row] = {}
        for q in (clean, ctx_clean):
            if not q:
                continue
            for r in self._candidates(namespace, self._match_expression(q), candidate_limit):
                by_id.setdefault(int(r["id"]), r)
        # Do NOT return early here: with an embedder, a query with zero lexical
        # candidates can still be answered by vector evidence alone. The vector
        # block below adds its own rows to by_id; an early return used to make
        # pure-vector recall always empty.
        rows = list(by_id.values())

        lexical_cur = self._lexical_scores(T.query_terms(clean), rows) if clean else {}
        lexical_ctx = self._lexical_scores(T.query_terms(ctx_clean), rows) if ctx_clean else {}

        sem: dict[int, float] = {}
        if self.embedder is not None and len(self.vectors) and clean:
            qvec = self.embedder.embed(clean)
            pool = [
                int(r["id"])
                for r in self._db.execute(
                    "SELECT id FROM items WHERE namespace = ? LIMIT ?", (namespace, candidate_limit)
                )
            ]
            for item_id, sim in self.vectors.search(qvec, pool).items():
                if sim > self.semantic_floor:
                    sem[item_id] = sim
                    if item_id not in by_id:
                        row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
                        if row is not None:
                            by_id[item_id] = row

        now = self.clock()
        idf, mean_idf = self._char_idf(namespace) if self.subject_bonus else ({}, 1.0)
        hits: list[Recalled] = []
        for item_id, row in by_id.items():
            lex = max(
                lexical_cur.get(item_id, 0.0),
                self.context_weight * lexical_ctx.get(item_id, 0.0),
            )
            semantic = sem.get(item_id, 0.0)
            # A hand-written title is the author's own summary of the row. When
            # the question names the topic but uses a different verb than the
            # body, a title match is real evidence (「盐吃多了」 vs 「每天盐摄入
            # 量」, 「深蹲怎么做」 vs 「深蹲正确动作」). Match the *first content
            # segment* of the question at character level: that is the topic,
            # while later segments are usually 处理 / 怎么办 scaffolding shared
            # by every first-aid entry. Shared characters are IDF-weighted so a
            # rare 盐 beats a common 多. Long topic segments (>4 chars) are not a
            # topic word but a whole sentence: 「量子计算机原理」 would otherwise
            # share 量 with 「力量训练基本原则」 and block the web fallback.
            # Off by default for personal memories, whose subjects are often
            # empty and must not add phantom evidence.
            subject_hit = 0.0
            if self.subject_bonus and clean and row["subject"]:
                topic_segments = [
                    run for kind, run in T._content_segments(clean) if kind == "cjk"
                ]
                if topic_segments and len(topic_segments[0]) <= 4:
                    want = set(topic_segments[0])
                    have = {
                        ch
                        for kind, run in T._content_segments(self.normalize(str(row["subject"])))
                        for ch in run
                    }
                    shared = want & have
                    if shared:
                        denominator = sum(idf.get(ch, mean_idf) for ch in want)
                        subject_hit = (
                            sum(idf.get(ch, mean_idf) for ch in shared) / denominator
                            if denominator
                            else 0.0
                        )
            if lex <= 0.0 and semantic <= 0.0 and subject_hit <= 0.0:
                continue
            if kinds is not None and row["kind"] not in kinds:
                continue
            memory = self._row_to_memory(row)
            parts = {
                "lexical": lex,
                "semantic": semantic,
                "recency": self._recency(memory.created_at, now),
                "salience": memory.salience,
                "subject": subject_hit,
            }
            score = sum(parts[k] * self.weights[k] for k in parts)
            hits.append(Recalled(memory=memory, score=score, parts=parts))

        hits.sort(key=lambda h: (-h.score, -h.memory.id))
        picked = hits[:limit]
        self._touch([h.memory.id for h in picked], now)
        return picked

    def _lexical_scores(
        self, want: dict[str, float], rows: Sequence[sqlite3.Row]
    ) -> dict[int, float]:
        """Score each candidate row, either by query coverage or by raw bm25.

        `bm25` is kept only as a baseline. Returning it raw is deliberate: its
        range depends on corpus size, and normalising it per candidate set is the
        exact step that turned a corpus-wide zero into a perfect 1.0 and let
        recency decide the ranking.
        """
        if self.scorer == "bm25":
            # bm25 is negative and lower-is-better: the most relevant row has the
            # most negative score. Map it to 1.0 and the least relevant to 0.0.
            # The old (v-min)/span formula did the opposite and put the best match
            # last, which is why bm25 looked even worse than its real baseline.
            ranks = {
                int(r["id"]): float(r["rank"] or 0.0)
                for r in rows
                if "rank" in r.keys()
            }
            if not ranks:
                return {}
            hi = max(ranks.values())   # least relevant (closest to zero)
            lo = min(ranks.values())   # most relevant (most negative)
            span = hi - lo
            if span <= 0:
                # A single FTS candidate has no spread to normalise against,
                # but it *did* match the query — that is real evidence. Give it
                # full credit instead of dropping it (which made a one-row
                # relevant match look like no evidence at all).
                return {k: 1.0 for k in ranks}
            return {k: (hi - v) / span for k, v in ranks.items()}

        out: dict[int, float] = {}
        for row in rows:
            score = T.coverage(want, self._have(row), char_tier=self.char_tier)
            if score > 0.0:
                out[int(row["id"])] = score
        return out

    def _recency(self, created_at: float, now: float) -> float:
        if self.half_life_days <= 0:
            return 1.0
        age_days = max(0.0, (now - created_at) / 86400.0)
        return float(0.5 ** (age_days / self.half_life_days))

    def _touch(self, ids: Sequence[int], now: float) -> None:
        if not ids:
            return
        self._db.executemany(
            "UPDATE items SET last_hit_at = ?, hit_count = hit_count + 1 WHERE id = ?",
            [(now, i) for i in ids],
        )
        self._db.commit()

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def render(hits: Sequence[Recalled], *, max_chars: int = 1800, max_items: int = 8) -> str:
        """Render hits as a prompt block.

        Weak items are dropped rather than truncated: a memory cut in half reads
        like a fact with a missing clause, and the model has no way to know the
        rest was never sent.
        """
        lines: list[str] = []
        used = 0
        for hit in list(hits)[:max_items]:
            line = f"- {hit.body}"
            if used + len(line) + 1 > max_chars:
                continue
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
