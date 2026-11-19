"""Synonym folding, applied to the index and the query by the same function.

The failure this module exists to prevent is subtle and easy to ship: fold
synonyms on the query side only, and recall *drops*. A memory stored as
「最近睡不着」 gets indexed verbatim; the query 「老人家失眠」 gets rewritten to
「老人家失眠」 → the canonical form; FTS5 then finds nothing, and the vector
score measured 0.169, below any usable floor. Nothing raises, nothing logs — the
memory is simply unreachable.

So `normalise()` is the single entry point, and the store calls it on both sides.
That is also why `normalise` lives on the store rather than at each call site:
one place to be wrong, one place to fix.

A real embedding model does not need this (it already knows 失眠 ≈ 睡不着), which
is why folding is opt-in per store rather than baked into tokenisation.
"""

from __future__ import annotations

__all__ = ["CANONICAL", "FOLD", "fold", "longest_match_len"]


def _pairs(*groups: tuple[str, str, tuple[str, ...]]) -> dict[str, str]:
    """Build variant → canonical from (canonical, canonical_again, variants).

    Every canonical form maps to itself. Without that, a second pass over the
    output would fold a canonical word's *substring*: 拉肚子 contains 肚子, so it
    became 拉腹部 — a word that exists in no dictionary and matches nothing.
    """
    table: dict[str, str] = {}
    for canonical, alias, variants in groups:
        table[canonical] = canonical
        table[alias] = canonical
        for v in variants:
            table[v] = canonical
    return table


# Groups are semantic, not thesaural: each one is a set of words a family member
# would plausibly use for the same fact on different days.
_BODY = _pairs(
    ("膝盖", "膝关节", ("膝",)),
    ("肚子", "腹部", ("小腹", "肠胃", "胃")),
    ("头", "脑袋", ("头部",)),
    ("牙", "牙齿", ("牙床",)),
    ("腰椎", "腰", ("腰背",)),
)
_PAIN = _pairs(
    ("痛", "疼", ("不舒服", "不适", "难受")),
)
_FEELING = _pairs(
    ("失眠", "睡不着", ("睡眠不好", "睡不好", "无法入睡", "睡觉", "入睡")),
    ("高兴", "开心", ("愉快", "快乐", "心情好")),
    ("难过", "伤心", ("低落", "心情不好")),
    ("累", "疲劳", ("乏力", "没力气")),
)
_DAILY = _pairs(
    ("散步", "走路", ("遛弯", "走一走")),
    ("就诊", "看病", ("去医院", "挂号", "复诊")),
)
_FAMILY = _pairs(
    ("孩子", "儿子", ("女儿", "娃")),
    ("老伴", "伴侣", ("丈夫", "妻子", "老公", "老婆")),
)
# Everyday co-reference that no character-level scorer can bridge: 茶 is the
# family name for 普洱/浓茶, and a blood-sugar question is about 糖尿病. Both
# were measured as total misses (coverage 0.0) on the eval set.
_DRINK = _pairs(
    ("茶", "茶叶", ("普洱", "浓茶", "红茶", "绿茶")),
    ("咖啡", "美式", ("拿铁", "意式", "espresso")),
)
_CONDITION = _pairs(
    ("糖尿病", "血糖", ("血糖高", "糖尿", "高血糖")),
)

CANONICAL: dict[str, str] = {
    **_BODY, **_PAIN, **_FEELING, **_DAILY, **_FAMILY, **_DRINK, **_CONDITION
}
FOLD = CANONICAL

_BY_LENGTH = sorted(CANONICAL, key=len, reverse=True)
_MAX_LEN = max(len(k) for k in _BY_LENGTH)


def longest_match_len() -> int:
    return _MAX_LEN


def fold(text: str) -> str:
    """Replace known variants with canonical forms, left to right.

    Single pass taking the longest match at each position, because sequential
    `str.replace` re-folds its own output: with 肚子→腹部 and 拉肚子→拉肚子 both
    in the table, replacing 肚子 first turns 拉肚子 into 拉腹部 and the correct
    rule never fires.
    """
    if not text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        for size in range(min(_MAX_LEN, n - i), 0, -1):
            chunk = text[i : i + size]
            canonical = CANONICAL.get(chunk)
            if canonical is not None:
                out.append(canonical)
                i += size
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
