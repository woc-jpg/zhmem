"""Tokenisation and scoring rules, each tied to a measured failure."""

from zhmem import text as T


def test_two_character_word_survives_as_one_token():
    # trigram cannot represent 普洱; the bigram index must keep it whole.
    assert "普洱" in T.terms("喜欢喝普洱")


def test_bigram_does_not_span_a_generic_character():
    # 了 is generic: a bigram must never straddle it (膝了, 了盖 are not words).
    segs = T._content_segments("膝了盖")
    assert [s for _, s in segs] == ["膝", "盖"]
    assert "膝了" not in T.terms("膝了盖")
    assert "了盖" not in T.terms("膝了盖")


def test_content_characters_that_build_words_are_kept():
    # Removing 上/下/日 would silently destroy 上周/下午/日本.
    for word in ("上周", "下午", "日本", "时间"):
        assert word in T.terms(word)


def test_question_scaffolding_is_stripped_whole():
    assert T.strip_query_noise("我膝盖疼怎么办") == "我膝盖疼"
    assert T.strip_query_noise("成都在哪") == "成都在哪"  # 哪 alone stripped, 成都 intact


def test_isolated_character_gets_word_weight_but_word_character_does_not():
    q = T.query_terms(T.strip_query_noise("咖啡怎么喝"))
    # 喝 stands alone after stripping → full topic weight.
    assert q.get("喝") == 1.0
    # 学 inside 量子力学 is not evidence on its own.
    q2 = T.query_terms("量子力学")
    assert "学" not in q2
    assert "力学" in q2


def test_coverage_prefers_words_and_only_falls_back_to_characters():
    have = set(T.index_text("喝美式不加糖").split())
    # 咖啡怎么喝 shares only 喝; word layer finds nothing, char tier gives weak evidence.
    score = T.coverage(T.query_terms("咖啡怎么喝"), have)
    assert 0 < score <= 0.25


def test_coverage_never_exceeds_one_and_is_absolute():
    want = T.query_terms("膝盖疼")
    have = set(T.index_text("膝盖不太好，阴雨天会疼").split())
    assert 0 < T.coverage(want, have) <= 1.0


def test_single_character_noise_does_not_match_unrelated_memory():
    want = T.query_terms("量子力学")
    have = set(T.index_text("女儿家的孩子今年上小学二年级").split())
    # they share 学, but 学 is inside a word and gets no weight.
    assert T.coverage(want, have) == 0.0


def test_isolated_character_survives_char_tier_off():
    # An isolated single char is the whole topic ("喝" in 咖啡怎么喝).
    # With the weak char tier disabled it must still score; the old code dropped
    # it because coverage only looked at multi-char tokens in that mode.
    want = T.query_terms("咖啡怎么喝")
    have = set(T.index_text("喝美式不加糖").split())
    assert T.coverage(want, have, char_tier=False) > 0


def test_has_topic_requires_a_real_content_segment():
    # 膝了盖 is three isolated single chars separated by a generic wall — no
    # two-char segment, so there is nothing specific to search on.
    assert not T.has_topic("膝了盖")
    # 膝盖疼 has a real content run.
    assert T.has_topic("膝盖疼")
