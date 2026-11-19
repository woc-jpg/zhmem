"""Synonym folding must be single-pass and applied on both sides."""

from zhmem import fold
from zhmem.synonyms import CANONICAL


def test_fold_is_longest_match_and_single_pass():
    # 肚子 is a canonical form and maps to itself, so a word containing it is
    # never corrupted. Sequential replace without self-mapping turned 拉肚子
    # into the non-word 拉腹部.
    assert fold("拉肚子") == "拉肚子"
    assert fold("腹部") == "肚子"
    assert CANONICAL["肚子"] == "肚子"


def test_canonical_forms_map_to_themselves():
    for word in ("肚子", "膝盖", "失眠", "茶", "糖尿病"):
        assert fold(word) == word


def test_family_phrases_fold_to_one_vocabulary():
    assert "孩子" in fold("女儿")
    assert "就诊" in fold("去医院看病")
    assert "失眠" in fold("睡不着")


def test_new_synonym_groups_bridge_world_knowledge():
    assert fold("普洱") == fold("茶")
    assert fold("血糖高") == fold("糖尿病")
    assert fold("美式") == fold("咖啡")
