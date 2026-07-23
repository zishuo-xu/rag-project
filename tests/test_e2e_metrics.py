"""端到端答案正确性指标单测（纯函数，零依赖零LLM）

覆盖 F5 harness 使用的 normalize_answer / tokenize_zh / answer_f1 /
normalized_exact_match / answer_hit。
"""
from app.evaluation.metrics import (
    normalize_answer,
    tokenize_zh,
    answer_f1,
    normalized_exact_match,
    answer_hit,
)


# ---- normalize_answer ----

def test_normalize_strips_punct_and_lowercases():
    assert normalize_answer("Hello, World! 你好。") == "helloworld你好"


def test_normalize_keeps_digits_and_chinese():
    assert normalize_answer("1963年。") == "1963年"


def test_normalize_empty():
    assert normalize_answer("") == ""
    assert normalize_answer(None) == ""


# ---- tokenize_zh ----

def test_tokenize_splits_chinese_groups_alnum():
    assert tokenize_zh("1963年") == ["1963", "年"]


def test_tokenize_mixed():
    assert tokenize_zh("使用API接口") == ["使", "用", "api", "接", "口"]


def test_tokenize_empty():
    assert tokenize_zh("") == []


# ---- answer_f1 ----

def test_f1_identical():
    assert answer_f1("1963年", "1963年") == 1.0


def test_f1_no_overlap():
    assert answer_f1("1963年", "完全无关的内容") == 0.0


def test_f1_both_empty():
    assert answer_f1("", "") == 1.0


def test_f1_gold_empty_pred_not():
    assert answer_f1("", "有内容") == 0.0


def test_f1_pred_empty_gold_not():
    assert answer_f1("1963年", "") == 0.0


def test_f1_verbose_pred_full_recall():
    """gold 短答案被冗长 pred 完全覆盖：recall=1，precision 较低，F1 居中"""
    f1 = answer_f1("1963年", "范廷颂于1963年被任命为主教，这是重要的历史事件。")
    assert 0.0 < f1 < 1.0
    # gold 的两个 token(1963,年)都在 pred 中 → recall 应为 1
    # 通过 f1 = 2pr/(p+r)，recall=1 时 f1 = 2p/(p+1)


def test_f1_partial_overlap():
    f1 = answer_f1("天主教河内总主教", "河内总主教")  # 部分覆盖
    assert 0.0 < f1 < 1.0


# ---- normalized_exact_match ----

def test_em_equal_after_norm():
    assert normalized_exact_match("1963年。", "1963年") is True


def test_em_different():
    assert normalized_exact_match("1963年", "1964年") is False


def test_em_empty_gold_false():
    assert normalized_exact_match("", "任意") is False


# ---- answer_hit ----

def test_hit_gold_substring_of_pred():
    assert answer_hit("1963年", "范廷颂于1963年被任为主教。") is True


def test_hit_not_present():
    assert answer_hit("1963年", "文档中未涉及该信息。") is False


def test_hit_ignores_punct_case():
    assert answer_hit("API 接口", "本系统提供api接口服务") is True


def test_hit_empty_gold_false():
    assert answer_hit("", "任意答案") is False
