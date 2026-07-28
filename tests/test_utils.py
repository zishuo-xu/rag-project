"""app/utils 共享工具测试（extract_json / norm_key）

这两个函数被 crag / faithfulness / query_transform / 图谱匹配共用，
解析失败 → None → 上游静默降级，属于正确性敏感点，锁定行为。
"""
from app.utils import extract_json, norm_key


# ============ extract_json ============

def test_extract_json_plain():
    assert extract_json('{"grade": "correct", "score": 0.9}') == {
        "grade": "correct", "score": 0.9
    }


def test_extract_json_code_fence():
    text = '思考...\n```json\n{"a": 1}\n```\n完毕'
    assert extract_json(text) == {"a": 1}


def test_extract_json_surrounding_noise():
    assert extract_json('前缀噪音 {"key": "值"} 后缀噪音') == {"key": "值"}


def test_extract_json_trailing_comma():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_extract_json_nested():
    text = '{"outer": {"inner": [1, 2]}, "ok": true}'
    assert extract_json(text) == {"outer": {"inner": [1, 2]}, "ok": True}


def test_extract_json_garbage_returns_none():
    assert extract_json("完全不是 JSON") is None
    assert extract_json('{"unclosed": ') is None
    assert extract_json("") is None


def test_extract_json_chinese_content():
    assert extract_json('{"reason": "证据充分，含1963年任命信息"}')[
        "reason"
    ] == "证据充分，含1963年任命信息"


# ============ norm_key ============

def test_norm_key_strips_whitespace_and_case():
    assert norm_key("B+ 树") == norm_key("b+树")
    assert norm_key("  Hello   World ") == "helloworld"


def test_norm_key_empty_and_none():
    assert norm_key("") == ""
    assert norm_key(None) == ""


def test_norm_key_consistency_across_spaces():
    """归一化口径：空格差异不影响匹配键（修复历史'仅 lower 漏去空白'失配）。"""
    assert norm_key("范廷颂") == norm_key("范廷颂 ")
    assert norm_key("Redis 集群") == norm_key("redis集群")
