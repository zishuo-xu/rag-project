"""F12 重写层评估脚本测试（离线）

覆盖：
1. eval_rewrite：触发 + 改写 + gold 关键词全命中 → ok
2. gold 关键词未命中 → ok=False 且记入 missed
3. 非追问样本（无需重写）→ ok=False 且 changed=False
4. 无 rewrite_gold 的样本不因关键词判负
5. 真实多轮数据集：启发式路径全量跑通且结构完整
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from app.retrieval.conversation import ConversationRewriter
from run_rewrite_eval import eval_rewrite


def _rewriter():
    s = MagicMock()
    s.use_history_rewrite = True
    s.history_rewrite_use_llm = False
    s.history_rewrite_max_turns = 4
    return ConversationRewriter(settings=s)


HISTORY = [
    {"role": "user", "content": "什么是 Redis 的缓存穿透？"},
    {"role": "assistant", "content": "查询不存在的数据。"},
]


def test_hit_all_keywords_ok():
    r = eval_rewrite(_rewriter(), {
        "id": "x1", "question": "它的解决方案有哪些？",
        "history": HISTORY, "rewrite_gold": ["Redis", "缓存穿透"],
    })
    assert r["triggered"] is True
    assert r["changed"] is True
    assert r["missed"] == []
    assert r["ok"] is True


def test_missed_keyword_not_ok():
    r = eval_rewrite(_rewriter(), {
        "id": "x2", "question": "它的解决方案有哪些？",
        "history": HISTORY, "rewrite_gold": ["Redis", "布隆过滤器"],
    })
    assert r["ok"] is False
    assert r["missed"] == ["布隆过滤器"]


def test_non_followup_not_rewritten():
    r = eval_rewrite(_rewriter(), {
        "id": "x3", "question": "什么是数据库索引？",
        "history": HISTORY, "rewrite_gold": [],
    })
    assert r["triggered"] is False
    assert r["changed"] is False
    assert r["ok"] is False


def test_empty_gold_not_penalized():
    r = eval_rewrite(_rewriter(), {
        "id": "x4", "question": "它的解决方案有哪些？",
        "history": HISTORY, "rewrite_gold": [],
    })
    assert r["missed"] == []
    assert r["ok"] is True


def test_real_dataset_heuristic_full_pass():
    """真实多轮集在启发式路径下应全部触发且关键词全命中（数据与算法联合回归）。"""
    dataset = json.loads(Path("data/eval_multiturn.json").read_text(encoding="utf-8"))
    results = [eval_rewrite(_rewriter(), s) for s in dataset["samples"]]
    assert len(results) >= 10
    failed = [r["id"] for r in results if not r["ok"]]
    assert failed == [], f"重写层未通过样本: {failed}"
