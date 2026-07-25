"""评估数据集完整性防线（离线）

背景：F6b 多跳集曾把 ground_truth="TODO_核对知识库" 的占位提交进仓库，
导致多跳收益长期未量化。本测试扫描 data/ 下所有评估数据集，断言：
1. 无 TODO/占位类 gold
2. question / ground_truth 非空
3. multiturn 样本的 rewrite_gold 非空且 history 结构合法
4. metadata.source 指向真实存在的文档（多源用 | 分隔）

只校验数据集（含 samples 列表且样本带 question/ground_truth 的 JSON），
跳过评估报告等结果文件。
"""

import json
from pathlib import Path

import pytest

DATA_DIR = Path("data")
PLACEHOLDERS = ("TODO", "待定", "占位", "PLACEHOLDER", "TBD")


def _dataset_files():
    """收集 data/ 顶层所有疑似数据集的 JSON（含 samples 且样本有 question 字段）。"""
    for path in sorted(DATA_DIR.glob("eval_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        samples = payload.get("samples")
        if isinstance(samples, list) and samples and "question" in samples[0]:
            yield path, samples


def test_dataset_files_found():
    """至少应发现主测试集 + 多跳集 + 多轮集。"""
    names = {p.name for p, _ in _dataset_files()}
    assert "eval_dataset_cmrc.json" in names
    assert "eval_multihop.json" in names
    assert "eval_multiturn.json" in names


def test_no_placeholder_gold():
    for path, samples in _dataset_files():
        for s in samples:
            gold = s.get("ground_truth", "")
            assert gold and gold.strip(), f"{path.name}:{s.get('id')} gold 为空"
            for ph in PLACEHOLDERS:
                assert ph not in gold, f"{path.name}:{s.get('id')} gold 含占位符 {ph}"


def test_questions_nonempty():
    for path, samples in _dataset_files():
        for s in samples:
            assert s.get("question", "").strip(), f"{path.name}:{s.get('id')} question 为空"


def test_metadata_source_exists():
    for path, samples in _dataset_files():
        for s in samples:
            source = s.get("metadata", {}).get("source", "")
            if not source:
                continue
            for src in source.split("|"):
                src = src.strip()
                if not src:
                    continue
                found = any((DATA_DIR / sub / src).exists()
                            for sub in ("sample_docs", "cmrc_docs"))
                assert found, f"{path.name}:{s.get('id')} source 不存在: {src}"


def test_multiturn_schema():
    path = DATA_DIR / "eval_multiturn.json"
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
    for s in samples:
        sid = s.get("id")
        assert s.get("slice") == "multiturn", f"{sid} slice 应为 multiturn"
        assert s.get("rewrite_gold"), f"{sid} rewrite_gold 为空"
        history = s.get("history")
        assert isinstance(history, list) and history, f"{sid} history 为空"
        for msg in history:
            assert msg.get("role") in ("user", "assistant"), f"{sid} history role 非法"
            assert msg.get("content", "").strip(), f"{sid} history content 为空"


def test_multihop_schema():
    path = DATA_DIR / "eval_multihop.json"
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
    assert len(samples) >= 10, "多跳集应至少 10 条"
    for s in samples:
        sid = s.get("id")
        assert s.get("slice") == "multihop", f"{sid} slice 应为 multihop"
        assert "chain" in s, f"{sid} 缺 chain 字段"
        assert s.get("hops"), f"{sid} hops 推理链为空"
