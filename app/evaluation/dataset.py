"""评估数据集管理 - 测试集构建与加载"""

import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# 默认评估数据集路径
DEFAULT_DATASET_PATH = Path("data/eval_dataset.json")


def create_eval_dataset(
    questions: List[str],
    ground_truths: List[str],
    metadata: Optional[List[dict]] = None,
    save_path: Optional[str] = None,
) -> dict:
    """
    创建评估数据集并保存。

    Args:
        questions: 测试问题列表
        ground_truths: 标准答案列表
        metadata: 额外元数据（如来源文档、难度等级）
        save_path: 保存路径

    Returns:
        数据集字典
    """
    dataset = {
        "version": "1.0",
        "num_samples": len(questions),
        "samples": [],
    }

    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        sample = {
            "id": i,
            "question": q,
            "ground_truth": gt,
        }
        if metadata and i < len(metadata):
            sample["metadata"] = metadata[i]
        dataset["samples"].append(sample)

    # 保存
    path = Path(save_path) if save_path else DEFAULT_DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"评估数据集已保存: {path} ({len(questions)} 条)")
    return dataset


def load_eval_dataset(path: Optional[str] = None) -> dict:
    """
    加载评估数据集。

    Args:
        path: 数据集文件路径

    Returns:
        数据集字典
    """
    file_path = Path(path) if path else DEFAULT_DATASET_PATH

    if not file_path.exists():
        logger.warning(f"评估数据集不存在: {file_path}")
        return {"samples": [], "num_samples": 0}

    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"加载评估数据集: {file_path} ({dataset.get('num_samples', 0)} 条)")
    return dataset


def get_questions_and_answers(path: Optional[str] = None) -> tuple:
    """
    从数据集提取问题和答案列表。

    Returns:
        (questions, ground_truths) 元组
    """
    dataset = load_eval_dataset(path)
    samples = dataset.get("samples", [])

    questions = [s["question"] for s in samples]
    ground_truths = [s["ground_truth"] for s in samples]

    return questions, ground_truths


# 示例评估数据集（可手动扩充）
SAMPLE_EVAL_DATA = {
    "questions": [
        "什么是 RAG（检索增强生成）？",
        "向量检索和关键词检索各有什么优缺点？",
        "什么是 RRF（Reciprocal Rank Fusion）？",
        "Cross-Encoder 和 Bi-Encoder 有什么区别？",
        "如何评估一个 RAG 系统的效果？",
    ],
    "ground_truths": [
        "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术框架，通过从外部知识库检索相关文档来增强大语言模型的回答能力，减少幻觉。",
        "向量检索（稠密检索）基于语义相似度，能理解同义词和语义关系，但对精确匹配较弱；关键词检索（稀疏检索/BM25）基于词频匹配，对专有名词和术语精确匹配效果好，但无法理解语义。",
        "RRF 是一种多路检索结果融合策略，公式为 score = Σ 1/(k+rank)，仅利用排名信息而非原始分数，避免了不同检索系统分数不可比的问题。",
        "Bi-Encoder 将 query 和 document 分别编码为向量再计算相似度，速度快适合大规模检索；Cross-Encoder 将 query 和 document 拼接后一起输入模型，精度更高但速度慢，适合对少量候选做精排。",
        "可以使用 RAGAS 框架评估，主要指标包括：Faithfulness（忠实度）、Answer Relevancy（答案相关性）、Context Precision（上下文精确度）、Context Recall（上下文召回率）。",
    ],
}


def init_sample_dataset():
    """初始化示例评估数据集"""
    create_eval_dataset(
        questions=SAMPLE_EVAL_DATA["questions"],
        ground_truths=SAMPLE_EVAL_DATA["ground_truths"],
    )
    logger.info("示例评估数据集初始化完成")
