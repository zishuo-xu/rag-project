"""RAGAS 评估指标 - Faithfulness, Relevancy, Precision, Recall"""

import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import get_settings

logger = logging.getLogger(__name__)


def evaluate_rag(
    questions: List[str],
    ground_truths: Optional[List[str]] = None,
) -> dict:
    """
    使用 RAGAS 框架评估 RAG 系统。

    评估指标：
    - Faithfulness（忠实度）：回答是否忠于检索到的上下文
    - Answer Relevancy（答案相关性）：回答与问题的相关程度
    - Context Precision（上下文精确度）：检索结果中相关文档的排名
    - Context Recall（上下文召回率）：检索结果覆盖标准答案的程度

    Args:
        questions: 测试问题列表
        ground_truths: 标准答案列表（可选，用于计算 recall）

    Returns:
        评估报告字典
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset

    from app.generation.chain import RAGChain

    settings = get_settings()
    logger.info(f"开始 RAG 评估: {len(questions)} 个问题")

    # 初始化 RAG Chain
    chain = RAGChain(use_query_transform=False, use_rerank=True)

    # 收集 RAG 输出
    answers = []
    contexts = []

    for question in questions:
        response = chain.invoke(question)
        answers.append(response.answer)
        contexts.append([doc.page_content for doc in response.sources])

    # 构建评估数据集
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }

    if ground_truths:
        eval_data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(eval_data)

    # 选择评估指标
    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths:
        metrics.append(context_recall)

    # 配置评估 LLM 和 Embedding
    eval_llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    eval_embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_base_url,
    )

    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # 构建报告
    report = {
        "metrics": {
            "faithfulness": result.get("faithfulness"),
            "answer_relevancy": result.get("answer_relevancy"),
            "context_precision": result.get("context_precision"),
            "context_recall": result.get("context_recall"),
        },
        "num_samples": len(questions),
        "details": [
            {
                "question": q,
                "answer": a,
                "num_contexts": len(c),
            }
            for q, a, c in zip(questions, answers, contexts)
        ],
    }

    logger.info(f"评估完成: {report['metrics']}")
    return report


def quick_evaluate(questions: List[str], ground_truths: List[str]) -> dict:
    """
    快速评估 - 仅计算不依赖 LLM 的指标。

    适用于快速验证检索质量。
    """
    from app.generation.chain import RAGChain

    chain = RAGChain(use_query_transform=False, use_rerank=True)

    results = []
    for q, gt in zip(questions, ground_truths):
        response = chain.invoke(q)

        # 简单的关键词覆盖率检查
        gt_keywords = set(gt.lower().split())
        answer_keywords = set(response.answer.lower().split())
        overlap = len(gt_keywords & answer_keywords) / max(len(gt_keywords), 1)

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": response.answer,
            "keyword_overlap": overlap,
            "num_sources": len(response.sources),
            "retrieval_time_ms": response.retrieval_result.retrieval_time_ms,
        })

    avg_overlap = sum(r["keyword_overlap"] for r in results) / len(results)

    return {
        "metrics": {"keyword_overlap": avg_overlap},
        "num_samples": len(questions),
        "details": results,
    }
