"""RAGAS 评估指标 - 自实现 Faithfulness, Relevancy, Precision, Recall

不依赖 ragas 包（其 0.4.x 有依赖冲突），直接用 LLM-as-Judge + Embedding 相似度实现。
"""

import json
import logging
import re
from typing import List, Optional

import numpy as np
from openai import OpenAI

from config import get_settings

logger = logging.getLogger(__name__)


def _get_judge_llm() -> OpenAI:
    """获取评估用 LLM 客户端（DeepSeek）"""
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _get_judge_model() -> str:
    return get_settings().openai_model


def _llm_judge(prompt: str) -> str:
    """调用 Judge LLM 获取评估结果"""
    client = _get_judge_llm()
    resp = client.chat.completions.create(
        model=_get_judge_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


def _get_embeddings(texts: List[str]) -> np.ndarray:
    """使用本地 Embedding 模型获取向量"""
    from app.ingestion.indexer import get_embeddings
    embedder = get_embeddings()
    return np.array(embedder.embed_documents(texts))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _extract_json(text: str) -> Optional[dict]:
    """鲁棒地从 LLM 输出中提取 JSON，处理常见格式问题"""
    # 尝试直接匹配最外层 JSON
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        return None
    raw = json_match.group()
    # 第一次尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 修复常见问题：尾部多余逗号
    cleaned = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 尝试只提取 score 字段
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
    if score_match:
        return {"score": float(score_match.group(1))}
    return None


# ==================== Faithfulness（忠实度） ====================

def _eval_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    """
    忠实度：回答中的每个声明是否都能从上下文中找到依据。
    方法：让 LLM 提取回答中的关键声明，逐一检查是否有上下文支持。
    """
    if not answer or answer.strip() == "":
        return 0.0

    context_text = "\n".join(contexts)
    prompt = f"""你是一个严格的评估专家。请评估以下回答的忠实度。

## 任务
判断回答中的每个关键声明（claim）是否能从给定上下文中找到支持依据。

## 上下文
{context_text}

## 问题
{question}

## 回答
{answer}

## 评估步骤
1. 从回答中提取所有关键事实性声明（忽略过渡词、格式词）
2. 逐一判断每个声明是否有上下文支持（supported / not_supported）
3. 计算：faithfulness = supported数量 / 总声明数量

## 输出格式（严格JSON）
{{"claims": [{{"claim": "...", "verdict": "supported"或"not_supported"}}], "score": 0.0到1.0之间的小数}}"""

    try:
        result = _llm_judge(prompt)
        data = _extract_json(result)
        if data:
            return float(data.get("score", 0.0))
    except Exception as e:
        logger.warning(f"Faithfulness 评估失败: {e}")
    return 0.5  # 默认中间值


# ==================== Answer Relevancy（答案相关性） ====================

def _eval_answer_relevancy(question: str, answer: str) -> float:
    """
    答案相关性：回答与问题的语义相关程度。
    方法：LLM 直接评判回答是否切题、是否完整覆盖了问题的各个方面。
    """
    if not answer or answer.strip() == "":
        return 0.0

    prompt = f"""你是一个严格的评估专家。请评估以下回答与问题的相关性。

## 问题
{question}

## 回答
{answer}

## 评估标准
- 1.0：回答完全切题，直接且完整地回答了问题的所有方面
- 0.8：回答基本切题，覆盖了问题的主要内容，但有少量冗余或遗漏
- 0.6：回答部分相关，但偏离了问题重点或遗漏了重要方面
- 0.4：回答相关性较弱，大部分内容与问题无关
- 0.0：回答完全不相关或为空

注意：
- 回答中包含来源标注（如[来源: xxx]）不影响相关性评分
- 回答使用列表/分段组织不影响评分
- 只关注回答是否切题地回答了问题本身

## 输出格式（严格JSON）
{{"reasoning": "一句话说明评分理由", "score": 0.0到1.0之间的小数}}"""

    try:
        result = _llm_judge(prompt)
        data = _extract_json(result)
        if data:
            return float(data.get("score", 0.5))
    except Exception as e:
        logger.warning(f"Answer Relevancy 评估失败: {e}")
    return 0.5


# ==================== Context Precision（上下文精确度） ====================

def _eval_context_precision(question: str, contexts: List[str], ground_truth: str) -> float:
    """
    上下文精确度：相关文档是否排在前面（加权精确度）。
    方法：让 LLM 判断每个 context 是否对回答问题有帮助，计算加权得分。
    """
    if not contexts:
        return 0.0

    contexts_formatted = "\n".join([f"[文档{i+1}] {c[:600]}" for i, c in enumerate(contexts)])
    prompt = f"""你是一个评估专家。请判断每个检索到的文档对回答问题的帮助程度。

## 问题
{question}

## 标准答案
{ground_truth}

## 检索到的文档
{contexts_formatted}

## 任务
对每个文档判断：它是否包含对回答该问题有用的信息？（relevant / not_relevant）

## 输出格式（严格JSON）
{{"verdicts": [{{"doc": 1, "verdict": "relevant"或"not_relevant"}}], "num_relevant": 数字}}"""

    try:
        result = _llm_judge(prompt)
        data = _extract_json(result)
        if data:
            verdicts = data.get("verdicts", [])
            if not verdicts:
                return 0.5

            # 计算加权精确度（排名靠前的权重更大）
            n = len(contexts)
            relevant_flags = []
            for v in verdicts:
                relevant_flags.append(1 if v.get("verdict") == "relevant" else 0)

            # 补齐
            while len(relevant_flags) < n:
                relevant_flags.append(0)

            # Weighted Precision@K: sum(precision@k * rel(k)) / num_relevant
            num_relevant = sum(relevant_flags)
            if num_relevant == 0:
                return 0.0

            weighted_sum = 0.0
            cumulative_relevant = 0
            for k in range(n):
                cumulative_relevant += relevant_flags[k]
                if relevant_flags[k] == 1:
                    precision_at_k = cumulative_relevant / (k + 1)
                    weighted_sum += precision_at_k

            return weighted_sum / num_relevant
    except Exception as e:
        logger.warning(f"Context Precision 评估失败: {e}")
    return 0.5


# ==================== Context Recall（上下文召回率） ====================

def _eval_context_recall(contexts: List[str], ground_truth: str) -> float:
    """
    上下文召回率：标准答案中的信息有多少能从检索结果中找到。
    方法：将标准答案拆分为多个声明，检查每个声明是否能从 context 中推导出来。
    """
    if not contexts or not ground_truth:
        return 0.0

    context_text = "\n".join(contexts)
    prompt = f"""你是一个评估专家。请评估检索结果对标准答案的覆盖程度。

## 标准答案
{ground_truth}

## 检索到的上下文
{context_text}

## 任务
1. 将标准答案拆分为独立的关键信息点（claims）
2. 逐一判断每个信息点是否能从上下文中找到或推导出来（attributable / not_attributable）
3. 计算：recall = attributable数量 / 总信息点数量

## 输出格式（严格JSON）
{{"claims": [{{"claim": "...", "verdict": "attributable"或"not_attributable"}}], "score": 0.0到1.0之间的小数}}"""

    try:
        result = _llm_judge(prompt)
        data = _extract_json(result)
        if data:
            return float(data.get("score", 0.0))
    except Exception as e:
        logger.warning(f"Context Recall 评估失败: {e}")
    return 0.5


# ==================== 主评估函数 ====================

def evaluate_rag(
    questions: List[str],
    ground_truths: Optional[List[str]] = None,
) -> dict:
    """
    评估 RAG 系统的四维指标。

    使用 LLM-as-Judge 方法：
    - Faithfulness：回答是否忠于检索到的上下文（不编造）
    - Answer Relevancy：回答与问题的语义相关程度
    - Context Precision：相关文档是否排在前面
    - Context Recall：检索结果覆盖标准答案的程度

    Args:
        questions: 测试问题列表
        ground_truths: 标准答案列表（用于 Precision 和 Recall）

    Returns:
        评估报告字典
    """
    from app.generation.chain import RAGChain

    logger.info(f"开始 RAG 评估: {len(questions)} 个问题")

    # 初始化 RAG Chain（开启 rerank + query_transform 提升召回率）
    chain = RAGChain(use_query_transform=True, use_rerank=True)

    # 收集 RAG 输出
    answers = []
    contexts_list = []

    for i, question in enumerate(questions):
        logger.info(f"  [{i+1}/{len(questions)}] 正在获取 RAG 回答: {question[:30]}...")
        response = chain.invoke(question)
        answers.append(response.answer)
        contexts_list.append([doc.page_content for doc in response.sources])

    # 逐样本评估
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []
    details = []

    for i, (q, a, ctxs) in enumerate(zip(questions, answers, contexts_list)):
        logger.info(f"  [{i+1}/{len(questions)}] 正在评估: {q[:30]}...")

        # Faithfulness
        f_score = _eval_faithfulness(q, a, ctxs)
        faithfulness_scores.append(f_score)

        # Answer Relevancy
        r_score = _eval_answer_relevancy(q, a)
        relevancy_scores.append(r_score)

        # Context Precision & Recall（需要 ground_truth）
        p_score = 0.0
        rc_score = 0.0
        if ground_truths and i < len(ground_truths):
            p_score = _eval_context_precision(q, ctxs, ground_truths[i])
            rc_score = _eval_context_recall(ctxs, ground_truths[i])
        precision_scores.append(p_score)
        recall_scores.append(rc_score)

        details.append({
            "question": q,
            "answer": a[:200] + "..." if len(a) > 200 else a,
            "num_contexts": len(ctxs),
            "faithfulness": round(f_score, 3),
            "answer_relevancy": round(r_score, 3),
            "context_precision": round(p_score, 3),
            "context_recall": round(rc_score, 3),
        })

    # 汇总
    metrics = {
        "faithfulness": round(float(np.mean(faithfulness_scores)), 4),
        "answer_relevancy": round(float(np.mean(relevancy_scores)), 4),
    }
    if ground_truths:
        metrics["context_precision"] = round(float(np.mean(precision_scores)), 4)
        metrics["context_recall"] = round(float(np.mean(recall_scores)), 4)

    report = {
        "metrics": metrics,
        "num_samples": len(questions),
        "details": details,
    }

    logger.info(f"评估完成: {metrics}")
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
