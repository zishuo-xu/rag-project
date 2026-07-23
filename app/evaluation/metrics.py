"""RAGAS 评估指标 - 自实现 Faithfulness, Relevancy, Precision, Recall

不依赖 ragas 包（其 0.4.x 有依赖冲突），直接用 LLM-as-Judge + Embedding 相似度实现。
"""

import json
import logging
import re
from typing import List, Optional

import numpy as np
from openai import OpenAI

from config import get_settings, get_llm_extra_body

logger = logging.getLogger(__name__)


def _get_judge_llm() -> OpenAI:
    """获取评估用 LLM 客户端（DeepSeek）

    设置 timeout + max_retries，避免单次 judge 调用挂起拖垮整轮评估
    （OpenAI 客户端默认超时 600s，过长）。
    """
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=120.0,
        max_retries=2,
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
        extra_body=get_llm_extra_body(),
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


def _tokenize_for_eval(text: str) -> List[str]:
    """
    评估用分词：中文按字/词切分，英文/数字保持完整 token。
    支持数字（如 "1963"）、英文（如 "API"）、中文词（如 "分布式"）。
    """
    import jieba
    tokens = jieba.lcut(text.lower())
    result = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        # 保留：英文/数字（含小数点）、中文>=2字
        if re.fullmatch(r'[a-zA-Z0-9]+(?:\.[0-9]+)?', t):
            result.append(t)
        elif len(t) >= 2 and any('\u4e00' <= c <= '\u9fff' for c in t):
            result.append(t)
        elif re.fullmatch(r'\d+[\u4e00-\u9fff]?[a-zA-Z\u4e00-\u9fff]*', t) and len(t) >= 2:
            result.append(t)
    return result


def _token_f1(prediction: str, reference: str) -> float:
    """
    Token 级 F1 分数（含数字/英文/中文词）。
    比纯空格分词更准确，适用于中文 QA 评估。
    """
    pred_tokens = _tokenize_for_eval(prediction)
    ref_tokens = _tokenize_for_eval(reference)

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    common = pred_set & ref_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(ref_set)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def quick_evaluate(questions: List[str], ground_truths: List[str]) -> dict:
    """
    快速评估 - 仅计算不依赖 LLM 的指标（Token F1 + 关键词覆盖率）。

    优化：使用 token 级 F1（含数字/英文/中文词），替代原始空格分词 overlap，
    消除数字型答案（如 "1963年"）的误判。

    适用于快速验证检索质量，零 LLM 调用。
    """
    from app.generation.chain import RAGChain

    chain = RAGChain(use_query_transform=False, use_rerank=True)

    results = []
    for q, gt in zip(questions, ground_truths):
        response = chain.invoke(q)

        # Token 级 F1（含数字/英文/中文词）
        f1 = _token_f1(response.answer, gt)

        # 关键词覆盖率（ground_truth 中的 token 在 answer 中出现的比例）
        gt_tokens = set(_tokenize_for_eval(gt))
        answer_tokens = set(_tokenize_for_eval(response.answer))
        overlap = len(gt_tokens & answer_tokens) / max(len(gt_tokens), 1)

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": response.answer,
            "token_f1": round(f1, 4),
            "keyword_overlap": round(overlap, 4),
            "num_sources": len(response.sources),
            "retrieval_time_ms": response.retrieval_result.retrieval_time_ms,
        })

    avg_f1 = sum(r["token_f1"] for r in results) / len(results)
    avg_overlap = sum(r["keyword_overlap"] for r in results) / len(results)

    return {
        "metrics": {
            "token_f1": round(avg_f1, 4),
            "keyword_overlap": round(avg_overlap, 4),
        },
        "num_samples": len(questions),
        "details": results,
    }


# ==================== 端到端答案正确性指标（F5，纯函数零依赖零LLM） ====================
# 与上方 jieba 版 _token_f1 不同：本组函数不依赖外部分词器，按字符切分中文，
# 保证单测可复现、跨环境一致，供 run_e2e_eval.py 端到端 harness 使用。

def normalize_answer(text: str) -> str:
    """端到端评测归一化：转小写，仅保留汉字与英数（去除所有标点与空白）。"""
    return "".join(re.findall(r"[\u4e00-\u9fffa-z0-9]", (text or "").lower()))


def tokenize_zh(text: str) -> List[str]:
    """评测用轻量中文分词：每个汉字一个 token + 连续英数一个 token（零依赖、确定）。"""
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", (text or "").lower())


def answer_f1(gold: str, pred: str) -> float:
    """端到端答案 Token 级 F1（多重集计数）。

    gold 为短答案 span，pred 为完整回答；以 token 重叠衡量 pred 对 gold 的覆盖。
    """
    from collections import Counter
    gold_toks = tokenize_zh(gold)
    pred_toks = tokenize_zh(pred)
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    common = sum((Counter(gold_toks) & Counter(pred_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def normalized_exact_match(gold: str, pred: str) -> bool:
    """严格 EM：归一化后完全相等。"""
    g = normalize_answer(gold)
    return bool(g) and g == normalize_answer(pred)


def answer_hit(gold: str, pred: str) -> bool:
    """宽松命中：归一化后的 gold 作为子串出现在 pred 中。

    RAG 完整回答通常长于短答案 span，子串命中是更贴近实际的端到端正确性信号。
    """
    g = normalize_answer(gold)
    return bool(g) and g in normalize_answer(pred)
