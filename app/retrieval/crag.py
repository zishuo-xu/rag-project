"""CRAG (Corrective RAG) - 自纠正检索：评估检索质量并自动补救"""

import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document

from config import get_settings, build_chat_llm
from app.retrieval.router import is_numeric_question
from app.utils import extract_json

logger = logging.getLogger(__name__)

# CRAG 评估 Prompt
CRAG_SHOULD_RETRIEVE_PROMPT = """判断以下问题是否需要从知识库中检索信息才能回答。

## 判断标准
- 需要检索：涉及特定领域知识、技术细节、文档内容、具体数据
- 不需要检索：简单问候、闲聊、通用常识（如"1+1等于几"）、对之前回答的追问澄清

## 问题
{question}

## 输出（严格JSON）
{{"need_retrieval": true或false, "reason": "一句话理由"}}"""

CRAG_EVALUATE_PROMPT = """你是一个检索质量评估专家。请判断以下检索结果对回答问题的帮助程度。

## 问题
{question}

## 检索到的文档（按相关性排序）
{documents}

## 评估标准
- "correct"：文档中包含直接回答问题的关键信息，足以生成高质量回答
- "ambiguous"：文档中部分相关，但信息不完整或需要补充
- "incorrect"：文档与问题基本无关，无法用于回答

## 输出（严格JSON）
{{"grade": "correct"或"ambiguous"或"incorrect", "relevant_indices": [相关文档编号], "reason": "一句话理由"}}"""


class CRAGEvaluator:
    """
    CRAG 自纠正检索评估器。

    功能：
    1. should_retrieve: 判断问题是否需要检索
    2. evaluate_relevance: 评估检索结果质量 (correct/ambiguous/incorrect)
    3. 根据评估结果决定后续动作
    """

    def __init__(self):
        settings = get_settings()
        self.threshold = settings.crag_relevance_threshold
        # 2026-07-28 零 LLM 精简：门控规则化 + 分级用 rerank 分数阈值。
        # 原 LLM 门控/分级 prompt 含检索内容触发内容审查 + qwen thinking 慢，
        # 按性能效果均衡原则精简（效果损：分级从语义判断降为分数阈值，诚实记录）。

    def should_retrieve(self, question: str) -> Tuple[bool, str]:
        """
        判断问题是否需要检索（零 LLM，规则）。

        闲聊/空 query 不检索；其余检索（含通用常识默认检索，漏判少数可接受）。
        原 LLM 版判通用常识不检索，规则难覆盖，默认检索更安全。
        """
        q = (question or "").strip()
        if len(q) < 2:
            return False, "query 过短，无需检索"
        chitchat = {
            "你好", "您好", "谢谢", "感谢", "再见", "拜拜", "你是谁",
            "早上好", "晚上好", "嗨", "哈喽", "在吗", "在不在",
        }
        if q in chitchat or any(q.startswith(w) for w in ("你好", "您好", "谢谢", "再见", "拜拜")):
            return False, "闲聊/问候，无需检索"
        return True, "默认需要检索"

    def evaluate_relevance(
        self, question: str, documents: List[Document]
    ) -> Tuple[str, List[int], str]:
        """
        评估检索结果相关性（零 LLM，用 cross-encoder rerank 分数）。

        documents 已按 rerank_score 降序（pipeline rerank 后），取 top1 sigmoid
        归一化判级。阈值初始经验值（0.5 correct / 0.3 incorrect），基线后看
        grade 分布再校准。
        """
        import math

        if not documents:
            return "incorrect", [], "无检索结果"

        # 取前 5 篇 rerank 分数（已降序，top1 最高）
        scores = [d.metadata.get("rerank_score") for d in documents[:5]]
        scores = [s for s in scores if isinstance(s, (int, float))]
        if not scores:
            return "correct", list(range(len(documents))), "无 rerank 分数，默认通过"

        top1 = float(scores[0])
        p = 1.0 / (1.0 + math.exp(-top1))  # sigmoid → (0,1)
        if p >= 0.5:
            grade = "correct"
        elif p < 0.3:
            grade = "incorrect"
        else:
            grade = "ambiguous"
        reason = f"rerank_sigmoid={p:.2f} (top1={top1:.2f})"
        logger.info(f"CRAG evaluate(零LLM): grade={grade}, {reason}")
        return grade, list(range(len(documents))), reason

    def filter_relevant_docs(
        self, documents: List[Document], relevant_indices: List[int]
    ) -> List[Document]:
        """根据评估结果过滤出相关文档"""
        if not relevant_indices:
            return documents  # 无索引信息时保留全部
        filtered = []
        for idx in relevant_indices:
            # 索引从1开始
            doc_idx = idx - 1
            if 0 <= doc_idx < len(documents):
                filtered.append(documents[doc_idx])
        return filtered if filtered else documents

    @staticmethod
    def validate_numeric_answer(question: str, documents: List[Document]) -> bool:
        """
        数字型答案精确匹配校验（零LLM调用）。

        当问题询问时间/数量/年份时，检查检索结果中是否包含对应的数字信息。

        Args:
            question: 用户问题
            documents: 检索到的文档

        Returns:
            True 表示检索结果中包含数字答案，False 表示缺失
        """
        # 判断问题是否在询问数字型信息（与 router 共用唯一判定）
        if not is_numeric_question(question):
            return True  # 非数字型问题，跳过校验

        # 检查检索结果中是否包含数字信息
        all_context = " ".join(doc.page_content for doc in documents)
        has_numbers = bool(re.search(r'\d{2,}', all_context))
        return has_numbers

