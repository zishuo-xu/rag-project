"""CRAG (Corrective RAG) - 自纠正检索：评估检索质量并自动补救"""

import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import get_settings, get_llm_extra_body
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
        # #17: LLM 添加超时和重试
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            max_tokens=256,
            request_timeout=30,
            max_retries=2,
            extra_body=get_llm_extra_body(),
        )
        self.threshold = settings.crag_relevance_threshold

    def should_retrieve(self, question: str) -> Tuple[bool, str]:
        """
        判断问题是否需要检索。

        Returns:
            (need_retrieval: bool, reason: str)
        """
        try:
            prompt = CRAG_SHOULD_RETRIEVE_PROMPT.format(question=question)
            response = self.llm.invoke(prompt)
            data = extract_json(response.content)
            if data:
                need = data.get("need_retrieval", True)
                reason = data.get("reason", "")
                logger.info(f"CRAG should_retrieve: {need} ({reason})")
                return need, reason
        except Exception as e:
            logger.warning(f"CRAG should_retrieve 失败: {e}")
        # 默认需要检索
        return True, "默认需要检索"

    def evaluate_relevance(
        self, question: str, documents: List[Document]
    ) -> Tuple[str, List[int], str]:
        """
        评估检索结果与问题的相关性。

        Args:
            question: 用户问题
            documents: 检索到的文档列表

        Returns:
            (grade, relevant_indices, reason)
            grade: "correct" | "ambiguous" | "incorrect"
        """
        if not documents:
            return "incorrect", [], "无检索结果"

        # 格式化文档（只取前5篇，每篇截断300字）
        docs_text = ""
        for i, doc in enumerate(documents[:5]):
            content = doc.page_content[:300]
            source = doc.metadata.get("source", "未知")
            docs_text += f"[文档{i+1}] (来源: {source})\n{content}\n\n"

        try:
            prompt = CRAG_EVALUATE_PROMPT.format(
                question=question,
                documents=docs_text,
            )
            response = self.llm.invoke(prompt)
            data = extract_json(response.content)
            if data:
                grade = data.get("grade", "ambiguous")
                indices = data.get("relevant_indices", [])
                reason = data.get("reason", "")
                logger.info(f"CRAG evaluate: grade={grade}, relevant={indices} ({reason})")
                return grade, indices, reason
        except Exception as e:
            logger.warning(f"CRAG evaluate 失败: {e}")

        # 默认认为相关
        return "correct", list(range(len(documents))), "评估失败，默认通过"

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

