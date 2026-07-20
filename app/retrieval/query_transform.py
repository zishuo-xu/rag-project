"""查询改写 - Multi-Query 与 HyDE 策略"""

import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings

logger = logging.getLogger(__name__)

# Multi-Query 改写 Prompt
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """你是一个 AI 搜索查询优化器。你的任务是从不同角度改写用户的问题，生成 3 个不同版本的查询，
以便在向量数据库中检索到更全面的相关文档。

通过生成多个不同视角的查询，可以克服用户原始措辞的局限性，
提高检索到相关文档的概率。

请生成 3 个不同版本的查询，每行一个，不要添加编号或其他格式。

原始问题: {question}"""
)

# HyDE (Hypothetical Document Embeddings) Prompt
HYDE_PROMPT = ChatPromptTemplate.from_template(
    """请针对以下问题，写一段简短的文字来回答它。
这段文字不需要完全准确，但应该包含可能出现在相关文档中的关键术语和概念。

问题: {question}

回答（200字以内）："""
)


class QueryTransformer:
    """
    查询改写器 - 优化用户查询以提高检索效果。

    支持两种策略：
    1. Multi-Query: 生成多个查询变体，扩大召回面
    2. HyDE: 生成假设性文档，用其 embedding 做检索
    """

    def __init__(self, llm=None):
        settings = get_settings()
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.7,
        )

    def multi_query(self, question: str, num_queries: int = 3) -> List[str]:
        """
        Multi-Query 改写：生成多个查询变体。

        原理：用户原始查询可能措辞不够精确或角度单一，
        通过 LLM 生成多个不同视角的查询，分别检索后合并结果，
        可以显著提高召回率。

        Args:
            question: 用户原始问题
            num_queries: 生成的查询数量

        Returns:
            查询变体列表（包含原始查询）
        """
        chain = MULTI_QUERY_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"question": question})
            queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
            queries = queries[:num_queries]
        except Exception as e:
            logger.warning(f"Multi-Query 生成失败: {e}")
            queries = []

        # 始终包含原始查询
        all_queries = [question] + queries
        # 去重
        all_queries = list(dict.fromkeys(all_queries))

        logger.info(f"Multi-Query: 原始 + {len(all_queries) - 1} 个变体")
        return all_queries

    def hyde(self, question: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings) 策略。

        原理：让 LLM 先生成一个"假设性回答"，
        然后用这个假设回答的 embedding 去检索，
        因为假设回答在语义空间中更接近真实文档。

        Args:
            question: 用户原始问题

        Returns:
            假设性文档文本（用于 embedding 检索）
        """
        chain = HYDE_PROMPT | self.llm | StrOutputParser()

        try:
            hypothetical_doc = chain.invoke({"question": question})
            logger.info(f"HyDE: 生成假设文档 ({len(hypothetical_doc)} 字符)")
            return hypothetical_doc
        except Exception as e:
            logger.warning(f"HyDE 生成失败: {e}, 回退到原始查询")
            return question

    def transform(
        self,
        question: str,
        strategy: str = "multi_query",
    ) -> List[str]:
        """
        查询改写统一入口。

        Args:
            question: 用户原始问题
            strategy: 改写策略 ("multi_query" | "hyde" | "none")

        Returns:
            改写后的查询列表
        """
        if strategy == "multi_query":
            return self.multi_query(question)
        elif strategy == "hyde":
            return [self.hyde(question)]
        else:
            return [question]
