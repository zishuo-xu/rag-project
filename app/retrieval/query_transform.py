"""查询改写 - Multi-Query 与 HyDE 策略"""

import logging
import time
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings, get_llm_extra_body

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

# Self-RAG 查询精化 Prompt（迭代检索：基于已有证据与缺口生成补充查询）
REFINE_PROMPT = ChatPromptTemplate.from_template(
    """你是一个检索查询优化器。当前检索到的证据还不足以完整回答问题，
请基于"已有证据"和"检索缺口"，生成一个更精准的新查询，用于补充检索缺失的信息。

要求：
1. 新查询应聚焦于尚未覆盖的缺失信息，不要重复已有证据的内容。
2. 只输出一行查询文本，不要编号、不要解释。

原始问题: {question}

已有证据摘要:
{evidence}

检索缺口/不足: {gap}

新查询："""
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
        # #17: LLM 添加超时和重试
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.7,
            request_timeout=30,
            max_retries=2,
            extra_body=get_llm_extra_body(),
        )
        # #7: 查询改写结果缓存 {question: (timestamp, queries)}
        self._transform_cache: dict[str, tuple[float, List[str]]] = {}
        self._cache_ttl = 3600  # 1小时缓存

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

    def refine(
        self, question: str, evidence_summary: str, gap_reason: str = ""
    ) -> str:
        """
        Self-RAG 查询精化：迭代检索时基于已有证据与缺口生成补充查询。

        原理：当首轮检索证据不足（CRAG 评为 ambiguous/incorrect）时，
        让 LLM 看到"已检索到什么 + 还缺什么"，针对性生成聚焦缺失信息的
        新查询，避免重复召回已有内容。

        Args:
            question: 用户原始问题
            evidence_summary: 已有证据的摘要文本
            gap_reason: CRAG 评估给出的缺口/不足理由

        Returns:
            精化后的查询文本；失败时降级到 HyDE，再失败回退原问题
        """
        chain = REFINE_PROMPT | self.llm | StrOutputParser()
        try:
            refined = chain.invoke({
                "question": question,
                "evidence": evidence_summary or "（暂无）",
                "gap": gap_reason or "信息不完整",
            })
            refined = refined.strip().split("\n")[0].strip()
            if refined:
                logger.info(f"查询精化: {refined[:40]}...")
                return refined
        except Exception as e:
            logger.warning(f"查询精化失败: {e}, 降级到 HyDE")
            return self.hyde(question)
        return question

    def transform(
        self,
        question: str,
        strategy: str = "multi_query",
    ) -> List[str]:
        """
        查询改写统一入口（带缓存）。

        Args:
            question: 用户原始问题
            strategy: 改写策略 ("multi_query" | "hyde" | "none")

        Returns:
            改写后的查询列表
        """
        if strategy == "none":
            return [question]

        # #7: 检查缓存
        cache_key = f"{strategy}:{question}"
        if cache_key in self._transform_cache:
            ts, cached_queries = self._transform_cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                logger.debug(f"查询改写缓存命中: {question[:30]}...")
                return cached_queries
            else:
                del self._transform_cache[cache_key]

        # 执行改写
        if strategy == "multi_query":
            queries = self.multi_query(question)
        elif strategy == "hyde":
            queries = [self.hyde(question)]
        else:
            queries = [question]

        # 写入缓存（限制缓存大小）
        if len(self._transform_cache) > 500:
            # 简单清理：删除最旧的一半
            sorted_keys = sorted(self._transform_cache.keys(),
                               key=lambda k: self._transform_cache[k][0])
            for k in sorted_keys[:len(sorted_keys) // 2]:
                del self._transform_cache[k]
        self._transform_cache[cache_key] = (time.time(), queries)

        return queries
