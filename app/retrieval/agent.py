"""F13 Agentic RAG - ReAct 状态机自主检索

把固定七阶段管道升级为 agent 自主决策：LLM 逐步决定调哪个工具、用什么查询、
何时停止（thought → action → observation 循环，概念对齐 LangGraph 的
State/Node/条件边，但零新依赖手写实现，全离线可 mock 测试）。

工具集（混合粗粒度，复用管道已有阶段）：
- search(query)    ：召回 + RRF 融合 + 重排子管道
- decompose()      ：F6b 多跳分解（agent 自主决定，不再依赖 F4 路由判定 multi_hop）
- grade()          ：CRAG 相关性分级，评估当前证据是否充分
- finish()         ：证据足够，结束检索

护栏：max_steps 硬上限 / 决策解析失败即停 / 工具异常写入 observation /
整体异常或空证据由调用方（pipeline）降级回七阶段管道。
时延：每步 1 次小 token 决策调用；use_agentic 默认关。
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

from config import get_settings, get_llm_extra_body
from app.retrieval.fusion import chunk_key

logger = logging.getLogger(__name__)

ACTIONS = ("search", "decompose", "grade", "finish")

DECISION_PROMPT = """你是一个检索智能体，通过调用工具为用户问题收集证据。

可用工具（严格按条件使用）：
- search(query)：用给定查询检索知识库。query 可与原问题不同，但**禁止与已执行查询重复或语义雷同**。
- decompose()：当问题需要**组合两个或以上事实/实体**才能回答时（如"A和B分别…""X的那个Y在哪里""先…还是…"），**优先使用**，拆成子问题分别检索后合并。
- grade()：评估当前证据是否足以回答问题；search 之后建议调用一次。
- finish()：已有 ≥1 篇相关证据时**立即结束**；observation 中**新增为 0** 说明检索已收敛，应 finish 或改用 decompose/grade，不要无目的重复 search；**最后一步无论证据如何都必须 finish**。

示例1（单事实查询）：
问题：范廷颂哪一年成为主教？
{{"thought": "单事实查询，直接检索", "action": "search", "args": {{"query": "范廷颂 任命 主教"}}}}
（观察：[新增5篇/累计5] search 返回 5 篇证据，含 1963 年任命信息）
{{"thought": "证据已含答案", "action": "finish", "args": {{}}}}

示例2（需组合两个事实）：
问题：范廷颂担任总主教的那个教区在哪里？
{{"thought": "需先定位教区名称再找位置，两个事实，优先分解", "action": "decompose", "args": {{}}}}
（观察：[新增8篇/累计8] decompose 拆出 2 个子问题，合并 8 篇证据）
{{"thought": "证据充分", "action": "finish", "args": {{}}}}

当前问题：{question}
进度：第 {step_num}/{max_steps} 步
已执行步骤：
{history}
当前证据摘要：
{evidence}

根据当前状态决定下一步，只输出 JSON（不要输出其他内容）：
{{"thought": "你的推理", "action": "search|decompose|grade|finish", "args": {{"query": "search 时的查询"}}}}"""


@dataclass
class AgentStep:
    """单步决策记录（观测与归因用）"""
    thought: str = ""
    action: str = ""
    args: dict = field(default_factory=dict)
    observation: str = ""


@dataclass
class AgentResult:
    """agent 检索结果：证据文档 + 决策轨迹"""
    documents: List[Document] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    stop_reason: str = ""            # agent_done / max_steps / decision_error / converged / no_evidence
    decomposed_subqueries: List[str] = field(default_factory=list)
    decomposition_chain: bool = False
    queries_used: List[str] = field(default_factory=list)
    crag_grade: str = ""


class AgenticRetriever:
    """ReAct 状态机检索器。pipeline 提供工具实现（召回/融合/重排/分解/评估）。"""

    def __init__(self, pipeline, llm=None, settings=None):
        self._pipeline = pipeline
        self._settings = settings or get_settings()
        self.llm = llm  # 可注入（测试/复用 chain.llm）；None 时按配置自建小预算客户端

    # ---- 主循环 ----

    def run(
        self,
        question: str,
        top_k: int | None = None,
        trace_id: str | None = None,
    ) -> AgentResult:
        """ReAct 循环：决策 → 执行 → 观察，直到 finish / 硬上限 / 决策失败。"""
        settings = self._settings
        top_k = top_k or settings.retrieval_top_k
        result = AgentResult()
        evidence: List[Document] = []
        evidence_ids: set = set()
        consecutive_empty_search = 0  # 连续零新增 search 计数（收敛护栏）
        finish_rejected = False       # 空手 finish 只驳回一次，防无限循环

        for step_num in range(1, settings.agentic_max_steps + 1):
            decision = self._decide(
                question, result.steps, evidence,
                step_num, settings.agentic_max_steps,
                grade=result.crag_grade,
                empty_streak=consecutive_empty_search,
            )
            if decision is None:
                result.stop_reason = "decision_error"
                break

            step = AgentStep(
                thought=decision.get("thought", ""),
                action=decision.get("action", ""),
                args=decision.get("args") or {},
            )
            result.steps.append(step)

            if step.action == "finish":
                # finish 门控：空手 finish 驳回一次给纠错机会（最后一步除外，已是最后机会）
                if (
                    not evidence
                    and not finish_rejected
                    and step_num < settings.agentic_max_steps
                ):
                    step.observation = "尚无证据，拒绝结束：请先 search 或 decompose 收集证据"
                    finish_rejected = True
                    continue
                step.observation = "agent 判定证据充分"
                result.stop_reason = "agent_done"
                break

            docs, observation = self._execute(step, question, top_k, evidence, result)
            step.observation = observation
            logger.info(f"F13 step {step_num}: {step.action} {step.args} -> {observation[:60]}")
            new_count = 0
            for d in docs:
                key = chunk_key(d)
                if key not in evidence_ids:
                    evidence_ids.add(key)
                    evidence.append(d)
                    new_count += 1
            # 新增量前缀：让 LLM 看到真实收敛信号（前缀位置不被 history 百字截断裁掉）
            if step.action in ("search", "decompose"):
                step.observation = (
                    f"[新增{new_count}篇/累计{len(evidence)}] " + step.observation
                )

            # 零 LLM 收敛护栏：连续两次 search 零新增证据 → 强制停止（复用 F2 收敛概念）
            if step.action == "search":
                consecutive_empty_search = consecutive_empty_search + 1 if new_count == 0 else 0
                if consecutive_empty_search >= 2:
                    step.observation += "（连续两次检索零新增证据，强制收敛停止）"
                    result.stop_reason = "converged"
                    break
            else:
                consecutive_empty_search = 0

        else:
            result.stop_reason = "max_steps"

        if not result.stop_reason:
            result.stop_reason = "max_steps"

        # 证据重排压缩到 top_k（复用管道重排，异常时直接截断）
        result.documents = self._final_rerank(question, evidence, top_k)
        # 空证据标 no_evidence，但不覆盖更具诊断价值的 decision_error / converged
        if not result.documents and result.stop_reason in ("agent_done", "max_steps"):
            result.stop_reason = "no_evidence"
        logger.info(
            f"F13 agentic 检索完成: steps={len(result.steps)} "
            f"stop={result.stop_reason} docs={len(result.documents)}"
        )
        return result

    # ---- 决策节点（LLM，JSON 输出，失败即停） ----

    def _decide(
        self,
        question: str,
        steps: List[AgentStep],
        evidence: List[Document],
        step_num: int,
        max_steps: int,
        grade: str = "",
        empty_streak: int = 0,
    ) -> Optional[dict]:
        """调 LLM 产出下一步决策。任何异常/解析失败返回 None（触发 decision_error 终止）。

        step_num/max_steps 使 LLM 感知步数预算；grade/empty_streak 使收敛信号可见。
        """
        try:
            llm = self._get_llm()
            prompt = DECISION_PROMPT.format(
                question=question,
                step_num=step_num,
                max_steps=max_steps,
                history=self._format_history(steps),
                evidence=self._format_evidence(
                    evidence, grade=grade, empty_streak=empty_streak
                ),
            )
            out = llm.invoke(prompt)
            content = getattr(out, "content", str(out)).strip()
            decision = self._parse_decision(content)
            if decision is None:
                logger.warning(f"F13 决策 JSON 解析失败: {content[:100]}")
                return None
            if decision.get("action") not in ACTIONS:
                logger.warning(f"F13 非法 action: {decision.get('action')}")
                return None
            return decision
        except Exception as e:
            logger.warning(f"F13 决策调用异常: {e}")
            return None

    @staticmethod
    def _parse_decision(content: str) -> Optional[dict]:
        """从 LLM 输出提取 JSON 决策（容忍 ```json 包裹与前后噪声）。"""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            decision = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return decision if isinstance(decision, dict) else None

    # ---- 工具执行节点 ----

    def _execute(
        self,
        step: AgentStep,
        question: str,
        top_k: int,
        evidence: List[Document],
        result: AgentResult,
    ) -> tuple[List[Document], str]:
        """分发到具体工具；工具异常转为 observation 文本（循环不中断）。"""
        try:
            if step.action == "search":
                return self._tool_search(step, question, top_k, result)
            if step.action == "decompose":
                return self._tool_decompose(question, top_k, result)
            if step.action == "grade":
                return self._tool_grade(question, evidence, result)
        except Exception as e:
            logger.warning(f"F13 工具 {step.action} 异常: {e}")
            return [], f"工具 {step.action} 执行失败: {e}"
        return [], f"未知工具: {step.action}"

    def _tool_search(self, step, question, top_k, result) -> tuple[List[Document], str]:
        query = (step.args.get("query") or "").strip() or question
        duplicate = query in result.queries_used
        result.queries_used.append(query)
        docs = self._pipeline.search(question, [query], top_k)
        snippet = docs[0].page_content[:80] if docs else ""
        observation = f"search('{query}') 返回 {len(docs)} 篇证据。首篇摘要: {snippet}"
        if duplicate:
            observation += "（警告：该查询已执行过，请改用 decompose/grade 或 finish）"
        return docs, observation

    def _tool_decompose(self, question, top_k, result) -> tuple[List[Document], str]:
        pipeline = self._pipeline
        if not pipeline.query_transformer:
            return [], "decompose 不可用（无 query_transformer）"
        decomposition = pipeline.query_transformer.decompose(question)
        subs = decomposition.sub_questions
        if len(subs) <= 1:
            return [], "分解结果为单一问题，无需分解检索"
        result.decomposed_subqueries = subs
        result.decomposition_chain = decomposition.chain
        docs = pipeline._decompose_retrieve(question, decomposition, top_k)
        return docs, f"decompose 拆出 {len(subs)} 个子问题（chain={decomposition.chain}），合并 {len(docs)} 篇证据"

    def _tool_grade(self, question, evidence, result) -> tuple[List[Document], str]:
        if not self._pipeline.crag_evaluator:
            return [], "grade 不可用（无 crag_evaluator）"
        grade, scores, reason = self._pipeline.evaluate(question, evidence)
        result.crag_grade = grade
        relevant = sum(1 for s in scores if s and s > 0)
        return [], f"grade 评估: {grade}（{reason}），相关 {relevant}/{len(evidence)} 篇"

    # ---- 收尾 ----

    def _final_rerank(self, question, evidence, top_k) -> List[Document]:
        """把累积证据重排压缩到 top_k；异常时直接截断（优雅降级）。"""
        if not evidence:
            return []
        try:
            return self._pipeline.rerank(question, evidence, top_k)
        except Exception as e:
            logger.warning(f"F13 证据收尾重排失败，直接截断: {e}")
            return evidence[:top_k]

    # ---- 辅助 ----

    @staticmethod
    def _format_history(steps: List[AgentStep]) -> str:
        if not steps:
            return "（尚无步骤）"
        return "\n".join(
            f"{i}. action={s.action} args={s.args} -> {s.observation[:100]}"
            for i, s in enumerate(steps, 1)
        )

    @staticmethod
    def _format_evidence(
        evidence: List[Document],
        max_docs: int = 3,
        grade: str = "",
        empty_streak: int = 0,
    ) -> str:
        if not evidence:
            return "（尚无证据）"
        lines = [f"- {d.page_content[:100]}" for d in evidence[:max_docs]]
        # 状态行：把已计算的收敛信号（grade/连续零新增）暴露给 LLM，零新增 LLM 调用
        status = [f"共 {len(evidence)} 篇"]
        if grade:
            status.append(f"grade={grade}")
        if empty_streak:
            status.append(f"连续 {empty_streak} 步零新增")
        lines.append("（" + "｜".join(status) + "）")
        return "\n".join(lines)

    def _get_llm(self):
        """决策 LLM：注入优先；否则按配置自建（小 token 预算 + 短超时）。"""
        if self.llm is not None and self.llm is not True:
            return self.llm
        from langchain_openai import ChatOpenAI
        s = self._settings
        self.llm = ChatOpenAI(
            model=s.openai_model, api_key=s.openai_api_key,
            base_url=s.openai_base_url, temperature=0,
            max_tokens=s.agentic_decision_max_tokens,
            request_timeout=15, max_retries=1,
            extra_body=get_llm_extra_body(),
        )
        return self.llm
