"""F13 Agentic RAG / ReAct 状态机测试（离线，LLM mock）

覆盖：
1. 决策解析：合法 JSON / markdown 包裹 / 噪声文本 / 非法 action
2. 主循环：search→finish 正常路径 / max_steps 硬上限 / 决策失败即停
3. 工具：search 用 args.query / decompose 触发分解检索 / grade 写入 crag_grade /
   工具异常转 observation 不中断循环
4. 收尾：证据重排到 top_k / 空证据 → no_evidence
5. 管道接线：use_agentic 开且有证据 → 走 agent 结果；agent 异常/空证据 → 降级七阶段
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieval.agent import AgenticRetriever, AgentResult, AgentStep
from app.retrieval.pipeline import RetrievalPipeline


def _settings(max_steps=4):
    s = MagicMock()
    s.use_agentic = True
    s.agentic_max_steps = max_steps
    s.agentic_decision_max_tokens = 256
    s.retrieval_top_k = 5
    s.rerank_top_n = 20
    s.recall_max_workers = 4
    s.use_summary_recall = False
    s.use_crag_gate = False
    s.use_autocut = False
    s.autocut_min_docs = 2
    s.use_query_router = False
    s.use_decomposition = False
    s.use_iterative_retrieval = False
    s.use_contextual_chunks = False
    return s


def _llm(responses):
    """按顺序返回决策 JSON 的 mock LLM。"""
    llm = MagicMock()
    llm.invoke.side_effect = [SimpleNamespace(content=r) for r in responses]
    return llm


def _pipeline():
    """工具方法全部为 mock 的 stub 管道。"""
    p = MagicMock()
    doc = Document(page_content="范廷颂1994年任河内总教区总主教", metadata={"chunk_id": "c1"})
    p.recall.return_value = {"dense": [doc], "sparse": []}
    p.fuse.return_value = [doc]
    p.rerank.return_value = [doc]
    p.evaluate.return_value = ("correct", [0], "证据充分")
    return p


# ============ 决策解析 ============

def test_parse_decision_plain_json():
    d = AgenticRetriever._parse_decision('{"thought": "t", "action": "search", "args": {"query": "q"}}')
    assert d["action"] == "search"


def test_parse_decision_markdown_wrapped():
    d = AgenticRetriever._parse_decision('思考中...\n```json\n{"action": "finish", "args": {}}\n```')
    assert d["action"] == "finish"


def test_parse_decision_garbage_returns_none():
    assert AgenticRetriever._parse_decision("我不知道") is None
    assert AgenticRetriever._parse_decision('{"action": ') is None


# ============ 主循环 ============

def test_loop_search_then_finish():
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"thought": "先检索", "action": "search", "args": {"query": "范廷颂 总主教"}}',
        '{"thought": "证据够了", "action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("范廷颂担任总主教的教区在哪里？")
    assert result.stop_reason == "agent_done"
    assert len(result.steps) == 2
    assert result.steps[0].action == "search"
    assert result.queries_used == ["范廷颂 总主教"]
    assert len(result.documents) == 1


def test_loop_max_steps_cap():
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "search", "args": {"query": "q"}}',
        '{"action": "search", "args": {"query": "q"}}',
    ]), settings=_settings(max_steps=2))
    result = agent.run("问题")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 2


def test_loop_decision_error_stops():
    agent = AgenticRetriever(_pipeline(), llm=_llm(["非 JSON 输出"]), settings=_settings())
    result = agent.run("问题")
    assert result.stop_reason == "decision_error"  # 不被 no_evidence 覆盖
    assert result.documents == []


def test_loop_invalid_action_stops():
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "hack", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert result.stop_reason in ("decision_error", "no_evidence")
    assert len(result.steps) == 0  # 非法 action 不入步


def test_llm_exception_stops_loop():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("api down")
    agent = AgenticRetriever(_pipeline(), llm=llm, settings=_settings())
    result = agent.run("问题")
    assert result.stop_reason in ("decision_error", "no_evidence")


# ============ 工具 ============

def test_search_tool_uses_args_query():
    p = _pipeline()
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "search", "args": {"query": "改写后的查询"}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    agent.run("原问题")
    # recall 收到的是 (原问题, [改写查询])
    args, _ = p.recall.call_args
    assert args[0] == "原问题"
    assert args[1] == ["改写后的查询"]


def test_search_tool_empty_query_falls_back_to_question():
    p = _pipeline()
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "search", "args": {}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("原问题")
    assert result.queries_used == ["原问题"]


def test_decompose_tool_triggers_subquery_retrieval():
    p = _pipeline()
    decomp = SimpleNamespace(sub_questions=["子问题1", "子问题2"], chain=False)
    p.query_transformer.decompose.return_value = decomp
    p._decompose_retrieve.return_value = [
        Document(page_content="子问题证据", metadata={"chunk_id": "c2"})
    ]
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "decompose", "args": {}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("多跳问题")
    p._decompose_retrieve.assert_called_once()
    assert result.decomposed_subqueries == ["子问题1", "子问题2"]
    assert "拆出 2 个子问题" in result.steps[0].observation


def test_decompose_tool_unavailable_without_transformer():
    p = _pipeline()
    p.query_transformer = None
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "decompose", "args": {}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert "不可用" in result.steps[0].observation


def test_grade_tool_records_crag_grade():
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "search", "args": {"query": "q"}}',
        '{"action": "grade", "args": {}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert result.crag_grade == "correct"
    assert "correct" in result.steps[1].observation


def test_tool_exception_becomes_observation():
    p = _pipeline()
    p.recall.side_effect = RuntimeError("chroma down")
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "search", "args": {"query": "q"}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert "chroma down" in result.steps[0].observation
    assert result.stop_reason == "no_evidence"  # 工具失败无证据


def test_evidence_dedup_across_steps():
    p = _pipeline()  # 每次 search 返回同一 chunk_id 文档
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "search", "args": {"query": "q1"}}',
        '{"action": "search", "args": {"query": "q2"}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert len(result.documents) == 1  # 按 chunk_id 去重


# ============ 收敛护栏与重复查询警告（F13 prompt 优化） ============

def test_converged_stop_after_two_empty_searches():
    """连续两次 search 零新增证据 → 强制收敛停止，不再消耗 max_steps。"""
    p = _pipeline()
    p.recall.return_value = {"dense": [], "sparse": []}
    p.fuse.return_value = []
    p.rerank.return_value = []  # 所有 search 都返回空
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "search", "args": {"query": "q1"}}',
        '{"action": "search", "args": {"query": "q2"}}',
        '{"action": "search", "args": {"query": "q3"}}',  # 不应执行到
        '{"action": "search", "args": {"query": "q4"}}',
    ]), settings=_settings(max_steps=4))
    result = agent.run("问题")
    assert result.stop_reason == "converged"
    assert len(result.steps) == 2  # 两次空 search 即停


def test_no_converged_stop_when_new_evidence_arrives():
    """search 有新增证据则重置收敛计数，继续循环。"""
    p = _pipeline()  # 每次返回同一篇 → 第一次有新增，第二次起零新增
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "search", "args": {"query": "q1"}}',   # +1 新证据
        '{"action": "search", "args": {"query": "q2"}}',   # 0 新增（重复 chunk）
        '{"action": "search", "args": {"query": "q3"}}',   # 0 新增 → converged
        '{"action": "search", "args": {"query": "q4"}}',   # 不应执行
    ]), settings=_settings(max_steps=4))
    result = agent.run("问题")
    assert result.stop_reason == "converged"
    assert len(result.steps) == 3


def test_duplicate_query_warns_in_observation():
    """重复语义的查询（与已执行完全相同）在 observation 中被警告，引导 finish。"""
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "search", "args": {"query": "相同查询"}}',
        '{"action": "search", "args": {"query": "相同查询"}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert "已执行过" in result.steps[1].observation


def test_prompt_has_fewshot_and_decompose_condition():
    """prompt v2：含 few-shot 示例与 decompose 硬条件、finish 引导。"""
    from app.retrieval.agent import DECISION_PROMPT
    assert "示例" in DECISION_PROMPT
    assert "两个或以上事实" in DECISION_PROMPT
    assert "finish" in DECISION_PROMPT


# ============ 管道接线（降级链） ============

def _real_pipeline(settings, **kwargs):
    return RetrievalPipeline(
        MagicMock(), MagicMock(), MagicMock(),
        settings=settings, **kwargs,
    )


def test_pipeline_uses_agent_result_when_available():
    settings = _settings()
    doc = Document(page_content="证据", metadata={"chunk_id": "c1"})
    agentic = MagicMock()
    agentic.run.return_value = AgentResult(
        documents=[doc], steps=[AgentStep(action="search")],
        stop_reason="agent_done", queries_used=["q"],
    )
    pipeline = _real_pipeline(settings, agentic=agentic)
    result = pipeline.run("问题")
    assert result.documents == [doc]
    assert result.agent_stop_reason == "agent_done"
    assert result.agent_steps[0]["action"] == "search"


def test_pipeline_falls_back_when_agent_raises():
    settings = _settings()
    agentic = MagicMock()
    agentic.run.side_effect = RuntimeError("agent boom")
    pipeline = _real_pipeline(settings, agentic=agentic)
    pipeline.indexer.hierarchical_search.return_value = []
    pipeline.dense_retriever.retrieve.return_value = []
    pipeline.sparse_retriever.retrieve.return_value = []
    result = pipeline.run("问题")  # 不抛出，走完整七阶段（空结果）
    assert result.agent_stop_reason == ""  # 未走 agent 分支
    assert result.documents == []


def test_pipeline_falls_back_when_agent_no_evidence():
    settings = _settings()
    agentic = MagicMock()
    agentic.run.return_value = AgentResult(documents=[], stop_reason="no_evidence")
    pipeline = _real_pipeline(settings, agentic=agentic)
    pipeline.indexer.hierarchical_search.return_value = []
    pipeline.dense_retriever.retrieve.return_value = []
    pipeline.sparse_retriever.retrieve.return_value = []
    result = pipeline.run("问题")
    assert result.agent_stop_reason == ""


def test_pipeline_skips_agent_when_flag_off():
    settings = _settings()
    settings.use_agentic = False
    agentic = MagicMock()
    pipeline = _real_pipeline(settings, agentic=agentic)
    pipeline.indexer.hierarchical_search.return_value = []
    pipeline.dense_retriever.retrieve.return_value = []
    pipeline.sparse_retriever.retrieve.return_value = []
    pipeline.run("问题")
    agentic.run.assert_not_called()
