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
    # search 复合原语委托回三阶段 mock，各测试对 recall/fuse/rerank 的覆写继续生效
    p.search.side_effect = (
        lambda question, queries, top_k, **kw:
        p.rerank(question, p.fuse(p.recall(question, queries)), top_k)
    )
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
        '{"action": "finish", "args": {}}',   # 空手 finish → 驳回一次（v3 门控）
        '{"action": "finish", "args": {}}',   # 二次 finish → 接受
    ]), settings=_settings())
    result = agent.run("问题")
    assert "chroma down" in result.steps[0].observation
    assert "拒绝" in result.steps[1].observation
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


# ============ 收敛优化 v3：信号可见 + finish 门控（2026-07-26） ============

def test_prompt_includes_step_budget_and_rules():
    """决策 prompt 含步数预算（第 i/N 步）与最后一步必停/新增0即收敛规则。"""
    from app.retrieval.agent import DECISION_PROMPT
    assert "最后一步" in DECISION_PROMPT
    assert "新增为 0" in DECISION_PROMPT
    llm = _llm([
        '{"action": "search", "args": {"query": "q"}}',
        '{"action": "finish", "args": {}}',
    ])
    agent = AgenticRetriever(_pipeline(), llm=llm, settings=_settings(max_steps=4))
    agent.run("问题")
    prompts = [c.args[0] for c in llm.invoke.call_args_list]
    assert "第 1/4 步" in prompts[0]
    assert "第 2/4 步" in prompts[1]


def test_search_observation_shows_new_count_prefix():
    """search observation 前缀展示真实新增量与累计量（前缀位置不被百字截断裁掉）。"""
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "search", "args": {"query": "q1"}}',
        '{"action": "search", "args": {"query": "q2"}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert result.steps[0].observation.startswith("[新增1篇/累计1]")
    assert result.steps[1].observation.startswith("[新增0篇/累计1]")


def test_finish_rejected_without_evidence_then_accepted():
    """零证据首次 finish 被驳回一次；取到证据后 finish 正常接受。"""
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "finish", "args": {}}',          # 空手 finish → 驳回
        '{"action": "search", "args": {"query": "q"}}',
        '{"action": "finish", "args": {}}',          # 有证据 → 接受
    ]), settings=_settings())
    result = agent.run("问题")
    assert "拒绝" in result.steps[0].observation
    assert result.stop_reason == "agent_done"
    assert len(result.steps) == 3
    assert len(result.documents) == 1


def test_second_finish_without_evidence_accepted():
    """二次 finish 即使仍无证据也接受（防无限循环），空证据覆写为 no_evidence。"""
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "finish", "args": {}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert len(result.steps) == 2
    assert result.stop_reason == "no_evidence"  # agent_done → 空证据覆写


def test_finish_not_rejected_on_last_step():
    """最后一步的空手 finish 不驳回（已是最后机会），交由 no_evidence 覆写语义。"""
    agent = AgenticRetriever(_pipeline(), llm=_llm([
        '{"action": "finish", "args": {}}',
    ]), settings=_settings(max_steps=1))
    result = agent.run("问题")
    assert len(result.steps) == 1
    assert "拒绝" not in result.steps[0].observation
    assert result.stop_reason == "no_evidence"


def test_evidence_format_includes_grade_and_streak():
    """证据摘要附带 grade 与连续零新增步数（LLM 可见收敛信号）。"""
    out = AgenticRetriever._format_evidence(
        [Document(page_content="证据文本", metadata={})],
        grade="correct", empty_streak=1,
    )
    assert "grade=correct" in out
    assert "连续 1 步零新增" in out
    # 无 grade / 无 streak 时不显示噪声子句
    plain = AgenticRetriever._format_evidence([Document(page_content="x", metadata={})])
    assert "grade=" not in plain
    assert "零新增" not in plain


def test_grade_observation_includes_relevant_count():
    """grade observation 附相关文档数（scores 不再丢弃）。"""
    p = _pipeline()
    p.evaluate.return_value = ("ambiguous", [1, 1, 0], "部分相关")
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "search", "args": {"query": "q"}}',
        '{"action": "grade", "args": {}}',
        '{"action": "finish", "args": {}}',
    ]), settings=_settings())
    result = agent.run("问题")
    assert "相关 2/" in result.steps[1].observation


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


# ============ 硬门控（证据充分度，零 LLM） ============

def _gate_settings(max_steps=4):
    s = _settings(max_steps=max_steps)
    s.agentic_evidence_gate = True
    return s


def _scored_pipeline(score=2.0):
    """返回带真实 rerank_score 证据的 stub 管道（sigmoid(2.0)=0.88 → CRAG correct）。"""
    p = _pipeline()
    doc = Document(
        page_content="范廷颂1963年被任命为主教",
        metadata={"chunk_id": "c1", "rerank_score": score},
    )
    p.recall.return_value = {"dense": [doc], "sparse": []}
    p.fuse.return_value = [doc]
    p.rerank.return_value = [doc]
    p.evaluate.return_value = ("correct", [0], "rerank_sigmoid=0.88")
    return p


def _never_finish_llm(n=4):
    """每步都返回新查询的 search，从不 finish——模拟 finish 率不达标的 LLM。"""
    return _llm([
        '{"action": "search", "args": {"query": "查询%d"}}' % i for i in range(n)
    ])


def _fresh_docs_pipeline(score=None, n=4):
    """每次 search 返回一篇全新文档（不同 chunk_id，避免 dedup 触发收敛护栏）。"""
    p = _pipeline()
    docs = [
        Document(
            page_content=f"证据{i}",
            metadata={"chunk_id": f"c{i}", **({"rerank_score": score} if score is not None else {})},
        )
        for i in range(n)
    ]
    p.recall.side_effect = [{"dense": [d], "sparse": []} for d in docs]
    p.fuse.side_effect = [[d] for d in docs]
    p.rerank.side_effect = [[d] for d in docs]
    return p


def test_hard_gate_forces_finish_when_evidence_correct():
    """LLM 从不 finish，但证据 CRAG=correct → 硬门控第 1 步即强制结束。"""
    p = _scored_pipeline()
    agent = AgenticRetriever(p, llm=_never_finish_llm(), settings=_gate_settings())
    result = agent.run("范廷颂哪一年成为主教？")
    assert result.stop_reason == "evidence_sufficient"
    assert len(result.steps) == 1                    # 无需耗满 max_steps
    assert result.crag_grade == "correct"
    assert "硬门控" in result.steps[0].observation
    assert len(result.documents) == 1


def test_hard_gate_not_fire_on_ambiguous():
    """CRAG=ambiguous（证据不足）→ 门控不触发，循环照常走到 max_steps。"""
    p = _fresh_docs_pipeline(score=2.0)
    p.evaluate.return_value = ("ambiguous", [0], "rerank_sigmoid=0.40")
    agent = AgenticRetriever(p, llm=_never_finish_llm(), settings=_gate_settings())
    result = agent.run("问题")
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 4
    assert result.crag_grade == "ambiguous"


def test_hard_gate_skipped_without_rerank_scores():
    """证据无真实 rerank 分数（如 decompose 纯 RRF 结果）→ 门控不启用，不误调 evaluate。"""
    p = _fresh_docs_pipeline(score=None)
    agent = AgenticRetriever(p, llm=_never_finish_llm(), settings=_gate_settings())
    result = agent.run("问题")
    p.evaluate.assert_not_called()
    assert result.stop_reason == "max_steps"


def test_hard_gate_disabled_by_switch():
    """agentic_evidence_gate=False → 完全旁路，行为回到纯 LLM 决策。"""
    p = _fresh_docs_pipeline(score=2.0)
    s = _gate_settings()
    s.agentic_evidence_gate = False
    agent = AgenticRetriever(p, llm=_never_finish_llm(), settings=s)
    result = agent.run("问题")
    p.evaluate.assert_not_called()
    assert result.stop_reason == "max_steps"


def test_hard_gate_fires_after_decompose():
    """decompose 返回带分数证据且 CRAG=correct → 门控同样接管停止权。"""
    p = _scored_pipeline()
    decomp = SimpleNamespace(sub_questions=["子问题1", "子问题2"], chain=False)
    p.query_transformer.decompose.return_value = decomp
    scored_docs = [
        Document(page_content="证据A", metadata={"chunk_id": "a", "rerank_score": 2.0}),
        Document(page_content="证据B", metadata={"chunk_id": "b", "rerank_score": 1.5}),
    ]
    p._decompose_retrieve.return_value = scored_docs
    agent = AgenticRetriever(p, llm=_llm([
        '{"action": "decompose", "args": {}}',
        '{"action": "search", "args": {"query": "多余的一步"}}',
    ]), settings=_gate_settings())
    result = agent.run("多跳问题")
    assert result.stop_reason == "evidence_sufficient"
    assert len(result.steps) == 1                    # 第 2 步 search 不会执行
    assert result.decomposed_subqueries == ["子问题1", "子问题2"]


def test_hard_gate_evaluates_scored_docs_in_descending_order():
    """累积证据乱序到达 → 门控按 rerank_score 降序送入 evaluate（CRAG 取 top1 判级）。"""
    p = _pipeline()
    low = Document(page_content="弱证据", metadata={"chunk_id": "lo", "rerank_score": -1.0})
    high = Document(page_content="强证据", metadata={"chunk_id": "hi", "rerank_score": 3.0})
    p.recall.side_effect = [
        {"dense": [low], "sparse": []},
        {"dense": [high], "sparse": []},
    ]
    p.fuse.side_effect = [[low], [high]]
    p.rerank.side_effect = [[low], [high]]
    # 第 1 步证据不足（不触发门控），第 2 步充分（触发）
    p.evaluate.side_effect = [
        ("ambiguous", [0], "rerank_sigmoid=0.27"),
        ("correct", [0, 1], "rerank_sigmoid=0.95"),
    ]
    agent = AgenticRetriever(p, llm=_never_finish_llm(), settings=_gate_settings())
    result = agent.run("问题")
    assert result.stop_reason == "evidence_sufficient"
    # 第 2 步触发门控：送入 evaluate 的首篇应为最高分文档（降序）
    docs_arg = p.evaluate.call_args[0][1]
    assert docs_arg[0].metadata["chunk_id"] == "hi"
