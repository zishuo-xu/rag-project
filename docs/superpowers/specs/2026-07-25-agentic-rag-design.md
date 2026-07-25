# F13 Agentic RAG 设计：ReAct 状态机自主检索

日期：2026-07-25
状态：已实施（用户批准方案 A 后落地）

## 背景

RAG 1.0–3.0 的检索是**固定七阶段管道**：所有查询走同一套 gate→transform→recall→fuse→rerank→evaluate→remediate，策略差异仅靠 F4 规则路由做参数微调。上一轮评估闭环同时暴露：F6b 分解触发依赖 F4 路由判定 multi_hop（15 条多跳仅判中 1 条），固定管道的"编排智能"已到瓶颈。

Agentic RAG 把编排决策从固定代码交给 LLM：**逐步决定调哪个工具、用什么查询、何时停止**。

## 关键决策（对话评审记录）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 实现框架 | 手写状态机（概念对齐 LangGraph） | 零新依赖、全离线 mock 可测、符合项目"自实现"人设；面试双叙事 |
| 与现有管道关系 | 并存开关 + 降级 | `use_agentic` 默认关；agent 异常/空证据降级回七阶段管道，回归安全 |
| 工具粒度 | 混合粗粒度 3 工具 | 决策空间适中、步数可控；细粒度全工具 LLM 决策不稳、时延膨胀 |
| 循环架构 | ReAct 单循环 | thought→action→observation 逐步调整；优于 Plan-and-Execute（失去观察适应）与 LLM 路由器（非真 agent） |

## 架构

```
┌─ plan_step（LLM 决策，JSON: {thought, action, args}）←┐
│        ↓                                              │
│  execute_tool ── search(query)  → recall+fuse+rerank  │
│              ── decompose()     → F6b 分解+合并       │
│              ── grade()         → CRAG 相关性分级     │
│        ↓                                              │
│  observe（写入 state.steps）──────────────────────────┘
        ↓ finish / max_steps / decision_error
  final_rerank（证据压缩到 top_k）→ RetrievalResult
```

- **State**：`AgentResult{documents, steps[], stop_reason, decomposed_subqueries, queries_used, crag_grade}`
- **工具实现全部复用管道阶段**（`pipeline.recall/fuse/rerank/_decompose_retrieve/evaluate`），agent 只做编排
- **一致性卖点**：agent 自主决定 decompose → 不再依赖 F4 路由，消解"分解触发依赖路由"缺陷

## 护栏与降级链

1. `agentic_max_steps=4` 硬上限（stop_reason=max_steps）
2. 决策 JSON 解析失败/非法 action/LLM 异常 → decision_error 即停
3. 工具异常 → 写入 observation，循环不中断（LLM 可看到失败并调整）
4. 空证据 → no_evidence（不覆盖 decision_error）
5. 整体异常或空证据 → **pipeline 降级回固定七阶段**
6. 决策调用小预算：`agentic_decision_max_tokens=256`、15s 超时、max_retries=1

## 观测

- `RetrievalResult.agent_steps`（thought/action/args/observation 全轨迹）+ `agent_stop_reason`
- e2e harness 汇总 `agentic.avg_steps / stop_reasons`；F13 仅 `--only F13` 显式评估，不随 baseline/full 批量切换

## 非目标（YAGNI）

- 不引入 LangGraph 依赖（概念对齐即可）
- 不做工具结果流式、不做多 agent 协作
- agentic 分支不做门控（agent 总是检索；闲聊直接回答由 chain 层既有路径处理）
