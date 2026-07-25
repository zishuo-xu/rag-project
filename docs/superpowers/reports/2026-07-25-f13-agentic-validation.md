# F13 Agentic RAG 验证报告

日期：2026-07-25
范围：F13 ReAct 状态机自主检索（手写，概念对齐 LangGraph，零新依赖）
设计文档：`docs/superpowers/specs/2026-07-25-agentic-rag-design.md`

## 1. 交付物

| 项 | 内容 |
|---|---|
| 核心模块 | `app/retrieval/agent.py`（~250 行）：ReAct 循环 + 3 工具（search/decompose/grade）+ 决策 JSON 解析 + 护栏 |
| 接线 | `RetrievalPipeline.run` 顶部 agentic 分支（异常/空证据降级七阶段）；`chain.py` 构建注入；`config.py` 3 个开关 |
| 观测 | `RetrievalResult.agent_steps/agent_stop_reason`；harness 汇总 `avg_steps/stop_reasons/action_dist` |
| 测试 | `tests/test_agentic.py` 19 个（决策解析/主循环/工具/降级链），**全量 292 passed** |

## 2. 实测（15 条多跳集，DeepSeek，两次复跑）

| 指标 | baseline | full | --only F6 | **--only F13** |
|---|---|---|---|---|
| 端到端 F1 | 0.255 | 0.276 | 0.260 | **0.286 / 0.283** |
| 检索命中率 | 0.80 | 0.80 | 0.80 | 0.80 |
| answer_hit | 0.067 | 0.067 | 0.067 | 0.067 |
| 平均延迟 | 7.8s | 10.3s | 6.6s | 9.3s / 9.6s |
| 分解触发 | 0 | 6.7% | 0 | decompose 决策 3/52 |

**F13 是四种模式里 F1 最高**（+3pp vs baseline），且延迟低于 full（少 F2/F3 的 LLM 调用）。

## 3. 诚实发现（面试核心素材）

1. **过度检索倾向**：10/15 样本打满 max_steps=4，agent 不善于主动 finish（仅 5 次）。
2. **工具选择 search 主导**：52 次决策中 search 43 次、grade 1 次、decompose 3 次；
   且**跨运行不稳定**（首轮复跑 decompose 0 次）——LLM 编排器的工具选择是随机的，
   需要 prompt 工程（few-shot 示例/工具描述/决策约束）引导，这是下一步优化点。
3. **与路由解耦生效**：F13 下 decompose 不再依赖 F4 路由（--only F13 时路由关闭，
   agent 仍自主触发分解 3 次），验证了设计动机；但触发率仍低，瓶颈转移到 LLM 的工具偏好。

## 4. 口径与局限

- 样本量 15，±3pp 为方向性结论；两次复跑 F1 差 0.003，非确定性来自 LLM 决策与生成。
- agentic 分支不做门控（agent 总是检索），闲聊场景未在本评估覆盖。
- 每步决策 1 次 LLM 调用（256 tokens 上限），15 样本共 52 次决策调用，成本可控但非零。

## 5. 复现

```bash
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --only F13 \
  --output data/eval_e2e_multihop_f13.json
uv run pytest tests/test_agentic.py -q   # 19 个离线测试
```
