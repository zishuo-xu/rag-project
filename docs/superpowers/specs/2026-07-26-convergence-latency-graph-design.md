# F13 收敛优化 + 延迟治理 + Graph RAG 升级 · 设计文档（2026-07-26）

> 承接 [2026-07-26 Prompt/路由优化报告](../reports/2026-07-26-prompt-router-optimization-report.md) 留下的三个诚实记录开放项。执行顺序 ①→②→④ 串行，每阶段独立 TDD + 评估 + 提交。

## 探索发现（代码审计结论，驱动设计）

### ① F13 finish 率 47% 的根因：信号不可见

- `new_count`（去重后真实新增量）在 `agent.py:129-135` 算出但**从不进 observation**——LLM 只看到 `len(docs)`（原始返回数），无法感知检索已收敛；
- `crag_grade`、`consecutive_empty_search` 均已计算但 `_format_evidence`（:283-289）不展示；
- **步数预算完全不可见**：`step_num`/`agentic_max_steps` 从不进 prompt，LLM 不知道只剩最后一步；
- finish 分支（:121-124）**零门控**：空证据 finish 被事后 `no_evidence` 覆写掩盖，无纠错机会；
- `_tool_grade`（:253-258）丢弃 `scores` 列表（pipeline.evaluate 已返回每文档相关性）。

**结论**：全部可用**零新增 LLM 调用**的信号注入修复，符合模块时延预算原则。

### ② 42.6s 的真相：单点离群 + 结构浪费

- `data/eval_e2e_multihop_full_v2.json` 逐样本：**14/15 样本 7.1-15.2s（均值 10.9s），1 个样本 486s**（44.5x），单点把均值拉到 42.6s；
- 离群机制：无全局时延预算，generation `request_timeout=60, max_retries=2`（chain.py:137-146）可叠加 180s，各阶段超时各自为政；
- 结构浪费：分解路径（14/15 触发）仍投机执行 `multi_query` transform 后**丢弃结果**（pipeline.py:457-467）；chain 模式串行 refine（每跳界 1 次 LLM，hops=3 → 最多 2 次）；F3 忠实度均值 0.6 < 阈值 0.7 → 重生率 40%（每次 +2 串行调用）；
- 最坏路径 LLM 调用数：14 次（chain 3 跳 + F2 两轮 + 重生）。

**结论**：主攻「消灭离群尾」（Deadline 熔断 + 超时收紧），辅以结构优化（router 前置跳过无用 transform、hops 3→2、max_tokens 封顶）。典型路径已 11s，无需大改。

### ④ 图的真实状态：比文档描述更弱

- 生产图（data/knowledge_graph.json，675 节点/2354 边）**全部关系仅「共现/上下文关联」**——由零 LLM 的 `build_fast`（jieba TF-IDF）构建，LLM 三元组路径（EXTRACTION_PROMPT）写了从未用于当前产物；
- 无实体/边类型 schema，边仅 source 文件名**无 chunk 溯源**；
- **分解路径完全跳过 graph 通道**（`_retrieve_subquery` 只用 dense+sparse）——14/15 多跳样本走分解，图通道对多跳实际零贡献；
- 图 Document 无 `chunk_id`，无法参与 F7 引用与 RRF 内容去重；
- 零图单测（仅 pipeline 测试中 MagicMock）；
- corpus 规模 137 块/21 文档，NetworkX+JSON 存储完全够用，瓶颈在**抽取质量与类型化**。

**结论**：Option A——类型化三元组（LLM 路径升级）+ chunk 溯源 + 接入分解路径。不做社区摘要（137 块 ROI 低）。

---

## 阶段 1 · ① F13 收敛优化（`app/retrieval/agent.py`）

全部零新增 LLM 调用：

| # | 改动 | 位置 |
|---|---|---|
| 1 | `_decide` 增参 `step_num, max_steps`；DECISION_PROMPT 加 `进度：第 i/N 步` + 规则「最后一步必须 finish」「新增为 0 即收敛，应 finish 或改 decompose/grade」 | :33-60, :166-189 |
| 2 | search observation **前缀** `[新增N篇/累计M] `（前缀不被 `_format_history` 百字截断裁掉） | run() :135 后 |
| 3 | `_format_evidence` 尾部追加 `（共 N 篇｜grade=X｜连续 K 步零新增）` | :283-289 |
| 4 | finish 门控：零证据首次 finish 驳回一次（observation 说明原因，`finish_rejected=True`，continue）；有证据或二次 finish 接受 | :121-124 |
| 5 | `_tool_grade` observation 附相关文档数（scores>0 计数，不再丢弃） | :253-258 |

**验收**：`--only F13` 复跑两次 → finish 率 ≥50%（现 47%）、avg_steps ≤3.5（现 3.53-3.6）、F1 ≥0.29（现 0.298/0.291）；未达标如实记录。

## 阶段 2 · ② 延迟治理（目标 full 均值 <20s、F1 ≥0.285）

| # | 改动 | 位置 |
|---|---|---|
| 1 | 轻量 `Deadline`（`app/core/deadline.py`：budget_ms/exceeded()/remaining_ms()，注入时钟可测）+ config `latency_budget_ms=25000`；pipeline.run() 起始建 Deadline 存入 RetrievalResult；F2 迭代入口（pipeline.py:578-595）与 F3 重生循环（chain.py:320-349）前检查，超预算跳过记 `budget_skipped` | 新文件 + pipeline/chain |
| 2 | 超时收紧：generation 60s×2 → 30s×1；query_transform 30s×2 → 20s×1 | chain.py:137-146, query_transform.py:95-103 |
| 3 | max_tokens 封顶：config `answer_max_tokens=1024` 传 generation；decompose/refine/multi_query/hyde 统一 512 | config.py, query_transform.py |
| 4 | router 前置：零 LLM 路由提到 gate∥transform 之前；multi_hop+分解时跳过 multi_query（gate 单跑），非 multi_hop 不变 | pipeline.py:446-500 |
| 5 | `decomposition_max_hops` 默认 3→2 | config.py:86 |

**验收**：full 多跳重跑 → 均值 <20s、无 >120s 离群、F1 ≥0.285、记录 budget_skipped 触发率与 num_failed。

## 阶段 3 · ④ Graph RAG 升级 Option A

| # | 改动 | 位置 |
|---|---|---|
| 1 | EXTRACTION_PROMPT 升级 JSON：`{"triples":[{head,head_type,relation,tail,tail_type}]}`，类型集 person/work/place/org/position/event/other，CMRC 域 few-shot；`_parse_triples` JSON 解析 + 旧正则兜底；边属性增 head_type/tail_type/chunk_id | graph_extractor.py:31-46, :191-227, :285-415 |
| 2 | ENTITY_EXTRACT_PROMPT 去技术域措辞改通用 | graph_retriever.py:29-36 |
| 3 | graph Document metadata 带首个来源 chunk_id（接通 F7/RRF 去重） | graph_retriever.py:236-286 |
| 4 | `_retrieve_subquery` channels 加 `"graph"` | pipeline.py:340-347 |
| 5 | 重建生产图：先抽 3-5 块试跑验三元组质量，再 `/api/graph/build` 全量；build_fast 保留为零 LLM 兜底 | 在线步骤 |
| 6 | 新建 tests/test_graph.py ~8 条 | 新文件 |

**验收**：重建后 full 多跳重跑 → 对比 F1 0.289 / hit 0.80 / coverage 0.68，trace graph_hits>0 的分解样本占比；F1 不升也如实记录（小 corpus 图通道是召回多样性，收益可能有限）。

## 风险与诚实边界

1. ① 门控可能让 agent 携弱证据提前结束 → F1 微降风险，以评估为准；
2. ② `answer_max_tokens=1024` 截断长答案风险 → 评估验证，必要时上调；
3. ② 超时收紧在 API 抖动时失败率可能上升 → 观察 num_failed；
4. ④ 中文传记体三元组抽取质量未知 → 小批试跑再全量；
5. ④ 图重建是在线 LLM 步骤（非离线测试），耗时数分钟。

## 复现命令

```bash
uv run pytest tests/ -q
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --mode full --only F13 --output data/eval_e2e_multihop_f13_v3a.json
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --mode full --output data/eval_e2e_multihop_full_v3.json
```
