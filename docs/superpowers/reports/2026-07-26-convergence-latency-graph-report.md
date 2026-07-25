# F13 收敛优化 + 延迟治理 + Graph RAG 升级 · 验证报告（2026-07-26）

> 承接 [Prompt/路由优化报告](./2026-07-26-prompt-router-optimization-report.md) 留下的三个诚实开放项，
> 按 ①→②→④ 串行推进，每阶段独立 TDD + 评估 + 提交。设计文档：
> [specs/2026-07-26-convergence-latency-graph-design.md](../specs/2026-07-26-convergence-latency-graph-design.md)。

## 阶段 1 · F13 收敛优化（commit b507e58）

### 根因与改动（`app/retrieval/agent.py`，零新增 LLM 调用）

上轮 finish 率 47% < 50% 的根因是**信号不可见**：`new_count` 算出未展示、`crag_grade` 不进 prompt、
步数预算从不进 prompt、finish 分支无任何门控。本轮四 lever：

1. **步数预算可见**：DECISION_PROMPT v3 加「第 i/N 步」+ 规则「最后一步必须 finish」「新增为 0 即收敛」。
2. **新增量可见**：observation 前缀 `[新增N篇/累计M]`（前缀位置不被百字截断裁掉）。
3. **证据状态可见**：`_format_evidence` 尾部 `（共N篇｜grade=X｜连续K步零新增）`。
4. **finish 门控**：零证据首次 finish 驳回一次（末步除外），二次或有证据即接受。
5. `_tool_grade` observation 附相关文档数（scores 不再丢弃）。

### 评估（--only F13，两次复跑，诚实记录）

| 指标 | v2 基线 | v3a | v3b | 目标 |
|---|---|---|---|---|
| avg F1 | 0.291–0.298 | 0.290 | 0.250 | ≥0.29 |
| avg steps | — | 3.47 | 3.70 | ≤3.5 |
| 主动 finish 率 | 47% | 40%（6/15） | 33%（5/15） | ≥50% |
| 平均延迟 | 12.4–12.8s | **9.6s** | 9.8s | — |

**结论（未达标如实记录）**：finish 率 40%/33% **未达 ≥50% 目标**，F1 方差增大（0.250–0.290）。
信号可见化降低了延迟（↓23%，agent 更早收敛/更少无效 search）但**无法强制 LLM 早停**——
finish 决策仍由模型自主。保留代码：正确性门控（零证据驳回）+ 延迟收益实在，finish 率记为开放项，
后续需更强机制（如证据充分度阈值硬门控）而非 prompt 信号。

## 阶段 2 · 延迟治理（commit 427ffa4）

### 根因与改动

full 模式 42.6s 均值实为**单点 486s 离群拖拽**（14/15 样本均值 10.9s）。治理主线：消灭离群尾 + 结构浪费。

1. **Deadline 时延预算**（`app/retrieval/deadline.py` 新增）：查询级 `latency_budget_ms=25000`，
   F2 迭代入口与 F3 重生循环超预算跳过并记 `budget_skipped`；clock 可注入（离线可测）。
2. **超时/重试收紧**：generation 60s×2→30s×1；query_transform 30s×2→20s×1。
3. **max_tokens 封顶**：`answer_max_tokens=1024`；改写 LLM 512（原均无界）。
4. **router 前置短路**：零 LLM 的 router 提到 gate∥transform 之前；multi_hop+分解时跳过
   multi_query（分解路径本就丢弃其结果），分解失败延迟补跑不丢召回。
5. `decomposition_max_hops` 3→2（每跳省 1 次串行 refine）。

### 评估（full 多跳 15 样本）

| 指标 | 基线 | 治理后 | 目标 |
|---|---|---|---|
| 平均延迟 | 42.6s | **9.6s（↓77%）** | <20s ✅ |
| p50 / p90 / max | —/—/486s | 8.3 / 15.1 / **16.9s** | 无 >120s 离群 ✅ |
| avg F1 | 0.289 | **0.291** | ≥0.285 ✅ |
| hit / coverage | 0.80 / 0.68 | 0.80 / 0.676 | 不降 ✅ |
| num_failed | — | 0 | 超时收紧未增失败 ✅ |
| budget_skipped 触发 | — | 0/15 | — |

**结论**：全部达标。budget_skipped 0/15 是设计意图——25s 预算仅兜底离群尾，正常路径（max 16.9s）不受限。
42.6s→9.6s 的降幅主要来自离群尾消灭（超时收紧杜绝 180s 级叠加重试）+ router 前置省去分解路径无用改写。

## 阶段 3 · Graph RAG 升级 Option A（待评估填充）

### 改动

1. **类型化三元组**（`app/ingestion/graph_extractor.py`）：EXTRACTION_PROMPT 升级 JSON 输出
   `{"triples":[{head,head_type,relation,tail,tail_type}]}`，类型集 person/work/place/org/position/event/other
   + 传记体 few-shot；`_parse_triples` JSON 解析（容忍代码块）+ 旧版逐行兜底；边属性增
   head_type/tail_type/**chunk_id 溯源**（同步+异步构建路径）。
2. **查询侧**（`app/retrieval/graph_retriever.py`）：ENTITY_EXTRACT_PROMPT 去技术域措辞改通用；
   图检索文档带 `graph:` 前缀 chunk_id + `source_chunk_ids`（溯源但不与真实分块在 RRF 同 key 互覆盖）。
3. **接入分解路径**（`app/retrieval/pipeline.py`）：`_retrieve_subquery` channels 加 `"graph"`，
   以**子问题**（而非原问题）做实体匹配——多跳分解的价值正在于每跳独立命中各自关系链。
4. **测试**：`tests/test_graph.py` +12 条（JSON 解析+兜底、类型归一、边溯源、实体匹配、分解路径含 graph）。

### 重建（在线 LLM 步骤）

小批试跑 4 块验质量通过后全量重建（`scripts/rebuild_graph_typed.py`）：

| 维度 | 旧图（jieba 共现） | 新图（LLM 类型化） |
|---|---|---|
| 节点 / 边 | 675 / 2354 | 1214 / 985 |
| 关系 | 共现 1609 + 上下文关联 745 | 属于 122 / 用于 69 / 包含 66 / 实现 / 使用 / 担任 18 … |
| 实体类型 | 无（全 entity） | other 1000 / place 51 / person 46 / org 42 / work 28 / event 24 / position 23 |
| chunk 溯源 | 无 | **985/985（100%）** |
| 构建耗时 | <2s（零 LLM） | 47.6s（74 合并单元，并发 5） |

### 评估（full 多跳 15 样本，对比重建前 v3）

| 指标 | v3（jieba 共现图） | v4（类型化图） | 判定 |
|---|---|---|---|
| avg F1 | 0.2914 | 0.2871 | −0.004，在跨运行波动带内（v3a/v3b 为 0.250–0.290），**无显著变化** |
| hit_rate | 0.80 | 0.80 | 持平 |
| coverage | 0.6756 | 0.6756 | 持平 |
| 平均延迟 | 9.6s | 10.8s（+12%） | 子问题级图检索开销 |
| num_failed | 0 | 0 | — |

**通道贡献验证**（trace 盲点说明）：`graph_hits` 仅统计主管道 recall，multi_hop 走分解路径
（`_decompose_retrieve`）时主管道被跳过，故 trace 中 graph_hits=0/15 是**统计口径问题而非通道失效**。
单点诊断（q0「范廷颂担任总主教的那个教区在哪里？」）证实：

- 快速实体匹配 0ms（零 LLM）命中「范廷颂」；
- 子问题「范廷颂担任什么职务？」→ 图文档带 `graph:tmpqnfbjx_x_2` chunk 溯源；
- 全管道 17 篇 fused 文档中 **2 篇 knowledge_graph**，含类型化关系「范廷颂枢机 →[担任]→ 总主教」。

**结论（未升如实记录）**：图升级在结构上达成目标（类型化关系 + 100% chunk 溯源 + 分解路径接入并
实际贡献文档），但 15 样本集上 F1 无显著变化（−0.004 ∈ 波动带），延迟 +12%——开销来自未命中图节点
的子问题触发 LLM 实体抽取回退。图检索为补充通道，不降即安全；净贡献需更大评估集与子问题级
graph_hits 统计才能判定。

**开放项**：① 分解路径图检索的 LLM 实体回退是延迟主因，可考虑子问题路径仅走快速匹配（零 LLM）；
② 评估 trace 缺分解路径的逐通道命中统计，下轮补 `decompose_graph_hits` 字段。

## 全量测试

336 passed（阶段1 +7 → 312；阶段2 +12 → 324；阶段3 +12 → 336）。

## 诚实边界与开放项

- **① finish 率未达标**：信号可见 ≠ 决策控制，需硬门控机制（证据充分度阈值）后续跟进。
- **② budget_skipped 0/15**：本评估集未触发熔断（max 16.9s < 25s）；熔断价值在极端 API 抖动场景，
  评估集无法覆盖，以单测保证正确性。
- **③ 图升级 F1 无显著变化**：结构目标达成（类型化 + 溯源 + 接入），但 15 样本集 F1 −0.004（波动带内）、
  延迟 +12%。诚实结论：本轮图升级是**基础设施投资**（关系质量、溯源能力、通道接线），
  检索收益未在当前评估集兑现；判定受限于样本量与 trace 统计盲点（见阶段 3 开放项）。
- `build_fast`（jieba 零 LLM）保留为兜底路径，生产图现为 LLM 类型化版本。
