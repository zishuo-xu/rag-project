# F13 Prompt 优化 + 路由 multi_hop 修复 · 验证报告（2026-07-26）

> 承接 [评估闭环报告](./2026-07-25-eval-closure-report.md) 与 [F13 验证报告](./2026-07-25-f13-agentic-validation.md) 暴露的两个精确优化点：
> ① F13 agent 工具选择 search 主导、decompose 不稳定；② 路由 multi_hop 仅判中 1/15 导致 F6b 分解无法触发。

## 1. A · F13 Prompt 工程（agent 工具选择引导）

### 改动（`app/retrieval/agent.py`）

1. **DECISION_PROMPT v2**：硬工具条件——「需要**组合两个或以上事实/实体**时**优先使用** decompose」「已有 ≥1 篇相关证据时**立即结束**」「**禁止与已执行查询重复或语义雷同**」；追加 2 条 few-shot（单事实：search→finish；双事实：decompose→finish）。
2. **收敛护栏**：`consecutive_empty_search` 计数，连续 2 次 search 零新增证据 → `stop_reason="converged"` 强制停。
3. **重复查询警告**：`_tool_search` 检测到已执行过的 query，在 observation 追加「（警告：该查询已执行过，请改用 decompose/grade 或 finish）」。
4. **stop_reason 优先级修复**：空证据仅在 stop_reason ∈ {agent_done, max_steps} 时才覆写为 no_evidence，保留 decision_error/converged 语义。

### 复测（15 条多跳集，--only F13，两次复跑）

| 指标 | 优化前 | v2a | v2b |
|---|---|---|---|
| decompose 次数 | 3 | **13** | **14** |
| search 次数 | 43 | 37 | 31 |
| 主动 finish（agent_done） | 5/15 | 3/15 | **7/15（47%）** |
| max_steps 打满 | 10/15 | 12/15 | 8/15 |
| avg F1 | 0.283–0.286 | **0.298** | 0.291 |
| 平均延迟 | 9.3–9.6s | 12.8s | 12.4s |

**结论**：工具选择偏好可被 prompt 工程校正（decompose 从偶发 3 次到稳定 13–14 次，跨运行稳定），F1 不降反升。**未达标项（诚实记录）**：finish 率目标 >50%，v2b 仅 47%；延迟 +30%（分解 LLM 调用）；步数收敛仍需更强停止信号（如证据充分度打分）。

## 2. B · 路由 multi_hop 修复（`app/retrieval/router.py`）

### 改动

1. **优先级调整**：`multi_hop > numeric > comparative > conceptual > factual`（原 numeric 优先会截胡多跳问题，使 F6b 永不触发）。
2. **多跳信号从 1 类扩到 7 类 + 双疑问词规则**：关系链（的…的，距离 15）/ 并行标记（分别·各自）/ 指代链（的那个·的那位）/ 先后比较（先…还是…先，无需疑问词）/ 时间序列（之后·以前·之前）/ **序数唯一指代**（第一个·唯一·最早…哪）/ **排他计数**（除…外还有，无需疑问词）/ **双疑问词**（哪…哪… = 一句话问两个事实）。
3. **CMRC 回归防线**：数字型 + 所有格关系链的单事实提问（「《X的Y》是哪一年…」）需**第二事实标记**（还/也/并/同时/以及）才判多跳，否则落回 numeric——优先级互换引入的 2 条 CMRC 回归误判由此修掉。

### 离线指标

| 指标 | 修复前 | 修复后 | 目标 |
|---|---|---|---|
| 多跳集路由召回 | 1/15 | **14/15** | ≥10 ✅ |
| CMRC multi_hop 误判 | 3/31 | **3/31** | ≤4 ✅ |

唯一漏判 mh9（「…哪位皇帝**遇刺后**死里逃生…」）：裸「后」字时序信号放宽会大面积误伤 CMRC（之后/最后/然后），权衡后接受为已知边界。

### 端到端复测（15 条多跳集）

| 模式 | F1（旧→新） | 分解触发率 | 说明 |
|---|---|---|---|
| `--only F6`（F4 关，负对照） | 0.260 → 0.281 | **0**（不变） | 涨幅来自单样本生成方差（mh1 0.274→0.549），**非分解收益**——再次实证「分解触发依赖路由输出」 |
| `full`（F4+F6 联动） | **0.276 → 0.289** | **1/15 → 14/15**（chain 率 67%） | 逐样本 3 胜（mh1/mh8/mh11）1 负（mh12）11 平 |
| full 平均延迟 | 10.3s → **42.6s** | — | 链式分解的在线 LLM 调用；分解是「时延换精度」的显式交易，路由前置保证只对 multi_hop 查询付费 |

## 3. 测试

- 新增 9 条路由测试（七类信号单测 + 数据集召回 ≥10/15 + CMRC 误判 ≤4 回归防线 + 优先级互换行为不变验证）。
- 新增 4 条 agent 测试（收敛护栏、新证据重置计数、重复查询警告、prompt 硬条件+few-shot 存在性）。
- **全量 305 passed**（296 → 305），全离线 mock，零 LLM。

## 4. Caveats（诚实记录）

1. **F1 提升幅度小（+0.013）且样本量 15**：在 LLM 生成方差内（负对照单样本波动达 +0.275）；结论应表述为「分解收益从不可测变为可测且为正」，而非「显著提升」。
2. **full 模式延迟 4x**：chain 分解每跳一次 LLM，生产需配合超时/降级或仅对高价值查询启用。
3. **F13 finish 率 47% < 50% 目标**：prompt 能校正工具偏好但不足以收敛步数，下一步是证据充分度打分（grade 强制化）。
4. **路由规则仍是启发式**：14/15 召回靠七类正则信号堆叠，泛化到域外问法会衰减；根本解是分解触发与路由解耦（agent 路径已实现——F13 的 decompose 不依赖路由）。

## 5. 复现命令

```bash
uv run pytest tests/ -q                                    # 305 passed
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --mode full --output data/eval_e2e_multihop_full_v2.json
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --mode full --only F6 --output data/eval_e2e_multihop_f6_v2.json
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --mode full --only F13 --output data/eval_e2e_multihop_f13_v2a.json
```
