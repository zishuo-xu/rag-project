# RAG 3.0 生产级增强 · 验证报告（F7–F12）

> 日期：2026-07-24 ｜ 分支：`master`
> 测试集：CMRC 31 道真实阅读理解题（带 gold）
> 模型：DeepSeek（思考模式关闭）｜ Embedding/Rerank：本地 bge（bge-small-zh-v1.5 / bge-reranker-base）
> 设计文档：[docs/superpowers/rag3-design.md](../rag3-design.md)
> 目标对照：在 RAG 2.0（F1–F6）基础上补齐**生产级 RAG** 能力，**≥5 个大特性**、充分测试、文档输出、时延优先、可自我迭代。本轮交付 **6 个**（F7–F12）。

---

## 0. 结论速览（先说诚实结论）

| 维度 | baseline（特性全关） | full（RAG3.0 全开） | 变化 |
|------|:---:|:---:|:---:|
| 检索命中率 | 100% | 100% | 持平（CMRC 已饱和） |
| docs→LLM（上下文噪声） | 8.00 篇 | 4.26 篇 | **−47%** ✅（F1 Autocut） |
| **short_answer F1（F10）** | 0.0（未抽取） | **0.520** | **答案抽取 +48%** ✅ |
| **short_answer EM（F10）** | **0.0** | **0.097** | **打破 EM=0** ✅ |
| 端到端 F1（完整答案） | 0.349 | 0.352 | ≈持平（完整答案仍冗长） |
| 生成忠实度（F3 测得） | 未测（F3 关） | 0.774 | 双层防线 ✅ |
| 严格重生成率 | 0% | 22.6% | 幻觉防线生效 ✅ |
| 平均引用数（F7） | 0 | 1.52 | 答案可溯源到块 ✅ |
| 答案段进 top 上下文率（F6） | 96.8% | 96.8% | 持平 |
| 平均延迟 | 5.97s | 8.11s | **+36%**（归因见 §5） |

**一句话**：RAG 3.0 没有去刷一个已饱和的检索基准，而是补齐生产级能力并**正面攻击上一轮如实报告的 EM=0**——F10 零 LLM 答案抽取把 short_answer 的 **EM 从 0 → 0.097、F1 从完整答案的 0.35 → 0.52（+48%）**，闭环了「答案表达没和短答案对齐」这个我自己定位的瓶颈；同时 F7 让答案平均带 **1.5 条块级引用**、F1 降噪 47%、F3+F8 给出 0.77 忠实度与 22.6% 重生成。六特性默认路径**零在线 LLM 增量**（时延优先），+36% 延迟几乎全部来自被 baseline 关闭的 F3 judge / F2 迭代（RAG2.0 特性），而非 F7–F12 本身。

---

## 1. 交付的六个大特性（F7–F12）

| 特性 | 模块 | 解决的问题 | 时延策略 | 默认 |
|------|------|-----------|---------|:---:|
| **F7 引用溯源** | `app/generation/citation.py` | 答案不可验证、无块级出处 | embedding 余弦，零在线 LLM | 开 |
| **F8 投机流式** | `app/generation/streaming.py` | F3 使流式 TTFT≈完整生成 | 忠实度检查移到流末 | 开 |
| **F9 多级缓存** | `app/retrieval/caches.py` | 重复 embedding/rerank 抬高 P95 | L1+L2 LRU，命中省算 | 开 |
| **F10 答案质量增强** | `app/generation/answer_boost.py` | EM=0、答案埋在解释里 | 聚焦 prompt+零 LLM 抽取；自一致性默认关 | 开（自一致性关） |
| **F11 可观测/加固** | `app/observability/metrics.py` + `app/api/security.py` | 无指标/鉴权/限流/结构化日志 | 内存计数 <1µs，O(1) | 指标开，鉴权/限流关 |
| **F12 对话记忆** | `app/retrieval/conversation.py` | 多轮指代/省略检索丢上下文 | 启发式零 LLM；LLM 路径默认关 | 开（LLM 关） |

> 每项独立开关、异常优雅降级到 RAG 2.0 行为（见 `docs/architecture.md §12` 降级表）。

---

## 2. 三层评估 A/B（CMRC 31 题，单次运行）

### 2.1 检索层
- 命中率两模式均 **100%**，`answer_in_top_context_rate` 均 **96.8%**（F6 已把答案段召回率拉满，本轮不再动检索粒度）。
- **docs→LLM：8.00 → 4.26（−47%）**：baseline 关闭 F1 Autocut 固定取 top_k=8，full 膝点自适应截断到 4.26 篇。

### 2.2 生成层
- **忠实度 0.774**（F3 LLM-judge；baseline 关 F3 故为 null）。
- **重生成率 22.6%**（7/31）：F3+F8 双层——非流式走 `_generate_faithful` 严格重生成，流式走 F8 投机流式在流末补 `correction`。

### 2.3 端到端 + 答案质量（本轮重点）
- **完整答案 F1：0.349 → 0.352**，≈持平——完整回答依旧冗长，与短 gold span 对不齐（与 RAG2.0 结论一致）。
- **F10 short_answer：F1 0.520 / EM 0.097 / hit 0.419**。这是本轮核心证据：把答案抽取成 short span 后，**EM 从恒 0 提升到 0.097（3/31 严格相等）、F1 从 0.35 提升到 0.52（+48%）**。
  - baseline 的 answer_quality 全 0.0 是「F10 抽取关闭、不产出 short_answer」所致，**非答案错误**；有意义的对比是 full 内部「完整答案 F1 0.352 vs short_answer F1 0.520」。
- **F7 引用：平均 1.52 条/答案**，每条含 `chunk_id + confidence + snippet`，答案可溯源到具体块。

---

## 3. 测试证据（充分测试）

全量 `uv run pytest tests/ -q`：**256 passed**（10.69s，全离线 mock，无真实 LLM/网络）。本轮新增 **101** 个用例：

| 测试文件 | 用例数 | 覆盖 |
|---------|:---:|------|
| `tests/test_f9_caches.py` | 18 | LRU 淘汰/线程安全/命中率、EmbeddingCache 包装、RerankCache key 正确性（chunk_id 集合变化即失效） |
| `tests/test_f11_observability.py` | 19 | 指标计数/直方图/百分位/Prometheus+JSON 导出、API Key 校验、限流固定窗口、豁免路径 |
| `tests/test_f10_answer_boost.py` | 17 | 数字型/事实型抽取、填充词剥离、自一致性投票与平票稳定、类型过滤 |
| `tests/test_f12_conversation.py` | 16 | 指代检测、主题回填、省略型重写、max_turns 窗口、无需重写回退 |
| `tests/test_f7_citation.py` | 15 | claim 切分（来源标注先剥离）、余弦关联、bigram snippet、阈值/空输入降级 |
| `tests/test_f_integration.py` | 10 | F7–F12 接入 `RAGChain.invoke/invoke_stream` 的端到端事件序与降级 |
| `tests/test_f8_streaming.py` | 6 | 投机流式 token 序、流末检查、不忠实 correction、重生成上限 |

每个特性均覆盖：**正常路径 / 降级路径（异常、开关关闭）/ 边界（空输入）/ 时延相关（缓存命中、投机流式事件序）**。

---

## 4. 自我迭代证据（闭环上一轮的诚实结论）

RAG 2.0 验证报告如实写下：「检索命中 100% 但端到端 EM=0、F1≈0.35，瓶颈是**答案 span 与短答案对不齐**，不在检索排序作用域」。本轮 **F10 正面回应**：

1. **定位**（上一轮）：EM=0 是「答案表达」问题，不是「检索」问题。
2. **下药**（本轮）：答案聚焦 prompt 把答案前置 + 零 LLM 抽取 short_answer。
3. **验证**（本轮）：short_answer EM 0 → 0.097、F1 0.35 → 0.52。**自己提的问题，自己下一轮闭环**，而非换基准刷分。

---

## 5. 时延预算与 +36% 延迟的诚实归因

| 特性 | 在线 LLM 增量 | 默认路径时延影响 |
|---|---|---|
| F7 引用 | 0（embedding） | +一次批量编码 ~20–80ms |
| F8 投机流式 | 0（检查移到流末） | **TTFT 大幅下降**，总时延不变 |
| F9 多级缓存 | 0 | **命中时省 100–700ms** |
| F10 答案增强 | 聚焦/抽取 0；自一致性默认关 | 默认 ~0 |
| F11 加固 | 0 | <1µs/请求 |
| F12 重写 | 启发式 0；LLM 默认关 | 默认 ~0 |

**+36%（5.97s→8.11s）归因**：baseline 把 **F1–F4 全关**（含 F3 忠实度 judge 与 F2 迭代检索），full 全开。延迟增量几乎全部来自 F3 的 LLM-judge 与 F2 迭代（RAG 2.0 特性，与 RAG2.0 报告 +26% 同源），**F7–F12 自身默认路径零在线 LLM 增量**。F9 缓存在重复查询下 P95 反而更低；F8 把 TTFT 从「完整生成」降到「首 token」。

---

## 6. Caveats（测量严谨性，诚实记录）

1. **检索已饱和**：CMRC 命中率 100%、`answer_in_top_context` 96.8%，F7–F12 不作用于检索排序，其价值在可溯源/答案质量/加固/时延，而非该基准 F1。
2. **F12 未被 CMRC 触发**：CMRC 为**单轮**数据集（无 chat history），eval 中 `rewrite_rate=0`。F12 由 16 个离线单测验证（指代/省略/主题回填），**多轮收益未在端到端 eval 量化**——诚实标注为待补（需多轮数据集）。
   → **已于 2026-07-25 闭环**：12 组多轮集双层评估（重写层 12/12 命中、检索命中率 0.67→0.75、双刃剑归因），详见 [2026-07-25-eval-closure-report.md](./2026-07-25-eval-closure-report.md)。
3. **F10 EM 仍偏低（0.097）**：抽取把 EM 从 0 拉起，但多数 gold 是长 span（如「1990年被擢升为…宗座署理」），short_answer 难以逐字相等；EM 在 verbose RAG 上本就区分度低，应以 **F1（0.52）+ 子串命中（0.42）** 为准。
4. **单次运行 + LLM 非确定性**：temperature=0 但 judge/改写含轻微方差；跨模型对比需注明口径（沿用 Task11/RAG2.0 教训）。
5. **缓存/鉴权/限流为进程内单机实现**：F9 LRU、F11 指标与限流均在单进程内，多副本部署需换 Redis/共享存储——已在面试指南路线图标注。

---

## 7. 复现命令

```bash
# 单元测试（全离线）
uv run pytest tests/ -q                      # 256 passed

# 端到端 A/B（CMRC 31 题）
uv run python run_e2e_eval.py --mode baseline --output data/eval_e2e_rag3_baseline.json
uv run python run_e2e_eval.py --mode full     --output data/eval_e2e_rag3_full.json

# 单特性归因（示例：仅 F9）
uv run python run_e2e_eval.py --mode full --only F9 --limit 5
```

产物：`data/eval_e2e_rag3_full.json`（31 样本）、`data/eval_e2e_rag3_baseline.json`（31 样本）。
