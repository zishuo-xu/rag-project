# RAG 2.0 深度增强设计文档

> 日期：2026-07-23 分支：`feat/rag2-depth-enhancements`
> 缘起：公众号文章《滴滴二面：你的 RAG 项目太老了——大模型时代，学得慢就不用学了》
> 目标：把项目从「RAG 1.0 三件套（调包+存库+拼 prompt）」推进到「RAG 2.0 深度优化」，
> 新增 ≥4 个**可离线实现、可测试、能当面试亮点**的大特性，并用模拟真实用户测试验证收益。

---

## 0. 文章框架 → 特性映射

文章核心论点：RAG 2.0 = 多路检索 + Rerank + **自适应截断** + **Agent 驱动的迭代检索（Self-RAG）**；
并提出「三层评估」（检索层→生成层→端到端）与三个深度问题（embedding 高相似但答案不在召回 / LLM 幻觉 / 离线指标 vs 线上体验）。

本项目**已具备**：五路召回、RRF 融合、CrossEncoder 重排、CRAG 门控+补救、语义缓存、并发治理、CMRC 检索评估。

本次**新增 5 个大特性**，覆盖文章「三层评估」框架：

| # | 特性 | 层 | 对应文章点 | 新文件 |
|---|------|----|-----------|--------|
| F1 | Autocut 自适应截断（Kneedle 膝点） | 检索 | 「动态截断替代固定 TopK，解决噪声注入」 | `app/retrieval/autocut.py` |
| F2 | Self-RAG 迭代检索（质量驱动终止） | 检索 | 「Agent 驱动的迭代检索 Self-RAG」 | `pipeline.py` 增强 + `query_transform.refine()` |
| F3 | 生成忠实度自检（幻觉检测+重生成） | 生成 | 深度问题二「召回准但生成错=幻觉」 | `app/generation/faithfulness.py` |
| F4 | 查询路由 / 类型自适应 | 检索 | 「什么场景非用 RAG / 自适应策略」 | `app/retrieval/router.py` |
| F5 | 端到端三层评估 harness（含 A/B） | 评估 | 取舍五「三层评估」+ 模拟真实用户测试 | `run_e2e_eval.py` |

设计原则（沿用项目既有风格）：
- 每个特性**独立配置开关**（默认开），可单独 A/B，关掉后行为与现状完全一致（保护既有 48 个测试）。
- 每个阶段是**可独立测试的纯单元**（mock 友好，离线运行）。
- 失败**优雅降级**（任何新特性异常都回退到原行为，不让管道崩溃）。
- 不引入死代码；新增字段进 `RetrievalResult` / `RAGResponse` 供观测与评估。

---

## F1 · Autocut 自适应截断（Kneedle 膝点检测）

### 问题
`reranker.py:73` 是 `ranked_docs[:top_k]` —— CrossEncoder 已给每个候选打了 `rerank_score`，
但无论第 `top_k+1` 名之后分数是否断崖式下跌，都固定塞 `top_k` 篇给 LLM，造成**噪声注入**
（无关文档稀释上下文、干扰生成、浪费 token）。

### 算法选型（自主决定）
对比三种候选：

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 相对阈值（归一化后保留 ≥ratio×最高分） | 极简、稳定 | ratio 是魔法数；是「固定相对_cutoff」而非检测结构性断点，"自适应"味淡 | 备选 |
| 最大间隙（找相邻分数最大落差） | 贴合「断崖」叙事、简单 | 对单点离群突变脆弱，易切过早 | 备选 |
| **Kneedle 膝点检测** | 看**整体曲率**而非单点，抗噪；公认 elbow 算法；真正自适应；面试亮点强 | 需归一化、代码略多 | **选用** |

### 算法定义
输入：已按 `rerank_score` 降序的候选 `docs[0..n-1]`。

1. 若 `n <= min_docs`：全部保留（下界）。
2. 取分数 `s_i`，min-max 归一化到 `[0,1]`：
   `y_i = (s_i - s_min) / (s_max - s_min)`；`x_i = i / (n-1)`。
   若 `s_max == s_min`（全等分/平坦）→ 无膝点 → 回退 `top_k`。
3. 首尾连线（chord）：`line_i = y_0 + (y_{n-1} - y_0) * x_i`。
4. 差值曲线：`d_i = y_i - line_i`（内点）。
5. **膝点** `k = argmax(d_i)`。若 `max(d_i) <= 0`（曲线不外凸，无结构性拐点）→ 回退 `top_k`。
6. 截断：保留 `docs[0..k]`（含膝点），再施加上下界：
   `keep = clamp(k+1, min_docs, top_k)`。

> 直观：相关性曲线「先高后跌入噪声平台」时，膝点正是高相关区与噪声尾的分界。
> 上下界保证：永不返回空（≥min_docs）、永不超过原 top_k（纯降噪，不扩容）。

### 集成点
`pipeline.rerank()` 阶段（`pipeline.py:180`）：
```
if use_rerank and reranker:
    if use_autocut:
        scored = reranker.rerank(q, documents, top_k=len(documents))  # 打分+排序全部候选（计算量不变，仅多返回）
        return autocut_truncate(scored, top_k=top_k, min_docs=..., score_key="rerank_score")
    return reranker.rerank(q, documents, top_k=top_k)
return documents[:top_k]
```
`Reranker` 保持不变（纯打分+排序+截断器）；Autocut 是管道级决策，纯函数可单测。

### 配置（config.py）
```
use_autocut: bool = True
autocut_min_docs: int = 2          # 下界
# 上界复用 retrieval_top_k
```

### 观测
`RetrievalResult` 新增 `pre_autocut_count: int = 0`（截断前候选数）。
`docs - to - LLM = len(result.documents)`，评估对比 ON/OFF 即可量化降噪幅度。
tracer `rerank` span 记录 `{autocut: bool, pre: N, post: M}`。

### 测试（TDD）
`tests/test_autocut.py`（纯函数，零依赖）：
- 明显膝点曲线 `[.95,.90,.85,.30,.25,.20]` → 保留 3 篇
- 平坦/线性曲线（无膝点）→ 回退 top_k
- 全等分 → 回退 top_k
- 候选数 ≤ min_docs → 全保留
- 膝点 < min_docs → 提升到 min_docs（下界）
- 膝点 > top_k → 压到 top_k（上界）
- 单篇 / 空列表 → 边界正确
- 负分/原始 logit（CrossEncoder 可输出负值）→ 归一化后仍正确
`tests/test_pipeline.py` 增补：use_autocut 开/关时 rerank 阶段行为。

---

## F2 · Self-RAG 迭代检索（质量驱动终止）

### 问题
现状 CRAG 补救是**单次** HyDE 重检索（`pipeline.remediate`）。文章指出的 RAG 2.0 方向是
**Agent 驱动的迭代检索（Self-RAG）**：检索→反思→精化查询→再检索，循环直到证据充分。

### 核心：终止判断标志（用户特别强调，非资源兜底）
迭代**不**以「模型预算/调用次数耗尽」为终止，而以**检索质量信号**终止：

1. **充分性（主信号）**：CRAG 判定累积证据 `grade == "correct"`（足以高质量回答）→ 停。
2. **收敛性（主信号）**：本轮迭代**未召回到任何新的相关文档**（边际增益=0，查询精化已无效）→ 停。
3. **安全兜底**：`max_retrieval_iterations` 硬上限（默认 2），仅防 ①② 均不触发时跑飞。

> 语义：「证据够了就停；精化不再带来新证据就停」。硬上限只是保险丝，不是终止依据。

### 流程
```
accumulated = 主检索结果（recall+fuse+rerank，已过 Autocut）
grade, indices, reason = CRAG.evaluate(accumulated)
if grade == "correct": return accumulated          # 首检即充分，零迭代

for it in range(max_iterations):                    # 安全兜底
    if grade == "correct": break                    # ① 充分性
    refined_q = query_transformer.refine(question, evidence_summary(accumulated), reason)
    new_docs = recall(refined_q, channels=dense+sparse) -> fuse -> rerank
    new_relevant = [d for d in new_docs if d.chunk_id not in accumulated_ids]
    if not new_relevant: break                      # ② 收敛性（无新增相关证据）
    accumulated = rerank(accumulated + new_relevant)  # 合并去重后重排
    grade, indices, reason = CRAG.evaluate(accumulated)

return accumulated[:top_k]
```
- `refine()` 是 `QueryTransformer` 新方法（LLM）：基于「问题 + 已有证据摘要 + CRAG 缺口理由」
  生成一个**针对缺失信息**的精化查询（区别于 HyDE 的「假设文档」与 multi_query 的「多角度」）。
- 与现有 CRAG 单次补救的关系：迭代检索**取代** `incorrect→单次 HyDE` 分支（当 `use_iterative_retrieval=True`）；
  关闭时行为与现状逐字节一致。

### 配置
```
use_iterative_retrieval: bool = True
max_retrieval_iterations: int = 2     # 安全兜底上限
```

### 观测
`RetrievalResult` 新增 `iterations_used: int = 0`、`iterative_stop_reason: str = ""`
（取值：`sufficient` / `converged` / `max_iterations` / `disabled`）。

### 测试（TDD）
`tests/test_iterative_retrieval.py`（全 mock）：
- 首检 correct → 0 迭代，stop_reason=`sufficient`
- ambiguous→精化召回到新相关文档→correct → 停于 `sufficient`，iterations_used=1
- 精化召回**无**新文档 → 停于 `converged`
- 一直 ambiguous 且每轮都有新文档 → 撞 `max_iterations` 兜底停
- 关闭开关 → 走原 CRAG 单次补救路径（回归保护）
- refine() LLM 异常 → 优雅降级（用原问题/HyDE 兜底，不崩）

---

## F3 · 生成忠实度自检（幻觉检测 + 重生成）

### 问题
文章深度问题二：**召回 100% 准确，LLM 仍可能生成错误答案**——这是**生成层幻觉**，工程上检索再准也根除不了。
现状只有 prompt 约束（`RAG_SYSTEM_PROMPT` 强调忠实），无**生成后校验**。

### 设计
新模块 `app/generation/faithfulness.py`：
```
class FaithfulnessChecker:
    def check(question, context_docs, answer) -> FaithfulnessResult
        # FaithfulnessResult{faithful: bool, score: float[0,1], unsupported: list[str], reason: str}
```
- LLM-judge：让模型从 answer 抽取关键论断，逐条判断是否被 context 支撑，
  返回 `score = 支撑论断数 / 总论断数`；`faithful = score >= faithfulness_threshold`。
- 严格 JSON 输出，复用 CRAG 的 `_extract_json` 容错解析。
- LLM 带 timeout/retry（沿用项目模式），异常 → 返回 `faithful=None`（未知，放行，不阻断）。

### 集成（chain.py）
`invoke()` / `invoke_stream()` 生成后：
```
if use_faithfulness_check and not gate_skipped and documents:
    fb = checker.check(question, documents, answer)
    if fb.faithful is False and fb.score < threshold:
        # 不忠实 → 用更严格 prompt 重生成一次（最多 faithfulness_max_regen 次）
        answer = generate(question, documents, chat_history, strict=True)
        fb = checker.check(...)   # 复检（可选）
    response.faithful = fb.faithful; response.faithfulness_score = fb.score
```
- 新增 `STRICT_RAG_PROMPT`（更强约束：信息不足必须明说「文档未涉及」，绝不补全）。
- 重生成**最多 1 次**（bound 成本）；流式路径：先非流式生成+自检，通过后再逐字 yield（或先 yield 再标注，取前者保证用户看到的是已校验答案）。

### 配置
```
use_faithfulness_check: bool = True
faithfulness_threshold: float = 0.7
faithfulness_max_regen: int = 1
```

### 观测
`RAGResponse` 新增 `faithful: bool | None = None`、`faithfulness_score: float = 0.0`、
`regenerated: bool = False`。

### 测试（TDD）
`tests/test_faithfulness.py`（mock LLM）：
- 答案全被支撑 → faithful=True, score=1.0
- 答案含未支撑论断 → faithful=False, score<threshold → 触发重生成（strict prompt 被调用）
- LLM 异常 → faithful=None，放行不崩
- 关闭开关 → 不调 checker（回归保护）
- 重生成上限：至多 regen 次

---

## F4 · 查询路由 / 类型自适应

### 问题
现状对所有查询用同一套召回策略（固定 channels/top_k）。文章强调「什么场景非用 RAG、自适应策略」。
现有 `CRAG.validate_numeric_answer` 已是「数字型特判」的雏形——把它**泛化**为 principled 的查询路由器。

### 设计
新模块 `app/retrieval/router.py`（**零 LLM、规则驱动**，确定可测）：
```
class QueryRouter:
    def route(question) -> RoutingDecision
        # RoutingDecision{query_type, channels, top_k, autocut_min_docs, strategy_hint}
```
查询类型（正则/关键词规则）：
| 类型 | 触发信号 | 策略 |
|------|---------|------|
| `numeric` | 什么时候/哪年/多少/几个/when/how many | 强调 sparse（精确词匹配），autocut 更紧（min_docs 小） |
| `comparative` | 区别/对比/优劣/vs/哪个好 | 提高 top_k（需更多候选对比） |
| `conceptual` | 什么是/原理/为什么/如何 | dense + graph（语义+关系） |
| `multi_hop` | 含多个实体/「A 的 B 的 C」 | 提高 top_k + 全通道 |
| `factual`(默认) | 其他 | 默认五路 |

### 集成（pipeline.run）
```
if use_query_router:
    decision = router.route(question)
    channels = decision.channels; top_k = decision.top_k or top_k
    # 用 decision 参数化 recall + rerank(autocut)
```
- 路由**只调参**（channels/top_k/autocut），不改管道结构，低风险。
- `CRAG.validate_numeric_answer` 的数字快路保留（生成层校验），路由是检索层策略选择，二者互补。

### 配置
```
use_query_router: bool = True
```

### 观测
`RetrievalResult` 新增 `query_type: str = ""`。tracer 记录路由决策。

### 测试（TDD）
`tests/test_router.py`（纯规则，零依赖）：
- 各类型触发词 → 正确 query_type 与策略参数
- 默认 fallback → factual
- 边界：空串、纯英文、混合
`tests/test_pipeline.py` 增补：路由开/关时 channels/top_k 被正确参数化。

---

## F5 · 端到端三层评估 harness（+ 模拟真实用户测试）

### 问题
文章取舍五：**好的 RAG 评估不只看「检索准不准」，还要看「生成好不好」和「端到端用户满意度」**。
现状只有检索层（CMRC 命中率）+ 生成层（RAGAS，跨模型不可比）。缺**端到端答案正确性**与**特性 A/B**。

### 设计
新脚本 `run_e2e_eval.py`，对 CMRC 31 题（带 gold answer）跑 `chain.invoke()` 全链路，度量**三层**：

1. **检索层**：命中率（keyword_coverage≥0.5 或 source_hit，沿用现有口径）、平均检索篇数。
2. **生成层**：忠实度（`response.faithfulness_score`，F3 提供；关闭时用 judge 兜底）。
3. **端到端**：答案正确性 vs gold——
   - **中文 F1**：gold 与 pred 分词（汉字字符 + 连续英数 token），算 precision/recall/F1。
   - **Exact Match**：归一化后完全匹配率。
   - 附：平均 docs-to-LLM（噪声指标，验证 F1 降噪）、平均延迟。

### A/B 与特性归因
- 通过环境变量/配置覆盖，分别以「全特性 OFF（baseline）」与「全特性 ON」跑同一数据集，对比三层指标。
- 单特性归因（可选）：逐个开关，量化每个特性对端到端 F1 / 降噪 / 忠实度的边际贡献。

### 模拟真实用户测试（用户硬性要求）
- **真实题集**：CMRC 31 道真实阅读理解题（带 gold）——端到端跑，量化。
- **真实口语化查询**：另拟 ~10 条模拟真实用户的查询，覆盖边缘场景：
  口语化措辞 / 追问 / **不可答问题**（应诚实拒答而非幻觉）/ 数字型 / 概念型 / 多跳 / 对比型。
  逐条捕获完整响应（答案+来源+忠实度+路由类型+迭代次数+autocut 降噪），定性检视。
- **健壮性**：所有 LLM 调用带 timeout/retry；增量保存结果（断点不丢）；长时间运行用 `nohup ... & disown` 脱离会话；
  LLM 失败优雅降级并如实计入报告（不隐藏）。
- **诚实报告**：`docs/superpowers/reports/2026-07-23-rag2-e2e-validation-report.md`，
  含 A/B 数字、每特性归因、cross-model/LLM 非确定性 caveat、失败案例剖析。

### 测试
`tests/test_e2e_metrics.py`：中文 F1 / EM 计算函数的单测（纯函数，零 LLM）。
脚本本身靠真实运行验证（非单测）。

---

## 实施顺序与风险控制

按风险从低到高、依赖从前到后：
1. **F1 Autocut**（纯函数，零风险，建立模式）
2. **F4 查询路由**（纯规则，零风险）
3. **F2 迭代检索**（改 `pipeline.run`，中风险 → 用配置开关 + 更新 test_pipeline mock 保护回归）
4. **F3 忠实度自检**（改 `chain`，加 LLM，中风险 → 开关 + mock）
5. **F5 评估 harness**（依赖 F1-F4 就位以度量）
6. **模拟真实用户测试**（运行 F5）
7. **文档同步**（README + interview_guide 亮点）+ 全量测试绿 + code review + 提交

每特性严格 TDD：先写失败测试 → 实现 → 验证通过 → 下一特性。
既有的 48 个测试必须全程保持绿（新特性默认行为通过 mock 显式置 OFF 保护回归）。

## 验收标准
- [ ] 5 个特性全部实现、配置可控、默认开、异常优雅降级
- [ ] 每个特性有独立单测；全量测试绿（既有 48 + 新增）
- [ ] F5 harness 产出三层指标 + 特性 ON/OFF A/B 对比
- [ ] 模拟真实用户测试：CMRC 31 题端到端 + ~10 条口语化查询，诚实报告落盘
- [ ] 端到端 F1 不退化（目标：ON ≥ OFF）；docs-to-LLM 下降（F1 降噪有效）
- [ ] README + interview_guide 同步新亮点；无死代码；提交干净
