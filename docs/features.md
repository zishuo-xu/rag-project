# 功能文档(特性清单与操作手册)

> **本文档的定位**:13 项特性 + 基础设施能力的**速查手册**——每项特性的开关、默认值、在线成本、降级行为、遥测字段与实测数字,一页可查。
> 分工:[architecture.md](./architecture.md) 讲实现, [technical_design.md](./technical_design.md) 讲设计决策, 本文讲「怎么用 / 怎么观测 / 出问题时退到哪」。
> 口径:数字为 deepseek、300 题 CMRC 全量基线(2026-07-29),除标注外。

## 1. 全特性速查表

| 特性 | 开关 | 默认 | 在线 LLM 成本 | 降级行为 |
|---|---|---|---|---|
| F1 Autocut 自适应截断 | `use_autocut` | ✅ | 0 | 回退固定 TopK |
| F2 Self-RAG 迭代检索 | `use_iterative_retrieval` | ✅ | 0~1(refine) | 单轮检索;超预算跳过 |
| F3 忠实度自检 | `use_faithfulness_check` | ✅ | 1(裁判)+0~1(重生成) | 异常放行(标"未知");超预算跳过重生成 |
| F4 查询路由 | `use_query_router` | ✅ | 0 | 默认检索深度 |
| F5 端到端评估体系 | (离线工具) | — | — | — |
| F6a 上下文增强分块 | `use_contextual_chunks` | ✅ | 0(索引期 LLM) | 降级 warning,主索引不阻断 |
| F6b 多跳分解 | `use_decomposition` | ✅ | 1(分解)+0~N(链式 refine) | 分解失败走原查询 |
| F7 引用溯源 | `use_citations` | ✅ | 0 | 无引用,答案照常 |
| F8 投机流式 | `use_speculative_streaming` | ✅ | 0(时序重排) | 回退阻塞流式 |
| F9 多级缓存 | `use_embedding_cache` / `use_rerank_cache` / `cache_enabled` | ✅ | 0 | miss 即正常计算 |
| F10 答案增强 | `use_answer_extraction` / `use_self_consistency` | ✅ / ❌ | 0 / 3×生成 | 抽取失败返回原答案;自洽默认关 |
| F11 生产加固 | `api_key` / `rate_limit_rpm` / `log_json` | 关/关/关 | 0 | 不配置=不启用,零影响 |
| F12 历史感知重写 | `use_history_rewrite` | ✅ | 0(启发式)/1(LLM 路径默认关) | 重写失败用原查询 |
| F13 Agentic RAG | `use_agentic` | ❌ | 每步 1(决策) | 任何异常/空证据→七阶段管道 |

**默认路径合计:每题 2 次在线 LLM 调用(生成 + F3 裁判)**,检索全链路零 LLM。实测均值延迟 4.5s(p50 3.9s / p90 4.6s / max 6.5s,300 题零失败)。

## 2. 特性卡片

---

### F1 · Autocut 自适应截断

- **定位**:Kneedle 膝点截断重排噪声尾,替代固定 TopK。
- **开关/参数**:`use_autocut[True]`、`autocut_min_docs[2]`(下界;上界 = `retrieval_top_k`)。
- **机制**:在 rerank 分数曲线上找膝点;必须工作在重排**之后**。
- **降级**:异常/关闭 → 固定 TopK 截断。
- **遥测**:`pre_autocut_count`(截断前候选数,实测均值 61.9)。
- **实测**:入 LLM 上下文 61.9→6.0 篇/题;`answer_in_top_context` 保持 1.0(截断不丢答案)。

### F2 · Self-RAG 迭代检索

- **定位**:CRAG 判不足时精化查询补召回,质量驱动终止。
- **开关/参数**:`use_iterative_retrieval[True]`、`max_retrieval_iterations[2]`(硬兜底)。
- **终止条件**:① sufficient(grade=correct)② converged(精化零新增)③ 硬上限。
- **降级/熔断**:超延迟预算 `check_skip` 跳过(记 `budget_skipped`);refine LLM 失败沿用原查询。
- **遥测**:`iterations_used`、`iterative_stop_reason`。
- **实测**:当前基线下平均迭代 0 次(零 LLM CRAG 几乎都判充分)——特性在线但基本休眠,留作难查询的安全网。

### F3 · 忠实度自检(F3+F8 核心叙事)

- **定位**:LLM-judge 逐 claim 判上下文支撑,不忠实有界重生成。
- **开关/参数**:`use_faithfulness_check[True]`、`faithfulness_threshold[0.7]`、`faithfulness_max_regen[1]`。
- **契约**:**单一事实源**——裁判与生成器用同一份格式化上下文;流式路径经 F8 在流末执行,共用 `regen_until_faithful` 循环体。
- **降级**:裁判异常 → `faithful=None` 放行(标"未知",不谎报已校验);超预算跳过重生成。
- **遥测**:`faithfulness`(score)、`faithful`、`regenerated`。
- **实测**:忠实度 **0.986**,重生成率 1.7%;口语化不可答问题 5/5 诚实拒答零幻觉。

### F4 · 查询路由(零 LLM)

- **定位**:正则信号判查询类型,自适应检索深度/降噪。
- **开关/参数**:`use_query_router[True]`;类型:numeric / comparative / multi_hop / conceptual / factual。
- **降级**:无匹配类型 → 默认深度。
- **遥测**:`query_type`。
- **实测**:多跳召回 1/15→14/15(修复后);已知边界:mh9 类漏判,登记为接受边界。

### F5 · 端到端评估体系

- **定位**:三层 e2e 评测 + 特性 A/B + 质量门(离线工具,非在线特性)。
- **入口**:`run_e2e_eval.py`( `--mode baseline|full`、`--only F*`、`--gate`、`--judge`、`--slice`、`--limit`、`--update-baseline`)。
- **分层**:零 LLM 检索评测(秒级)/ 50 题快速切片(~4min)/ 专项切片 / 300 题全量(~20min)。
- **判定**:gate smoke 层绝对阈值(检索 hit≥0.8、延迟≤30s、链路健康);full 层叠加相对基线(容差 F1±0.05 / hit±0.08)。
- **文档**:[technical_design.md](./technical_design.md) §9。

### F6a · 上下文增强分块

- **定位**:索引期 LLM 为每块生成上下文前缀,增强嵌入判别力。
- **开关/参数**:`use_contextual_chunks[True]`、`contextual_max_chars[80]`;独立 chroma 集合 `chunks_contextual`。
- **成本**:**索引期** LLM(每块 1 次,全量 ~3700 次),在线零增量——成本转移原则的典型实例。
- **降级**:构建失败 → warning,主索引不阻断;无 key 复现得到"无增强版"索引(数字偏低,README 注明)。

### F6b · 多跳查询分解

- **定位**:复合问题拆子问题分别检索合并;并行(无依赖)/链式(有依赖,上跳证据精化下跳)。
- **开关/参数**:`use_decomposition[True]`、`decomposition_max_subquestions[4]`、`decomposition_max_hops[2]`。
- **触发**:F4 判 multi_hop,或 F13 agent 自主 decompose。
- **降级**:分解失败/单子问题 → 原查询常规检索。
- **遥测**:`decomposed_subqueries`、`decomposition_chain`。
- **实测**:多跳分解触发率 93%(F13 路径);增益在方差内(如实记录,见报告),图通道子问题检索走零 LLM `fast_only`。

### F7 · 引用溯源(零在线 LLM)

- **定位**:答案切 claim,与源块 embedding 余弦关联,输出块级引用。
- **开关/参数**:`use_citations[True]`、`citation_threshold[0.5]`、`citation_max_claims[6]`。
- **降级**:异常 → 无引用,答案照常返回。
- **遥测**:`num_citations`、`source_hit`。
- **实测**:0.91 条引用/题;graph 源块经 `graph:` 前缀 + `source_chunk_ids` 同样可溯源。

### F8 · 投机流式(F3+F8 核心叙事)

- **定位**:先逐 token 流式(快 TTFT),流末忠实度自检,不忠实追加 `correction` 事件。
- **开关/参数**:`use_speculative_streaming[True]`。
- **事件协议**:`token`(逐字)→ `correction`(不忠实时的严格重生成,~3% 触发)→ `final`(答案 + 忠实度元信息)。
- **降级**:checker=None → 直接放行(与 F3 关闭一致)。
- **已知代价**:3% 概率、约 1 秒的错误曝光窗口(详见 [投机流式文章](./articles/03-speculative-streaming.md));强合规场景关闭本开关回退阻塞校验。

### F9 · 多级缓存

- **定位**:语义响应缓存(整答)+ embedding 缓存 + rerank 缓存。
- **开关/参数**:`cache_enabled[True]`、`cache_threshold[0.92]`、`cache_ttl[3600]`、`cache_max_size[200]`;`use_embedding_cache[True]/512`、`use_rerank_cache[True]/256`。
- **降级**:miss/异常 → 正常计算,对用户透明。
- **限制**:单进程内存实现;多副本部署需外置 Redis(登记为路线图,超出当前范围)。

### F10 · 答案增强

- **定位**:零 LLM 短答案 span 抽取(打破 EM 恒 0)+ 可选自适应自洽。
- **开关/参数**:`use_answer_extraction[True]`;`use_self_consistency[False]`(默认关:无数据撑 + 3× 生成成本,**软摘**——代码留、不主动讲)。
- **降级**:抽取失败 → 返回原答案。
- **遥测**:`short_answer`、`f1_short`、`em_short`、`hit_short`。
- **实测**:短答案 F1 **0.610**、EM 0→0.167(历史性地打破 verbose 答案 EM 恒 0)。

### F11 · 生产加固与可观测

- **定位**:API key 鉴权 + 限流(中间件,仅配置时注册)+ 结构化日志 + 指标。
- **开关/参数**:`api_key[""]`、`rate_limit_rpm[0]`、`log_json[False]`;指标常开无开关。
- **降级/设计**:不配置 = 完全不启用(默认零影响);限流触发返回 429,鉴权失败 401,健康检查等路径豁免。
- **端点**:`/api/metrics`(Prometheus + JSON 双格式)。

### F12 · 历史感知查询重写

- **定位**:多轮对话指代/省略消解,默认零 LLM 启发式。
- **开关/参数**:`use_history_rewrite[True]`、`history_rewrite_use_llm[False]`、`history_rewrite_max_turns[4]`。
- **降级**:重写异常 → 原查询。
- **遥测**:`rewritten`、`rewritten_query`。
- **实测**:重写层 12/12 全通过,检索 hit +8pp;端到端 F1 持平(启发式双刃剑:mt1 覆盖 0→0.60 vs mt7 0.83→0,如实记录)。

### F13 · Agentic RAG(核心叙事)

- **定位**:零依赖手写 ReAct 状态机,agent 自主决定检索/分解/停止。
- **开关/参数**:`use_agentic[False]`(默认关)、`agentic_max_steps[4]`、`agentic_decision_max_tokens[256]`、`agentic_evidence_gate[True]`(硬门控)。
- **停止机制**:LLM 主动 finish / 证据充分硬门控(CRAG correct 即停,零 LLM)/ 收敛护栏 / 步数上限。
- **降级**:任何异常或空证据 → 自动回退七阶段管道主路径。
- **遥测**:`agent_steps`(决策轨迹)、`agent_stop_reason`、`agent_actions`。
- **实测(多跳切片)**:finish 率 **93%**(修复前 33-53%)、平均 1.53 步、F1 0.566、延迟 5.9s。

## 3. 基础设施能力

| 能力 | 机制 | 关键参数 | 观测 |
|---|---|---|---|
| **延迟治理** | 查询级 `Deadline` 预算,可选阶段超预算熔断 | `latency_budget_ms[25000]`、`answer_max_tokens[1024]` | `budget_skipped`(实测几乎不触发——防离群尾不压均值) |
| **并发闸门** | `asyncio.Semaphore` 限流,排队超时 503 | `max_concurrent_requests[4]`、`request_queue_timeout[30]` | 压测:`run_concurrency_bench.py` |
| **可观测性** | 进程内计数器/直方图 + span tracing | 常开 | `/api/metrics`;遥测字段见 §4 |
| **评估门** | eval-as-gate,退化即非零退出 | `--gate [--gate-mode smoke\|full]` | pre-push hook(opt-in) |

## 4. 遥测字段速查

每次请求的 `RetrievalResult` / `RAGResponse` 携带(扩展新信号请加字段,勿 ad-hoc 打日志):

| 字段 | 含义 |
|---|---|
| `crag_grade` | CRAG 分级(correct/ambiguous/incorrect/recovered) |
| `query_type` | F4 路由判定类型 |
| `pre_autocut_count` | F1 截断前候选数 |
| `iterations_used` / `iterative_stop_reason` | F2 迭代次数与终止原因 |
| `faithfulness` / `regenerated` | F3 忠实度得分 / 是否触发重生成 |
| `num_citations` / `source_hit` | F7 引用数 / 源块命中 |
| `agent_steps` / `agent_stop_reason` | F13 决策轨迹 / 停止原因 |
| `budget_skipped` | 被延迟预算熔断的阶段列表 |
| `rewritten` / `rewritten_query` | F12 重写触发与结果 |
| `decomposed_subqueries` / `decomposition_chain` | F6b 分解子问题与链式标志 |

## 5. 特性组合与模式

- **延迟最优路径(默认)**:F1+F4+F7+F9+F10 开,检索零 LLM,每题 2 次 LLM(生成+裁判),~4.5s;
- **质量最大化路径**:`--mode full` 评测口径 = 默认 + F2/F3/F6 全开(F13 独立 `--only F13`);
- **RAG1.0 基线路径**:`--mode baseline`(F1-F4 全关)——A/B 的对照组,永远保留可跑;
- **零 key 路径**:服务拒绝启动(硬要求),但零 LLM 评测(检索/重写/graph fast)与单测全部可跑。

---

*新特性合入时:本文加卡片 + [architecture.md](./architecture.md) 加实现节 + config 加开关,三处同步,缺一不合并。*
