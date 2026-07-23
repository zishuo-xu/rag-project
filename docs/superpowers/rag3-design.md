# RAG 3.0 设计文档 — 生产级增强（F7–F12）

> 目标：在 RAG 2.0（F1–F6）基础上，补齐**生产级 RAG** 的关键能力。每个特性独立开关、异常优雅降级、**优先考虑时延**、可离线单测。
>
> 现状痛点（来自 RAG 2.0 验证报告）：
> 1. **答案正确性弱**：端到端 EM=0、F1≈0.36（检索命中 100%，但生成答案与短答案 span 对不齐）。
> 2. **时延回退**：F3 忠实度自检使 full 模式 ~8.8s vs baseline ~7.0s；且流式路径在 F3 开启时退化为"先非流式生成再整体吐出"，丢失逐 token 体验。
> 3. **生产加固缺失**：无指标导出、无鉴权、无限流、无结构化日志。
> 4. **重复计算浪费**：相同/相似查询重复 embedding、重复 cross-encoder 重排。
> 5. **答案不可溯源**：用户无法知道某句话来自哪个文档块。
> 6. **多轮指代未解**："它的原理呢？" 无法独立检索。

---

## 设计原则

1. **时延预算优先**：任何新增在线 LLM 调用都必须可关闭，且默认走零 LLM 快速路径；重计算尽量移到索引期或后台。
2. **优雅降级**：每个特性模块异常时回退到 RAG 2.0 行为，绝不中断主链路（与 F1–F6 一致）。
3. **可观测**：每个特性向 `RetrievalResult` / `RAGResponse` 写入观测字段，并接入 F11 指标。
4. **可离线单测**：LLM 全部 mock，纯函数/规则路径零依赖。
5. **独立开关**：`config.py` 每项一个布尔开关，默认开启但可被 eval harness 的 A/B 关闭。

---

## F7 引用溯源与答案定位（Citation & Grounding）

**问题**：生产级 RAG 必须让答案可验证。当前答案只有 `[来源: 文档名]` 文本标注，无法定位到具体块、无置信度。

**方案**：生成后把答案切成句子级 claim，用 **embedding 余弦相似度**（零在线 LLM）把每个 claim 关联到最相关的源文档块，输出结构化引用。

- 新模块 `app/generation/citation.py`：`CitationBuilder.build(question, answer, docs) -> List[Citation]`
- `Citation` dataclass：`{claim, source, chunk_id, doc_index, confidence, snippet}`
- claim 切分：正则按 `。！？.!?\n` 切句，过滤来源标注与过短句。
- 关联：claim 向量 vs 各 doc 向量余弦相似度；取最高分块，`confidence = cosine`；低于 `citation_threshold` 标 `confidence` 偏低但仍返回最近块。
- **时延**：复用索引期已有 doc embedding（若 metadata 带 `embedding`）否则即时编码；claim 数上限 `citation_max_claims` 控制编码量。零 LLM。
- 接入：`RAGResponse.citations`；`/api/chat` 返回 `citations`。
- 开关：`use_citations`（默认 True）、`citation_threshold=0.5`、`citation_max_claims=6`。

---

## F8 低延迟流式 + 投机忠实度（Speculative Streaming Faithfulness）

**问题**：F3 让流式路径先完整生成+自检再整体吐出，TTFT（首 token 时延）≈ 完整生成时延，用户体验差，且 full 模式时延 +1.8s。

**方案**：**投机流式**——先把答案逐 token 流给用户（快 TTFT），答案完成后在后台/同协程内做忠实度检查；若不忠实，追加一个 `correction` 事件携带严格重生成答案。用户既快又能看到已校验结果。

- 改造 `chain.invoke_stream`：F3 开启时不再阻塞式 `_generate_faithful`，而是：
  1. 流式生成累积 `full_answer`，逐 token yield；
  2. 流结束后跑忠实度检查；不忠实则 strict 重生成，yield `{"type":"correction","data":new_answer}`，并把最终 `done` 的 answer 替换为重生成结果。
- 新增事件类型 `correction`；`/api/chat` SSE 透传。
- **时延**：TTFT 从 ~完整生成 降到 ~首 token；忠实度检查与重生成只在流末发生，不阻塞首屏。
- 开关：`use_speculative_streaming`（默认 True）。关闭时回退旧阻塞行为。
- 观测：`RAGResponse.regenerated`、`faithful` 不变；新增 trace span `speculative_faithfulness`。

---

## F9 多级缓存（Multi-level Caching）

**问题**：相同/相似查询重复做 embedding 编码（~50–200ms）与 cross-encoder 重排（~100–500ms），浪费且抬高 P95。

**方案**：在现有语义响应缓存（L3）之上新增两级：

- **L1 Embedding 缓存**：`EmbeddingCache`，LRU，key=查询文本，value=向量。包装 `embeddings.embed_query`。
- **L2 Rerank 缓存**：`RerankCache`，LRU，key=`hash(query + sorted(chunk_ids))`，value=排序后 chunk_id→score。命中则跳过 cross-encoder，按缓存分排序。
- 新模块 `app/retrieval/caches.py`：通用 `LRUCache` + `EmbeddingCache` + `RerankCache`。
- 接入：`Reranker.rerank` 先查 L2；`RAGChain`/`pipeline` 的 query embedding 走 L1。
- **时延**：重复查询命中 L1/L2 直接省掉编码与重排，P95 显著下降；LRU O(1)。
- 开关：`use_embedding_cache`（True）、`embedding_cache_size=512`、`use_rerank_cache`（True）、`rerank_cache_size=256`。
- 正确性：rerank key 含 chunk_id 集合，文档变化即 key 变化，不会返回过期排序。

---

## F10 答案质量增强（Answer Quality Boost）

**问题**：EM=0、F1≈0.36。根因：完整回答冗长、把答案埋在解释里，与短答案 span 对不齐；数字/事实型问题未做答案聚焦。

**方案**（三件套，均可独立开关）：

1. **答案聚焦 Prompt**（`answer_focused_prompt`）：要求模型"先用一句话给出直接答案，再展开"，把答案前置。
2. **答案抽取后处理**（`extract_answer`）：零 LLM，从回答首句/含关键数字句抽取核心答案 span，写入 `RAGResponse.short_answer`，供评测与前端高亮。
3. **自适应自一致性**（`self_consistency`）：仅对 `numeric`/`factual` 短答案型查询，采样 N 次（temperature>0）抽取短答案并投票，取多数；其余类型跳过以保时延。
- 新模块 `app/generation/answer_boost.py`。
- **时延**：聚焦 prompt 与抽取零额外调用；自一致性仅短答案型触发且 N 小（默认 3），可关。
- 开关：`use_answer_focus`（True）、`use_answer_extraction`（True）、`use_self_consistency`（False，默认关以保时延，评测时可开）、`self_consistency_samples=3`、`self_consistency_types="numeric,factual"`。
- 观测：`RAGResponse.short_answer`、`self_consistency_used`。

---

## F11 可观测性与生产加固（Observability & Hardening）

**问题**：无指标、无鉴权、无限流、无结构化日志，距生产级有差距。

**方案**：

1. **指标注册表** `app/observability/metrics.py`：进程内计数器/直方图（零依赖，不引 prometheus 客户端）：
   - `requests_total{endpoint,status}`、`request_latency_ms` 直方图、`cache_hits_total{level}`、`faithfulness_score` 汇总、`errors_total`、`tokens_generated`。
   - `GET /api/metrics` 导出 Prometheus 文本格式 + JSON。
2. **API Key 鉴权** `app/api/security.py`：`api_key` 配置为空则关闭；非空时校验 `X-API-Key` 头（/api/health、/api/metrics 豁免）。
3. **限流**：令牌桶 `RateLimiter`（按客户端 IP，`rate_limit_rpm`），超限 429。
4. **结构化日志**：JSON 行格式开关 `log_json`。
- **时延**：指标为内存原子计数，开销 <1µs；鉴权/限流 O(1)。
- 开关：`api_key=""`（默认关）、`rate_limit_rpm=0`（0=关）、`enable_metrics=True`、`log_json=False`。

---

## F12 多轮对话记忆 / 历史感知查询重写（Conversational Memory）

**问题**：多轮追问含指代/省略（"它的原理呢？""那区别呢？"），直接检索会丢上下文。

**方案**：用对话历史把当前问题重写为自包含查询。

- **零 LLM 启发式**：检测指代词（它/这个/那/其/前者/后者/英文 it/this/that）或省略型短句；用最近一轮的实体/主题词回填。
- **可选 LLM 重写**：`history_rewrite_use_llm` 开启时用一次小调用做指代消解（默认关，保时延）。
- 新模块 `app/retrieval/conversation.py`：`ConversationRewriter.rewrite(question, history) -> str`。
- 接入：`RAGChain.invoke/invoke_stream` 在检索前对带历史的查询重写；观测字段 `rewritten_query`。
- **时延**：启发式零 LLM；LLM 路径默认关。
- 开关：`use_history_rewrite`（True）、`history_rewrite_use_llm`（False）、`history_rewrite_max_turns=4`。

---

## 时延预算总览

| 特性 | 在线 LLM 增量 | 默认路径时延影响 |
|---|---|---|
| F7 引用 | 0（embedding） | +一次批量编码（可复用），~20–80ms |
| F8 投机流式 | 0（检查移到流末） | **TTFT 大幅下降**，总时延不变或略降 |
| F9 多级缓存 | 0 | **命中时省 100–700ms** |
| F10 答案增强 | 聚焦/抽取 0；自一致性默认关 | 默认 ~0；开启自一致性 +N 次生成 |
| F11 加固 | 0 | <1µs/请求 |
| F12 重写 | 启发式 0；LLM 默认关 | 默认 ~0 |

净效果：默认配置下 **TTFT 与 P95 下降**，答案正确性与可溯源性提升，生产加固到位。

---

## 测试策略

- 每个特性一个 `tests/test_fX_*.py`，LLM 全 mock，离线可跑。
- 覆盖：正常路径 / 降级路径（异常、开关关闭）/ 边界（空输入）/ 时延相关（缓存命中、投机流式事件序）。
- 全量 `pytest tests/ -v` 必须绿。
- eval harness（`run_e2e_eval.py`）接入新开关，跑 A/B 验证答案质量与时延。
