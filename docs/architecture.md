# 架构与核心实现文档

> 本文档描述系统当前的真实架构与重要功能实现，所有结论均对照代码（含 `file:line` 引用）。
> 配套阅读：`README.md`（快速上手）、`docs/interview_guide.md`（设计决策与面试叙事）、
> `docs/superpowers/reports/2026-07-23-rag2-e2e-validation-report.md`（A/B 实测报告）。

---

## 目录

1. [系统概览与设计原则](#1-系统概览与设计原则)
2. [分层架构总览](#2-分层架构总览)
3. [请求生命周期](#3-请求生命周期)
4. [摄入层 Ingestion](#4-摄入层-ingestion)
5. [检索层 Retrieval（核心）](#5-检索层-retrieval核心)
6. [生成层 Generation](#6-生成层-generation)
7. [RAG 2.0 五大特性实现](#7-rag-20-五大特性实现)
7b. [RAG 3.0 生产级增强（F7–F12）](#7b-rag-30-生产级增强f7f12)
8. [并发治理](#8-并发治理)
9. [可观测性 Observability](#9-可观测性-observability)
10. [评估体系](#10-评估体系)
11. [配置开关速查](#11-配置开关速查)
12. [降级与容错设计](#12-降级与容错设计)
13. [技术选型](#13-技术选型)

---

## 1. 系统概览与设计原则

本系统是一个**生产级 RAG（Retrieval-Augmented Generation）系统**，覆盖
**摄入 → 检索 → 生成 → 评估** 全链路，并在标准 RAG 之上落地了 RAG 2.0 的五项深度增强
（Autocut 自适应截断、Self-RAG 迭代检索、忠实度自检、查询路由、端到端三层评估）。

### 设计原则

| 原则 | 体现 |
|------|------|
| **分层解耦** | 摄入 / 检索 / 生成 / 评估各层独立，可单独替换与测试 |
| **管道化** | 检索被重构为 7 阶段 `RetrievalPipeline`，每阶段职责单一、可开关、可替换 |
| **策略可插拔** | 分块、召回通道、查询改写、重排、截断策略均由配置开关驱动 |
| **优雅降级** | 几乎所有增强模块异常时回退到安全默认值，绝不阻断主链路 |
| **可观测** | 每个阶段记录耗时与中间结果，内置 Trace 瀑布图 |
| **eval 驱动** | 三层评估（检索 / 生成 / 端到端）+ A/B 特性归因，所有结论可复现 |

---

## 2. 分层架构总览

后端依赖方向自上而下：

```
main.py (FastAPI 入口 / lifespan)
  └─ app/api/routes.py (HTTP 端点) + app/api/schemas.py (Pydantic 模型)
       └─ app/generation/chain.py (RAGChain 编排中枢)
            ├─ app/retrieval/pipeline.py (RetrievalPipeline 七阶段)
            │    ├─ dense.py / sparse.py / graph_retriever.py / parent_child.py  (五路召回)
            │    ├─ fusion.py (RRF) / reranker.py (CrossEncoder) / autocut.py (Kneedle)
            │    ├─ query_transform.py / crag.py / router.py  (改写 / 评估 / 路由)
            │    └─ app/ingestion/indexer.py (层级索引)
            ├─ app/generation/faithfulness.py + prompts.py  (生成 / 忠实度自检)
            ├─ app/retrieval/cache.py (语义缓存)
            └─ app/observability/tracing.py (追踪)
  app/ingestion/ (loader / chunker / indexer / graph_extractor  摄入层)
  app/evaluation/metrics.py (评估指标)
config.py (全局配置单例)
frontend/app.py (Streamlit 前端，5 个 Tab)
```

**关键设计**：`chain.py` 是**薄编排层**，只负责「调 pipeline 拿文档 → 压缩上下文 → 拼 prompt →
生成 → 自检」；所有检索复杂度都收敛在 `RetrievalPipeline` 内。

---

## 3. 请求生命周期

### 3.1 入口与启动（`main.py`）

- `lifespan(app)`（`main.py:25-58`）启动时：
  1. 校验 `OPENAI_API_KEY`，缺失则抛 `RuntimeError`（`:31-35`）；
  2. 构造全局 `RAGChain(use_query_transform=True, use_rerank=True, query_strategy="multi_query")`
     并通过 `set_rag_chain` 注入路由（`:38-42`）；
  3. 用 `asyncio.Semaphore(settings.max_concurrent_requests)` 建立**并发闸门**（`:46-48`）；
  4. 尝试 `sparse_retriever.build_index()` 预热 BM25 索引（`:51-54`，失败仅告警）。
- 顶部设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`（`:4-5`），跳过 HuggingFace 联网检查，实现离线加速。

### 3.2 非流式提问完整调用链

```
POST /api/chat → chat()                          [routes.py:75]
  → _acquire_gate (Semaphore，排队超时返回 503)    [routes.py:61]
  → asyncio.to_thread(chain.invoke)               [routes.py:102]   ← 卸载到线程池，不阻塞事件循环
    → RAGChain.invoke                             [chain.py:287]
        1. tracer.start_trace                     [chain.py:295]
        2. _check_cache (语义缓存命中则直接返回)    [chain.py:298, 423]
        3. self.retrieve → pipeline.run (七阶段)   [chain.py:304]
        4. 生成分支：
           - gate_skipped      → generate_direct  (门控判定无需检索)
           - 有 checker + 有文档 → _generate_faithful (生成 + 忠实度自检 + 有界重生成)
           - 否则              → generate
        5. tracer.end_trace + _write_cache        [chain.py:328-331]
        6. return RAGResponse                     [chain.py:333]
  → _build_chat_response                          [routes.py:718]
```

### 3.3 主要 API 端点（`app/api/routes.py`）

| 端点 | 方法 | 行号 | 说明 |
|------|------|------|------|
| `POST /api/chat` | `chat` | :75 | 对话；`stream=True` 走 SSE，否则闸门内同步执行 |
| `POST /api/documents/upload` | `upload_document` | :211 | 校验后缀 → load → smart_chunk → index → 追加 BM25 |
| `GET /api/documents` | `list_documents` | :269 | 按 `doc_id` 分组统计 |
| `GET /api/documents/{id}/chunks` | `get_document_chunks` | :298 | 全链路索引详情（分块 + BM25 词项 + 向量 + L1 摘要） |
| `POST /api/retrieval/compare` | `retrieval_compare` | :366 | Dense/Sparse/Hybrid-RRF/Hybrid+Rerank 四策略对比 |
| `GET /api/traces` / `/api/traces/stats` | `get_traces` / `get_trace_stats` | :460/:468 | 瀑布图数据 / 各阶段平均耗时 |
| `POST /api/evaluate` | `evaluate` | :478 | 调 `evaluate_rag`（RAGAS 四维） |
| `POST /api/graph/build` | `build_knowledge_graph` | :510 | 后台异步构建知识图谱（增量） |
| `GET /api/graph/{stats,triples,visual}` | … | :567+ | 图谱统计 / 三元组 / 可视化数据 |
| `POST /api/graph/query` / `GET /api/graph/path` | … | :586/:603 | 图检索 / 实体路径 |
| `GET /api/health` | `health_check` | :688 | 健康检查 |

**SSE 流式**（`_stream_response`，`routes.py:121-206`）：用 `asyncio.to_thread(next, event_iter, None)`
逐事件把同步生成器移出事件循环（`:147`）；事件类型 `cache_hit` / `retrieval` / `token` / `done`；
闸门在 `finally` 中释放（`:204-206`）。

---

## 4. 摄入层 Ingestion

```
原始文件 → Loader → Chunker → Indexer → ChromaDB (L1 摘要 + L2 明细)
                         └─→ GraphExtractor → NetworkX 知识图谱
```

### 4.1 Loader（`app/ingestion/loader.py`）

- `LOADER_MAPPING`（`:14-19`）：`.pdf → PyPDFLoader`，`.txt/.md/.markdown → TextLoader`。
- `load_document(file_path)`（`:22-59`）：加载并补元数据 `source / file_path / doc_id / chunk_index`。
- `load_directory(dir_path)`（`:62-89`）：glob 批量加载，单文件失败仅告警不中断。

### 4.2 Chunker（`app/ingestion/chunker.py`）

| 策略 | 实现 | 适用 |
|------|------|------|
| **递归字符分块** | `recursive_chunk`（`:15-53`），`chunk_size=512 / overlap=64`，分隔符优先级 `\n\n > \n > 。 > . > 空格` | 结构规整的通用文档 |
| **语义分块** | `semantic_chunk`（`:56-135`），先切句再算相邻句 embedding 余弦相似度，`sim < 0.75` 处切分 | 主题频繁变化的长文档 |

- `smart_chunk`（`:138-181`）：**短文档（≤1000 字符）整段保留不切**，长文档按开关选语义或递归分块。
- 每块写入 `chunk_id = f"{doc_id}_{i}"` 与 `position`，作为后续去重与 Parent-Child 关联的主键。

### 4.3 层级索引 Indexer（`app/ingestion/indexer.py`）

- **双 Chroma collection**：`chunk_store`（L2 明细，collection `chunks`）+ `summary_store`（L1 摘要，collection `summaries`）。
- `index_documents(chunks, build_summary=True)`（`:115-150`）：L2 写入分块；L1 按 `doc_id` 分组，
  用 LLM 为每篇文档生成 ≤200 字摘要后写入（摘要失败仅告警）。
- `hierarchical_search(query, top_k=5)`（`:170-205`）：先 L1 取 top-3 摘要定位 `doc_id`，
  再 L2 用 `filter={"doc_id": {"$in": ...}}` 精检；过滤失败回退全局检索。
- Embedding：`get_embeddings()`（`:15-32`）默认本地 `HuggingFaceEmbeddings`（`normalize_embeddings=True`），
  模型由 `embedding_model` 配置（默认 `all-MiniLM-L6-v2`，`.env` 可覆盖为中文优化的 bge 系列）。

### 4.4 知识图谱抽取（`app/ingestion/graph_extractor.py`）

> **诚实说明**：被 `/api/graph/build` 调用的主构建路径 `build_from_documents_async`（`:285-415`）
> 是 **LLM 抽取**三元组；另有一条**零 LLM** 备用路径 `build_fast`（`:97-168`，jieba TF-IDF 关键词作实体 +
> 共现边），目前未接线到任何 API 端点。

- `KnowledgeGraphBuilder` 基于 `nx.DiGraph`（面向十篇级文档量，故用 NetworkX 而非 Neo4j）。
- LLM 路径：`extract_triples`（`:170-189`，每块最多 15 个三元组）→ `_parse_triples`（`:191-227`）解析。
- 异步构建：`asyncio.Semaphore(5)` 并发（`:340`）、两两合并 chunk 减少 LLM 调用（`_merge_chunks`）、
  增量跳过已处理 `chunk_id`（`:311-322`）、每 5 单元检查点持久化（`:368-377`）。
- 检索支持：`get_subgraph(entities, max_hops=2)`（`:481-514`，双向 BFS 扩展）、
  `get_entity_relations`（`:516-544`）、`_fuzzy_match_nodes`（`:582-605`，完全匹配插队首）。

---

## 5. 检索层 Retrieval（核心）

### 5.1 七阶段统一管道 `RetrievalPipeline`（`app/retrieval/pipeline.py`）

```
gate(要不要检索) → transform(查询改写) → recall(五路并行召回) → fuse(RRF 融合)
→ rerank(精排 + Autocut) → evaluate(CRAG 分级) → remediate / iterate(不合格则补救或迭代)
```

`RetrievalResult`（`:22-41`）承载全链路中间产物：`documents / dense_results / sparse_results /
graph_results / summary_results / fused_results / reranked_results / queries_used /
retrieval_time_ms / crag_grade / crag_action / gate_skipped / pre_autocut_count /
query_type / iterations_used / iterative_stop_reason`。

| 阶段 | 方法（行号） | 职责 |
|------|--------------|------|
| **① gate** | `gate`（`:79-87`） | 判断是否需要检索；未启用或异常默认 `True`（保守，不漏检索） |
| **② transform** | `transform`（`:91-96`） | 委托 `QueryTransformer` 生成查询变体，否则 `[question]` |
| **③ recall** | `recall`（`:100-172`） | `ThreadPoolExecutor(max_workers=6)` 并行五路召回，单路失败仅告警 |
| **④ fuse** | `fuse`（`:176-184`） | 固定 dense+sparse，其余通道非空才加入，调 RRF |
| **⑤ rerank** | `rerank`（`:188-208`） | CrossEncoder 精排；开启 Autocut 时先全量打分再膝点截断 |
| **⑥ evaluate** | `evaluate`（`:212-219`） | 先零 LLM 数字校验，再 CRAG LLM 分级 |
| **⑦ remediate** | `remediate`（`:223-234`） | HyDE 改写 → 双路召回 → 融合 → 重排（管道中再嵌一条小管道） |

**主编排 `run()`（`:330-505`）的关键优化**：

- **投机并行**：当门控与改写都启用时，用 `ThreadPoolExecutor(2)` 让 gate 与 transform **并行执行**
  （`:351-364`）——即使门控最终判定跳过检索，改写也已并发完成，不增加串行延迟。
- **F4 路由**：`query_router.route` 动态调整 `effective_top_k` 与 `effective_autocut_min`（`:385-401`）。
- **CRAG 分支**：开启 F2 时统一走 `_iterative_retrieve`（`:453-470`）；否则
  `incorrect → remediate`、`ambiguous → filter_relevant_docs`、`correct → 直接使用`。

### 5.2 五路召回（`ALL_CHANNELS`，`pipeline.py:19`）

| 通道 | 实现文件 | 原理 | 关键点 |
|------|----------|------|--------|
| **Dense** | `dense.py` | embedding + Chroma 向量相似度 | 多查询变体批量预算 embedding（`pipeline.py:119-124`），复用 `similarity_search_by_vector` |
| **Sparse** | `sparse.py` | BM25Okapi + jieba 中文分词 | `_tokenize`（`:103-134`）词级分词 + 数字单位正则（`1963年/100ms`）；索引持久化到 `bm25_index.pkl` |
| **Graph** | `graph_retriever.py` | 实体抽取 → 关系/子图多跳召回 | `_fast_entity_match`（`:88-120`）零 LLM 优先，失败回退 LLM；`max_hops=2` |
| **Parent-Child** | `parent_child.py` | 小块检索、大块返回 | child(200) 检索 `top_k*3` → 按 `parent_id` 去重 → 批量取 parent(1024)；需 `has_index()` |
| **Summary** | `indexer.py` | L1 摘要召回 | 由 `use_summary_recall` 开关，提供文档级语义 |

> **注意**：`RAGChain` 初始化会构造 Parent-Child 检索器，但上传端点未调用其 `index_documents`，
> 故默认无 child 索引，recall 中 `has_index()` 为 False 即自动跳过该路（优雅降级）。

### 5.3 RRF 融合（`app/retrieval/fusion.py`）

```
RRF_score(d) = Σ 1 / (k + rank_i(d)),   k = 60 (settings.retrieval_rrf_k)
```

- `reciprocal_rank_fusion`（`:13-64`）：仅利用**排名**而非原始分数，规避 Dense 与 BM25 分数不可比问题；
  多路都命中的文档自然获得更高分；`doc_key = chunk_id or hash(content)` 去重，结果写入 `metadata["rrf_score"]`。

### 5.4 CrossEncoder 重排（`app/retrieval/reranker.py`）

- 模型 `cross-encoder/ms-marco-MiniLM-L-6-v2`（`.env` 可覆盖为 bge-reranker）。
- `rerank(query, documents, top_k)`（`:33-79`）：构造 `(query, doc.page_content)` 对，
  `model.predict(pairs)` 打分后降序取 top_k，写入 `metadata["rerank_score"]`。
- **为什么用 Cross-Encoder 而非 Bi-Encoder**：query-doc 交互注意力精度更高，但速度慢，
  故只对融合后的少量候选（`rerank_top_n=20`）精排。

### 5.5 查询改写（`app/retrieval/query_transform.py`）

| 策略 | 方法 | 原理 |
|------|------|------|
| **Multi-Query** | `multi_query`（`:83-114`） | LLM 生成 3 个不同角度变体，**始终含原始查询**并去重 |
| **HyDE** | `hyde`（`:116-138`） | LLM 生成假设性回答，用其 embedding 检索；失败回退原问题 |
| **Refine**（F2 用） | `refine`（`:140-172`） | 基于已有证据与缺口精化查询；失败降级 HyDE，再失败回退原问题 |

- 内置 `_transform_cache`（TTL 3600s）避免重复 LLM 调用；缓存超 500 条删最旧一半（`:211-216`）。

### 5.6 CRAG 门控 / 评估 / 补救（`app/retrieval/crag.py`）

- `should_retrieve(question)`（`:70-89`）：LLM 判断是否需要检索，异常默认 `(True, "默认需要检索")`。
- `evaluate_relevance(question, documents)`（`:91-132`）：取前 5 篇各截断 300 字，LLM 输出
  `{grade, relevant_indices, reason}`；异常默认 `("correct", all_indices, "评估失败，默认通过")`。
- `validate_numeric_answer`（静态，`:148-174`）：**零 LLM** 数字校验——问题匹配数字型模式
  （`什么时候/哪一年/多少/when/how many`）但上下文无 `\d{2,}` 时直接判 `incorrect`，避免 LLM 编造数字。
- `_extract_json`（静态，`:176-191`）：正则取最外层 `{...}` 并修复尾逗号，被多个模块复用。

### 5.7 语义缓存（`app/retrieval/cache.py`）

- `SemanticCache`（`:26`）：`threshold=0.92 / max_size=200 / ttl=3600`。
- `get(query_embedding)`（`:52-88`）：O(n) 遍历，跳过 TTL 过期项，算余弦相似度取最高，
  `>= threshold` 命中并 `hit_count += 1`、刷新 timestamp（LRU）。
- **效果**：重复/语义相近查询绕过整条管道，毫秒级返回（实测 ~20-50ms，约 225 QPS）。
- 仅在**无对话历史**时启用（`chain.py:_check_cache`），避免多轮上下文污染。

---

## 6. 生成层 Generation

### 6.1 编排中枢 `RAGChain`（`app/generation/chain.py`）

- `RAGResponse`（`:37-48`）：`answer / sources / retrieval_result / total_time_ms / cache_hit /
  faithful / faithfulness_score / regenerated`。
- `__init__`（`:59-138`）：按配置开关注入各组件（graph / parent_child / crag / router /
  faithfulness_checker / semantic_cache）。LLM 为 `ChatOpenAI(temperature=0, streaming=True,
  request_timeout=60, max_retries=2, extra_body=get_llm_extra_body())`——`extra_body` 在关闭思考模式时
  下发 `{"thinking": {"type": "disabled"}}`。
- `invoke`（`:287-341`）与 `invoke_stream`（`:343-419`）：均为
  `start_trace → _check_cache → retrieve → 生成分支 → end_trace → _write_cache`。
  **F3 开启时流式会先非流式生成 + 自检，再一次性输出已校验答案**（`:383-388`），保证用户看到的是通过忠实度校验的内容。

### 6.2 上下文压缩（零 LLM）

- `compress_context`（`:153-190`）：正则 `[一-鿿]{2,}|[a-zA-Z0-9]+` 提取 query 关键词，
  按 `(?<=[。！？.!?\n])` 分句，按关键词重叠度为每篇文档保留 top `max_sentences_per_doc=3` 句。
  在生成前进一步降低上下文噪声，**完全不消耗 LLM**。

### 6.3 Prompt 模板（`app/generation/prompts.py`）

| 模板 | 行号 | 用途 |
|------|------|------|
| `RAG_SYSTEM_PROMPT` / `RAG_SIMPLE_PROMPT` | :6 / :42 | 标准 RAG（6 条核心原则：仅基于文档 / 不补充 / 精确引用 / 诚实拒答 / 聚焦 / 结构清晰） |
| `RAG_CHAT_PROMPT` | :26 | 多轮对话（`MessagesPlaceholder("chat_history")`） |
| `STRICT_RAG_PROMPT` / `STRICT_RAG_CHAT_PROMPT` | :86 / :45 | F3 严格重生成（信息不足必须明说） |
| `DIRECT_ANSWER_PROMPT` | :80 | 门控跳过检索时的通用回答 |
| `FAITHFULNESS_CHECK_PROMPT` | :106 | F3 LLM-judge，输出 `{score, unsupported[], reason}` |
| `FALLBACK_RESPONSE` | :71 | 无文档兜底文案 |

---

## 7. RAG 2.0 五大特性实现

这是本轮迭代的核心，针对「RAG 1.0 三件套（切分 + 向量 + 拼接）已过时」的痛点，
落地了多路检索 + Rerank + Autocut + Agent 迭代检索 + 三层评估的 RAG 2.0 能力。

### F1 · Autocut 自适应截断（`app/retrieval/autocut.py`）

**问题**：固定 `top_k` 要么截断掉有用文档，要么引入噪声文档。

**实现**：用 **Kneedle 膝点算法**在重排分数曲线上找「拐点」，自适应决定保留篇数。

- `find_knee(scores)`（`:40-79`）：
  1. min-max 归一化分数 `ys`，横轴 `xs = i/(n-1)`；
  2. 连首尾两点成直线，取一般式 `A·x + B·y + C = 0`；
  3. 遍历内点算到直线的距离 `|A·xs[i] + B·ys[i] + C|`，距离最大者即膝点；
  4. 平局稳定取靠前者（`_TIE_TOL`）；分数全等或近似线性（`best_dist < _EPS`）返回 `None`（无膝点）。
- `autocut_truncate(documents, top_k, min_docs=2)`（`:82-120`）：
  `knee is None → keep = min(top_k, n)`，否则 `keep = knee + 1`；
  最终 `keep = max(min_docs, min(keep, top_k, n))`——**纯降噪、绝不扩容**。

**效果**：上下文平均篇数 8 → 4.19（噪声 −48%），同时命中率不变。

### F2 · Self-RAG 迭代检索（`pipeline.py:_iterative_retrieve`，`:238-305`）

**问题**：单次检索可能证据不足，但盲目多轮又浪费延迟。

**实现**：**质量驱动**的迭代检索，三种终止条件（**非 token 预算**）：

| 终止条件 | 判定 | stop_reason |
|----------|------|-------------|
| ① **充分性** | CRAG 评级 `grade == "correct"` | `sufficient` |
| ② **收敛性** | 精化召回未带来任何新增 `chunk_id` | `converged` |
| ③ **安全兜底** | 达到 `max_retrieval_iterations`（硬上限，非主信号） | `max_iterations` |

每轮循环：`_refine_query`（基于证据摘要 + 缺口精化查询）→ 双路 recall → rerank →
合并去重重排 → 重新 CRAG 评估。`crag_grade` 仅在「由非 correct 转为 correct」时标记 `recovered`
（`:466`，避免首检即正确被误标）。

### F3 · 忠实度自检 + 有界严格重生成（`app/generation/faithfulness.py` + `chain.py:_generate_faithful`）

**问题**：LLM 可能脱离检索上下文产生幻觉，且难以量化。

**实现**：

- `FaithfulnessChecker.check(question, context_docs, answer)`（`:57-88`）：
  用 `FAITHFULNESS_CHECK_PROMPT` 让 LLM 作 judge，输出 `{score, unsupported[], reason}`，
  `score = 被支撑论断数 / 总论断数`，clamp 到 [0,1]，`faithful = score >= threshold(0.7)`；
  **任何异常 / 不可解析 → `faithful = None` 放行，绝不阻断主链路**。
- `_generate_faithful`（`chain.py:256-285`）：生成 → 自检 → 若 `faithful is False` 且仍有配额，
  用 `strict=True` 重新生成，最多 `faithfulness_max_regen(1)` 次；返回
  `(answer, faithful, score, regenerated)`。

**效果**：可量化忠实度 0.80，22.6% 的回答触发重生成；不可答问题 5/5 诚实拒答、零幻觉。

### F4 · 查询路由（`app/retrieval/router.py`，零 LLM）

**问题**：不同查询类型对召回广度 / 降噪的需求不同。

**实现**：规则引擎按优先级 `numeric > comparative > multi_hop > conceptual > factual` 分类，
**只调 `top_k` 与 `autocut_min_docs`，不削减召回路**：

| 类型 | 触发信号 | 策略 |
|------|----------|------|
| numeric | `什么时候/哪一年/多少/几个/\d+年` | `autocut_min_docs=1`（收紧降噪，突出唯一答案） |
| comparative | `区别/差异/对比/优劣/vs` | `top_k+3, autocut_min_docs=3`（放宽，覆盖多方） |
| multi_hop | `的.{1,10}的` 关系链 + 疑问词 | `top_k+3, autocut_min_docs=3` |
| conceptual | `什么是/原理/为什么/如何` | 默认 |
| factual | 兜底 | 默认 |

### F5 · 端到端三层评估 + A/B 归因（`app/evaluation/metrics.py` + `run_e2e_eval.py`）

**问题**：单层指标无法定位问题出在检索还是生成；特性效果难以归因。

**实现**：

- **三层指标**（`metrics.py` 末尾纯函数，零依赖零 LLM）：
  - 检索层：命中率 / 关键词覆盖率 / 来源命中；
  - 生成层：忠实度 / 重生成率；
  - 端到端：`answer_f1`（多重集 token F1，`:485-502`）、`normalized_exact_match`（`:505-508`）、
    `answer_hit`（gold 归一化后作子串命中，`:511-517`）。
- **A/B 特性归因**（`run_e2e_eval.py`）：`FEATURE_FLAGS = {F1: use_autocut, F2: use_iterative_retrieval,
  F3: use_faithfulness_check, F4: use_query_router}`；`apply_feature_mode(mode, only)` 支持
  `baseline`（全关）/ `full`（全开）/ `--only Fx`（单特性），并**强制 `cache_enabled=False`** 防止跨模式污染。
- 另有 `COLLOQUIAL_QUERIES`（10 条口语化 / 数字 / 不可答 / 闲聊查询）做定性检视。

> **诚实结论**（详见验证报告）：检索命中率已饱和至 100%，F1-F4 对端到端 F1 提升有限（0.347→0.357）；
> 真正收益是**上下文噪声 −48%、可量化忠实度 0.80、不可答问题诚实拒答**，代价是延迟 +26%（6.95s→8.80s）。

---

## F6 · 答案定位增强

**问题**：F5 的诚实结论指出瓶颈是『答案 span 不在 top 块』的**召回粒度**问题——源文档命中了，但精确答案所在的那一小段没被排进 top 上下文；同时 multi_hop 类问题（答案需跨多个事实拼接）单轮召回难以覆盖。F6 从**召回粒度**与**查询结构**两个方向补这块短板。

### F6a · 细粒度召回 + 上下文增强（`app/ingestion/contextual.py` + `indexer.py`）

- **Parent-Child 接线**：此前 `parent_child_retriever` 是 dead feature（写了没接进主链路）。F6 在 `routes.py` 摄入与 `POST /api/documents/reindex` 中真正调用 `index_documents`，并在 `chain.py:rebuild_parent_child_index` 提供历史索引重建入口；子块（200 字）精确匹配、父块（1024 字）回传上下文。
- **Contextual Chunking**（Anthropic Contextual Retrieval）：索引期用 LLM 为每个 chunk 生成「文档级上下文」，**embed 时用「上下文+原文」提升语义可检索性，但 `page_content` 仍存原文**（上下文只进 metadata），从而在线检索阶段**零 LLM 成本**。见 `contextual.py:generate_chunk_context / build_chunk_contexts`。
- **双 Chroma 集合 + `detail_store` 切换**：contextual 结果写入独立集合 `chunks_contextual`；`indexer.detail_store` 仅在 `use_contextual_chunks=True` **且** contextual 集合非空时返回它，否则回退原 `chunk_store`——保证未重建索引时行为与旧版完全一致（回归安全）。`dense.py` 的向量/文本两路均改走 `detail_store`。

### F6b · 多跳查询分解（`query_transform.py:decompose` + `pipeline.py` 多跳分支）

- `decompose(question) -> Decomposition(sub_questions, chain)`：LLM 输出 JSON，复用 `CRAGEvaluator._extract_json` 解析；子问题数截断到 `decomposition_max_subquestions`，≤1 个子问题则回退 `Decomposition([question], False)`。
- pipeline 仅在 `query_type=="multi_hop"` 且 `use_decomposition` 时进入多跳分支：**并行优先**（各子问题走轻量 `recall`（仅 dense+sparse，跳过门控/改写）后 RRF 合并）；`chain=True` 时**链式**（用上一跳证据经 `refine()` 生成下一跳）。
- 观测字段：`RetrievalResult.decomposed_subqueries / decomposition_chain / answer_localization_method`（`parent_child` / `contextual`）。

> **设计准则**：F6a 在线零 LLM（成本前置到索引期）、F6b 仅 multi_hop 触发（避免给简单查询加分解开销）；两者异常均优雅降级到原召回路径。

---

## 7b. RAG 3.0 生产级增强（F7–F12）

RAG 2.0（F1–F6）解决了「检索得准、生成不编造」的问题；RAG 3.0 在此基础上补齐**生产级 RAG** 的六项关键能力。设计准则与 F1–F6 一脉相承：**每项独立开关、异常优雅降级、默认路径零在线 LLM 增量（时延优先）、可离线单测**。完整设计见 `docs/superpowers/rag3-design.md`。

> 痛点驱动（来自 RAG 2.0 验证报告）：端到端 EM=0 / F1≈0.36（检索命中 100% 但答案与短答案 span 对不齐）；F3 使流式退化为「先整体生成再吐」TTFT 高；重复 embedding/rerank 抬高 P95；答案不可溯源；多轮指代未解；无指标/鉴权/限流/结构化日志。

### F7 · 引用溯源与答案定位（`app/generation/citation.py`，零在线 LLM）

- 生成后把答案切成句子级 claim（`split_claims`：先整体剥离 `[来源: …]` 标注再按 `。！？.!?\n` 切句，过滤过短句），用 **embedding 余弦相似度**把每个 claim 关联到最相关源块，输出结构化 `Citation{claim, source, chunk_id, doc_index, confidence, snippet}`。
- `_tokens` 用「中文 bigram + 英数单词」做 snippet 重叠匹配，避免整句成单 token 导致零重叠。
- **时延**：claim 数上限 `citation_max_claims(6)` 控制编码量；零 LLM。接入 `RAGResponse.citations` 与 `/api/chat`。
- 开关：`use_citations[True]`、`citation_threshold[0.5]`、`citation_max_claims[6]`。

### F8 · 低延迟流式 + 投机忠实度（`app/generation/streaming.py`）

- **投机流式**：先逐 token 把答案流给用户（快 TTFT），流结束后再跑忠实度检查；不忠实则 strict 重生成并追加 `{"type":"correction"}` 事件，`done` 的 answer 替换为校验后结果。用户「既快又看到已校验答案」。
- 解决 F3 在流式路径「先非流式生成再整体吐出」导致 TTFT≈完整生成时延的回退。
- **时延**：TTFT 从 ~完整生成 降到 ~首 token；检查/重生成只在流末发生，不阻塞首屏。
- 开关：`use_speculative_streaming[True]`（关闭回退旧阻塞行为）。

### F9 · 多级缓存（`app/retrieval/caches.py`）

- 在既有语义响应缓存（L3）之上新增两级：**L1 Embedding 缓存**（`EmbeddingCache` 包装 `embed_query/embed_documents`，key=文本）与 **L2 Rerank 缓存**（`RerankCache`，key=`hash(query+sorted(chunk_ids))`，命中即跳过 cross-encoder 按缓存分排序）。
- 通用 `LRUCache`（线程安全 OrderedDict，O(1)，统计 hits/misses/hit_rate）；`reranker.rerank` 先查 L2，`RAGChain._embed_query` 走 L1。
- **正确性**：rerank key 含 chunk_id 集合，文档变化即 key 变化，不返回过期排序。
- **时延**：重复查询命中省掉编码（~50–200ms）与重排（~100–500ms），P95 显著下降。
- 开关：`use_embedding_cache[True]`、`embedding_cache_size[512]`、`use_rerank_cache[True]`、`rerank_cache_size[256]`；`GET /api/cache/stats` 暴露命中率。

### F10 · 答案质量增强（`app/generation/answer_boost.py`）

- 三件套（均可独立开关）：**① 答案聚焦 Prompt**（`RAG_FOCUS_PROMPT`：要求「第一行先用一句话直接给出答案」把答案前置）；**② 零 LLM 答案抽取**（`extract_short_answer`：数字型问题抽年份/数值，否则抽首个实质句去填充词，写入 `short_answer`）；**③ 自适应自一致性**（`self_consistency_vote`：仅 `numeric/factual` 短答案型采样 N 次投票取多数，其余跳过保时延）。
- 直接攻击 EM=0：把冗长回答里的答案 span 对齐到短答案。冒烟评测 `short_answer` 使 EM 0.0→0.5、F1 0.44→0.95。
- **时延**：聚焦/抽取零额外调用；自一致性默认关。
- 开关：`use_answer_focus[True]`、`use_answer_extraction[True]`、`use_self_consistency[False]`、`self_consistency_samples[3]`、`self_consistency_types[numeric,factual]`。

### F11 · 可观测性与生产加固（`app/observability/metrics.py` + `app/api/security.py`）

- **指标注册表**：进程内计数器/直方图（零依赖，不引 prometheus 客户端；直方图有界 deque 采样，线程安全），`GET /api/metrics` 导出 Prometheus 文本 + JSON。
- **API Key 鉴权**：`api_key` 为空即关闭；非空校验 `X-API-Key`（`/api/health`、`/api/metrics`、`/docs` 等豁免）。
- **限流**：`RateLimiter` 固定窗口按客户端 IP 每分钟计数，超限 429。
- **结构化日志**：`log_json` 开启 JSON 行格式（`main.py:_setup_logging`）。
- **时延**：指标为内存原子计数 <1µs；鉴权/限流 O(1)。安全中间件仅在 `api_key` 或 `rate_limit_rpm>0` 时注册（默认关，不影响测试/流式）。
- 开关：`enable_metrics[True]`、`api_key[""]`、`rate_limit_rpm[0]`、`log_json[False]`。

### F12 · 多轮对话记忆 / 历史感知查询重写（`app/retrieval/conversation.py`）

- 用对话历史把含指代/省略的当前问题（「它的原理呢？」「那区别呢？」）重写为自包含查询：`needs_rewrite` 检测指代词/省略型短句，`ConversationRewriter.rewrite` 用最近一轮主题词回填（启发式零 LLM），`history_rewrite_use_llm` 开启时走一次小调用做指代消解（默认关）。
- 接入 `RAGChain.invoke/invoke_stream` 检索前重写；观测字段 `rewritten_query`。
- **时延**：启发式零 LLM；LLM 路径默认关。
- 开关：`use_history_rewrite[True]`、`history_rewrite_use_llm[False]`、`history_rewrite_max_turns[4]`。

> **RAG 3.0 时延预算**：默认配置下 F7/F9/F10/F11/F12 在线零 LLM 增量，F8 把忠实度检查移到流末——净效果是 **TTFT 与 P95 下降**、答案正确性与可溯源性提升、生产加固到位。每项异常均回退 RAG 2.0 行为。

---

## 8. 并发治理

`/api/chat` 是 CPU（embedding/rerank）+ IO（LLM）混合负载，高并发下会打满事件循环与下游 LLM。
系统采用四重治理：

1. **`asyncio.Semaphore` 准入闸门**：`max_concurrent_requests(4)` 限制同时在途请求数（`main.py:46-48`）。
2. **`asyncio.to_thread` 卸载**：把同步的检索 / 生成放到线程池，**解除对事件循环的阻塞**（`routes.py:102,147`）。
3. **排队超时 503 快速失败**：`request_queue_timeout(30s)` 内拿不到信号量即返回 503，而非无限堆积（`routes.py:61-70`）。
4. **语义缓存吸收热点**：重复查询绕过整条管道毫秒级返回。

**召回层并行**：`recall` 用 `ThreadPoolExecutor(max_workers=6)` 并行五路（`pipeline.py:147`）；
门控 + 改写投机并行（`pipeline.py:357-364`）；图谱构建 `asyncio.gather + Semaphore(5)`（`graph_extractor.py:340`）。

**实测**：并发 1→20 下错误率 0%；缓存命中 ~225 QPS。

---

## 9. 可观测性 Observability

`app/observability/tracing.py` 实现轻量级、线程安全的链路追踪：

- `Span`（`:11-25`）：`name / start_ms(相对 trace 起点偏移) / duration_ms / metadata`。
- `Trace`（`:28-46`）：`trace_id / question / timestamp / spans[] / total_ms / answer_preview`。
- `Tracer`（`:49`）：`_traces = deque(maxlen=50)` 环形归档，`threading.Lock()` 保证线程安全。
- 管道中实际记录的 span：`cache_hit / gate_transform / query_routing / multi_recall /
  rrf_fusion / rerank / crag_evaluation / generation`。
- 通过 `GET /api/traces`（瀑布图）与 `GET /api/traces/stats`（各阶段平均耗时）暴露给前端「观测台」Tab。

---

## 10. 评估体系

| 脚本 | 职责 | 是否依赖 LLM judge |
|------|------|--------------------|
| `run_eval.py` | RAGAS 四维（faithfulness / answer_relevancy / context_precision / context_recall） | 是 |
| `run_retrieval_eval.py` | CMRC 检索命中率 / 覆盖率 / 平均检索耗时 | **否**（与 LLM 解耦，作为检索质量决定性指标） |
| `run_e2e_eval.py` | 端到端三层 + A/B 特性归因 + 口语化定性检视 | 部分（忠实度用 LLM） |
| `run_concurrency_bench.py` | QPS / P50 / P95 / P99 / 错误率 / 瓶颈分析（并发 1/3/5/10/20） | — |

**RAGAS 四维自实现**（`app/evaluation/metrics.py`，不依赖 ragas 包）：

- `_eval_faithfulness`（`:94-132`）：LLM 抽 claim 逐条判 supported，`score = supported / 总`。
- `_eval_answer_relevancy`（`:137-175`）：按 1.0/0.8/0.6/0.4/0.0 档评分。
- `_eval_context_precision`（`:180-240`）：Weighted Precision@K = `Σ(precision@k · rel(k)) / num_relevant`。
- `_eval_context_recall`（`:245-277`）：拆标准答案为 claim 判 attributable，`recall = attributable / 总`。

> **测量严谨性**：换 LLM 会导致 RAGAS「生成模型 + 评判模型」同时变化、与历史基线不可比。
> 故以**与 LLM 解耦的 CMRC 检索评估**（命中率 100%）作为检索质量的决定性指标。

---

## 11. 配置开关速查（`config.py`）

`Settings(BaseSettings)` + `@lru_cache` 单例 `get_settings()`；优先级：环境变量 > `.env` > 默认值。

| 分类 | 关键配置（默认值） |
|------|--------------------|
| LLM / Embedding | `openai_model`、`embedding_provider[local]`、`embedding_model[all-MiniLM-L6-v2]` |
| 检索 | `retrieval_top_k[5]`、`retrieval_rrf_k[60]`、`rerank_top_n[20]`、`rerank_model` |
| 分块 | `chunk_size[512]`、`chunk_overlap[64]`、`semantic_chunk_threshold[0.75]` |
| Graph RAG | `graph_enabled[True]`、`graph_max_hops[2]`、`graph_max_entities[5]` |
| Parent-Child | `use_parent_child[True]`、`parent_chunk_size[1024]`、`child_chunk_size[200]` |
| CRAG | `use_crag[True]`、`use_crag_gate[True]`、`crag_relevance_threshold[0.5]` |
| 管道接线 | `use_summary_recall[True]`、`recall_max_workers[6]` |
| **F1 Autocut** | `use_autocut[True]`、`autocut_min_docs[2]` |
| **F2 迭代检索** | `use_iterative_retrieval[True]`、`max_retrieval_iterations[2]` |
| **F3 忠实度** | `use_faithfulness_check[True]`、`faithfulness_threshold[0.7]`、`faithfulness_max_regen[1]` |
| **F4 路由** | `use_query_router[True]` |
| **F6a 上下文增强** | `use_contextual_chunks[True]`、`contextual_max_chars[80]`、`chroma_contextual_collection[chunks_contextual]` |
| **F6b 查询分解** | `use_decomposition[True]`、`decomposition_max_subquestions[4]`、`decomposition_max_hops[3]` |
| **F7 引用溯源** | `use_citations[True]`、`citation_threshold[0.5]`、`citation_max_claims[6]` |
| **F8 投机流式** | `use_speculative_streaming[True]` |
| **F9 多级缓存** | `use_embedding_cache[True]`、`embedding_cache_size[512]`、`use_rerank_cache[True]`、`rerank_cache_size[256]` |
| **F10 答案增强** | `use_answer_focus[True]`、`use_answer_extraction[True]`、`use_self_consistency[False]`、`self_consistency_samples[3]` |
| **F11 可观测/加固** | `enable_metrics[True]`、`api_key[""]`、`rate_limit_rpm[0]`、`log_json[False]` |
| **F12 对话记忆** | `use_history_rewrite[True]`、`history_rewrite_use_llm[False]`、`history_rewrite_max_turns[4]` |
| 并发 | `max_concurrent_requests[4]`、`request_queue_timeout[30.0]` |
| 思考模式 | `llm_thinking_enabled[False]`（关思考避免 reasoning 吃 max_tokens） |
| 语义缓存 | `cache_enabled[True]`、`cache_threshold[0.92]`、`cache_ttl[3600]`、`cache_max_size[200]` |

> 注：`.env` 中 `RETRIEVAL_TOP_K=8` 为生产取值（高于默认 5），这也是 A/B 中 baseline `num_docs=8` 的来源。

---

## 12. 降级与容错设计

系统的一条核心准则是：**任何增强模块失败，都回退到安全默认值，绝不阻断主链路。**

| 模块 | 降级行为 | 位置 |
|------|----------|------|
| CRAG 门控 | 异常默认「需要检索」 | `pipeline.py:85-87` |
| CRAG 评估 | 异常默认 `correct` 放行 | `crag.py:132` |
| 忠实度自检 | 异常 / 不可解析返回 `faithful=None` 放行 | `faithfulness.py:86-88` |
| 查询精化 | 异常回退原问题 | `pipeline.py:316-319` |
| HyDE | 失败回退原查询 | `query_transform.py:136-138` |
| BM25 索引 | 加载失败自动重建 | `sparse.py:147-150` |
| Parent-Child | 批量查询失败回退逐个；无索引则跳过该路 | `parent_child.py:176-193` |
| 单路召回 | 任一通道失败仅告警，其余照常融合 | `pipeline.py:162-167` |
| **F7 引用** | 异常返回空引用列表，不影响答案 | `citation.py:CitationBuilder.build` |
| **F8 投机流式** | 检查/重生成异常透传原答案，`correction` 不发 | `streaming.py:speculative_faithful_stream` |
| **F9 多级缓存** | 缓存异常直落原编码/重排路径 | `caches.py` / `reranker.py` |
| **F10 答案增强** | 抽取失败 `short_answer=""`；自一致性异常用原答案 | `answer_boost.py` |
| **F11 加固** | 指标异常静默；鉴权/限流默认关不阻断 | `metrics.py` / `security.py` |
| **F12 重写** | 异常/无需重写回退原问题 | `conversation.py:ConversationRewriter.rewrite` |

**零 LLM 快速路径**（降低延迟与成本）：图实体匹配、数字校验、查询路由、Autocut、上下文压缩、语义缓存命中。

---

## 13. 技术选型

| 选择 | 理由 | 替代方案 |
|------|------|----------|
| ChromaDB | 轻量、本地持久化、零运维、Python 原生 | Milvus（重）、FAISS（无持久化） |
| LangChain | 生态完善、组件丰富 | LlamaIndex |
| FastAPI | 异步高性能、自动 OpenAPI、类型安全 | Flask（无异步） |
| Streamlit | 快速搭建可视化 demo、Python 原生 | React（开发成本高） |
| CrossEncoder (ms-marco / bge-reranker) | 本地免费、精度高 | Cohere Rerank（需 API 费用） |
| NetworkX | 十篇级文档量下零运维，无需图数据库 | Neo4j（重） |
| DeepSeek（关思考模式） | 成本低、中文强；`extra_body` 关思考避免吃 max_tokens | GPT-4o-mini |
| 自实现 RAGAS 四维 + 纯函数端到端指标 | 可控、可离线、避免跨模型口径混淆 | ragas 包 |

---

## 附：理解本系统的核心文件清单

1. `app/generation/chain.py` — 编排中枢
2. `app/retrieval/pipeline.py` — 七阶段检索管道（含 F2 迭代）
3. `app/api/routes.py` — HTTP 端点与并发闸门
4. `app/ingestion/indexer.py` — 层级索引（含 F6a 双集合 + `detail_store` 切换）
5. `app/retrieval/fusion.py` / `reranker.py` / `autocut.py` — 融合 / 重排 / F1 截断
6. `app/retrieval/crag.py` / `router.py` / `query_transform.py` — 评估 / F4 路由 / 改写（含 F6b `decompose`）
6b. `app/ingestion/contextual.py` — F6a Contextual Chunking 上下文生成（索引期 LLM）
7. `app/generation/faithfulness.py` / `prompts.py` — F3 忠实度自检与模板
7b. `app/generation/citation.py` / `streaming.py` / `answer_boost.py` — F7 引用 / F8 投机流式 / F10 答案增强
7c. `app/retrieval/caches.py` / `conversation.py` — F9 多级缓存 / F12 历史感知重写
7d. `app/observability/metrics.py` / `app/api/security.py` — F11 指标注册表 / 鉴权限流
8. `app/ingestion/graph_extractor.py` / `app/retrieval/graph_retriever.py` — 图谱
9. `config.py` — 全部开关与默认值
10. `app/observability/tracing.py` — 追踪
11. `app/evaluation/metrics.py` + `run_e2e_eval.py` — F5 评估
12. `frontend/app.py` — 前端五 Tab（对话 / 对比实验 / 观测台 / 索引透视 / 知识图谱）
