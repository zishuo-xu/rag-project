# F6 答案定位增强设计文档

> 日期：2026-07-23 分支：`feat/f6-answer-localization`
> 缘起：RAG 2.0 五特性（F1-F5）落地后，A/B 实测发现**检索命中率已饱和至 100%，但端到端 F1 几乎不动**（0.347→0.357）。
> 目标：把优化重心从「召不召得回」转向「**召回的内容里有没有答案、排序对不对**」，新增 F6 答案定位增强（两个支柱），
> 并用专门的**多跳 / 细粒度评估子集**验证收益（避免重蹈「饱和指标掩盖真实问题」的覆辙）。

---

## 0. 背景与问题定位

### 实测确认的硬问题

上一轮 A/B（`docs/superpowers/reports/2026-07-23-rag2-e2e-validation-report.md`）的关键发现：

- 检索命中率 100%（31/31），F1-F4 对端到端 F1 提升有限（0.347→0.357）——**命中率已饱和，不是瓶颈**。
- 18/31 子串未命中里，存在一类典型失败（id=2/4）：**源文档已召回，但承载答案的块没排进 top 上下文**。
  这正是文章三个深度问题里的「embedding 高相似但答案不在召回」。
- 真实收益来自上下文噪声 −48%、可量化忠实度、不可答诚实拒答；代价是延迟 +26%。

### 现状中的三个空洞

| 空洞 | 位置 | 说明 |
|------|------|------|
| Parent-Child 是死特性 | `routes.py:211` 上传端点未调 `parent_child.index_documents` | `has_index()` 恒 False，该召回路被静默跳过 |
| 块缺乏文档级上下文 | `chunker.py` 切出的块是「裸文本」 | 块向量不含文档语境，排序精度受限 |
| 多跳只识别不分解 | `router.py:_is_multi_hop` 识别后仅调大 `top_k` | 多跳问题仍被当单跳整句召回 |

### 特性映射

| # | 特性 | 层 | 对应问题 | 新/改文件 |
|---|------|----|----------|-----------|
| F6a | 细粒度召回 + 上下文增强 | 摄入/检索 | 「答案不在 top 块」 | `parent_child.py` 接线 + 新增 `contextual.py` + `indexer.py` |
| F6b | 多跳查询分解 | 检索 | 多跳被当单跳召回 | `query_transform.py` + `pipeline.py` 增强 |

### 设计原则（沿用项目既有风格）

- 每个支柱**独立配置开关**（默认开），可单独 A/B；关掉后行为与现状一致（保护既有 126 个测试）。
- 每个新单元**可独立测试**（mock 友好，离线运行）。
- 失败**优雅降级**：分解失败回退原问题、上下文生成失败回退裸块、Parent-Child 无索引则跳过该路。
- **F6a 零在线 LLM 增量**（上下文增强是索引时一次性）；**F6b 仅对多跳查询**触发，并行时约 +2 次 LLM 调用（链式按跳数增加，有硬上限）。
- 新增观测字段进 `RetrievalResult`，供评估与瀑布图使用。

---

## F6a · 细粒度召回 + 上下文增强

### 问题

命中率 100% 但答案段不在 top 块，根因有二：
1. **召回粒度错配**：dense/sparse 命中的小块（512 字递归块）可能恰好不含完整答案句，而答案在相邻文本里。
2. **块向量缺语境**：一个脱离文档上下文的块，其 embedding 无法表达「这是讲 X 的文档里关于 Y 的段落」，导致排序时与 query 的语义匹配不够精准。

### (a) 接线 Parent-Child（小块检索，大块返回）

`parent_child.py` 已完整实现，只差接线：

- **现状**：`ParentChildRetriever.index_documents`（parent=1024/overlap128，child=200/overlap30）、
  `retrieve`（child 检索 `top_k*3` → 按 `parent_id` 去重 → 批量取 parent 还原顺序）、`has_index()` 均已就绪；
  `RAGChain.__init__`（`chain.py:79-86`）也构造了它，`pipeline.recall` 已把它列为召回通道（需 `has_index()`）。
- **改动**：
  1. 上传端点 `routes.py:upload_document`（:211-266）在 `indexer.index_documents` 之后补
     `rag_chain.parent_child_retriever.index_documents(chunks)`（受 `settings.use_parent_child` 控制）。
  2. 提供**一次性重建入口**（如 `POST /api/documents/reindex` 或启动时检测），为已索引的 21 篇文档补建 child/parent 索引。
- **效果**：child（200）精确命中查询相关句，返回的 parent（1024）提供**包含答案段的宽窗口**——直接修复「答案不在 top 块」。
- **调参**：parent/child 尺寸沿用 1024/200；parent 仍会经过 rerank + Autocut，不会无脑放大噪声。

### (b) 上下文增强分块（Contextual Chunking，索引时一次性 LLM）

采用 Anthropic「Contextual Retrieval」思路，模型无关：

- **新增** `app/ingestion/contextual.py`：`generate_chunk_context(doc_text, chunk_text) -> str`
  用 LLM 生成一句简洁的文档级上下文（文档主题 + 本块在文中的定位，≤80 字）；失败返回空串（降级为裸块）。
- **索引时**：对每个块计算 `contextualized = context + "\n" + chunk_text`，**用 contextualized 去 embedding**，
  但 Chroma 中 `page_content` **仍存原文**（避免上下文前缀污染生成）。
  - 实现要点：`HierarchicalIndexer` 增加 `index_documents_contextual(chunks, contexts)`，
    手动 `embeddings.embed_documents([contextualized...])` 后 `collection.add(ids, embeddings, documents=[原文], metadatas)`，
    并把 context 存入 `metadata["context"]` 供观测。
  - 由 `settings.use_contextual_chunks` 选择查询哪个 collection：新增 `chunks_contextual` collection 与现有 `chunks` 并行，索引时两套都建；`HierarchicalIndexer.search_chunks` 按开关决定查询 `contextual_store` 还是 `chunk_store`，从而实现**无需重建即可 A/B**。
- **成本**：137 块 × 1 次 LLM，**仅索引时一次**；在线检索零增量。
- **效果**：块向量携带文档级语义 → 真正含答案的块排序上升。

### 数据流（F6a）

```
摄入：chunks → [generate_chunk_context ×N（一次性 LLM）] → contextualized 嵌入（存原文）+ Parent-Child 索引
检索：recall 五路（含 parent_child：child 命中→返回 parent）→ fuse → rerank → autocut
```

### 配置开关（F6a）

| 开关 | 默认 | 说明 |
|------|------|------|
| `use_parent_child` | True | 已存在；接线后真正生效 |
| `use_contextual_chunks` | True | 查询时使用上下文增强嵌入 |
| `contextual_max_chars` | 80 | 上下文片段长度上限 |

### 降级与边界

- 上下文生成失败 → 该块降级为裸块嵌入（不阻断索引）。
- Parent-Child 无 child 索引（`has_index()==False`）→ 跳过该召回路（现状行为）。
- 关闭 `use_contextual_chunks` → 查询裸块嵌入，行为与现状一致。

---

## F6b · 多跳查询分解

### 问题

`router.py:_is_multi_hop`（`的.{1,10}的` 关系链 + 疑问词）能识别多跳，但识别后只调大 `top_k`，
多跳问题仍被当**单跳整句**召回。需要组合多个事实（或链式依赖）的问题，单次整句召回往往召不全。

### 分解器设计

`query_transform.py` 新增：

- `DECOMPOSE_PROMPT`：让 LLM 输出 JSON
  `{"sub_questions": ["q1", "q2", ...], "chain": true|false}`，
  `chain=true` 表示子问题存在依赖链（后一跳需要前一跳的答案）。
- `decompose(question) -> Decomposition`（dataclass：`sub_questions: List[str]`, `chain: bool`）：
  - 解析 JSON（复用 `CRAGEvaluator._extract_json`）；
  - 子问题数裁剪到 `settings.decomposition_max_subquestions`（默认 4）；
  - **任何异常 / 解析失败 / 仅 1 个子问题 → 回退 `Decomposition([question], chain=False)`**（即退化为单跳，绝不阻断）。

### 检索与合并（并行优先，依赖时链式）

`pipeline.py:run` 在路由判定 `query_type == "multi_hop"` 且 `settings.use_decomposition` 时进入分解分支：

- **无依赖（`chain=False`）→ 并行**：
  - 各子问题走**轻量检索路径** `_retrieve_subquery(q)`：跳过逐子问题的门控与 multi-query 改写，
    直接以子问题做 dense+sparse 召回（复用现有召回通道）→ 子问题内 fuse。
  - `ThreadPoolExecutor` 并行执行所有子问题（复用 `recall_max_workers`）。
  - 所有子问题结果**统一 RRF 合并**成一个候选集。
- **有依赖（`chain=True`）→ 链式**：
  - 按序检索第一跳 → 用一次轻量 LLM 抽取拿到 hop-1 的关键事实/实体 → 代入下一跳模板构造 hop-2 查询 → 检索……
  - 受 `settings.decomposition_max_hops`（默认 3）硬上限保护。
- 合并后的候选集**继续走主管道**：rerank → autocut → **一次** CRAG evaluate（不合格走 F2 迭代/补救）。

### 延迟控制（关键约束）

- **并行（默认）**：子问题用轻量路径（无逐子问题门控/改写/评估），额外 LLM 调用约束为
  **≈ 1 次分解 + 1 次最终 CRAG 评估**；子问题召回是 embedding/BM25（无 LLM）且并发，延迟≈最慢单子问题。
- **链式（仅依赖时）**：每跳需一次轻量 LLM 抽取 hop 答案以构造下一跳，额外调用 ≈ 1 分解 +（抽取 × 跳数）+ 1 最终评估，
  受 `decomposition_max_hops` 硬上限约束；仅在分解器明确标出依赖链时启用，避免默认串行拖慢。

### 与 F2 的关系

F2 是「**纵向**精化同一问题」（refine→重召回），F6b 是「**横向**拆成多个子问题」。二者正交共存：
先横向分解并合并，合并后的候选集仍可由 F2 的 evaluate/迭代/补救兜底。

### 数据流（F6b）

```
multi_hop 问题 → decompose（1 次 LLM）
  ├─ chain=False：子问题并行轻量召回 → 各自 fuse → 统一 RRF 合并
  └─ chain=True ：hop1 召回→抽关键信息→hop2…（≤max_hops）→ 合并
→ rerank → autocut → 一次 CRAG evaluate →（不合格）F2 迭代/补救 → 生成
```

### 配置开关（F6b）

| 开关 | 默认 | 说明 |
|------|------|------|
| `use_decomposition` | True | 多跳查询是否分解 |
| `decomposition_max_subquestions` | 4 | 子问题数上限 |
| `decomposition_max_hops` | 3 | 链式分解跳数硬上限 |

### 降级与边界

- 分解失败 / 仅 1 子问题 → 退化为单跳整句召回（现状行为）。
- 链式抽取 hop 信息失败 → 用上一跳原始查询继续，不中断。
- 关闭 `use_decomposition` → 多跳问题走现状（仅调大 top_k）。

### 观测字段（进 `RetrievalResult`）

`decomposed_subqueries: List[str]`、`decomposition_chain: bool`、`answer_localization_method: str`
（如 `"parent_child"` / `"contextual"` / `""`），供评估与瀑布图。

---

## 评估设计（吸取「指标饱和」教训）

**核心原则**：命中率已饱和，必须用**专门子集**才能测出 F6 的收益，否则又会被 100% 命中率掩盖。

### 多跳测试集（新建）

- 从知识库（21 篇文档）人工/半自动构造 **10-15 道 2-hop 题** + gold 答案，
  覆盖「并行组合」与「链式依赖」两类（如「X 的 Y 的 Z」）。
- 存 `data/eval_multihop.json`（字段：`question / gold / hops / chain`）。

### 细粒度子集（新建）

- 从 CMRC 31 题中挑出 id=2/4 类「**源命中但答案不在 top 块**」的题，标注为细粒度子集，
  额外记录「答案段所在 chunk_id」，用于直接度量**答案段是否进入 top 上下文**（新指标 `answer_in_top_context`）。

### harness 复用与 A/B

- `run_e2e_eval.py` 的 `FEATURE_FLAGS` 增加 `F6: use_decomposition`；
  F6a 的上下文增强通过 `use_contextual_chunks` 切换查询的嵌入集合做 A/B，Parent-Child 通过 `use_parent_child` 切换。
- 新增切片 `--slice multihop|finegrained`，对比 baseline vs F6 的：
  - `answer_f1` / `answer_hit`（端到端）
  - `answer_in_top_context`（细粒度专属：答案段是否进 top 上下文）
  - 检索命中率（**回归保护：必须保持 100%**）
  - 延迟（多跳查询的额外开销）

---

## 测试策略（TDD + 回归保护）

- **先写测试再实现**，新增：
  - `tests/test_decomposition.py`：decompose 正常/JSON 异常/单子问题回退/子问题裁剪/链式标记。
  - `tests/test_contextual_chunking.py`：context 生成/失败降级裸块/contextualized 嵌入但存原文。
  - `tests/test_parent_child_wiring.py`：上传后 `has_index()==True`、child 命中返回 parent。
  - `tests/test_pipeline_multihop.py`：multi_hop 分支并行合并/链式/分解失败退化/关闭走现状。
- **回归保护**：新开关在 `test_pipeline.py` 的 mock 里**默认关闭**（沿用 F1-F5 的保护模式），确保既有 126 测试全绿。
- 评估脚本变更配套纯函数测试（如 `answer_in_top_context` 计算）。

---

## 实施顺序与风险

### 顺序

1. **Pillar 1（F6a）先行**：风险低、接线死特性、零在线延迟。
   - 先 Parent-Child 接线（含重建入口）→ 验证命中率不掉、细粒度子集 `answer_in_top_context` 提升。
   - 再上下文增强分块（新增 contextual.py + indexer 双嵌入 + 重建索引）→ 回归验证命中率 100%。
2. **Pillar 2（F6b）后续**：分解器 → 管道并行/链式分支 → 多跳评估集 → A/B。

### 风险

| 风险 | 缓解 |
|------|------|
| 上下文增强改变全部块嵌入，命中率可能波动 | 重建索引后强制回归 CMRC，命中率 <100% 则回滚该支柱 |
| 多跳评估集质量决定结论可信度 | 人工核对 gold 答案，覆盖并行/链式两类 |
| 链式分解串行延迟高 | 默认并行，仅明确依赖时链式；`max_hops` 硬上限 |
| Parent 过大引入噪声 | parent 仍经 rerank + Autocut；必要时调小 parent_chunk_size |
| 双嵌入索引翻倍存储 | 21 篇文档量级可接受；规模化时再评估按开关只建一套 |

---

## 备选方案（已否决）

| 方案 | 否决理由 |
|------|----------|
| Late Chunking（Jina） | 需特定长上下文 embedding 模型；Contextual Chunking 模型无关、可离线 |
| 纯规则分解（无 LLM） | 表达能力有限，复杂多跳拆不准；改用 LLM 分解 + 规则回退 |
| 句级答案抽取（生成侧） | 与检索侧定位正交；本期先做检索侧定位，句级抽取列为后续 |
| 只做分解不做细粒度 | 用户选择两者一起做；细粒度直接修复实测的 id=2/4，杠杆高且零在线延迟 |
