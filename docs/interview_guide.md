# RAG 系统面试与学习指南

## 一、项目定位（30秒电梯演讲）

> "这是一个**生产级 RAG 系统**，覆盖文档摄入→多路检索→融合重排→LLM生成→自动评估全链路。
> 核心亮点是 **Dense+BM25 多路召回 + RRF 融合 + Cross-Encoder 精排** 的三级检索架构，
> 配合 **RAGAS 四维评估**和**内置 Trace 可观测性**，实现了可量化、可追踪的检索增强生成。"

---

## 二、系统架构全景

### 2.1 请求生命周期

```
用户问题
    │
    ▼
┌─────────────────┐
│  Query Transform │  ← Multi-Query: 生成 3 个查询变体
│  (可选)          │  ← HyDE: 生成假设性文档
└────────┬────────┘
         │ queries: [q1, q2, q3, q_original]
         ▼
┌─────────────────────────────────────────┐
│           Multi-Recall (多路召回)         │
│  ┌─────────────┐    ┌─────────────┐     │
│  │ Dense       │    │ Sparse      │     │
│  │ (Embedding  │    │ (BM25+jieba)│     │
│  │  + Chroma)  │    │             │     │
│  └──────┬──────┘    └──────┬──────┘     │
│         ▼                   ▼            │
│    top-20 docs         top-20 docs       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   RRF Fusion    │  ← score(d) = Σ 1/(k + rank_i(d)), k=60
└────────┬────────┘
         │ ~30 个候选
         ▼
┌─────────────────┐
│  Cross-Encoder  │  ← (query, doc) 对 → 精排 top-5
│  Reranker       │
└────────┬────────┘
         │ top-5 docs
         ▼
┌─────────────────┐
│  LLM Generation │  ← 构建 context + prompt → 流式输出
└────────┬────────┘
         ▼
    最终回答 + 引用来源 + Trace 记录
```

### 2.2 技术栈一览

| 层级 | 技术 | 选型理由 |
|------|------|----------|
| 编排框架 | LangChain | 生态完善、组件丰富、面试认可度高 |
| 后端 API | FastAPI | 异步高性能、自动 OpenAPI 文档、类型安全 |
| 前端 | Streamlit | 快速搭建 Demo、Python 原生、4个Tab可视化 |
| LLM | DeepSeek（OpenAI 兼容接口）| 中文能力强、可切换任意兼容服务、支持思考模式开关 |
| Embedding | BAAI/bge-small-zh-v1.5（本地）| 中文优化、零成本、低延迟 |
| 向量数据库 | ChromaDB | 轻量、本地持久化、零运维 |
| 重排序 | BAAI/bge-reranker-base（本地）| 中文 cross-encoder、免费本地运行、精度好 |
| 稀疏检索 | rank-bm25 + jieba | 中文词级分词、精确术语匹配 |
| 评估 | 自实现 RAGAS 四维 + CMRC 检索命中 | LLM-as-Judge + 零 LLM 检索评估双轨 |
| 可观测性 | 自研轻量 Tracer | 线程安全、瀑布图、阶段耗时统计 |

### 2.3 设计原则（面试必答）

| 原则 | 体现 |
|------|------|
| **分层解耦** | 摄入/检索/生成/评估各层独立，可单独替换测试 |
| **策略可插拔** | 分块策略、检索策略、改写策略均可通过配置切换 |
| **可观测性** | 每次调用自动记录 6 阶段耗时，支持瀑布图分析 |
| **渐进增强** | 基础 RAG 可运行，Rerank/Query Transform 可开关 |

---

## 三、核心模块深度解析

### 3.1 文档摄入管道

**文件**: `app/ingestion/loader.py` → `chunker.py` → `indexer.py`

#### 分块策略对比

| 策略 | 原理 | 适用场景 | 实现要点 |
|------|------|----------|----------|
| 递归字符分块 | 按 `\n\n` > `\n` > `。` > ` ` 层级切分 | 通用文档 | chunk_size=512, overlap=64 |
| 语义分块 | 相邻句子 embedding 余弦相似度 < 阈值处切分 | 主题频繁变化 | threshold=0.75 |

**面试话术**: "我实现了两种分块策略。递归分块适合结构规整的文档，语义分块通过计算相邻句子的 embedding 相似度来检测语义跳变点，在主题交叉的长文档上效果更好。"

#### 层级索引设计

```
L1 摘要索引 (summaries collection)
  └── LLM 为每篇文档生成 200 字摘要 → 向量化
  └── 用途：粗粒度定位相关文档

L2 明细索引 (chunks collection)
  └── 原始分块 → 向量化
  └── 用途：细粒度段落检索
```

**检索流程**: 先 L1 定位文档 → 再 L2 在目标文档内精细检索（`filter={"doc_id": {"$in": target_ids}}`）

### 3.2 多路检索

**文件**: `app/retrieval/dense.py`, `sparse.py`, `fusion.py`, `reranker.py`

#### 为什么需要多路召回？

| 检索方式 | 优势 | 劣势 | 典型场景 |
|----------|------|------|----------|
| Dense (向量) | 理解语义、同义词 | 对精确术语/编号弱 | "如何提升系统吞吐量" |
| Sparse (BM25) | 精确关键词匹配 | 无法理解语义关系 | "B+树的度是多少" |
| **混合** | **互补，提高召回率** | 需要融合策略 | 通用 |

#### RRF 融合公式

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

- `k = 60`（平滑常数，减少高排名的过度优势）
- **只用排名不用原始分数** → 避免 Dense 和 BM25 分数不可比
- 多路都命中的文档自然获得更高分数

**面试话术**: "RRF 的核心思想是只利用排名信息做融合，不依赖各路检索的原始分数。因为向量相似度是 0~1，BM25 分数可能是 0~30，直接加权没有意义。RRF 通过倒数排名来归一化，k=60 是论文推荐的经验值。"

#### Cross-Encoder vs Bi-Encoder

| 维度 | Bi-Encoder | Cross-Encoder |
|------|-----------|---------------|
| 输入 | query 和 doc 分别编码 | (query, doc) 拼接后一起编码 |
| 交互 | 无交互（余弦相似度） | 全交互（交叉注意力） |
| 速度 | 快（可预计算） | 慢（每对都要推理） |
| 精度 | 较低 | **高** |
| 用途 | 大规模初筛 | 少量候选精排 |

**面试话术**: "Bi-Encoder 快但粗，Cross-Encoder 慢但准。所以我的架构是先用 Bi-Encoder（向量检索）从几百个文档中召回 30 个候选，再用 Cross-Encoder 精排到 5 个。这样既保证了速度又保证了精度。"

#### BM25 中文分词

```python
# 使用 jieba 词级分词（非单字），过滤停用词
tokens = jieba.lcut(text.lower())
# 保留：英文/数字完整单词 + 中文≥2字的词
```

### 3.3 查询改写

**文件**: `app/retrieval/query_transform.py`

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| Multi-Query | LLM 生成 3 个不同角度的查询变体 | 用户问题模糊、角度单一 |
| HyDE | LLM 生成假设性回答，用其 embedding 检索 | 问题与文档措辞差异大 |

**面试话术**: "Multi-Query 解决的是用户措辞不够精确的问题，通过生成多个不同视角的查询来扩大召回面。HyDE 解决的是问题和文档在语义空间中距离远的问题，因为假设性回答在措辞上更接近真实文档。"

### 3.4 生成模块

**文件**: `app/generation/chain.py`, `prompts.py`

#### Prompt 设计要点

```
1. 仅基于文档回答（不编造）
2. 标注来源 [来源: 文档名]
3. 信息不足时诚实拒答
4. 聚焦问题本身（不扩展）
5. 结构清晰（列表/分段）
```

#### 流式输出 (SSE)

```
event: retrieval   → 检索元数据（查询变体、各路命中数、耗时）
event: token       → 逐 token 流式输出
event: done        → 完成信号 + 来源列表 + 总耗时
```

### 3.5 评估模块

**文件**: `app/evaluation/metrics.py`

#### RAGAS 四维指标

| 指标 | 评估对象 | 含义 | 实现方式 |
|------|----------|------|----------|
| Faithfulness | 生成质量 | 回答是否忠于上下文（不幻觉） | LLM 提取声明 → 逐一验证 |
| Answer Relevancy | 生成质量 | 回答与问题的相关程度 | LLM 直接评判切题度 |
| Context Precision | 检索质量 | 相关文档是否排在前面 | 加权 Precision@K |
| Context Recall | 检索质量 | 是否检索到了所有相关信息 | 标准答案声明 → 上下文归因 |

#### 实际评估结果

**检索质量（CMRC，零 LLM，决定性指标）**：

| 指标 | 得分 | 目标 | 状态 |
|------|------|------|------|
| 命中率 | **100%（31/31）** | ≥90.32% | ✓ |
| 关键词覆盖率 | **100%** | — | ✓ |
| 平均检索耗时 | ~5.1s/题 | — | 检索（embedding+rerank）为冷查询瓶颈 |

**生成质量（RAGAS 四维，DeepSeek 生成+评判，20 题）**：

| 指标 | 得分 | 说明 |
|------|------|------|
| Faithfulness | 0.82 | |
| Answer Relevancy | 0.32 | 关思考模式后答案偏简短所致（已对照验证 judge 正常）|
| Context Precision | 0.85 | |
| Context Recall | 0.82 | |

> ⚠️ **口径 caveat**：RAGAS 基线由 qwen 生成+评判，本轮换成 DeepSeek，**不同口径不可直接对比**。
> 检索质量以与 LLM 解耦的 CMRC 评估为准（100%）；RAGAS 反映的是「DeepSeek+关思考」配置下的生成质量。
> 详见 [验证报告](./superpowers/reports/2026-07-22-task11-validation-report.md)。

**面试话术**: "我用双轨评估：RAGAS 四维（LLM-as-Judge）衡量生成质量，CMRC 命中评估（零 LLM）衡量检索质量。换 LLM 后我特意用 CMRC 这个与模型解耦的尺子来公平地证明检索接线有效——命中率 100%。同时我诚实标注了 RAGAS 的跨模型口径问题，不拿不可比的数字直接报喜。"

### 3.6 可观测性

**文件**: `app/observability/tracing.py`

```
每次问答自动记录：
  query_transform → dense_retrieval → sparse_retrieval → rrf_fusion → rerank → generation

提供：
  - 单次调用瀑布图（各阶段起止时间、耗时、元数据）
  - 汇总统计（总调用数、平均耗时、瓶颈阶段）
  - 最近 50 次调用历史
```

### 3.7 检索对比实验

**文件**: `app/api/routes.py` → `/api/retrieval/compare`

同一问题并行执行 4 种策略：
1. Dense Only（向量检索）
2. Sparse Only（BM25 关键词）
3. Hybrid RRF（融合）
4. Hybrid + Rerank（最终方案）

返回各策略的**耗时、命中数、与最终结果重叠度**，直观证明每一层组件的增益。

### 3.8 统一检索管道与并发治理（本次迭代核心）

这是本项目第二轮迭代的主体工作，包含 4 个可作为面试叙事的设计决策。

#### 决策一：把检索重构为统一的 7 阶段管道 `RetrievalPipeline`

**问题**：原本检索逻辑散落在 `chain.py` 里，查询改写、多路召回、融合、重排、CRAG 耦合在一个大方法中，难以单独测试和扩展。

**做法**：抽出 `app/retrieval/pipeline.py`，把检索定义为 7 个职责单一的阶段：

```
gate(要不要检索) → transform(查询改写) → recall(多路并行召回) → fuse(RRF)
→ rerank(精排) → evaluate(CRAG分级) → remediate(不合格则补救)
```

`chain.py` 退化为**薄编排层**，只负责「调 pipeline 拿文档 → 拼 prompt → 生成」，`retrieve()` 委托给 `pipeline.run()`。

**面试话术**：「我把检索从生成链里解耦成独立的 `RetrievalPipeline`，7 个阶段各自可测、可开关、可替换。这样 CRAG 补救这种『管道中再嵌一条小管道』的逻辑才有清晰的落点，而不是往大方法里继续堆 if。」

#### 决策二：把 5 个『半成品』特性真正接进主管道

**问题**：代码库里已有 Summary 摘要索引、CRAG 门控、CRAG 补救、Parent-Child、Graph 等模块，但大多**写了没接进主链路**（dead feature）。

**做法**：用配置开关把它们逐一接入 `RetrievalPipeline`，并补离线行为测试：

| 特性 | 接入方式 | 开关 |
|------|----------|------|
| Summary 召回 | 作为**第 5 路**参与并行召回 + RRF | `use_summary_recall` |
| CRAG 门控 | `gate()` 判断是否需要检索，可跳过检索直接生成 | `use_crag_gate` |
| CRAG 补救 | `evaluate()` 分级后，不合格走 `remediate()`（HyDE+重召回+RRF+重排）| `use_crag` |
| Parent-Child / Graph | 作为召回通道并入 `ALL_CHANNELS` | `use_parent_child` / `graph_enabled` |

**验证**：与 LLM 解耦的 CMRC 检索评估 **命中率 100%（31/31）、覆盖率 100%**。

**面试话术**：「接半成品比写新功能更难，因为要在不破坏既有行为的前提下把它们织进主链路。我用配置开关 + 离线行为测试（含 mutation testing）保证每接一个特性都可验证、可回滚。」

#### 决策三：并发治理——闸门 + 解除事件循环阻塞 + 排队超时

**问题**：`/api/chat` 是 CPU（embedding/rerank）+ IO（LLM）混合负载，高并发下会打满事件循环和下游 LLM。

**做法**：
1. **`asyncio.Semaphore` 闸门**：`max_concurrent_requests` 限制同时在途请求数（lifespan 中创建）
2. **`asyncio.to_thread`**：把同步的检索/生成放到线程池，**解除对事件循环的阻塞**
3. **排队超时 503**：`request_queue_timeout` 内拿不到闸门信号量就返回 503，快速失败而非无限堆积
4. **语义缓存**：embedding 相似度命中缓存，重复查询毫秒级返回，吸收热点负载

**面试话术**：「FastAPI 是异步的，但我的检索是同步阻塞调用，直接 await 会卡死事件循环。所以我用 `to_thread` 卸载到线程池，再用 Semaphore 做准入控制、超时返回 503 做快速失败。语义缓存则让重复查询绕过整条管道。」

#### 决策四：eval 驱动 + 测量严谨（识别跨模型混淆）

**问题**：迭代中途把 LLM 从 qwen 换成 DeepSeek，导致 RAGAS 四维「生成模型 + 评判模型」同时变化，**与历史基线不可比**。

**做法**：
1. 识别出混淆后，改用**与 LLM 解耦的 CMRC 检索评估**作为检索质量的决定性指标（命中率 100%）
2. 对 `answer_relevancy=0.32` 的异常低值做**对照实验**：用好答案测 judge 能正确打 1.0 → 证明不是 bug，而是「关思考模式后答案偏简短」的真实信号
3. 落地 **思考模式开关** `llm_thinking_enabled`（DeepSeek reasoning 模型用 `extra_body={"thinking":{"type":"disabled"}}` 关闭，避免 reasoning_content 吃掉 max_tokens）

**面试话术**：「换模型后 RAGAS 全线下跌，我没有直接报喜或报忧，而是先定位这是『换模型导致的不同口径』混淆，再用无 LLM 的 CMRC 评估给出检索质量的公平结论。评估指标本身也要被审视，不能拿来就用。」

---

## 四、工程化亮点

### 4.1 并发性能评测

**文件**: `run_concurrency_bench.py`

- 使用 asyncio + httpx 模拟 1/3/5/10/20 并发
- 输出 QPS、P50/P95/P99 延迟、错误率
- 自动瓶颈分析（检索 vs 生成占比）
- 性能评级（S/A/B/C/D）

### 4.2 前端四 Tab 可视化

| Tab | 功能 |
|-----|------|
| 💬 对话问答 | 流式对话 + 来源引用 + 检索链路详情 |
| 🧪 对比实验 | 4 种检索策略并行对比 |
| 📊 观测台 | Trace 瀑布图 + 阶段耗时统计 |
| 🔬 索引透视 | 分块内容 + Embedding 向量 + BM25 词项 + L1 摘要 |

### 4.3 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_retrieval.py` | RRF 融合、加权融合、BM25 中英文分词 |
| `test_chunker.py` | 递归分块、重叠验证、元数据保留 |
| `test_api.py` | 健康检查、参数校验、文档上传格式校验 |

### 4.4 评估数据集设计

- **12 道题**覆盖 10 个技术领域
- 故意引入**主题交叉**（如"缓存"出现在 Redis/系统设计/Python 性能文档中）
- 包含 easy/medium/hard 三个难度
- 包含 single_doc 和 cross_doc 两种类型

---

## 五、面试高频 Q&A

### Q1: "为什么选择 RRF 而不是直接加权分数？"

> Dense 检索的余弦相似度范围是 [0,1]，BM25 分数可能是 [0,30+]，两者量纲完全不同，直接加权没有意义。RRF 只利用排名信息，天然解决了分数不可比问题。而且 RRF 有一个很好的性质：多路都命中的文档会自动获得更高分数，不需要手动调权重。

### Q2: "Cross-Encoder 这么慢，为什么不直接用 Bi-Encoder？"

> 精度差距很大。Bi-Encoder 是 query 和 doc 独立编码，没有交互；Cross-Encoder 是拼接后做交叉注意力，能捕获细粒度的语义匹配。但 Cross-Encoder 每对都要推理一次，所以只能对少量候选用。我的架构是 Bi-Encoder 初筛 30 个 → Cross-Encoder 精排到 5 个，兼顾速度和精度。

### Q3: "查询改写不会引入噪声吗？"

> 会。所以我有三个保护措施：
> 1. 始终包含原始查询（不会被改写结果完全替代）
> 2. 多路结果经过 RRF 融合和 Rerank，噪声文档会被自然淘汰
> 3. 可以通过配置关闭查询改写（渐进增强原则）

### Q4: "如何评估 RAG 系统效果？"

> 我使用 RAGAS 框架的四维指标：
> - **Faithfulness**（忠实度）：回答是否编造了文档中没有的内容
> - **Answer Relevancy**（相关性）：回答是否切题
> - **Context Precision**（精确度）：相关文档是否排在前面
> - **Context Recall**（召回率）：是否检索到了所有必要信息
>
> 实现方式是 LLM-as-Judge，让 LLM 提取回答中的声明逐一验证。在 12 题测试集上四项指标均 >0.9。

### Q5: "中文检索效果怎么保证？"

> 三个层面：
> 1. **分词**：BM25 使用 jieba 词级分词（非单字），过滤单字停用词
> 2. **测试集**：故意设计高干扰性中文问题（如"缓存穿透/击穿/雪崩"），验证语义精准理解
> 3. **主题交叉**：同一关键词出现在多篇文档中（如"缓存"在 Redis/系统设计/Python 性能），验证抗干扰能力

### Q6: "为什么用 ChromaDB 而不是 Milvus/FAISS？"

> 项目定位是展示 RAG 工程实践，ChromaDB 的优势是零运维、本地持久化、Python 原生集成。如果是生产环境需要亿级数据，会考虑 Milvus（分布式）或 Qdrant（Rust 实现，性能好）。FAISS 没有内置持久化和元数据过滤，需要额外封装。

### Q7: "系统的瓶颈在哪里？怎么优化？"

> 通过并发 bench 的瓶颈分析（冷查询，并发=1）：
> 1. **检索阶段是主要瓶颈**（~5s，占总延迟 84%）：本地 embedding + cross-encoder rerank 是 CPU 密集计算，rerank 尤其耗时
> 2. **生成阶段**（~1s）：DeepSeek 关思考模式后较快，已用流式输出优化体感
>
> 已落地的优化：
> - **语义缓存**：重复查询命中缓存，P95 从 ~8.9s 压到 ~53ms（bench 实测）
> - **多路召回并行**：线程池并发执行 5 路召回（`recall_max_workers`）
> - **to_thread 卸载**：同步检索不阻塞事件循环，保证高并发吞吐
>
> 进一步优化方向：reranker 蒸馏/量化、embedding 批处理、GPU 加速、热门查询预热。

### Q8: "如果让你重新设计，会怎么改？"

> 已经做了的（第二轮迭代）：
> - **Graph RAG**：零 LLM 快速构建知识图谱，实体多跳召回（作为召回通道之一）
> - **Parent-Child / CRAG 门控与补救 / 语义缓存**：均已接入统一管道
>
> 下一步想做的：
> 1. **自适应检索**：根据 query 复杂度动态决定是否改写、是否多路、是否精排（CRAG 门控是雏形）
> 2. **Agentic RAG**：引入 Tool-Use Agent，支持多步推理与迭代检索
> 3. **检索延迟优化**：reranker 蒸馏/量化、GPU 加速（当前冷查询瓶颈在 rerank）
> 4. **向量库升级**：迁移到 Qdrant/Milvus 支持规模化
> 5. **评估闭环**：线上反馈 → 自动标注 → 策略迭代

---

## 六、项目文件索引

| 文件 | 职责 | 代码行数 |
|------|------|----------|
| `main.py` | FastAPI 入口、生命周期管理 | 83 |
| `config.py` | pydantic-settings 配置管理 | 52 |
| `app/ingestion/loader.py` | 多格式文档加载（PDF/TXT/MD） | 90 |
| `app/ingestion/chunker.py` | 递归分块 + 语义分块 | 157 |
| `app/ingestion/indexer.py` | 层级索引（L1摘要 + L2明细） | 253 |
| `app/ingestion/graph_extractor.py` | 知识图谱抽取（零 LLM） | 649 |
| `app/retrieval/pipeline.py` | ★ RetrievalPipeline 7 阶段统一编排 | 359 |
| `app/retrieval/dense.py` | 向量检索 | 53 |
| `app/retrieval/sparse.py` | BM25 + jieba 中文分词 | 123 |
| `app/retrieval/graph_retriever.py` | Graph 多跳召回 | 363 |
| `app/retrieval/parent_child.py` | Parent-Child 检索 | 207 |
| `app/retrieval/fusion.py` | RRF 融合 + 加权融合 | 106 |
| `app/retrieval/reranker.py` | Cross-Encoder 精排 | 112 |
| `app/retrieval/query_transform.py` | Multi-Query + HyDE | 134 |
| `app/retrieval/crag.py` | CRAG 门控/评估/补救 | 191 |
| `app/retrieval/cache.py` | 语义缓存 | 159 |
| `app/generation/chain.py` | RAG 薄编排层（委托 pipeline + 生成 + Trace） | 422 |
| `app/generation/prompts.py` | Prompt 模板 | 61 |
| `app/observability/tracing.py` | 轻量级 Trace（线程安全） | 146 |
| `app/api/routes.py` | 全部 API 端点 | 460 |
| `app/api/schemas.py` | Pydantic 请求/响应模型 | 105 |
| `app/evaluation/metrics.py` | RAGAS 四维评估（LLM-as-Judge） | 403 |
| `app/evaluation/dataset.py` | 评估数据集管理 | 123 |
| `frontend/app.py` | Streamlit 前端（4 Tab） | 793 |
| `run_eval.py` | RAGAS 四维评估脚本 | 54 |
| `run_retrieval_eval.py` | CMRC 检索命中评估（零 LLM） | 90 |
| `run_concurrency_bench.py` | 并发性能评测 | 315 |

---

## 七、面试叙事结构（建议 5 分钟）

```
1. 项目定位（30s）
   "生产级 RAG 系统，覆盖摄入→检索→生成→评估全链路"

2. 核心设计决策（2min）
   ① 多路召回为什么必要 → 消融实验数据支撑
   ② 统一检索管道 RetrievalPipeline（7阶段重构）+ 5 个半成品接入
   ③ 并发治理（Semaphore闸门 + to_thread + 503）
   ④ 评估怎么做 → RAGAS 四维 + CMRC 零LLM检索评估双轨

3. 工程亮点（1min）
   ① 内置 Trace 可观测 / 瀑布图
   ② 检索对比实验（4策略并行）
   ③ 并发性能评测（QPS/P95/瓶颈分析）
   ④ 前端 4 Tab 全链路可视化

4. 评估结果（30s）
   "CMRC 检索命中率 100%（31/31）；并诚实标注 RAGAS 的跨模型口径问题"

5. 扩展思考（30s）
   "下一步想做自适应检索 / Agentic RAG / reranker 延迟优化"
```

---

## 八、扩展路线图

### 第一梯队：体现技术深度

| 方向 | 核心价值 | 关键技术 |
|------|----------|----------|
| 自适应检索 | 根据 query 复杂度动态选策略 | 分类器/规则引擎 |
| Graph RAG | 关系推理、多跳问答 | Neo4j + 知识图谱构建 |
| Agentic RAG | 多步推理、Tool-Use | LangGraph / ReAct |
| 评估闭环 | 线上反馈→策略迭代 | 自动标注 + A/B 测试 |

### 第二梯队：体现工程成熟度

| 方向 | 核心价值 | 关键技术 |
|------|----------|----------|
| Redis 缓存层 | 降低 LLM 调用成本 | 语义相似度缓存 |
| 异步任务队列 | 大文档不阻塞 API | Celery / arq |
| 多租户隔离 | SaaS 化 | Collection 隔离 + API Key |
| 向量库迁移 | 规模化 | Qdrant / Milvus |

### 第三梯队：差异化亮点

| 方向 | 核心价值 | 关键技术 |
|------|----------|----------|
| 多模态 RAG | 图片/表格检索 | ColPali / 多模态 Embedding |
| 知识蒸馏 Reranker | 大模型标注→小模型精排 | 教师-学生训练 |
| 在线 A/B 测试 | 真实效果对比 | 流量分桶 + 指标收集 |
