# 设计文档：检索管道接线补全 + 并发性能优化

- 日期：2026-07-22
- 状态：已获用户确认（方案 A：管道重构与接线同步进行）
- 范围：方向一（5 个半成品功能接线）+ 方向二（中度并发性能优化），合并为一个项目

## 1. 背景与目标

### 1.1 问题现状

代码探索发现主链路存在多处"做了一半"的功能：

1. `indexer.py` 的 `hierarchical_search()`（L1 摘要索引检索）已实现但无调用方——摘要索引建了不用
2. `crag.py` 的 `should_retrieve()` / `validate_numeric_answer()` 已实现但无调用方；`chain.py:60` 注释声称管道含 `[CRAG: Should Retrieve?]` 步骤，实际未接入
3. CRAG 补救链路不完整：incorrect 时 HyDE 重检索只做 dense 召回，且不经过 rerank 直接替换结果
4. `chain.py:177-182` 批量预计算的 `query_embeddings` 变量计算后未被使用
5. `semantic_chunk()` 已实现但上传 API 固定走递归分块，未暴露开关

性能方面：并发压测显示并发=10 时错误率 50%、P50 80s，根因是门控/改写/评估/生成多次 LLM 串行调用 + 无请求级并发控制。

### 1.2 成功标准（用户确认）

1. **检索质量指标提升**：context_recall 从 0.88 → ≥0.90，其余 RAGAS 指标不回退
2. **面试叙事素材**：每个改动沉淀为可讲的设计决策（为什么、权衡、数据验证）
3. **工程整洁度**：主链路无死代码、注释与实际行为一致、单测可离线运行、README 同步

注：并发性能改善是目标之一，但不设硬性达标线（用户未选此项作为成功标准），以压测前后对比数据呈现。

### 1.3 非目标（YAGNI）

- 不改 async 全链路（保留 ThreadPoolExecutor 召回模型）
- 不做语义缓存持久化、LLM 中间层（深度方案内容，本次不做）
- 不新增 RAG 范式特性（Agentic RAG / Self-RAG 等留待后续）

## 2. 架构设计

### 2.1 模块拆分

`chain.py`（649 行）目前混杂编排、召回、融合、评估、补救、压缩、缓存七种职责，拆分为：

```
app/generation/chain.py      → 保留：invoke / invoke_stream 编排、generate / generate_stream、
                                 compress_context、语义缓存检查与写入（提取公共私有方法消除重复）
app/retrieval/pipeline.py    → 新建 RetrievalPipeline：检索全链路七个阶段
```

`RetrievalPipeline` 通过构造函数注入全部组件（dense / sparse / graph / parent_child / indexer / transformer / crag / reranker），每个阶段是独立方法，可单独 mock 测试。`RAGChain.retrieve()` 变为对 pipeline 的薄委托，**对外签名不变**——routes.py、前端、评估脚本零改动。

### 2.2 新管道数据流

```
[缓存检查]
  → ① 门控 gate()         should_retrieve() 判断是否需要检索（闲聊/常识直接跳生成）
  → ② 改写 transform()    Multi-Query / HyDE（已有 1h TTL 缓存）
  → ③ 召回 recall()       五路并行：Dense + Sparse + Graph + Parent-Child + 【新】Summary(L1)
                           批量预计算 query_embeddings 真正分发给 Dense 各查询变体
  → ④ 融合 fuse()         RRF（五路输入）
  → ⑤ 重排 rerank()       CrossEncoder top-K
  → ⑥ 评估 evaluate()     【新】数字型问题先跑零 LLM 的 validate_numeric_answer() 快速路径，
                           缺失数字直接判 incorrect（省一次 LLM judge）；否则走 LLM 三档评级
  → ⑦ 补救 remediate()    incorrect → HyDE 重检索走【完整 mini-pipeline】：
                           改写 + dense/sparse 双路并行召回 → RRF → rerank
                           补救最多 1 次防循环；ambiguous → 过滤不相关文档（保持现状）
  → [生成] → [缓存写入]
```

### 2.3 四个接线点的具体做法

| 接线点 | 做法 | 配置开关（默认） |
|---|---|---|
| L1 摘要索引参与检索 | 复用 `indexer.hierarchical_search()` 作为第 5 路召回，结果进 RRF 融合 | `use_summary_recall`（True） |
| CRAG 门控 | `should_retrieve()` 放管道最前，false 时跳过 ②-⑦ 直接生成；调用失败默认检索 | `use_crag_gate`（True） |
| CRAG 补救链路补全 | HyDE 重检索复用 pipeline 的 recall/fuse/rerank 阶段（补 sparse 路 + rerank）；`validate_numeric_answer` 接入评估前置 | 复用现有 `use_crag` |
| 半成品优化收尾 | `DenseRetriever.retrieve()` 增加可选 `embedding` 参数复用预计算结果；上传 API 增加 `chunk_strategy=recursive/semantic` 参数 | — |

### 2.4 错误处理原则

- 每路召回独立 try/except 降级（保持现状），单路失败不影响整体
- 门控失败 → 默认需要检索（保守方向，不漏检索）
- 补救失败 → 保留原检索结果并在 `crag_action` 中标记
- 补救最多执行 1 次，防止评估-补救循环
- 所有降级路径写日志 + trace span，前端观测台可见

## 3. 并发与性能设计（中度方案）

### 3.1 请求级并发闸门

- `main.py` lifespan 创建全局 `asyncio.Semaphore`（新增配置 `max_concurrent_requests`，默认 4）
- `/api/chat` 入口 `async with semaphore:` 包裹，超出排队而非打爆 LLM
- 排队等待超过 30s 返回 503 + 友好提示

### 3.2 管道内消除串行

| 现状串行点 | 优化 |
|---|---|
| 门控① 与改写② 串行，各一次 LLM 调用 | 投机并行：两者同时发出，门控 false 时丢弃改写结果（浪费一次调用换 2-5s 延迟） |
| CRAG 评估⑥ 依赖重排⑤ 结果 | 保持串行（真实依赖），但数字型问题走零 LLM 快速路径跳过 LLM judge |
| 补救⑦ 内 HyDE 改写与重检索 | 补救 mini-pipeline 内部 dense/sparse 召回并行（复用阶段方法） |
| `invoke()` 与 `invoke_stream()` ~60 行重复缓存逻辑 | 提取公共私有方法 |

线程池大小配置化：新增 `recall_max_workers`（默认 6，五路召回后原 max_workers=4 不足）。

### 3.3 验证方式

`run_concurrency_bench.py` 跑改动前后对比（并发 1/5/10），记录错误率与 P50/P95，结果写入面试素材文档。

## 4. 测试与质量验证

### 4.1 离线单元测试（新增，不依赖 API key / 本地模型）

- `tests/test_pipeline.py`：门控三分支（true / false / 失败降级）、五路召回聚合与单路失败降级、补救链路（incorrect→HyDE 完整 mini-pipeline、最多 1 次、失败保留原结果）、数字校验快速路径
- `tests/test_dense_embedding.py`：预计算 embedding 复用（验证 embedding 调用次数从 N 降到 1）
- `tests/test_upload_chunk_strategy.py`：API 层 chunk_strategy 参数校验（mock RAGChain，不触发真实初始化）

### 4.2 质量验证（快速迭代 + 里程碑全量评估）

- 迭代期：`quick_evaluate()` 秒级回归（token F1 + 关键词覆盖率）
- 收尾：完整 RAGAS 四维评估（eval_dataset 12 题 + cmrc 31 题），目标 **context_recall ≥ 0.90**，faithfulness / precision 不回退；归档改动前基线报告
- 补回丢失的 CMRC 检索命中率评估脚本 `run_retrieval_eval.py`

### 4.3 回归保障

- 重构前先跑通现有 3 个测试文件作为基线
- `/api/retrieval/compare` 消融端点用于验证摘要召回路的贡献

## 5. 文档与面试素材

1. `docs/interview_guide.md`：Graph RAG 从"下一步"挪到已实现；新增 4 条设计决策记录：
   - 为什么摘要召回进 RRF 而不是前置路由
   - 门控投机并行的权衡
   - 补救链路为什么必须过 rerank
   - 并发闸门为什么用 Semaphore 而不是队列中间件
2. `README.md`：同步模型栈（qwen + bge 系列）、5 Tab 前端、模块结构、新增配置开关
3. `.env.example` 同步；确认 `.env` 未入库（密钥检查）

## 6. 实施顺序（概要）

1. 基线：跑通现有测试 + 归档当前评估报告
2. 重构：chain.py 拆出 pipeline.py（行为不变，配离线单测）
3. 接线：摘要召回路 → CRAG 门控 → 补救链路补全 → embedding 复用 + chunk_strategy
4. 并发：Semaphore 闸门 + 投机并行 + 线程池配置化
5. 验证：quick_evaluate 回归 → 全量 RAGAS 评估 → 并发压测对比
6. 文档：interview_guide / README / .env.example
