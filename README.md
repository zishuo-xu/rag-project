# RAG 智能问答系统

一个生产级检索增强生成（RAG）系统，展示 RAG 全链路的工程实践。

## 文档导航

| 文档 | 说明 |
|------|------|
| [README.md](./README.md) | 项目概览与快速开始 |
| [docs/architecture.md](./docs/architecture.md) | 架构设计文档（数据流、模块设计、技术选型理由） |
| [docs/api.md](./docs/api.md) | API 接口详细文档（请求/响应示例） |
| [docs/interview_guide.md](./docs/interview_guide.md) | 面试与学习指南（技术深度解析、高频Q&A、扩展路线） |
| [API Swagger UI](http://localhost:8000/docs) | 启动服务后的交互式 API 文档 |

## 系统架构

检索管道统一在 `RetrievalPipeline`，分 7 个阶段：

```
门控(gate) → 查询改写(transform) → 多路召回(recall) → RRF融合(fuse)
          → 重排(rerank) → CRAG评估(evaluate) → 补救(remediate)
```

```
┌──────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline (7 阶段)                         │
├──────────────────────────────────────────────────────────────────┤
│  并发闸门(Semaphore)                                                │
│      │                                                              │
│  ┌───▼─────┐  ┌──────────┐  ┌─────────────────────────────────┐  │
│  │ CRAG门控 │─▶│  Query   │─▶│  多路召回 (并行, 线程池)           │  │
│  │要不要检索│  │Transform │  │  Dense│BM25│Graph│ParentChild│Summary│
│  └─────────┘  └──────────┘  └───────────────┬─────────────────┘  │
│                                              │                      │
│  ┌───────────────────────────────────────────▼─────────────────┐  │
│  │  RRF 融合 → Cross-Encoder 重排 → CRAG 评估(相关性分级)         │  │
│  └───────────────────────────┬─────────────────────────────────┘  │
│                              │  若不合格                            │
│                  ┌───────────▼───────────┐                         │
│                  │  CRAG 补救(HyDE+重召回) │                         │
│                  └───────────┬───────────┘                         │
│                              ▼                                      │
│              ┌──────────────────────────────┐                      │
│              │  LLM 生成 (DeepSeek, 语义缓存) │                      │
│              └──────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 五路召回 | Dense + BM25 + Graph + Parent-Child + Summary 并行召回（线程池） |
| RRF 融合 | Reciprocal Rank Fusion 合并多路结果，避免分数不可比 |
| Cross-Encoder 重排序 | 对融合候选精排，显著提升 top-K 精度 |
| 查询改写 | Multi-Query 多变体 / HyDE 假设文档，扩大召回面 |
| Graph RAG | 零 LLM 快速构建知识图谱，实体多跳召回 |
| Parent-Child | 小块检索 + 大块返回，兼顾命中精度与上下文完整 |
| CRAG 自纠正 | 门控判断是否检索 + 相关性分级 + 不合格时 HyDE 补救 |
| 智能分块 | 递归字符分块 + 语义分块（基于 embedding 边界检测） |
| 层级索引 | L1 文档摘要索引（参与第 5 路召回）+ L2 段落明细索引 |
| 语义缓存 | embedding 相似度命中缓存，重复查询毫秒级返回 |
| 并发治理 | Semaphore 闸门 + to_thread 解除阻塞 + 排队超时 503 |
| 流式响应 | SSE 实时输出，提升用户体验 |
| 评估体系 | RAGAS 四维（LLM-judge）+ CMRC 检索命中（零 LLM）+ 并发 bench |

### RAG 2.0 深度增强（本次迭代核心）

对标「RAG 1.0 三件套（调包+存库+拼 prompt）已过时」，在**检索 / 生成 / 评估**三层各做深度增强，每项独立开关、异常均优雅降级到原行为：

| 特性 | 层级 | 说明 |
|------|------|------|
| **F1 · Autocut 自适应截断** | 检索层 | Kneedle 膝点检测动态截断重排噪声尾巴，替代固定 TopK（下界保护 + 上界不扩容） |
| **F2 · Self-RAG 迭代检索** | 检索层 | 证据不足时精化查询补充召回；**质量驱动终止**（充分性/收敛性，硬上限仅兜底） |
| **F3 · 生成忠实度自检** | 生成层 | LLM-judge 逐论断校验答案是否被上下文支撑，不忠实则严格 prompt 有界重生成（幻觉检测） |
| **F4 · 查询路由** | 检索层 | 规则驱动（零 LLM）识别 numeric/comparative/multi_hop/conceptual/factual，自适应检索深度与降噪强度 |
| **F5 · 端到端三层评估** | 评估层 | 检索命中率 + 生成忠实度 + 端到端中文 F1/EM/命中；支持特性 A/B 与单特性归因 |

配置开关（`config.py`，默认全开）：`use_autocut` / `use_iterative_retrieval` / `use_faithfulness_check` / `use_query_router`。

## 技术栈

- **框架**: LangChain + FastAPI + Streamlit
- **LLM**: DeepSeek（OpenAI 兼容接口，可切换任意兼容服务）
- **Embedding**: 本地 `BAAI/bge-small-zh-v1.5`（sentence-transformers，无需 API）
- **向量数据库**: ChromaDB
- **重排序**: 本地 `BAAI/bge-reranker-base`（cross-encoder，无需 API）
- **评估**: 自实现 RAGAS 四维（LLM-as-Judge）+ CMRC 检索命中（零 LLM）

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key（默认 DeepSeek，OpenAI 兼容接口）
# Embedding 与 Rerank 均为本地模型，无需额外 API Key
```

### 3. 启动 API 服务

```bash
python main.py
# 服务运行在 http://localhost:8000
# API 文档: http://localhost:8000/docs
```

### 4. 启动前端界面

```bash
streamlit run frontend/app.py
# 界面运行在 http://localhost:8501
```

### 5. 使用流程

1. 在前端侧边栏上传文档（支持 PDF/TXT/MD）
2. 在对话框输入问题
3. 系统自动执行：查询改写 → 多路召回 → 融合 → 重排 → 生成
4. 查看回答及引用来源

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 对话（支持流式） |
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 文档列表 |
| POST | `/api/evaluate` | 触发评估 |
| GET | `/api/health` | 健康检查 |

## 运行评估

提供三个评估脚本，覆盖质量、检索、性能三个维度：

```bash
# 1. RAGAS 四维质量评估（LLM-as-Judge，需 LLM API）
uv run python run_eval.py
#    → data/eval_report.json (faithfulness/relevancy/precision/recall)

# 2. CMRC 检索命中评估（零 LLM，直接衡量检索管道质量）
uv run python run_retrieval_eval.py
#    → data/eval_report_cmrc.json (命中率/覆盖率/检索耗时)
#    当前结果: 命中率 100% (31/31), 覆盖率 100%

# 3. 并发性能评测（需先启动服务，打真实 /api/chat）
uv run python main.py &            # 先起服务
uv run python run_concurrency_bench.py
#    → data/concurrency_report.json (QPS/P50/P95/错误率)

# 4. 端到端三层评估 + 特性 A/B（F5，需 LLM API）
uv run python run_e2e_eval.py --mode full --colloquial   # 全特性开 + 口语化查询检视
uv run python run_e2e_eval.py --mode baseline            # RAG1.0 基线（F1-F4 全关）
#    → data/eval_e2e_{full,baseline}.json (检索命中/生成忠实度/端到端F1/EM/命中 + A/B)
```

> 评估口径说明：检索质量以**与 LLM 解耦的 CMRC 评估**为准（命中率 100%）；
> RAGAS 四维受生成/评判模型影响，跨模型对比时需注明口径。详见
> [docs/superpowers/reports/2026-07-22-task11-validation-report.md](./docs/superpowers/reports/2026-07-22-task11-validation-report.md)。

## 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
├── main.py                     # FastAPI 入口（lifespan 初始化 + 并发闸门）
├── config.py                   # 统一配置（含管道接线/并发/思考模式开关）
├── run_eval.py                 # RAGAS 四维质量评估
├── run_retrieval_eval.py       # CMRC 检索命中评估（零 LLM）
├── run_concurrency_bench.py    # 并发性能评测
├── app/
│   ├── ingestion/              # 文档摄入
│   │   ├── loader.py           # 多格式加载
│   │   ├── chunker.py          # 智能分块
│   │   ├── indexer.py          # 层级索引
│   │   └── graph_extractor.py  # 知识图谱抽取（零 LLM）
│   ├── retrieval/              # 检索模块
│   │   ├── pipeline.py         # ★ RetrievalPipeline 7 阶段统一编排
│   │   ├── dense.py            # 稠密检索
│   │   ├── sparse.py           # BM25 稀疏检索
│   │   ├── graph_retriever.py  # Graph 多跳召回
│   │   ├── parent_child.py     # Parent-Child 检索
│   │   ├── fusion.py           # RRF 融合
│   │   ├── reranker.py         # Cross-Encoder 重排
│   │   ├── query_transform.py  # 查询改写（Multi-Query/HyDE）
│   │   ├── crag.py             # CRAG 门控/评估/补救
│   │   └── cache.py            # 语义缓存
│   ├── generation/             # 生成模块
│   │   ├── chain.py            # RAG Chain（薄编排层，委托 pipeline）
│   │   └── prompts.py          # Prompt 模板
│   ├── api/                    # API 层
│   │   ├── routes.py           # 路由（并发闸门 + SSE）
│   │   └── schemas.py          # 数据模型
│   ├── evaluation/             # 评估模块
│   │   ├── metrics.py          # RAGAS 指标（LLM-as-Judge）
│   │   └── dataset.py          # 测试数据集
│   └── observability/          # 可观测性
├── frontend/app.py             # Streamlit 前端
├── data/sample_docs/           # 示例文档
└── tests/                      # 单元测试（全离线，mock LLM）
```
