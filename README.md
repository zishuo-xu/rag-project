# RAG 智能问答系统

一个生产级检索增强生成（RAG）系统，展示 RAG 全链路的工程实践。

## 文档导航

| 文档 | 说明 |
|------|------|
| [README.md](./README.md) | 项目概览与快速开始 |
| [docs/architecture.md](./docs/architecture.md) | 架构设计文档（数据流、模块设计、技术选型理由） |
| [docs/api.md](./docs/api.md) | API 接口详细文档（请求/响应示例） |
| [API Swagger UI](http://localhost:8000/docs) | 启动服务后的交互式 API 文档 |

## 系统架构

```
用户查询 → 查询改写(Multi-Query/HyDE) → 多路召回(Dense+BM25) → RRF融合 → Cross-Encoder重排序 → LLM生成 → 回答
```

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │  Query   │───▶│  Multi-Recall │───▶│   RRF Fusion        │    │
│  │Transform │    │              │    │                     │    │
│  └──────────┘    │ ┌──────────┐ │    └──────────┬──────────┘    │
│                  │ │  Dense   │ │               │               │
│                  │ │(Embedding)│ │    ┌──────────▼──────────┐    │
│                  │ └──────────┘ │    │   Cross-Encoder      │    │
│                  │ ┌──────────┐ │    │   Reranker           │    │
│                  │ │  Sparse  │ │    └──────────┬──────────┘    │
│                  │ │  (BM25)  │ │               │               │
│                  │ └──────────┘ │    ┌──────────▼──────────┐    │
│                  └──────────────┘    │   LLM Generation    │    │
│                                     │   (GPT-4o-mini)     │    │
│                                     └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 多路召回 | Dense (向量相似度) + Sparse (BM25关键词) 双路检索 |
| RRF 融合 | Reciprocal Rank Fusion 合并多路结果，避免分数不可比 |
| Cross-Encoder 重排序 | 对融合候选精排，显著提升 top-K 精度 |
| 查询改写 | Multi-Query 多变体 / HyDE 假设文档，扩大召回面 |
| 智能分块 | 递归字符分块 + 语义分块（基于 embedding 边界检测） |
| 层级索引 | L1 文档摘要索引 + L2 段落明细索引 |
| 流式响应 | SSE 实时输出，提升用户体验 |
| RAGAS 评估 | Faithfulness / Relevancy / Precision / Recall 四维评估 |

## 技术栈

- **框架**: LangChain + FastAPI + Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: OpenAI text-embedding-3-small
- **向量数据库**: ChromaDB
- **重排序**: sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2)
- **评估**: RAGAS

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
# 编辑 .env，填入你的 OpenAI API Key
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

```python
from app.evaluation.dataset import init_sample_dataset
from app.evaluation.metrics import evaluate_rag

# 初始化示例测试集
init_sample_dataset()

# 运行评估
report = evaluate_rag(
    questions=["什么是RAG？", "如何评估RAG系统？"],
    ground_truths=["RAG是...", "使用RAGAS..."],
)
print(report["metrics"])
```

## 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
├── main.py                     # FastAPI 入口
├── config.py                   # 统一配置
├── app/
│   ├── ingestion/              # 文档摄入
│   │   ├── loader.py           # 多格式加载
│   │   ├── chunker.py          # 智能分块
│   │   └── indexer.py          # 层级索引
│   ├── retrieval/              # 检索模块
│   │   ├── dense.py            # 稠密检索
│   │   ├── sparse.py           # BM25 稀疏检索
│   │   ├── fusion.py           # RRF 融合
│   │   ├── reranker.py         # Cross-Encoder 重排
│   │   └── query_transform.py  # 查询改写
│   ├── generation/             # 生成模块
│   │   ├── chain.py            # RAG Chain
│   │   └── prompts.py          # Prompt 模板
│   ├── api/                    # API 层
│   │   ├── routes.py           # 路由
│   │   └── schemas.py          # 数据模型
│   └── evaluation/             # 评估模块
│       ├── metrics.py          # RAGAS 指标
│       └── dataset.py          # 测试数据集
├── frontend/app.py             # Streamlit 前端
├── data/sample_docs/           # 示例文档
└── tests/                      # 单元测试
```
