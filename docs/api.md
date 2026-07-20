# API 接口文档

> 基础 URL: `http://localhost:8000`
> 交互式文档: `http://localhost:8000/docs` (Swagger UI)

## 目录

- [对话接口](#1-对话接口)
- [文档管理](#2-文档管理)
- [评估接口](#3-评估接口)
- [健康检查](#4-健康检查)

---

## 1. 对话接口

### POST `/api/chat`

执行 RAG 问答，支持普通和流式（SSE）两种响应模式。

**请求体：**

```json
{
  "question": "什么是 RAG？",
  "chat_history": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
  ],
  "top_k": 5,
  "use_query_transform": true,
  "use_rerank": true,
  "query_strategy": "multi_query",
  "stream": false
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| question | string | 是 | - | 用户问题（不能为空） |
| chat_history | array | 否 | [] | 对话历史 |
| top_k | int | 否 | 5 | 最终返回的参考文档数 (1-20) |
| use_query_transform | bool | 否 | true | 是否启用查询改写 |
| use_rerank | bool | 否 | true | 是否启用 Cross-Encoder 重排序 |
| query_strategy | string | 否 | "multi_query" | 查询改写策略: `multi_query` / `hyde` / `none` |
| stream | bool | 否 | false | 是否流式返回 |

**普通响应 (stream=false)：**

```json
{
  "answer": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术框架... [来源: rag_technology.md]",
  "sources": [
    {
      "content": "RAG（Retrieval-Augmented Generation，检索增强生成）是一种...",
      "source": "rag_technology.md",
      "score": 0.8745,
      "metadata": {
        "chunk_id": "rag_technology_3",
        "doc_id": "rag_technology",
        "position": 3
      }
    }
  ],
  "retrieval_detail": {
    "queries_used": ["什么是RAG？", "RAG技术的定义和原理", "检索增强生成是什么"],
    "dense_count": 15,
    "sparse_count": 8,
    "fused_count": 18,
    "final_count": 5,
    "retrieval_time_ms": 1234.5
  },
  "total_time_ms": 3456.7
}
```

**流式响应 (stream=true)：**

返回 SSE (Server-Sent Events) 流，包含三种事件类型：

```
event: retrieval
data: {"queries_used": [...], "dense_count": 15, "sparse_count": 8, ...}

event: token
data: R

event: token
data: AG

event: token
data: （

event: done
data: {"sources": [...], "total_time_ms": 3456.7}
```

**cURL 示例：**

```bash
# 普通请求
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "stream": false}'

# 流式请求
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "stream": true}' \
  --no-buffer
```

---

## 2. 文档管理

### POST `/api/documents/upload`

上传文档并自动建立索引（分块 + 向量化 + BM25 索引）。

**请求：** `multipart/form-data`

| 字段 | 类型 | 说明 |
|------|------|------|
| file | File | 文档文件，支持 `.pdf` / `.txt` / `.md` |

**响应：**

```json
{
  "message": "文档 'rag_technology.md' 上传并索引成功",
  "filename": "rag_technology.md",
  "num_chunks": 12,
  "doc_id": "rag_technology"
}
```

**错误响应：**

```json
// 400 - 不支持的格式
{"detail": "不支持的文件格式: .exe，支持: .pdf, .txt, .md"}

// 500 - 处理失败
{"detail": "文档处理失败: ..."}
```

**cURL 示例：**

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@./data/sample_docs/rag_technology.md"
```

---

### GET `/api/documents`

获取已索引的文档列表。

**响应：**

```json
{
  "documents": [
    {
      "doc_id": "rag_technology",
      "source": "rag_technology.md",
      "num_chunks": 12
    }
  ],
  "total": 1
}
```

---

## 3. 评估接口

### POST `/api/evaluate`

触发 RAG 系统评估，使用 RAGAS 指标衡量检索和生成质量。

**请求体：**

```json
{
  "questions": [
    "什么是 RAG？",
    "RRF 融合的原理是什么？"
  ],
  "ground_truths": [
    "RAG 是检索增强生成技术...",
    "RRF 公式为 score = Σ 1/(k+rank)..."
  ]
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| questions | array[string] | 是 | 测试问题列表 |
| ground_truths | array[string] | 否 | 标准答案（提供则额外计算 Context Recall） |

**响应：**

```json
{
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.92,
    "context_precision": 0.78,
    "context_recall": 0.81
  },
  "num_samples": 2,
  "details": [
    {
      "question": "什么是 RAG？",
      "answer": "RAG 是...",
      "num_contexts": 5
    }
  ]
}
```

**指标含义：**

| 指标 | 范围 | 含义 |
|------|------|------|
| faithfulness | 0-1 | 回答忠于检索内容的程度（越高越好） |
| answer_relevancy | 0-1 | 回答与问题的相关性（越高越好） |
| context_precision | 0-1 | 相关文档排在前面的程度（越高越好） |
| context_recall | 0-1 | 检索覆盖标准答案的程度（越高越好） |

---

## 4. 健康检查

### GET `/api/health`

检查服务运行状态。

**响应：**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "indexed_documents": 3
}
```

---

## 错误处理

所有接口在出错时返回标准 HTTP 状态码：

| 状态码 | 含义 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 422 | 请求体验证失败（Pydantic） |
| 500 | 服务器内部错误 |
| 501 | 功能未实现（如评估依赖未安装） |

错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```
