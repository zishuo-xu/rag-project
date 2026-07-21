---
kind: external_dependency
name: ChromaDB 向量数据库
slug: chromadb
category: external_dependency
category_hints:
    - vendor_identity
    - client_constraint
scope:
    - '**'
---

### ChromaDB 向量数据库
- **角色**：作为项目的向量存储后端，承载文档分块（L2明细索引）和文档摘要（L1摘要索引）两个集合
- **集成点**：通过 `chroma_persist_dir`、`chroma_chunk_collection`、`chroma_summary_collection` 配置项管理持久化路径和集合名称
- **使用模式**：使用 LangChain 的 `langchain_chroma.Chroma` 包装器，支持 where 过滤查询和相似度搜索
- **约束**：本地持久化存储，数据保存在 `./data/chroma_db` 目录下，重启后数据不丢失
- **关键文件**：`indexer.py` 中的 `HierarchicalIndexer` 类管理两个 Chroma 实例的创建和操作