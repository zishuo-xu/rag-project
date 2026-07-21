---
kind: external_dependency
name: LangChain 框架生态
slug: langchain
category: external_dependency
category_hints:
    - framework_behavior
    - sdk_real_api
scope:
    - '**'
---

### LangChain 框架生态
- **角色**：作为 RAG 系统的核心编排框架，提供 Chain、PromptTemplate、Document、Embeddings 等基础抽象
- **集成模式**：使用 `langchain_core.documents.Document` 作为统一文档表示，`ChatOpenAI` 作为 LLM 客户端，`OpenAIEmbeddings` 作为嵌入接口
- **关键行为**：通过 PromptTemplate + LLM + StrOutputParser 构建链式处理管道；使用 BaseSettings 进行配置管理
- **扩展性**：通过继承和组合模式扩展检索策略，保持与 LangChain 生态的兼容性