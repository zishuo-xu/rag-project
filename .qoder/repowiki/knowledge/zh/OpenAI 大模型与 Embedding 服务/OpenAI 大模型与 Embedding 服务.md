---
kind: external_dependency
name: OpenAI 大模型与 Embedding 服务
slug: openai
category: external_dependency
category_hints:
    - vendor_identity
    - auth_protocol
scope:
    - '**'
---

### OpenAI 大模型与 Embedding 服务
- **角色**：提供 LLM（GPT-4o-mini）和文本嵌入（text-embedding-3-small）能力，用于 RAG 系统的生成、检索和评估环节
- **集成点**：通过 `openai_api_key`、`openai_base_url`、`openai_model`、`openai_embedding_model` 配置项注入
- **使用模式**：支持自定义 base_url（可指向兼容 OpenAI API 的第三方服务），实现模型供应商解耦
- **认证协议**：基于 API Key 的 Bearer Token 认证，通过 `api_key` 参数传递