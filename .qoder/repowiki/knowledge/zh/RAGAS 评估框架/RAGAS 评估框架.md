---
kind: external_dependency
name: RAGAS 评估框架
slug: ragas
category: external_dependency
category_hints:
    - migration_status
scope:
    - '**'
---

### RAGAS 评估框架
- **角色**：RAG 系统质量评估工具，提供 Faithfulness、Relevancy、Precision、Recall 等指标
- **替代方案**：使用 LLM-as-Judge 方法自实现四个核心指标，避免外部依赖冲突
- **评估维度**：Faithfulness（忠实度）、Answer Relevancy（答案相关性）、Context Precision（上下文精确度）、Context Recall（上下文召回率）