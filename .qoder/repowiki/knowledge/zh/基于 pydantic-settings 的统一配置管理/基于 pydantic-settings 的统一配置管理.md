---
kind: configuration_system
name: 基于 pydantic-settings 的统一配置管理
category: configuration_system
scope:
    - '**'
source_files:
    - config.py
    - .env.example
    - main.py
    - app/generation/chain.py
---

## 系统概述
本仓库采用 **pydantic-settings** 作为统一配置加载器，通过一个集中式 `Settings` 类提供类型化、带默认值的环境配置访问。所有业务模块通过全局单例 `get_settings()` 获取配置，实现“环境变量优先 + .env 文件 + 硬编码默认值”的三层覆盖策略。

## 核心机制
- **配置文件来源与优先级**：`SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")` 指定从项目根目录 `.env` 读取；若未设置则回退到 Python 进程环境中的同名大写变量；最终使用类字段默认值。
- **全局单例缓存**：`@lru_cache()` 装饰的 `get_settings()` 保证进程内只实例化一次 `Settings`，避免重复解析环境变量。
- **类型校验与文档生成**：依赖 pydantic 的类型注解（如 `int`、`bool`、`str`）在启动时自动校验并可在 OpenAPI 中暴露。

## 关键文件与包
- `config.py`：定义 `Settings` 模型及 `get_settings()` 单例，是配置系统的唯一入口。
- `.env.example`：提供完整的环境变量模板，包含 OpenAI/Embedding/ChromaDB/Retrieval/Chunking/Server/LangSmith 等分组键名。
- `main.py`：应用生命周期中调用 `get_settings()` 做启动期校验（如强制要求 `OPENAI_API_KEY`），并通过 `settings.api_host`、`settings.api_port` 驱动 uvicorn。
- 各业务模块（`app/generation/chain.py`、`app/retrieval/*.py`、`app/ingestion/*.py`、`app/evaluation/metrics.py`、`app/api/routes.py`）均通过 `from config import get_settings` 按需读取配置项。

## 架构约定与设计决策
1. **集中式 Settings 模型**：所有运行时可调参数集中在 `config.Settings` 中，按功能域分块注释（OpenAI、Embedding、ChromaDB、Retrieval、Chunking、Server、Graph RAG、LangSmith），新增配置应遵循此分组风格。
2. **环境变量命名规范**：全部使用全大写下划线命名（如 `RETRIEVAL_TOP_K`、`CHUNK_SIZE`、`LANGCHAIN_TRACING_V2`），与 pydantic-settings 的自动映射一致。
3. **默认值即文档**：每个字段都给出合理默认值，使系统在无 `.env` 时可本地运行（例如 `embedding_provider="local"`、`graph_enabled=True`），但关键密钥（如 `openai_api_key`）在 `main.lifespan` 中显式检查并抛出明确错误提示。
4. **按需启用特性开关**：通过布尔型配置项控制可选能力（`graph_enabled`、`langchain_tracing_v2`、`use_rerank` 等），便于在不同部署环境灵活裁剪。
5. **配置不可变消费**：业务代码仅通过 `get_settings().xxx` 读取，不修改 Settings 实例，保证跨模块一致性。

## 开发者应遵守的规则
- **新增配置项**：在 `config.Settings` 中添加字段，同步更新 `.env.example` 对应条目，并在需要处用 `get_settings().xxx` 消费。
- **敏感信息**：API Key 等机密一律走环境变量，禁止硬编码或提交到版本库；`.env` 已在 `.gitignore` 中忽略。
- **类型安全**：为数值型配置保留 pydantic 类型注解，让启动期获得自动校验与清晰的报错信息。
- **启动期校验**：对必须存在的配置（如 `OPENAI_API_KEY`）应在 `main.lifespan` 中显式断言，避免静默失败。
- **多环境切换**：通过切换不同 `.env` 文件或进程环境变量即可切换 OpenAI Base URL、Embedding Provider、ChromaDB 路径等，无需改动代码。