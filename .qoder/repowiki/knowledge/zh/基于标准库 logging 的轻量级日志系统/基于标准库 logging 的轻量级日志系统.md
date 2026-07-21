---
kind: logging_system
name: 基于标准库 logging 的轻量级日志系统
category: logging_system
scope:
    - '**'
source_files:
    - main.py
    - run_eval.py
    - app/api/routes.py
    - app/evaluation/metrics.py
    - app/ingestion/graph_extractor.py
---

## 1. 使用的系统与框架
- 采用 Python 标准库 logging，未引入 structlog、loguru、log4j 等第三方日志框架。
- 通过 logging.basicConfig() 在进程入口处统一配置根 logger，设置默认级别为 INFO，格式为：%(asctime)s - %(name)s - %(levelname)s - %(message)s
- 评估脚本 run_eval.py 使用独立格式：%(asctime)s [%(levelname)s] %(message)s，与主服务略有差异。

## 2. 核心文件与位置
- 日志初始化入口：main.py（FastAPI 应用启动时调用 basicConfig）
- 评估脚本入口：run_eval.py（单独 basicConfig）
- 各模块中通过 logger = logging.getLogger(__name__) 获取命名 logger，覆盖以下子包：
  - app/api/routes.py — API 层错误日志
  - app/evaluation/dataset.py、app/evaluation/metrics.py — 评估流程日志
  - app/generation/chain.py — RAG Chain 执行日志
  - app/ingestion/chunker.py、app/ingestion/graph_extractor.py、app/ingestion/indexer.py、app/ingestion/loader.py — 文档摄入管道日志
  - app/retrieval/dense.py、app/retrieval/fusion.py、app/retrieval/graph_retriever.py、app/retrieval/reranker.py、app/retrieval/sparse.py — 检索管线日志

## 3. 架构与约定
- 单点初始化：所有日志由 main.py 中的 basicConfig 一次性配置，子模块仅负责获取 logger 并输出消息，不重复配置。
- 命名空间划分：每个模块以 __name__ 作为 logger 名称，形成与包结构一致的层级命名空间（如 app.retrieval.dense），便于按模块过滤日志。
- 日志级别使用：
  - info：业务流程关键节点（索引构建进度、评估开始/完成、RAG 回答获取等）
  - warning：可恢复异常或降级路径（三元组抽取失败、指标评估失败、数据集缺失等）
  - error：不可恢复错误（文档上传失败、图检索失败、评估失败等）
  - debug：仅在 graph_extractor.py 中出现，用于调试三元组数量，其余模块极少使用
- 结构化程度低：日志消息均为字符串拼接 f-string，未使用结构化字段（JSON 字段、trace_id、请求上下文等），也不存在统一的中间件注入请求 ID。
- Sink 单一：默认输出到 stderr（控制台），仓库根目录下的 server.log、eval_output*.log、concurrency_output.log 是运行过程中重定向产生的文件，并非由 logging 框架自动路由。
- 无异步日志支持：当前实现全部基于同步 logging，未见 aiologging、loguru 异步 handler 等适配。

## 4. 开发者应遵循的规则
1. 不要在子模块中再次调用 basicConfig，只通过 logger = logging.getLogger(__name__) 获取 logger 并输出。
2. 选择合适的日志级别：业务正常流转用 info，可恢复异常用 warning，致命错误用 error；调试信息谨慎使用 debug。
3. 保持消息可读性：当前全为文本消息，建议包含关键上下文（如文件名、ID、计数），避免过长堆栈直接拼入消息体。
4. 如需结构化日志或集中收集：应在 main.py 的 basicConfig 处扩展 Handler（如 JSONFormatter、FileHandler、RotatingFileHandler），而非在各模块分散配置。
5. 评估脚本与主服务保持一致：若新增独立脚本，尽量复用 main.py 的日志格式，避免多套格式并存。