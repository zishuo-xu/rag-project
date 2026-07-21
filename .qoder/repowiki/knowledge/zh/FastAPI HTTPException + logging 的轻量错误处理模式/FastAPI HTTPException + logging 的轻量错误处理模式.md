---
kind: error_handling
name: FastAPI HTTPException + logging 的轻量错误处理模式
category: error_handling
scope:
    - '**'
source_files:
    - main.py
    - app/api/routes.py
    - app/ingestion/loader.py
    - app/evaluation/metrics.py
    - app/evaluation/dataset.py
---

本仓库采用路由层抛出 HTTPException、业务层抛出标准 Python 异常、logging 记录的轻量级错误处理方案，未引入自定义异常类、全局异常处理器或中间件统一包装。

## 1. 系统/框架与工具
- HTTP 层：全部通过 FastAPI 内置 fastapi.HTTPException 表达客户端可感知的错误（400/501/500），由 FastAPI 自动序列化为 JSON detail 响应体。
- 日志：使用 Python 标准库 logging，在 main.py 中通过 logging.basicConfig(level=logging.INFO, format=...) 统一初始化，各模块以 logger = logging.getLogger(__name__) 获取 logger，按 info/warning/error 级别输出到控制台。
- 配置校验：启动阶段通过 RuntimeError 直接中断进程（OPENAI_API_KEY 缺失时）。

## 2. 关键文件与位置
- main.py：应用入口，配置 logging、CORS 中间件；lifespan 中用 RuntimeError 阻断启动。
- app/api/routes.py：所有 API 端点集中抛 HTTPException，并在 try/except Exception 块中 logger.error 后向上抛出 500。
- app/ingestion/loader.py：业务函数抛出标准异常 FileNotFoundError / ValueError / NotADirectoryError，供上层捕获。
- app/evaluation/metrics.py、app/evaluation/dataset.py：评估模块用 logger.warning 记录单项指标失败，不中断整体评估流程。

## 3. 架构与约定
- 业务函数：抛出标准 Python 异常（FileNotFoundError、ValueError、NotADirectoryError），示例见 loader.load_document()。
- API 路由：捕获业务异常 -> logger.error -> raise HTTPException(status_code=500, detail=...)，示例见 /api/documents/upload、/api/graph/build。
- 参数校验：直接在路由内 raise HTTPException(400, ...)，如文件格式、空问题列表、Graph RAG 未启用。
- 可选依赖：ImportError 单独捕获并返回 501，示例见 /api/evaluate 缺少评估依赖。
- 健康检查：内部异常被吞掉，降级为 num_docs=0，不影响健康状态，示例见 /api/health。
- 启动期：关键配置缺失直接 raise RuntimeError，进程退出，示例 OPENAI_API_KEY 为空。

## 4. 开发者应遵循的规则
1. 不要在业务层抛 HTTPException，它只属于 API 边界；业务函数应抛标准异常。
2. 在路由层统一包裹 try/except Exception，先 logger.error 再 raise HTTPException(500)，避免裸异常泄露堆栈。
3. 参数合法性校验优先抛 400，而非进入业务逻辑后再报错。
4. 对可选依赖使用 ImportError 分支，返回 501 明确告知客户端功能不可用。
5. 不要吞掉异常而不记录，每个 except 至少写一条 logger.error，便于线上排查。
6. 健康检查等只读接口可在内部异常时降级返回默认值，保持服务可用性。

当前方案简单直观，但尚未实现以下增强：自定义异常类型体系、全局异常处理器 @app.exception_handler、结构化日志 JSON 格式、Sentry/告警集成、重试与熔断策略。后续如需在生产环境落地，建议在此基础上逐步补齐。