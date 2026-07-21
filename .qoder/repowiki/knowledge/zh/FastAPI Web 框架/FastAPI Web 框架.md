---
kind: external_dependency
name: FastAPI Web 框架
slug: fastapi
category: external_dependency
category_hints:
    - framework_behavior
scope:
    - '**'
---

### FastAPI Web 框架
- **角色**：提供 RESTful API 服务，支持同步和异步请求处理，内置 Swagger 文档自动生成
- **集成模式**：使用 `@asynccontextmanager` 管理应用生命周期，在启动时初始化 RAG Chain 和 BM25 索引
- **关键特性**：CORS 中间件配置允许跨域访问，SSE（Server-Sent Events）流式响应支持实时输出
- **依赖关系**：配合 uvicorn 作为 ASGI 服务器运行，sse-starlette 提供 SSE 支持，python-multipart 支持文件上传
- **路由设计**：RESTful 风格的路由组织，按功能模块划分 chat、documents、graph、evaluation 等端点