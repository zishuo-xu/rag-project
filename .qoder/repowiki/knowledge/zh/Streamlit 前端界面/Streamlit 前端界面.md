---
kind: external_dependency
name: Streamlit 前端界面
slug: streamlit
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
---

### Streamlit 前端界面
- **角色**：提供交互式 Web 界面，支持文档上传、问题输入、结果展示和可视化
- **集成模式**：独立的 Python 脚本运行，通过 HTTP 请求与 FastAPI 后端通信
- **部署方式**：通过 `streamlit run frontend/app.py` 命令启动，默认运行在 8501 端口
- **用户交互**：侧边栏文件上传、对话框问答、结果可视化展示等功能