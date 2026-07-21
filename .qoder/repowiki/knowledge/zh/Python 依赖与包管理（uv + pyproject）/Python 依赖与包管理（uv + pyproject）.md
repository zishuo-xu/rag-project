---
kind: build_system
name: Python 依赖与包管理（uv + pyproject）
category: build_system
scope:
    - '**'
source_files:
    - pyproject.toml
    - uv.lock
    - .python-version
    - .env.example
    - main.py
    - run_eval.py
    - run_concurrency_bench.py
---

本仓库采用现代 Python 工程化方案，以 `pyproject.toml` 声明项目元数据与依赖，使用 **uv** 作为包管理器并锁定 `uv.lock`，配合 `.python-version` 约束运行时版本。整体构建体系轻量、无 Makefile/Dockerfile/CI 配置，属于“脚本驱动 + uv 锁”的本地开发模式。

1. 使用的系统与工具
- 包清单：`pyproject.toml`（PEP 621），定义项目名、Python 版本要求、dependencies 与 optional-dependencies（dev）。
- 包解析与锁定：`uv`，通过 `uv.lock` 记录跨平台 resolution-markers 与每个包的 wheel/sdist 哈希，保证可复现安装。
- Python 版本：`.python-version` 固定为 3.11；`pyproject.toml` 中 `requires-python = ">=3.11"` 向上兼容。
- 测试运行：`pytest`，通过 `[tool.pytest.ini_options]` 指定 `pythonpath` 与 `testpaths`。
- 环境变量：`.env.example` 提供 OpenAI / ChromaDB / Reranker / Server / LangSmith 等运行时配置模板，由 `python-dotenv` + `pydantic-settings` 在代码中加载。

2. 关键文件与职责
- `pyproject.toml`：声明所有生产与开发依赖（LangChain、FastAPI、ChromaDB、RAGAS、Streamlit 等），以及 pytest 配置。
- `uv.lock`：uv 生成的完整依赖树与哈希锁定文件，包含多平台标记，是构建/部署时的确定性来源。
- `.python-version`：强制使用 Python 3.11，便于开发者与 CI 环境对齐。
- `.env.example`：集中列出所有外部服务密钥与检索/分块/服务器参数，避免硬编码。
- `main.py`、`run_eval.py`、`run_concurrency_bench.py`：顶层入口脚本，分别启动 FastAPI 服务、执行评估与并发基准测试。
- `tests/`：基于 pytest 的单元测试，覆盖 API、分块器、检索融合等核心路径。

3. 架构与约定
- 单仓库、单应用：后端、前端 Streamlit、评估脚本、示例文档均位于同一仓库，不拆子包或发布 PyPI 包。
- 依赖分层：生产依赖集中在 `dependencies`，开发依赖放入 `optional-dependencies.dev`，通过 `uv pip install -e .[dev]` 一次性安装。
- 可复现构建：新增/升级依赖后应通过 `uv lock` 更新 `uv.lock`，提交锁文件以保证团队与 CI 一致。
- 配置外置：所有敏感信息与可调参数统一写入 `.env`（从 `.env.example` 复制），不在代码中写死。

4. 开发者应遵循的规则
- 修改依赖时只改 `pyproject.toml`，然后运行 `uv lock` 生成新的 `uv.lock`，不要手动编辑锁文件。
- 保持 `.python-version` 与 `requires-python` 一致，避免本地与 CI 出现版本漂移。
- 新增环境变量请在 `.env.example` 中同步补充默认值，并在代码侧用 `pydantic-settings` 读取。
- 运行方式：
  - 安装：`uv pip install -e .[dev]`
  - 启动服务：`uv run python main.py`（或 `uvicorn` 直接调用）
  - 运行测试：`uv run pytest`
  - 运行评估：`uv run python run_eval.py`
- 当前仓库未包含 Dockerfile、Makefile、GitHub Actions 等 CI/容器化配置；如需引入，建议新增对应文件并在 PR 流程中校验 `uv.lock` 是否同步更新。