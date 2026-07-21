---
kind: dependency_management
name: 基于 uv 的 Python 依赖锁定与版本管理
category: dependency_management
scope:
    - '**'
source_files:
    - pyproject.toml
    - uv.lock
    - .python-version
---

本项目采用 uv（Rust 实现的高性能 Python 包管理器）作为统一的依赖管理与环境解析工具，配合 PEP 621 风格的 pyproject.toml 声明依赖、uv.lock 生成确定性锁文件，形成“声明 + 锁定”的完整闭环。

- 依赖声明：所有运行时与可选依赖集中在根目录 pyproject.toml 中，按功能域分组注释（LangChain 核心、向量检索、API 服务、前端、评估、文档加载、配置工具），并通过 [project.optional-dependencies] 将测试/开发依赖（pytest、httpx 等）与主依赖解耦。
- 版本策略：统一使用 >=X.Y.Z 宽松下限约束，不固定上限，由 uv 在首次解析时根据当前 Python 版本与平台标记（resolution-markers）选择兼容子版本，随后写入 uv.lock 固化具体版本与哈希。
- 锁文件：uv.lock 记录每个包的精确版本、来源（PyPI）、sdist/wheel 的 SHA256 校验值以及跨 Python 3.11–3.14、多平台的 wheel 列表，保证构建可重现；同时通过 requires-python = ">=3.11" 与 .python-version 共同约束运行环境。
- 私有源与代理：当前仓库未出现自定义 index URL、--index-url、PIP_INDEX_URL 或 uv 相关配置文件，表明依赖全部来自官方 PyPI，无私有镜像或 vendoring 策略。
- 更新流程：新增/升级依赖应修改 pyproject.toml 中的版本号，然后执行 uv lock 重新生成 uv.lock；提交时需同时提交两者以保持团队一致性。

开发者约定：
- 仅在 pyproject.toml 中声明依赖，禁止在代码中硬编码版本号或通过 pip install 手动安装。
- 变更依赖后必须同步提交 uv.lock，避免 CI 与本地环境产生漂移。
- 保持 requires-python >= 3.11 与 .python-version 一致，防止 uv 在不同 Python 下解析出不同依赖树。