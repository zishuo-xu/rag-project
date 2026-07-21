---
kind: frontend_style
name: Streamlit 前端主题与内联样式体系
category: frontend_style
scope:
    - '**'
source_files:
    - .streamlit/config.toml
    - frontend/app.py
---

本仓库的前端采用 Streamlit 单文件应用，未引入外部 CSS/SCSS/Tailwind 等独立样式工程。整体视觉风格通过以下两层实现：

1. **Streamlit 主题配置**（`.streamlit/config.toml`）
   - 使用 `[theme]` 段定义全局色板：主色 `#6366F1`、背景 `#0F172A`、次级背景 `#1E293B`、文本 `#E2E8F0`，字体为 sans serif。
   - 该配置提供暗色基调，作为所有组件的默认底色。

2. **页面内联 CSS + HTML 注入**（`frontend/app.py`）
   - 在应用启动时通过 `st.markdown(..., unsafe_allow_html=True)` 注入一段 `<style>` 块，覆盖 `.block-container`、侧边栏、聊天气泡、按钮、滚动条、Tab、Expander 等 Streamlit 内置组件样式。
   - 自定义类名包括 `status-pill`、`welcome-card`、`pipeline` / `pipe-node`、`vec-row` / `vec-bar`、`term-tag`、`chunk-card`、`summary-card`、`source-card`、`legend-item` 等，用于渲染管道流程图、向量条形图、分块卡片、来源卡片、Trace 瀑布图等可视化元素。
   - 颜色方案围绕 Tailwind 风格的语义色展开：`#F59E0B`（琥珀）、`#38BDF8`（天蓝）、`#A78BFA`（紫罗兰）、`#34D399`（翠绿）、`#F472B6`（粉红）、`#1E293B` / `#172033` / `#111827`（多层深蓝灰），并通过 `STAGE_COLORS` 列表循环取用，保证 Trace 阶段着色一致。
   - 布局上固定 `max-width: 1200px`，侧边栏深色背景并加右侧边框，聊天消息圆角 14px 带边框，按钮主色为琥珀渐变悬停效果。

**架构约定**
- 所有样式集中在 `frontend/app.py` 顶部 `st.set_page_config` 之后的 `<style>` 块中，不再拆分为独立 CSS 文件。
- 新增 UI 元素应复用已有类名模式（如 `*-card`、`*-row`、`*-tag`），保持视觉一致性。
- 颜色优先从 `STAGE_COLORS` 或主题色板选取，避免硬编码新色值。
- 图表与可视化通过拼接 HTML 字符串并用 `st.markdown(..., unsafe_allow_html=True)` 渲染，而非依赖第三方绘图库。

**约束与建议**
- 由于使用 `unsafe_allow_html=True`，向用户输入拼接 HTML 时需自行做转义（代码中已对 `& < >` 进行替换）。
- 当前无响应式断点策略，仅通过 flex-wrap 和百分比宽度适配；如需移动端优化应在内联 CSS 中补充媒体查询。
- 若未来需要多页或多组件拆分，建议将 `<style>` 块提取为独立 CSS 文件并通过 Streamlit 的 `st.css` 机制加载。