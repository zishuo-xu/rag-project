# RAG 智能问答系统

一个**教学样板**——借「生产级」叙事外壳讲 RAG 全链路工程故事。产物是更小但更硬的核：**7 条核心叙事**，诚实报告 + A/B 归因是核心卖点（见下「核心叙事」）。

## 文档导航

| 文档 | 说明 |
|------|------|
| [README.md](./README.md) | 项目概览与快速开始 |
| [docs/architecture.md](./docs/architecture.md) | 架构设计文档（数据流、模块设计、技术选型理由） |
| [docs/api.md](./docs/api.md) | API 接口详细文档（请求/响应示例） |
| [docs/interview_guide.md](./docs/interview_guide.md) | 面试与学习指南（技术深度解析、高频Q&A、扩展路线） |
| [docs/superpowers/reports/](./docs/superpowers/reports/) | 逐轮迭代验证报告（诚实记录，含负面/无效结果；最新：[2026-07-29 零 LLM 均衡化](./docs/superpowers/reports/2026-07-29-zero-llm-balance-report.md)） |
| [API Swagger UI](http://localhost:8000/docs) | 启动服务后的交互式 API 文档 |

## 核心叙事（面试 30 秒）

> **教学样板**——借「生产级」叙事外壳讲 RAG 全链路工程故事。产物是更小但更硬的核：**7 条核心叙事**，其余降级为素材。诚实报告 + A/B 归因是核心卖点：弱数字不是包袱，是「如实记录」的素材。

**7 条核心叙事**：

| 特性 | 一句话 | 关键证据 |
|---|---|---|
| F1 Autocut 自适应截断 | Kneedle 膝点动态截断重排噪声尾巴，替代固定 TopK | 入 LLM 上下文 61.9→6.0 篇/题（膝点截断） |
| F3+F8 投机流式忠实度 | 先逐 token 流式（快 TTFT），流末忠实度自检，不忠实追加 correction | 忠实度 0.986（重生成率 1.7%） |
| F7 引用溯源 | 答案切 claim，embedding 余弦关联源块，零在线 LLM | 0.91 条块级引用/题 |
| F10 answer_extraction | 零 LLM 抽取核心短答案 span | 短答案 F1 0.61、EM 0→0.167（打破恒 0） |
| Graph RAG | 零 LLM jieba 共现图 + chunk 溯源（回退自 LLM 类型化，权衡可复现） | 21856 节点/106483 边，1 秒重建 |
| F13 Agentic RAG | 零 LangGraph 手写 ReAct + 零 LLM 证据硬门控（CRAG 判充分即停） | finish 率 33-53%→**93%**（多跳切片，原诚实开放项已修复） |
| 延迟治理 | Deadline 查询级预算熔断离群尾，默认路径零在线 LLM 增量 | full 均值 42.6→4.5s（−89%） |

**软摘（代码留、不主动讲）**：F10 self_consistency（默认关、无数据撑）、多模型 qwen 切换（provider 切换卖点已由 `build_chat_llm` 收口覆盖）。判据：非高频考点 + 无数据撑 + 制造叙事噪音 → 软摘。

> 上表数字为 **300 题 CMRC 全量基线**（2026-07-29 复刷，deepseek 口径，300/300 零失败；F13 为 15 题多跳切片）。检索命中/覆盖 1.00/1.00、端到端 F1 0.600、EM 0.143。下文完整特性表为深挖参考，面试主线以本表 7 条为准。

## 系统架构

检索管道统一在 `RetrievalPipeline`，分 7 个阶段：

```
门控(gate) → 查询改写(transform) → 多路召回(recall) → RRF融合(fuse)
          → 重排(rerank) → CRAG评估(evaluate) → 补救(remediate)
```

```
┌──────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline (7 阶段)                         │
├──────────────────────────────────────────────────────────────────┤
│  并发闸门(Semaphore)                                                │
│      │                                                              │
│  ┌───▼─────┐  ┌──────────┐  ┌─────────────────────────────────┐  │
│  │ CRAG门控 │─▶│  Query   │─▶│  多路召回 (并行, 线程池)           │  │
│  │要不要检索│  │Transform │  │  Dense│BM25│Graph│ParentChild│Summary│
│  └─────────┘  └──────────┘  └───────────────┬─────────────────┘  │
│                                              │                      │
│  ┌───────────────────────────────────────────▼─────────────────┐  │
│  │  RRF 融合 → Cross-Encoder 重排 → CRAG 评估(相关性分级)         │  │
│  └───────────────────────────┬─────────────────────────────────┘  │
│                              │  若不合格                            │
│                  ┌───────────▼───────────┐                         │
│                  │  CRAG 补救(HyDE+重召回) │                         │
│                  └───────────┬───────────┘                         │
│                              ▼                                      │
│              ┌──────────────────────────────┐                      │
│              │  LLM 生成 (DeepSeek, 语义缓存) │                      │
│              └──────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 五路召回 | Dense + BM25 + Graph + Parent-Child + Summary 并行召回（线程池） |
| RRF 融合 | Reciprocal Rank Fusion 合并多路结果，避免分数不可比 |
| Cross-Encoder 重排序 | 对融合候选精排，显著提升 top-K 精度 |
| 查询改写 | Multi-Query 多变体 / HyDE 假设文档，扩大召回面 |
| Graph RAG | LLM 类型化三元组知识图谱（7 类实体 + chunk 溯源），实体多跳召回 |
| Parent-Child | 小块检索 + 大块返回，兼顾命中精度与上下文完整 |
| CRAG 自纠正 | 门控判断是否检索 + 相关性分级 + 不合格时 HyDE 补救 |
| 智能分块 | 递归字符分块 + 语义分块（基于 embedding 边界检测） |
| 层级索引 | L1 文档摘要索引（参与第 5 路召回）+ L2 段落明细索引 |
| 语义缓存 | embedding 相似度命中缓存，重复查询毫秒级返回 |
| 并发治理 | Semaphore 闸门 + to_thread 解除阻塞 + 排队超时 503 |
| 流式响应 | SSE 实时输出，提升用户体验 |
| 评估体系 | RAGAS 四维（LLM-judge）+ CMRC 检索命中（零 LLM）+ 并发 bench |

### RAG 2.0 深度增强（本次迭代核心）

对标「RAG 1.0 三件套（调包+存库+拼 prompt）已过时」，在**检索 / 生成 / 评估**三层各做深度增强，每项独立开关、异常均优雅降级到原行为：

| 特性 | 层级 | 说明 |
|------|------|------|
| **F1 · Autocut 自适应截断** | 检索层 | Kneedle 膝点检测动态截断重排噪声尾巴，替代固定 TopK（下界保护 + 上界不扩容） |
| **F2 · Self-RAG 迭代检索** | 检索层 | 证据不足时精化查询补充召回；**质量驱动终止**（充分性/收敛性，硬上限仅兜底） |
| **F3 · 生成忠实度自检** | 生成层 | LLM-judge 逐论断校验答案是否被上下文支撑，不忠实则严格 prompt 有界重生成（幻觉检测） |
| **F4 · 查询路由** | 检索层 | 规则驱动（零 LLM）识别 numeric/comparative/multi_hop/conceptual/factual，自适应检索深度与降噪强度 |
| **F5 · 端到端三层评估** | 评估层 | 检索命中率 + 生成忠实度 + 端到端中文 F1/EM/命中；支持特性 A/B 与单特性归因 |
| **F6 · 答案定位增强** | 检索层 | **F6a** 细粒度召回 + 上下文增强分块（Parent-Child 接线 + Contextual Chunking 双集合，索引期 LLM、在线零 LLM）；**F6b** 多跳查询分解（并行优先 / 依赖链式，仅 multi_hop 触发） |

配置开关（`config.py`，默认全开）：`use_autocut` / `use_iterative_retrieval` / `use_faithfulness_check` / `use_query_router` / `use_contextual_chunks` / `use_decomposition`。

### RAG 3.0 生产级增强（本次迭代核心）

在 RAG 2.0 基础上补齐**生产级 RAG** 的关键能力，重点攻克 RAG 2.0 验证暴露的三大痛点：**端到端 EM=0**、**F3 导致的时延回退与流式退化**、**生产加固缺失**。每项独立开关、异常优雅降级、**时延预算优先**（默认路径零新增在线 LLM）：

| 特性 | 层级 | 说明 | 时延影响 |
|------|------|------|---------|
| **F7 · 引用溯源** | 生成层 | 答案切句级 claim，用 embedding 余弦关联到最相关源块，输出结构化引用（来源/块id/置信度/证据片段），零在线 LLM | +一次批量编码 |
| **F8 · 投机流式忠实度** | 生成层 | 先逐 token 流式吐字（快 TTFT），流末做忠实度自检，不忠实追加 `correction` 事件——修复 F3 的流式退化与时延回退 | **TTFT 大幅下降** |
| **F9 · 多级缓存** | 检索层 | L1 Embedding 缓存 + L2 Rerank 缓存（线程安全 LRU），重复查询省掉编码与 cross-encoder | **命中省 100–700ms** |
| **F10 · 答案质量增强** | 生成层 | 零LLM 短答案抽取 + 自适应自一致性（仅短答案型查询）——**攻克 EM=0** | 默认 ~0 |
| **F11 · 可观测性与加固** | 平台层 | 进程内指标（计数/直方图，`/api/metrics` Prometheus+JSON）+ API Key 鉴权 + 限流 + 结构化日志 | <1µs/请求 |
| **F12 · 多轮对话记忆** | 检索层 | 历史感知查询重写，解析指代/省略（"它的原理呢？"→"缓存穿透的原理"），零LLM 启发式（LLM 可选） | 默认 ~0 |

配置开关（`config.py`，默认值见括号）：`use_citations`(True) / `use_speculative_streaming`(True) / `use_embedding_cache`(True) / `use_rerank_cache`(True) / `use_answer_extraction`(True) / `use_self_consistency`(False，保时延) / `use_history_rewrite`(True) / `api_key`(""=关) / `rate_limit_rpm`(0=关) / `log_json`(False)（指标常开无开关）。

> **关键验证结果**（CMRC 31 题端到端 A/B，详见 [RAG3 验证报告](./docs/superpowers/reports/2026-07-24-rag3-validation-report.md)）：F10 短答案抽取把 **short_answer 的 EM 从 0.0 提升到 0.097、F1 从完整答案的 0.35 提升到 0.52（+48%）**，闭环上一轮如实报告的 EM=0；F7 平均 **1.52 条块级引用**、F1 上下文降噪 **47%**（8.0→4.26 篇）、F3+F8 忠实度 **0.77**/重生成 22.6%；六特性默认路径**零在线 LLM 增量**；**256 项测试全绿**（本轮新增 101）。

### F13 · Agentic RAG（ReAct 状态机自主检索）

把固定七阶段管道的**编排决策交给 LLM**：ReAct 循环逐步决定调哪个工具、用什么查询、何时停止（thought→action→observation，概念对齐 LangGraph 但**零新依赖手写**，全离线可测）。工具集复用管道阶段：`search(query)`（召回+融合+重排）/ `decompose()`（F6b 分解，**agent 自主决定，不再依赖 F4 路由**）/ `grade()`（CRAG 分级）。

护栏：max_steps=4 硬上限 / 决策解析失败即停 / 工具异常写入 observation / **整体异常或空证据降级回七阶段管道**。默认关（`use_agentic=False`），每步 1 次小 token 决策调用（256 tokens / 15s 超时）。

**实测**（15 条多跳集，详见 [F13 验证报告](./docs/superpowers/reports/2026-07-25-f13-agentic-validation.md) 与 [Prompt/路由优化报告](./docs/superpowers/reports/2026-07-26-prompt-router-optimization-report.md)）：F1 **0.255→0.283+（四种模式最优）**，延迟 +20%。首轮诚实发现 agent 过度检索（10/15 打满步数）、search 主导（43/52 次决策）、decompose 仅 3 次且跨运行不稳定。**Prompt v2 优化后**（硬工具条件 + 2 条 few-shot + 连续空检索收敛护栏 + 重复查询警告）：decompose 稳定触发 **3→13/14 次**、F1 **0.286→0.298**、主动结束率 5/15→7/15（两次复跑）；代价是分解 LLM 调用使延迟升至 ~12.5s——工具选择行为可用 prompt 工程引导，但 finish 率仍未达 50%，如实记录为开放项。同期修复 F4 路由 multi_hop 召回（1/15→14/15，CMRC 误判维持 3/31），full 模式 F6b 分解触发 1/15→14/15、F1 0.276→0.289。

```bash
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --only F13   # 多跳集 agentic 评估
```

### 收敛优化 + 延迟治理 + Graph RAG 升级（2026-07-26）

承接上轮三个诚实开放项，①→②→④ 串行推进（详见 [整合报告](./docs/superpowers/reports/2026-07-26-convergence-latency-graph-report.md)）：

- **① F13 收敛优化 v3**（零新增 LLM）：步数预算/新增量/证据状态三类信号进 prompt + 零证据 finish 驳回门控。延迟 12.5→9.7s（↓23%）；**finish 率 40%/33% 仍未达 ≥50% 目标**——信号可见无法强制 LLM 早停，如实记为开放项。
- **② 延迟治理**：Deadline 查询级时延预算（25s，熔断 F2/F3 离群尾）+ 超时重试收紧（60s×2→30s×1）+ max_tokens 封顶 + router 前置短路（分解路径跳过无用改写）+ hops 3→2。full 均值 **42.6s→9.6s（↓77%）**（原单点 486s 离群消灭），max 16.9s，F1 0.291（基线 0.289），num_failed=0。
- **④ Graph RAG 升级（Option A）**：EXTRACTION_PROMPT 升级 JSON 类型化三元组（person/work/place/org/position/event/other + 传记体 few-shot）+ 边带 chunk_id 溯源 + 图检索文档 `graph:` 前缀溯源（不与真实分块 RRF 互覆盖）+ 分解路径子问题接入 graph 通道。重建生产图：675 节点/2354 共现边 → **1214 节点/985 类型化边，chunk 溯源 100%**。评估诚实记录：F1 0.291→0.287（−0.004 ∈ 波动带，**无显著变化**）、hit/coverage 持平、延迟 +12%——结构能力升级（关系质量 + 溯源 + 通道接线），检索收益未在 15 样本集兑现。

测试：312→324→**336 全绿**（每阶段 +12 / +12）。

## 技术栈

- **框架**: LangChain + FastAPI + Streamlit
- **LLM**: DeepSeek（OpenAI 兼容接口，可切换任意兼容服务）
- **Embedding**: 本地 `BAAI/bge-small-zh-v1.5`（sentence-transformers，无需 API）
- **向量数据库**: ChromaDB
- **重排序**: 本地 `BAAI/bge-reranker-base`（cross-encoder，无需 API）
- **评估**: 自实现 RAGAS 四维（LLM-as-Judge）+ CMRC 检索命中（零 LLM）

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）：依赖由已提交的 uv.lock 锁定版本，配合 .python-version=3.11 保证环境可复现
uv sync
uv sync --extra dev    # 需要跑测试时（pytest 在 dev extras 里）

# 或使用 pip（无 lockfile 保护，宽松解析，仅备选）
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key（默认 DeepSeek，OpenAI 兼容接口）
# Embedding 与 Rerank 均为本地模型，无需额外 API Key
```

### 3. 启动 API 服务

```bash
python main.py
# 服务运行在 http://localhost:8000
# API 文档: http://localhost:8000/docs
# 首跑会自动联网下载本地 embedding/rerank 模型（约 1.2GB），之后离线启动
```

### 4. 启动前端界面

```bash
streamlit run frontend/app.py
# 界面运行在 http://localhost:8501
```

### 5. 使用流程

1. 在前端侧边栏上传文档（支持 PDF/TXT/MD）
2. 在对话框输入问题
3. 系统自动执行：查询改写 → 多路召回 → 融合 → 重排 → 生成
4. 查看回答及引用来源

## 复现评测数字（复现链）

向量库 `data/chroma_db/` 不入库（gitignore），clone 后需先制备一次索引，才能复现 README 与评测报告中的数字：

```bash
# 1. 灌全量语料进索引（chroma / summary / parent-child / BM25 + 图谱增量）
#    语料 data/cmrc_docs/（165 篇 md）与题集 data/eval_dataset_cmrc_full.json（300 题）均已提交；
#    只有需要重新生成语料/题目时才跑 scripts/build_cmrc_eval.py（需联网拉 CMRC2018）。
#    注意：L1 摘要 + F6a 上下文增强在索引期用 LLM（一次性，不在线上调用）；
#    无 LLM key 时降级为 warning 不阻断，但索引为「无增强版」，端到端数字会偏低。
uv run python scripts/ingest_cmrc_full.py

# 2. 零 LLM 知识图谱（jieba 共现图，秒级重建；仓库已提交 data/knowledge_graph.json，可跳过）
uv run python scripts/rebuild_graph_fast.py

# 3. 复现数字
uv run python run_retrieval_eval.py            # 检索命中/覆盖率（零 LLM）
uv run python run_e2e_eval.py --mode full      # 端到端三层评估（需 LLM API，默认 deepseek 口径）
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 对话（支持流式） |
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 文档列表 |
| POST | `/api/evaluate` | 触发评估 |
| GET | `/api/health` | 健康检查 |

## 运行评估

**评测分层（按等待时间选层）**：

| 层 | 命令 | 耗时（deepseek） | 用途 |
|---|---|---|---|
| 零 LLM 检索 | `run_retrieval_eval.py` / `run_rewrite_eval.py` | 秒级 | 改检索层即跑 |
| 快速切片（50 题） | `run_e2e_eval.py --dataset data/eval_slice_fast.json --gate --baseline data/eval_slice_baseline.json` | ~4 分钟 | 日常迭代 A/B |
| 多跳 / 多轮切片 | `--dataset data/eval_multihop.json` / `data/eval_multiturn.json` | ~2 分钟 | F6b / F12 / F13 专项 |
| 全量基线（300 题） | `run_e2e_eval.py --mode full --gate` | ~20 分钟 | 发版前 + 基线重定 |

> 快速切片为步长 6 确定性抽样（跨次同题、可比），检测力弱于全量（容差 ±0.11 vs ±0.05），负结果不下强结论。全量基线为 **deepseek 口径**（2026-07-29 回填）；日常 provider 若为 qwen，跨 provider 不做 gate 退化比对（数字差异含模型差异，不全是代码退化）。

评估脚本覆盖质量、检索、性能三个维度：

```bash
# 1. RAGAS 四维质量评估（LLM-as-Judge，需 LLM API）
#    注意：默认 20 题抽样集 data/eval_dataset.json 未入库，主力评估请用 #4 端到端评估
uv run python run_eval.py
#    → data/eval_report.json (faithfulness/relevancy/precision/recall)

# 2. CMRC 检索命中评估（零 LLM，直接衡量检索管道质量）
uv run python run_retrieval_eval.py
#    → data/eval_report_cmrc.json (命中率/覆盖率/检索耗时)
#    当前数字以报告文件为准（300 题全量集口径）

# 3. 并发性能评测（需先启动服务，打真实 /api/chat）
uv run python main.py &            # 先起服务
uv run python run_concurrency_bench.py
#    → data/concurrency_report.json (QPS/P50/P95/错误率)

# 4. 端到端三层评估 + 特性 A/B（F5，需 LLM API）
uv run python run_e2e_eval.py --mode full --colloquial   # 全特性开 + 口语化查询检视
uv run python run_e2e_eval.py --mode baseline            # RAG1.0 基线（F1-F4 全关）
#    → data/eval_e2e_{full,baseline}.json (检索命中/生成忠实度/端到端F1/EM/命中 + A/B)

# 5. 切片评估：多跳（F6b）与多轮（F12）
uv run python run_e2e_eval.py --dataset data/eval_multihop.json --mode full      # 多跳 15 条
uv run python run_e2e_eval.py --dataset data/eval_multiturn.json --only F12      # 多轮 12 组
#    → data/eval_e2e_{multihop,multiturn}_*.json

# 6. F12 重写层评估（零 LLM，秒级，可进 CI）
uv run python run_rewrite_eval.py
#    → data/eval_rewrite_heuristic.json (触发率/改写率/gold 关键词命中率)

# 7. 质量卡点（eval-as-gate，需 LLM API）：真实链路退化即 exit 非零
uv run python run_e2e_eval.py --smoke --gate            # push 前：smoke 鲁棒集卡点
uv run python run_e2e_eval.py --mode full --gate        # 发版前：叠加端到端基线退化
uv run python run_e2e_eval.py --mode full --update-baseline  # 确认未退化后刷新基线

# 8. 语义裁判（--judge，需 LLM API）：每样本 +1 次 LLM 调用，比词法 F1/hit 更真的质量信号
uv run python run_e2e_eval.py --mode full --judge       # 追加 end_to_end.avg_correctness
```

**切片评估实测结论（2026-07-25，详见[验证报告](./docs/superpowers/reports/2026-07-25-eval-closure-report.md)）**：

- **F12 多轮**：重写层启发式路径 12/12 全命中（触发率/改写率/关键词命中率均 100%）；
  端到端检索命中率 0.67→0.75（+8pp），典型样本「它的解决方案有哪些？」覆盖率 0→0.60。
  但启发式话题回填是双刃剑——话题噪声词也会拉低已可检索样本（mt7 覆盖率 0.83→0），
  端到端 F1 持平（0.297→0.291）。**改写质量决定多轮收益**，LLM 重写路径留作后续优化。
- **F6b 多跳**：规则路由（F4）在本多跳集上仅判出 1/15 为 multi_hop，分解率 6.7%，
  分解收益无法有效测量；`--only F6` 实验同时证实分解触发依赖 F4 路由输出（路由关则分解率 0）。
  **多跳分解的瓶颈在路由召回而非分解器本身**，路由 multi_hop 规则优化留作下一步。

> 评估口径说明：检索质量以**与 LLM 解耦的 CMRC 评估**为准（命中率 100%）；
> RAGAS 四维受生成/评判模型影响，跨模型对比时需注明口径。详见
> [docs/superpowers/reports/2026-07-22-task11-validation-report.md](./docs/superpowers/reports/2026-07-22-task11-validation-report.md)。

### 质量卡点（eval-as-gate）

`pytest` 全 mock 了 LLM/embedding，只能保证代码不崩，**测不到真实检索/生成质量**。
`app/evaluation/gate.py` 对 `run_e2e_eval` 的汇总做"退化即 `exit 2`"判定，补这道真实链路闸门：

- **smoke**（`--smoke --gate`，每次 push 跑，约 1–2 分钟）：卡检索命中率 / 答案入 top 上下文率 /
  延迟上限 / 链路健康度——这些取决于检索+embedding，对生成措辞抖动不敏感，适合小样本。
- **full**（`--mode full --gate`，发版前手动跑）：在 smoke 之上叠加端到端 F1 / 命中率的
  **相对基线退化**判定（`data/eval_gate_baseline.json`，用容差吸收 LLM 非确定性抖动，避免误报）。
- **启用 pre-push 自动卡点**（默认不启用）：`git config core.hooksPath .githooks`；
  脚本会先检查 LLM key，无 key 友好跳过、不阻断无 key 协作者。

> ⚠️ **能力边界（诚实）**：smoke **有意不卡**端到端 F1——小样本+生成抖动会把真实信号淹没，
> 卡死只会变成误报工厂、被 `--no-verify` 绕过。生成 prompt 改坏的缓慢退化要靠 **full 在发版前拦**。
> gate 是回归的早期预警+发版闸门，不是"质量不退的银弹"；与 mock 单测**互补**。
> ⚠️ **更新基线陷阱**：`--update-baseline` 前必须人工确认本次质量未退化，否则等于废掉卡点。
> 后续接 CI：在 `.github/workflows` 加 `uv run python run_e2e_eval.py --smoke --gate`，
> 配 `OPENAI_API_KEY` secret 并用 `if: secrets.OPENAI_API_KEY != ''` 让无 key 时自动 skip。

### 生成质量（prompt 重构）

端到端 F1 偏低的瓶颈**不在检索**（命中率 1.0、答案入 top 上下文率 0.97），而在生成侧。
对 31 条失分样本逐条归因，两类损失各占约一半：

- **冗长/格式失配**：答案实质正确，却被「根据参考文档，…完整句… [来源: 文档1]」+ 列表分段稀释——
  字符级 F1 的 precision 被长答案压低，EM 因模板恒带额外汉字而**结构性恒 0**。
- **过度拒答**：上下文里明明有答案（入 top 上下文率 0.97 为证），模型却以「文档中未涉及」搪塞，
  根因是生成保守而非检索截断，集中在概念/多跳题。

为此把四个生成 prompt（`app/generation/prompts.py`）统一为四条约束：**① 答案优先**（首句直接给答案，
禁「根据参考文档」开场）、**② 简洁聚焦**（不堆列表分段）、**③ 去掉正文内联来源**（溯源交由 F7 结构化
引用 + 前端来源面板，不丢）、**④ 软化拒答**（只要文档含答案就作答，仅当确实完全没有才说明无法回答）。
防幻觉硬约束「仅基于文档」与忠实度自检（F3）**保留不动**，作为软化拒答后的安全网。
`tests/test_generation_prompts.py` 以渲染后文本锁定这四条，防回退。

**实测**（CMRC 30 条可评分集，同口径前后对比）：`avg_f1` **0.364→0.521（+43%，24/30 样本上升）**、
`em_rate` **0.0→0.033（破零）**、忠实度 **0.767→0.833（不降反升，软化拒答未牺牲防幻觉）**、疑似过度拒答
**12→9**；`hit_rate` 0.433→0.300，但 4 个掉点样本 F1 全部同步上升（长句型 gold 的子串假象，见下）。
正文内联来源移除后 F7 块级引用 1.53→0.97 条/答（溯源仍在）。

> **指标解读（诚实）**：EM 与 `hit_rate`（要求 gold 作为连续归一化子串精确出现）对「长句型 gold vs
> 聚焦/改写答案」天然苛刻——实测 prompt 重构后 4 个 hit 掉点样本的 `avg_f1` **同时全部上升**，即 hit 掉点
> 是测量假象而非质量回退。故以 **`avg_f1` 为主信号**，EM/hit 仅作参照，**不要追 EM=1 / hit=1**；
> `f1_short` / `em_rate_short`（F10 短答案）是有意义的 EM 参照。
> **语义裁判（更真的信号）**：词法 F1/hit 量的是「span 复现度」而非「答案质量」。`run_e2e_eval --judge`
> 追加 LLM-as-judge 的 `end_to_end.avg_correctness`（`app/evaluation/metrics.py::answer_correctness`）——
> 允许同义改写、语序调整、详略差异，只看语义是否等价于 gold，正是对上述苛刻性的纠偏。每样本 +1 次 LLM
> 调用，故**默认关、仅 full 用**；gate 以 full-only + 相对基线（容差 0.05）接入，未跑 judge 时键缺失自动
> 跳过、向后兼容。发版前建议跑 `--mode full --judge` 用语义正确率复核词法指标看不出的改写正确性。
> **数据卫生**：抽取式数据偶有 gold 为空占位（如 CMRC 源文档实体缺失留下的 `《""》`），归一化后为空 →
> F1/EM/hit 结构性恒 0，`run_e2e_eval` 会透明跳过并打印这些样本 id，不计入均值（非刷分，正当清洗）。
> **残差归因更正（compress_context 自伤，已修）**：上版曾把 prompt 重构后残存的 11 条拒答（id
> 2/4/9/10/12/13/14/17/18/21/27，`answer_in_top_context_rate=1.0`、`avg_correctness` 卡 0.61）定性为
> 「deepseek-v4-flash 抽取能力天花板、生成侧 prompt 杠杆用尽」。**该结论已被推翻**：两轮纯 prompt 实验阴性
> 是对的（prompt 确非杠杆），但根因不在模型，而在生成侧的上下文压缩 `compress_context`（`app/generation/chain.py`）
> ——旧版用「整段中文 token」粗粒度匹配、每篇硬留 top-3 句，密集段落里答案句与 query 的**词/子串重合**
> （如「双月十六日」）抓不到 → 重叠分 0 → **答案句被压缩丢弃**，模型只看到无关开场白才「合理拒答」。逐条实锤：
> 11/11 拒答样本的 gold 答案句曾被旧版丢弃。修复 = 换 jieba 词级分词 + 「相关句（重合>0）必留、无相关句整篇
> 兜底」，并统一 `generate`/`generate_stream` 两条路径（此前流式不调压缩、行为不一致）。同口径 A/B：
> `avg_correctness` **0.61→0.9533**、`avg_f1` **0.521→0.704**、忠实度 **0.833→0.9333**、重生成率
> **0.267→0.067**、**拒答 11→0**，检索命中率仍 1.0。`tests/test_compress_context.py` 锁定「答案句必留」防回退。
> **诚实残差**：修复后 0 拒答、语义正确率 0.95，剩余失分才真正属模型能力边界；再提需换更强生成模型，检索/压缩已非瓶颈。

## 运行测试

```bash
pytest tests/ -v
```

## 项目结构

```
├── main.py                     # FastAPI 入口（lifespan 初始化 + 并发闸门）
├── config.py                   # 统一配置 + build_chat_llm LLM 构造工厂
├── run_eval.py                 # RAGAS 四维质量评估
├── run_retrieval_eval.py       # CMRC 检索命中评估（零 LLM）
├── run_concurrency_bench.py    # 并发性能评测
├── run_e2e_eval.py             # 端到端三层评估 + 特性 A/B（支持多跳/多轮 slice）
├── run_rewrite_eval.py         # F12 重写层评估（零 LLM，秒级）
├── app/
│   ├── utils.py                # 零依赖共享工具（extract_json）
│   ├── ingestion/              # 文档摄入
│   │   ├── loader.py           # 多格式加载
│   │   ├── chunker.py          # 智能分块
│   │   ├── indexer.py          # 层级索引（含 F6a 双集合 + detail_store 切换）
│   │   ├── contextual.py       # F6a 上下文增强（索引期一次性 LLM）
│   │   ├── service.py          # 摄入编排（ingest_file / rebuild_enhanced_indexes）
│   │   └── graph_extractor.py  # 知识图谱抽取（LLM 类型化三元组 + chunk 溯源）
│   ├── retrieval/              # 检索模块
│   │   ├── pipeline.py         # ★ RetrievalPipeline 7 阶段统一编排 + search() 复合原语
│   │   ├── dense.py            # 稠密检索
│   │   ├── sparse.py           # BM25 稀疏检索
│   │   ├── graph_retriever.py  # Graph 多跳召回
│   │   ├── parent_child.py     # Parent-Child 检索
│   │   ├── fusion.py           # RRF 融合 + chunk_key/dedup_by_chunk_id 去重原语
│   │   ├── reranker.py         # Cross-Encoder 重排（先查 L2 缓存）
│   │   ├── autocut.py          # F1 Kneedle 膝点自适应截断
│   │   ├── query_transform.py  # 查询改写（Multi-Query/HyDE/Refine/F6b Decompose）
│   │   ├── crag.py             # CRAG 门控/评估/补救
│   │   ├── router.py           # F4 查询路由（is_numeric_question 权威定义）
│   │   ├── conversation.py     # F12 历史感知查询重写
│   │   ├── agent.py            # F13 Agentic ReAct 自主检索
│   │   ├── deadline.py         # 查询级时延预算（全局熔断可选阶段）
│   │   └── caching.py          # 三级缓存（L1 embedding / L2 rerank / L3 语义响应）
│   ├── generation/             # 生成模块
│   │   ├── chain.py            # RAG Chain（薄编排层：_rewrite_and_retrieve/_finalize）
│   │   ├── faithfulness.py     # F3 忠实度自检（regen_until_faithful 共用循环）
│   │   ├── citation.py         # F7 引用溯源（embedding 余弦，零 LLM）
│   │   ├── streaming.py        # F8 投机流式忠实度
│   │   ├── answer_boost.py     # F10 短答案抽取 + 自一致性
│   │   └── prompts.py          # Prompt 模板
│   ├── api/                    # API 层
│   │   ├── routes.py           # 路由（并发闸门 + SSE，HTTP 关切）
│   │   ├── schemas.py          # 数据模型
│   │   └── security.py         # F11 API Key 鉴权 + 限流
│   ├── evaluation/             # 评估模块
│   │   ├── metrics.py          # RAGAS 指标（LLM-as-Judge）
│   │   └── dataset.py          # 测试数据集
│   └── observability/          # 可观测性（tracing 链路追踪 / metrics 指标注册表）
├── frontend/app.py             # Streamlit 前端
├── data/sample_docs/           # 示例文档
└── tests/                      # 单元测试（全离线，mock LLM）
```
