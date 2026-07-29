# 简历项目条目(定稿)

> 数字口径:300 题 CMRC 全量基线(2026-07-30,deepseek),可一键复现。
> 按简历篇幅选用:一行版(项目列表)/ 三条版(重点项目)/ STAR 口述版(面试)。

## 一行版

> **生产级中文 RAG 系统(独立设计与实现)**:检索命中 1.0、忠实度 0.99、端到端 F1 0.63、延迟 42.6s→5s(−89%);13 项特性全独立开关 + A/B 归因,负面实验如实留档;uv.lock 锁定依赖,陌生人 clone 可复现全部评测数字。

## 三条版(推荐)

**生产级中文 RAG 系统** —— LangChain + FastAPI + Streamlit,本地 embedding/reranker(bge 系列),300 题 CMRC 评测体系
- **检索与生成**:五路并行召回 + RRF 融合 + 本地 cross-encoder 重排 + Kneedle 自适应截断,检索命中/覆盖 1.00;生成侧经失败模式驱动的定向优化,端到端 F1 0.600→0.633(+3.3pp),忠实度 0.987(LLM-judge 逐 claim 校验 + 投机流式校正)
- **延迟治理**:查询级时延预算熔断 + 默认路径零在线 LLM 增量(每题仅 2 次调用)+ 流式投机校验,full 均值 42.6s→~5s(−89%);零 LLM 化改造经 60 题受控 A/B 验证:延迟 −50%、质量方向全正
- **Agentic RAG**:零依赖手写 ReAct 状态机(工具复用管道阶段),经三轮迭代(含一轮如实记录的失败)定位"信号可见 ≠ 决策控制",以零 LLM 证据硬门控将 finish 率 33%→93%;464 个离线测试 + eval-as-gate 质量门 + 复现链闭环(uv.lock,一键复现评测)

## 英文版(三条)

**Production-grade Chinese RAG System** — LangChain + FastAPI + Streamlit; local embedding/reranker (BAAI bge); 300-question CMRC evaluation suite
- **Retrieval & generation**: 5-channel parallel recall + RRF fusion + local cross-encoder reranking + Kneedle adaptive truncation (retrieval hit/coverage 1.00); failure-mode-driven prompt optimization lifted end-to-end F1 from 0.600 to 0.633; faithfulness 0.987 via per-claim LLM-judge verification with speculative-stream correction
- **Latency governance**: query-level deadline budgets, zero online-LLM increments on the default path (2 LLM calls/query), speculative streaming — full-path latency 42.6s → ~5s (−89%); a zero-LLM refactor validated by controlled A/B: −50% latency with no quality regression
- **Agentic RAG**: dependency-free hand-written ReAct state machine; three iteration rounds (one documented failure) established that "visible signals ≠ decision control", and a zero-LLM evidence gate raised the finish rate from 33% to 93%; 464 offline tests, eval-as-gate quality gates, and a closed reproduction chain (uv.lock; every reported number reproducible from a fresh clone)

## STAR 口述版(面试 2 分钟)

**S(情境)**:我需要一个能讲完整 RAG 工程故事的作品,目标不是刷分,而是"每条优化都能证明有效、每个无效结果都敢写下来"。

**T(任务)**:13 项特性(RAG 2.0 深度增强 + 3.0 生产级 + Agentic)全部独立开关、全部有 A/B 归因数据,同时把 full 路径延迟从 42.6s 压到可用,并让陌生人 clone 后能复现每一个数字。

**A(行动,挑两个最硬的讲)**:
1. *F13 finish 率*:agent 打满步数上限、finish 率只有 33%。第一轮加 prompt 硬约束,工具选择修好了但 finish 率 47% 仍不达标;第二轮让收敛信号完全可见,finish 率不升反降——我由此得出"信号可见 ≠ 决策控制",改用零 LLM 的证据充分度硬门控(CRAG 判 correct 即强制停),finish 率 93%、多跳 F1 0.291→0.566。两轮失败是这条结论的前提,报告里原样保留。
2. *零 LLM 化*:把热路径上的 CRAG 分级、查询改写从 LLM 改成规则/分数。改完没有直接宣布成功,而是搭了同索引、同题、同模型的受控 A/B:延迟 −50%,质量方向全正,忠实度 −0.017 如实标注为待观察。

**R(结果)**:检索饱和(1.0)、忠实度 0.99、F1 0.633、延迟 −89%;Graph RAG 升级 F1 −0.004 的 null 结果触发回退并留档;464 测试 + 质量门 + 复现链闭环。我认为最能代表这个项目质量的不是 0.633 这个数字,而是"每个数字都有对照、每个无效实验都有记录"。

## 可能被追问的诚实答案(备询)

- **F1 0.633 算什么水平?** CMRC 抽取式问答的 verbose RAG 场景,词法 F1 天然被措辞差异压低;同一系统的语义裁判口径(旧基线)是 0.96。我没有和公开榜单对比,因为语料是自建的——可比性只在系统内部(A/B),这点我在报告里写明了。
- **为什么不上云/不做多副本?** 有意为之:目标是可复现的教学样板,不是 SLO 生产系统;F9/F11 是单进程内存实现,多副本需 Redis,登记为路线图而非已完成。
- **最大的遗憾?** 生成侧 F1 有已知的天花板(词法指标对措辞敏感 + gold 形态多样),prompt 层杠杆已用尽;再上去要换更强的生成模型(5× 延迟换 +0.11 F1),这是均衡取舍,不是没做到。
