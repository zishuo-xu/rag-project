# 评估闭环设计：F6b 多跳 / F12 多轮 量化补全

日期：2026-07-25
状态：已实施（与对话评审记录一致，用户批准方案 A 后落地）

## 背景与问题

RAG 3.0 验证报告有两处"未量化"缺口，直接削弱"每个特性都有数据支撑"的面试叙事：

1. **F6b 多跳分解**：`data/eval_multihop.json` 仅 3 条骨架，`ground_truth` 为 `TODO_核对知识库` 占位，且 mh2/mh3 知识库内本就无答案（构造缺陷），多跳收益从未量化。
2. **F12 多轮记忆**：CMRC 是单轮数据集，端到端评估从不携带 history，rewrite_rate=0，多轮收益未量化。

## 目标

- 多跳集扩到 12–15 条、gold 全部可核对，跑出 F6b 的 A/B 数据
- 新建多轮集 10–15 组，双层指标（重写层 + 端到端层）量化 F12
- 加数据集完整性防线测试，堵死"占位提交"流程漏洞

## 设计（已批准的关键决策）

### 多跳评估集（F6b）

- Schema 沿用现有字段，新增 `hops`（推理链记录，便于归因与面试讲解）
- 15 条：11 链式（chain=true）+ 4 并行（chain=false），全部来自 CMRC 三篇文档的真实实体链
- 修正原 3 条：mh1 可答保留；mh2/mh3 改写为知识库可答问题
- gold 由语料原文起草 → 人工核对 → 提交；`build_multihop_eval.py` 保留核对警告

### 多轮评估集（F12）

- 新建 `data/eval_multiturn.json`，样本含 `history`（前置轮 QA）、`rewrite_gold`（改写后查询必含关键词）、`rewrite_type`（pronoun/ellipsis/mixed）
- 12 组，基于 sample_docs 六篇技术文档；追问设计保证命中 F12 启发式触发规则（指代词/呢结尾/连接词开头）
- 双层指标：
  - 重写层：`run_rewrite_eval.py` 离线调 `ConversationRewriter.rewrite`，零 LLM 秒级，可进 CI
  - 端到端层：`--slice multiturn`，baseline vs `--only F12` 对比 F1/hit

### harness 最小扩展（run_e2e_eval.py）

1. `eval_sample` 透传 `sample.get("history")` → `chain.invoke(chat_history=...)`（chain 原生支持，无 history 样本行为不变）
2. 结果记录 `rewritten_query` 字符串（原仅 bool）
3. `--slice` 增加 `multiturn`

### 指标口径

| 切片 | 跑法 | 核心指标 |
|---|---|---|
| multihop | baseline / full / `--only F6` 三跑 | F1、hit、answer_in_top_context、decomposed_rate |
| multiturn | baseline / `--only F12` 两跑 + 重写层 | F1、hit、rewrite 关键词命中率 |

已知风险：`--only F6` 时 F4 路由关闭，若分解依赖路由判定 multi_hop 则测不到分解 → 以 baseline vs full 差值为主口径，`--only F6` 为辅，报告中如实注明交互关系。

### 测试（全离线）

- `test_eval_dataset_integrity.py`：扫描 data/eval_*.json，断言无 TODO 占位、gold 非空、source 真实存在、multiturn/multihop schema 合法
- `test_e2e_multiturn.py`：history 透传 / None 默认 / rewritten_query 记录 / 异常降级 / slice 过滤
- `test_rewrite_eval.py`：重写层判定逻辑 + 真实数据集联合回归

### 文档收尾

README 评估章节、interview_guide.md（两处"待补"改实测结论）、验证报告 `docs/superpowers/reports/2026-07-25-eval-closure-report.md`。

## 非目标（YAGNI）

- 不做通用切片评估框架（仅两个新切片，抽象过度）
- 不重构图谱/Neo4j、不引入 Redis（属后续生产化方向）
- 多跳集不做不可答样本（不可答已由口语化检视覆盖）
