# 让 LLM 学会停止:一个 Agentic RAG finish 率从 33% 到 93% 的三轮迭代

> Agent 最尴尬的失败方式不是答错,是停不下来:步数预算打满、重复检索、迟迟不肯 finish。这篇记录我如何花两轮迭代证明"提示词解决不了这个问题",再用一轮硬机制把它修好。包含一次反直觉的失败——那一轮我让信号完全可见,finish 率反而下降了。

## 背景:一个零依赖的 ReAct 状态机

我需要一个 Agentic 检索路径:面对"范廷颂担任总主教的那个教区在哪里?"这类需要组合两个事实的问题,固定七阶段管道力不从心,得让 agent 自己决定"检索什么、要不要拆、什么时候停"。

实现上我没引 LangGraph,而是按它的概念(State/Node/条件边)手写了约 200 行 ReAct 循环,零新依赖、全离线 mock 可测。工具集复用现有管道阶段:`search`(召回+融合+重排)、`decompose`(多跳分解)、`grade`(CRAG 评估)、`finish`。护栏:max_steps=4 硬上限、决策解析失败即停、任何异常降级回七阶段管道。

第一版跑多跳评测集(15 题),数据立刻打脸:

- F1 0.283,四种模式里最高——能力是有的;
- **finish 率 33%(5/15)**:三分之二的查询打满 4 步预算才停;
- 平均 3.5 步,search 动作 43 次,decompose 只触发 3 次。

agent 像一個不知道什么时候该离场的客人,反复 search 同一个意思的查询。问题明确:**停止决策失控**。

## 第一轮(v2):加强 prompt 约束——工具修好了,停止没修好

诊断:决策 prompt 太软。agent 不知道"什么情况下必须分解"、"什么情况下必须停"。

修复三件事:① prompt 加硬工具条件("需组合两个及以上事实时**优先 decompose**"、"已有 ≥1 篇相关证据**立即 finish**"、"禁止重复/雷同查询");② 加两条 few-shot(单事实 search→finish / 双事实 decompose→finish);③ 收敛护栏:连续 2 次空检索强制停。

复测:**decompose 从 3 次跳到 13-14 次(稳定触发),F1 0.286→0.298。工具选择修好了。**

但 finish 率:47%(7/15),**仍未达标**。8-12/15 的查询依然打满步数。

这一轮给了我第一个认知:prompt 工程能校正**工具选择偏好**(decompose 触发率翻四倍就是证据),但校正不了**停止决策**。为什么?因为"何时停"需要 agent 对自己的证据状态做判断,而它对状态的感知是残缺的。

## 第二轮(v3):让信号完全可见——一次反直觉的失败

顺着上面的推理,我把根因定位为"信号不可见":

- 每步新增了多少证据,算出来了但没给 LLM 看;
- CRAG 分级结果从不进 prompt;
- 步数预算(第几步/共几步)从不告知。

修复(全部零新增 LLM 调用):步数可见("第 i/N 步" + "末步必须 finish" 规则)、新增量可见(observation 前缀 `[新增N篇/累计M]`)、证据状态可见(`共N篇｜grade=X｜连续K步零新增`)、空手 finish 驳回一次给纠错机会。

预期:信号到位,agent 自然早停。

**实测:延迟 12.5s→9.7s(↓23%),agent 确实少做无效检索了。但 finish 率 40%/33%——不升反降。**

这是整个迭代里最值钱的一组数据。它推翻了一个看起来不言自明的假设:

> **给 LLM 看收敛信号 ≠ 能控制它的停止决策。**

信号可见化改变了 agent 的检索效率(延迟降了),但 finish 依然是一个"自主决策"——而自主决策意味着它可以看了所有信号之后,继续选择 search。提示词能影响偏好,不能强制执行。

我在报告里如实写下未达标,并把下一轮方向从"更好的提示词"改为"硬机制":**把停止权从 LLM 手里拿过来。**

## 第三轮:证据充分度硬门控——零 LLM 的接管

设计原则:停止判据必须是**机器可计算的证据充分度**,而不是模型的自我感觉。

系统里恰好有现成的充分度信号:CRAG 分级。它在 7/28 已经被改造成零 LLM 实现——取 top1 文档的 rerank 分数做 sigmoid 归一化,`p≥0.5` 判 correct(证据充分)。于是硬门控几乎是"接线"而非"新建":

```python
# search/decompose 带来新证据后:
if step.action in ("search", "decompose") and new_count > 0 and settings.agentic_evidence_gate:
    sufficient, grade = self._evidence_sufficient(question, evidence)
    if sufficient:
        step.observation += f"(硬门控:CRAG={grade},证据充分,强制结束)"
        result.stop_reason = "evidence_sufficient"
        break

def _evidence_sufficient(self, question, evidence):
    # 累积证据来自多步、可能乱序;CRAG 取 top1 判级,需按 rerank 分降序
    scored = [(float(d.metadata["rerank_score"]), d) for d in evidence
              if isinstance(d.metadata.get("rerank_score"), (int, float))]
    if not scored:
        return False, ""   # 无真实分数(纯 RRF 结果)不启用,避免"无分数默认通过"误判
    scored.sort(key=lambda x: x[0], reverse=True)
    grade, _, _ = self._pipeline.evaluate(question, [d for _, d in scored])
    return grade == "correct", grade
```

两个容易踩的坑,都在代码里处理了:其一,累积证据来自多步检索、顺序是乱的,而 CRAG 依赖 top1,所以必须重排降序再送入;其二,decompose 的合并结果走纯 RRF、没有 rerank 分数,而 CRAG 对无分数输入"默认通过",若不显式拦截,门控会在每次分解后误触发。

**复测(同 15 题多跳集):**

| 指标 | 历史最佳(v3b,纯 prompt 信号) | 硬门控 |
|---|---|---|
| finish 率 | 53%(8/15 agent_done) | **93%(14/15 evidence_sufficient)** |
| 平均步数 | 3.6 | **1.53** |
| 检索 hit | 0.80(四轮迭代没动过) | **0.93** |
| e2e F1 / EM | 0.291 / 0 | **0.566 / 0.133** |
| 平均延迟 | — | 5.9s |

finish 率之外,最意外的是检索 hit 从 0.80 跳到 0.93。机制是这样的:以前 agent 打满 4 步,反复 search 把大量弱相关文档堆进证据池,收尾重排的 top_k 被噪声挤占;**硬门控让它带着第一批高质量证据就停,证据池干净,top_k 里留下正确答案的概率反而高了。** 早停不只是省延迟,它是一次降噪。

## 诚实的边界

- n=15,单次运行,±5pp 内无统计意义;
- F1 0.291→0.566 的幅度远超方差,但同期还有语料扩容(100→165 篇)和 CRAG 零 LLM 化生效,**无法把全部收益单独归因给硬门控**——报告里写了全部混血因素;
- 门控阈值(0.5)复用的是 CRAG 的既有校准,没有为 F13 单独调参;若主路径换 provider/换 reranker,分数分布平移时这条阈值要重新看。

## 三点复盘

1. **两轮失败不是浪费,是排除法。** v2 证明 prompt 能修工具选择,v3 证明 prompt 修不了停止决策——合起来才推出"停止权必须外置"。如果第一轮失败就直接上硬门控,我写不出"信号可见 ≠ 决策控制"这个有数据支撑的结论。
2. **最好的修复是复用已有信号。** 硬门控没有引入任何新模型调用,它只是把 CRAG 的判级结果接到了循环的 break 上。零 LLM、零新增延迟、可离线测试——这类修复比"加一个评估模型"健壮得多。
3. **把失败写进叙事。** 这个项目最让我有底气的一段面试材料,不是 93% 这个数字,是"我先实证了提示词的边界,再按诊断换机制"这条路径——方向是从数据里推出来的,不是碰巧试对的。

---

*实现见 `app/retrieval/agent.py`(`_evidence_sufficient` + 主循环门控段),开关 `agentic_evidence_gate`,6 条离线单测覆盖(含乱序证据、无分数拦截、开关旁路)。切片数据:`data/eval_e2e_f13.json`;历史对照快照 `data/eval_e2e_multihop_f13_*.json`。*
