"""生成 prompt 契约测试（离线，零 LLM）

锁定生成侧重构后的四条核心约束，防止后续改动悄悄回退到"冗长/强制内联来源/
默认拒答"的旧行为（旧行为是端到端 F1 偏低、EM 结构性恒 0 的生成侧根因）：

1. 答案优先：首句直接给答案，禁止"根据参考文档"开场。
2. 简洁聚焦：不堆列表分段。
3. 去掉正文内联来源：不再强制正文写 [来源: 文档名]（溯源交由 F7 结构化引用）。
4. 软化拒答：只要文档包含答案就作答，仅当"确实"完全没有才说明无法回答。
5. 防幻觉硬约束保留：软化拒答不得以"仅基于文档/绝不补全"为代价。

断言对象是模板渲染后的最终文本（LLM 实际所见），而非源码字面量。
"""
import pytest

from app.generation import prompts

ALL = ["simple", "chat", "strict", "strict_chat"]


def _rendered() -> dict:
    """渲染四个生成 prompt 为最终文本（填充占位变量）。"""
    kw = {"context": "参考文档内容", "question": "示例问题", "chat_history": []}
    return {
        "simple": prompts.RAG_SIMPLE_PROMPT.format(**kw),
        "chat": prompts.RAG_CHAT_PROMPT.format(**kw),
        "strict": prompts.STRICT_RAG_PROMPT.format(**kw),
        "strict_chat": prompts.STRICT_RAG_CHAT_PROMPT.format(**kw),
    }


@pytest.mark.parametrize("name", ALL)
def test_answer_first(name):
    text = _rendered()[name]
    assert "直接给出答案" in text          # 答案优先指令
    assert "根据参考文档" in text and "开头" in text  # 明确禁止 boilerplate 开场


@pytest.mark.parametrize("name", ALL)
def test_concise(name):
    assert "不堆列表" in _rendered()[name]  # 简洁聚焦，不堆列表分段


@pytest.mark.parametrize("name", ALL)
def test_no_mandatory_inline_source(name):
    # 旧版强制正文内联 [来源: 文档名] 已移除（溯源交由 F7 结构化引用 + 前端来源面板）
    assert "[来源" not in _rendered()[name]


@pytest.mark.parametrize("name", ALL)
def test_softened_refusal(name):
    text = _rendered()[name]
    assert "只要文档包含答案" in text  # 有答案就作答，不把"未涉及"当默认反射
    assert "确实" in text              # 仅当文档"确实"完全没有相关信息才拒答


@pytest.mark.parametrize("name", ALL)
def test_anti_hallucination_retained(name):
    # 软化拒答不得牺牲防幻觉：硬约束"仅基于文档、绝不补全文档中没有的内容"保留
    assert "文档中没有" in _rendered()[name]
