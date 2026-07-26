"""compress_context 上下文压缩契约测试（离线，零 LLM）

锁定 Phase1 修复：词级分词 + 相关句必留，防止回退到旧版「整段中文 token + 硬留
top-3」把答案句误删的行为——实测旧版对 11/11 拒答样本丢弃了 gold 答案句（答案明明
在检索上下文里，模型却因看不到而拒答），是端到端正确率偏低（语义 0.61）的生成侧主因。
"""
from langchain_core.documents import Document

from app.generation.chain import RAGChain


def _chain() -> RAGChain:
    """裸 chain：compress_context 是纯方法（仅用静态分词），无需重型 __init__。"""
    return RAGChain.__new__(RAGChain)


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "t.md"})


# 密集段落：多句无关开场 + 埋藏的答案句（与 query 有词重合）+ 更多无关句
DENSE = (
    "《太平天历》是太平天国的历法。它由冯云山所创制。太平军于咸丰元年攻克永安。"
    "其定一年为366天，单月31天，双月30天。谷雨、夏至、处暑皆在单月十七日。"
    "立夏、小暑、白露皆在双月初一日。小满、大暑、秋分、小雪、大寒皆在双月十六日。"
    "用太平天国名号纪元。以天干地支纪年、月、日。星期顺序仿西法。"
)
Q_SOLAR = "哪些节气在双月十六日？"


def test_answer_sentence_preserved():
    """核心回归锁：与 query 有词重合的答案句即使埋在密集段落里也必须保留。"""
    out = _chain().compress_context(Q_SOLAR, [_doc(DENSE)])
    assert "小满、大暑、秋分、小雪、大寒皆在双月十六日" in out[0].page_content


def test_irrelevant_sentences_compressed():
    """无关句（零重合）应被压缩掉，证明压缩仍生效（不是全量保留）。"""
    out = _chain().compress_context(Q_SOLAR, [_doc(DENSE)])
    text = out[0].page_content
    assert "冯云山所创制" not in text   # 无关开场句被丢弃
    assert len(text) < len(DENSE)       # 整体确实被压缩


def test_fallback_keeps_whole_doc_when_no_overlap():
    """无任何相关句时整篇保留兜底，绝不盲删潜在答案。"""
    doc = _doc("甲乙丙丁戊。己庚辛壬癸。子丑寅卯辰。")
    out = _chain().compress_context("量子色动力学的基本假设是什么？", [doc])
    assert out[0].page_content == doc.page_content


def test_single_sentence_doc_unchanged():
    doc = _doc("只有一句话没有句末标点的短文档")
    out = _chain().compress_context("任意问题", [doc])
    assert out[0].page_content == doc.page_content


def test_original_order_preserved():
    """保留的多个相关句维持原文顺序（前句索引 < 后句），无关句被压缩。"""
    doc = _doc("北京是中国的首都。今天天气很好。上海是中国的经济中心。")
    text = _chain().compress_context("北京和上海哪个更大？", [doc])[0].page_content
    assert "北京是中国的首都" in text and "上海是中国的经济中心" in text
    assert "今天天气很好" not in text                # 无关句被压缩
    assert text.index("北京") < text.index("上海")   # 原文顺序保持
