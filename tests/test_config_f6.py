"""F6 配置开关默认值"""
from config import Settings


def test_f6a_defaults():
    s = Settings()
    assert s.use_contextual_chunks is True
    assert s.contextual_max_chars == 80
    assert s.chroma_contextual_collection == "chunks_contextual"


def test_f6b_defaults():
    s = Settings()
    assert s.use_decomposition is True
    assert s.decomposition_max_subquestions == 4
    assert s.decomposition_max_hops == 2  # 3→2：延迟治理（2026-07-26），每跳省 1 次串行 refine
