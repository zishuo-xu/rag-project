"""配置项测试"""
from config import Settings


def test_new_pipeline_settings_defaults():
    s = Settings()
    assert s.use_summary_recall is True
    assert s.use_crag_gate is True
    assert s.recall_max_workers == 6
    assert s.max_concurrent_requests == 4
    assert s.request_queue_timeout == 30.0
