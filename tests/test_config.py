"""配置项测试"""
from config import Settings


def test_new_pipeline_settings_defaults():
    # _env_file=None: 隔离本地 .env（如 MAX_CONCURRENT_REQUESTS=8），只验证代码默认值
    s = Settings(_env_file=None)
    assert s.use_summary_recall is True
    assert s.use_crag_gate is True
    assert s.recall_max_workers == 6
    assert s.max_concurrent_requests == 4
    assert s.request_queue_timeout == 30.0
