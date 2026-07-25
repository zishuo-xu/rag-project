"""查询级时延预算（Deadline）- 零依赖，注入时钟可测

动机（2026-07-26 延迟治理）：
    full 模式 42.6s 均值实为单点 486s 离群拖拽（14/15 样本均值 10.9s）。
    各阶段超时各自为政、无全局预算，超时/重试可叠加成风暴。
    Deadline 给整条查询链路一个全局预算：可选阶段（F2 迭代、F3 重生成）
    在超预算时整体跳过并记录 `skipped`，保证典型路径不受影响、离群尾被熔断。

设计取舍：
    - budget_ms <= 0 表示不启用（零行为变化，回归安全）
    - 只熔断「可选且可降级」的阶段，不中断主链路（召回/生成照常）
    - 时钟可注入（clock 参数），测试无需真实 sleep
"""

import time


class Deadline:
    """全局时延预算计时器。`check_skip(stage)` 超预算时记录并返回 True。"""

    def __init__(self, budget_ms: int, clock=None):
        # 非数值预算（如测试 MagicMock settings）视为关闭，保证回归安全
        self._budget_ms = budget_ms if isinstance(budget_ms, (int, float)) else 0
        self._clock = clock or time.monotonic
        self._start = self._clock()
        self.skipped: list[str] = []

    def elapsed_ms(self) -> float:
        return (self._clock() - self._start) * 1000

    def exceeded(self) -> bool:
        """budget_ms <= 0 时恒不超（功能关闭）。"""
        return self._budget_ms > 0 and self.elapsed_ms() >= self._budget_ms

    def check_skip(self, stage: str) -> bool:
        """超预算 → 记录被跳过的阶段名并返回 True（调用方据此降级）。"""
        if self.exceeded():
            self.skipped.append(stage)
            return True
        return False
