# 可观测性与监控体系

## 可观测性三大支柱

可观测性（Observability）是指通过系统的外部输出来理解系统内部状态的能力。现代可观测性建立在三大支柱之上：

### Metrics（指标）
- 定义：随时间聚合的数值测量（如 QPS、延迟 P99、CPU 使用率）
- 特点：低成本、可告警、适合趋势分析
- 类型：Counter（递增计数器）、Gauge（瞬时值）、Histogram（分布）、Summary（分位数）
- 工具：Prometheus、InfluxDB、Datadog

### Logging（日志）
- 定义：离散事件的文本记录（如请求日志、错误堆栈）
- 特点：信息丰富、可追溯、但存储成本高
- 最佳实践：结构化日志（JSON）、统一 TraceID 关联、合理日志级别
- 工具：ELK（Elasticsearch + Logstash + Kibana）、Loki + Grafana

### Tracing（链路追踪）
- 定义：记录请求在分布式系统中经过的每个服务和操作
- 核心概念：Trace（完整调用链）、Span（单个操作）、SpanContext（跨进程传播）
- 采样策略：头部采样（固定比例）、尾部采样（只保留慢/错误请求）
- 工具：Jaeger、Zipkin、OpenTelemetry、SkyWalking

## Prometheus 监控体系

### 架构设计
```
Service (/metrics) → Prometheus Server → PromQL → Grafana
                         ↓
                    Alertmanager → 通知渠道
```

### 数据模型
- 时间序列：metric_name{label1="value1", label2="value2"}
- 示例：http_requests_total{method="GET", handler="/api/chat", status="200"}
- 标签（Label）：用于过滤和聚合的键值对

### PromQL 常用查询
```promql
# QPS（每秒请求数）
rate(http_requests_total[5m])

# P99 延迟
histogram_quantile(0.99, rate(http_duration_seconds_bucket[5m]))

# 错误率
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# 内存使用率
container_memory_usage_bytes / container_spec_memory_limit_bytes * 100
```

### 告警规则
```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "API 错误率超过 5%"
```

## OpenTelemetry 标准

### 概述
OpenTelemetry（OTel）是 CNCF 的可观测性标准框架，统一了 Metrics、Logs、Traces 的采集方式。

### 核心组件
| 组件 | 说明 |
|------|------|
| API | 各语言的埋点接口（Java/Go/Python/JS） |
| SDK | API 的默认实现（采样、处理、导出） |
| Collector | 独立代理，接收/处理/导出遥测数据 |
| Exporter | 将数据发送到后端（Prometheus/Jaeger/OTLP） |

### Collector 管道
```
Receivers → Processors → Exporters
(OTLP/Jaeger)  (batch/filter)  (Prometheus/Jaeger/Loki)
```

### 自动埋点
- Java：Java Agent（-javaagent:opentelemetry-javaagent.jar）
- Python：opentelemetry-instrument 命令包装
- Go：手动埋点为主（otelhttp、otelgrpc 中间件）

## 日志工程实践

### 结构化日志
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "ERROR",
  "service": "order-service",
  "trace_id": "abc123def456",
  "span_id": "789ghi",
  "message": "支付超时",
  "order_id": "ORD-20240115-001",
  "duration_ms": 5023,
  "error": "PaymentGatewayTimeout"
}
```

### 日志级别规范
| 级别 | 使用场景 | 示例 |
|------|----------|------|
| ERROR | 需要人工介入的异常 | 数据库连接失败、支付异常 |
| WARN | 潜在问题但系统可自愈 | 重试成功、降级触发 |
| INFO | 关键业务节点 | 订单创建、用户登录 |
| DEBUG | 开发调试信息 | 请求参数、中间计算结果 |

### 日志采集架构
```
App → Filebeat/Fluentd → Kafka → Logstash/Flink → Elasticsearch → Kibana
```

## SLI / SLO / SLA

### 定义
- **SLI（Service Level Indicator）**：服务质量指标（如可用性 99.95%、P99 延迟 < 200ms）
- **SLO（Service Level Objective）**：内部目标（如月度可用性 ≥ 99.95%）
- **SLA（Service Level Agreement）**：对外合同承诺（如可用性 < 99.9% 赔偿）

### Error Budget（错误预算）
- 计算：1 - SLO = 允许不可用的比例
- 示例：SLO 99.95% → 月度错误预算 = 0.05% × 30天 = 21.6 分钟
- 用途：平衡发布速度和稳定性（预算耗尽则冻结发布）

### 四个黄金指标（Google SRE）
1. **Latency（延迟）**：请求耗时（区分成功和失败）
2. **Traffic（流量）**：系统负载量（QPS、并发连接数）
3. **Errors（错误）**：失败请求的比率
4. **Saturation（饱和度）**：资源使用率（CPU/内存/磁盘）

## 告警治理

### 告警分级
| 级别 | 响应时间 | 处理方式 | 示例 |
|------|----------|----------|------|
| P0 | 5分钟内 | 电话 + 即时响应 | 核心服务不可用 |
| P1 | 15分钟内 | 即时消息通知 | 错误率突增 |
| P2 | 1小时内 | 工单跟进 | 磁盘使用率 > 80% |
| P3 | 下个工作日 | 排期处理 | 非核心指标波动 |

### 告警收敛
- 去重：相同告警在恢复前只发一次
- 聚合：同一服务多个实例告警合并
- 抑制：上游故障时抑制下游级联告警
- 静默：维护窗口期间暂停告警

### On-Call 最佳实践
- 告警必须可操作（Actionable）：每条告警附带处理手册（Runbook）
- 避免告警疲劳：误报率 > 50% 的告警必须优化或删除
- 事后复盘（Postmortem）：P0/P1 事故必须写复盘文档
