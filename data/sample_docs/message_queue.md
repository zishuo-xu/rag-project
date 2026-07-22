# 消息队列核心原理与实战

## 消息队列概述

消息队列（Message Queue）是分布式系统中实现异步通信和解耦的核心中间件。它通过在生产者和消费者之间引入中间层，实现流量削峰、系统解耦和最终一致性。

核心应用场景：
- 异步处理：注册后异步发送邮件/短信，不阻塞主流程
- 流量削峰：秒杀场景下将突发请求写入队列，后端匀速消费
- 系统解耦：订单系统无需直接调用库存、物流、积分系统
- 数据管道：日志采集、CDC（Change Data Capture）实时同步

## Kafka 架构与原理

### 核心概念
| 概念 | 说明 |
|------|------|
| Broker | Kafka 服务节点，多个 Broker 组成集群 |
| Topic | 消息的逻辑分类，类似数据库中的表 |
| Partition | Topic 的物理分片，是并行度的基本单位 |
| Offset | 每条消息在 Partition 内的唯一递增编号 |
| Consumer Group | 消费者组，组内每个 Partition 只被一个消费者消费 |

### 存储设计
- 每个 Partition 对应磁盘上一组 Segment 文件（.log + .index + .timeindex）
- 顺序写磁盘，性能接近内存（600MB/s+）
- 零拷贝（sendfile）：数据从磁盘直接到网卡，不经过用户空间
- 页缓存（Page Cache）：依赖 OS 缓存热数据，JVM 堆外内存

### 副本与高可用
- 每个 Partition 有 N 个副本：1 个 Leader + (N-1) 个 Follower
- ISR（In-Sync Replicas）：与 Leader 保持同步的副本集合
- 写入策略：acks=all 时要求所有 ISR 副本确认
- Leader 选举：从 ISR 中选出新 Leader（Unclean 选举可能丢数据）

### 消费者机制
- Pull 模式：消费者主动拉取消息，可控制消费速率
- Rebalance：消费者加入/离开时重新分配 Partition
- Offset 提交：自动提交（可能重复/丢失）vs 手动提交（精确控制）
- 幂等消费：业务层通过唯一 ID 去重

## RocketMQ 特性

### 架构组件
- NameServer：轻量级注册中心（无状态，可集群部署）
- Broker：消息存储和转发（Master-Slave 模式）
- Producer：消息发送方（支持同步/异步/单向发送）
- Consumer：Push 模式（实际是长轮询）和 Pull 模式

### 消息类型
| 类型 | 说明 | 典型场景 |
|------|------|----------|
| 普通消息 | 无特殊语义 | 日志、通知 |
| 顺序消息 | 同一 Queue 内 FIFO | 订单状态变更 |
| 延迟消息 | 指定延迟级别后投递 | 超时取消订单（30min） |
| 事务消息 | 半消息 + 本地事务 + 回查 | 分布式事务最终一致 |

### 事务消息原理
1. Producer 发送半消息（Half Message）到 Broker
2. Broker 存储半消息（对 Consumer 不可见）
3. Producer 执行本地事务
4. 根据本地事务结果 Commit/Rollback
5. 若 Broker 长时间未收到确认，主动回查 Producer 事务状态

## 消息可靠性保证

### 生产端
- 同步发送 + 重试（Kafka: retries=3, retry.backoff.ms=100）
- 事务消息（RocketMQ）
- 发送确认机制（acks）

### Broker 端
- 多副本同步复制（Kafka: min.insync.replicas=2）
- 刷盘策略：同步刷盘（可靠但慢）vs 异步刷盘（快但可能丢）
- 集群部署 + 故障自动切换

### 消费端
- 手动 ACK：处理完业务逻辑后再提交 Offset
- 幂等设计：唯一键约束 / 状态机 / Token 机制
- 死信队列（DLQ）：多次消费失败的消息进入死信队列人工处理

## 消息顺序性

### 全局有序
- 单 Partition/Queue，牺牲并行度
- 适用场景极少（如 binlog 同步）

### 局部有序（推荐）
- 同一业务 Key 路由到同一 Partition（如 order_id % partition_num）
- 单 Partition 内消费者单线程消费
- Kafka：key-based partitioner
- RocketMQ：MessageQueueSelector

## 性能优化

### Kafka 调优
- batch.size：批量发送大小（默认 16KB，可调大到 64KB）
- linger.ms：等待凑批时间（默认 0，设为 5-10ms 提升吞吐）
- compression.type：lz4 或 snappy 压缩减少网络 IO
- num.io.threads / num.network.threads：IO/网络线程数
- log.retention.hours：日志保留时间（默认 168h = 7天）

### 监控指标
- 消费延迟（Consumer Lag）：未消费消息数，核心告警指标
- 吞吐量（Messages/s）：生产/消费速率
- 请求延迟（Produce/Consume Latency）：P99 延迟
- ISR 收缩频率：频繁收缩说明副本同步有问题

## 消息队列选型对比

| 维度 | Kafka | RocketMQ | RabbitMQ |
|------|-------|----------|----------|
| 吞吐量 | 百万级/s | 十万级/s | 万级/s |
| 延迟 | ms 级 | ms 级 | μs 级 |
| 消息回溯 | 支持（按 Offset） | 支持（按时间） | 不支持 |
| 事务消息 | 0.11+ 支持 | 原生支持 | 不支持 |
| 延迟消息 | 不原生支持 | 18 个级别 | 插件支持 |
| 适用场景 | 日志/大数据管道 | 电商/金融业务 | 中小规模异步 |
