# 分布式系统理论与实践

## CAP 定理

CAP 定理（Brewer 定理）指出，分布式系统不可能同时满足以下三个特性，最多只能同时满足其中两个：

- **Consistency（一致性）**：所有节点在同一时刻看到相同的数据
- **Availability（可用性）**：每个请求都能在合理时间内得到非错误响应
- **Partition Tolerance（分区容错性）**：网络分区发生时系统仍能继续运作

### 实际选择
由于网络分区在分布式环境中不可避免，实际系统只能在 CP 和 AP 之间选择：
- **CP 系统**：ZooKeeper、etcd、HBase（牺牲可用性保证一致性）
- **AP 系统**：Cassandra、DynamoDB、Eureka（牺牲强一致保证可用）
- **注意**：CAP 不是非黑即白，可以在不同操作上做不同取舍（如购物车用 AP，支付用 CP）

## BASE 理论

BASE 是对 CAP 中 AP 方案的延伸，是大型互联网系统的实践经验：
- **Basically Available（基本可用）**：允许损失部分可用性（如降级、限流）
- **Soft State（软状态）**：允许中间状态存在（如数据同步中）
- **Eventually Consistent（最终一致性）**：经过一段时间后数据最终达到一致

最终一致性的实现方式：
- 读时修复（Read Repair）：读取时发现不一致则修复
- 反熵（Anti-Entropy）：后台定期对比并同步数据
- 补偿事务（Saga）：通过反向操作实现业务级一致性

## 一致性协议

### Raft 协议
Raft 是一种易于理解的分布式一致性算法（etcd、Consul 使用）：

**Leader 选举：**
1. 节点启动为 Follower，超时未收到心跳则转为 Candidate
2. Candidate 递增 Term，向所有节点发送 RequestVote
3. 获得多数票（N/2+1）即成为 Leader
4. 每个 Term 最多一个 Leader（投票互斥保证）

**日志复制：**
1. 客户端请求发送给 Leader
2. Leader 追加日志并发送 AppendEntries 给 Follower
3. 多数节点确认后 Leader 提交（commit）
4. Leader 通知 Follower 提交

**安全性保证：**
- 选举限制：Candidate 的日志必须至少和投票者一样新
- Leader 完整性：已提交的日志不会丢失
- 状态机安全：所有节点按相同顺序应用日志

### Paxos 与 Multi-Paxos
- Basic Paxos：两阶段（Prepare + Accept），每次只决定一个值
- Multi-Paxos：选举稳定 Leader 后跳过 Prepare 阶段，提升性能
- 应用：Google Chubby、Apache ZooKeeper（ZAB 协议，类 Paxos）

## 分布式事务

### 2PC（两阶段提交）
1. **Prepare 阶段**：协调者询问所有参与者是否可以提交
2. **Commit 阶段**：全部 Yes 则提交，任一 No 则回滚

问题：
- 同步阻塞：Prepare 后参与者锁定资源等待
- 单点故障：协调者宕机导致参与者无限等待
- 数据不一致：部分参与者收到 Commit，部分未收到

### 3PC（三阶段提交）
在 2PC 基础上增加 CanCommit 阶段和超时机制，减少阻塞但不能完全解决一致性。

### TCC（Try-Confirm-Cancel）
- **Try**：预留资源（如冻结库存）
- **Confirm**：确认执行（扣减冻结的库存）
- **Cancel**：取消预留（释放冻结的库存）

优点：不依赖数据库锁，性能好
缺点：业务侵入大，需实现三个接口

### Saga 模式
将长事务拆分为多个本地事务 + 补偿操作：
- 编排式（Orchestration）：中心协调器控制流程
- 协同式（Choreography）：通过事件驱动，各服务监听消息
- 适用：微服务间的长流程业务（如订单→支付→物流）

## 分布式 ID 生成

### 方案对比
| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| UUID | 128位随机数 | 简单、无中心 | 无序、占空间、索引性能差 |
| 数据库自增 | 单表 AUTO_INCREMENT | 有序、简单 | 单点瓶颈 |
| 号段模式 | 批量获取 ID 段缓存在本地 | 高性能、容灾 | 重启浪费号段 |
| Snowflake | 时间戳+机器ID+序列号 | 有序、高性能 | 时钟回拨问题 |
| Leaf（美团） | 号段+Snowflake 双模式 | 生产验证 | 依赖 ZooKeeper |

### Snowflake 结构（64位）
```
0 | 41位时间戳 | 10位机器ID | 12位序列号
```
- 41位时间戳：可用约 69 年
- 10位机器ID：最多 1024 个节点
- 12位序列号：每毫秒每节点 4096 个 ID

## 分布式锁

### 基于 Redis
- 单节点：SET key value NX PX timeout
- RedLock：向 N 个独立节点加锁，多数成功才算获取
- 问题：时钟跳跃、GC 暂停导致锁过期

### 基于 ZooKeeper
- 创建临时顺序节点，最小节点获得锁
- 监听前一个节点的删除事件（避免惊群）
- 优势：无需设置过期时间（会话断开自动释放）

### 基于 etcd
- 利用 Lease（租约）+ Revision 全局递增
- 事务操作保证原子性
- 适用：Kubernetes 生态

## 负载均衡策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 轮询（Round Robin） | 依次分配 | 节点性能均匀 |
| 加权轮询 | 按权重比例分配 | 节点配置不同 |
| 最少连接 | 分配给当前连接最少的节点 | 长连接场景 |
| 一致性哈希 | 按 Key 哈希到固定节点 | 缓存、分片 |
| P2C（Power of 2 Choices） | 随机选2个取较优 | 大规模集群 |

## 服务治理

### 熔断器（Circuit Breaker）
三种状态：Closed（正常）→ Open（熔断）→ Half-Open（探测恢复）
- 触发条件：错误率 > 阈值（如 50%）或慢调用比例 > 阈值
- 实现：Hystrix（已停更）、Sentinel、resilience4j

### 限流
- 令牌桶：允许突发，恒定速率生成令牌
- 漏桶：恒定速率处理，平滑流量
- 滑动窗口：统计窗口内请求数

### 降级
- 自动降级：熔断触发后返回兜底数据
- 手动降级：大促前关闭非核心功能
- 分级降级：按业务重要性分 P0/P1/P2 级别
