# Redis 核心原理与实战

## 数据结构与底层实现

Redis 不仅仅是 KV 缓存，它提供 5 种基础数据结构和多种高级数据结构。

### 五种基础类型
| 类型 | 底层编码 | 典型场景 |
|------|----------|----------|
| String | SDS（简单动态字符串） | 缓存、计数器、分布式锁 |
| Hash | ziplist / hashtable | 对象存储（用户信息） |
| List | quicklist（ziplist + linkedlist） | 消息队列、最新列表 |
| Set | intset / hashtable | 去重、交集并集（共同好友） |
| ZSet | ziplist / skiplist + hashtable | 排行榜、延迟队列 |

### 跳表（Skip List）
ZSet 的核心数据结构，平均 O(log N) 查找：
- 多层链表，每层是下层的"快速通道"
- 插入时随机决定层数（概率 p=0.25）
- 相比平衡树，实现简单、范围查询友好

## 持久化机制

### RDB（快照）
- 触发：`SAVE`（阻塞）/ `BGSAVE`（fork 子进程）
- 原理：fork 时利用 COW（Copy-On-Write），子进程遍历内存生成 dump.rdb
- 优点：恢复快、文件紧凑
- 缺点：可能丢失最后一次快照后的数据

### AOF（追加日志）
- 每条写命令追加到 aof 文件
- 刷盘策略：always / everysec（推荐）/ no
- 重写（Rewrite）：后台生成最小命令集，压缩文件体积
- Redis 7.0：Multi Part AOF（base + incr）

### 混合持久化（推荐）
RDB + 增量 AOF 结合，兼顾恢复速度和数据安全。

## 高可用架构

### 主从复制
- 全量同步：从节点发送 PSYNC，主节点 BGSAVE + 发送 RDB + 增量命令
- 增量同步：基于 replication offset + 环形缓冲区（repl_backlog）
- 作用：读写分离、数据备份

### Sentinel（哨兵）
- 监控主节点健康（PING + INFO）
- 自动故障转移：选举新主（优先级 → offset → runid）
- 通知客户端新主地址
- 至少 3 个哨兵节点（奇数，防脑裂）

### Cluster（集群）
- 16384 个 slot，分配到多个主节点
- 客户端重定向：MOVED（永久迁移）/ ASK（迁移中）
- Gossip 协议：节点间交换状态信息
- 故障检测：节点互相 PING，超时标记 PFAIL → FAIL

## 缓存问题解决方案

### 缓存穿透（查不存在的数据）
- 布隆过滤器：在缓存前拦截不存在的 key
- 空值缓存：`SET key "" EX 60`
- 参数校验：ID 格式不合法直接拒绝

### 缓存击穿（热点 key 过期）
- 互斥锁：`SET lock_key 1 NX EX 10`，只允许一个线程回源
- 逻辑过期：不设 TTL，后台异步更新
- 永不过期 + 主动更新

### 缓存雪崩（大量 key 同时过期）
- 过期时间加随机偏移：`EX base + random(0, 300)`
- 多级缓存：本地缓存（Caffeine）+ Redis
- 熔断降级：DB 压力过大时返回兜底数据

### 缓存与数据库一致性
- **先更新 DB，再删缓存**（推荐）
- 延迟双删：更新 DB → 删缓存 → sleep(500ms) → 再删缓存
- 订阅 binlog（Canal）异步更新缓存
- 最终一致性：设置合理 TTL 兜底

## 分布式锁

### 基本实现
```bash
# 加锁（原子操作）
SET resource_lock unique_value NX PX 30000

# 解锁（Lua 脚本保证原子性）
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
```

### RedLock（多节点）
向 N 个独立 Redis 节点加锁，超过半数成功才算获取锁。
争议：Martin Kleppmann 指出时钟跳跃问题，实际生产中需权衡。

### 锁续期
看门狗机制（Redisson 实现）：后台线程每 TTL/3 续期，防止业务未完成锁就过期。

## 性能优化

### 大 Key 问题
- 定义：String > 10KB，Hash/List/Set/ZSet 元素 > 5000
- 危害：阻塞其他请求、网络带宽、内存不均
- 解决：拆分、压缩、异步删除（UNLINK）

### 热 Key 问题
- 发现：`redis-cli --hotkeys`、监控平台
- 解决：本地缓存、读写分离、Key 打散（加后缀）

### Pipeline 与 Lua
- Pipeline：批量命令一次网络往返，减少 RTT 开销
- Lua 脚本：服务端原子执行多条命令，避免竞态
