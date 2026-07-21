# 计算机网络核心知识

## TCP/IP 四层模型

| 层级 | 协议 | 职责 |
|------|------|------|
| 应用层 | HTTP, DNS, FTP, SMTP | 用户交互 |
| 传输层 | TCP, UDP | 端到端通信 |
| 网络层 | IP, ICMP, ARP | 路由寻址 |
| 链路层 | Ethernet, WiFi | 相邻节点传输 |

## TCP 三次握手

```
Client → Server: SYN, seq=x
Server → Client: SYN+ACK, seq=y, ack=x+1
Client → Server: ACK, seq=x+1, ack=y+1
```

为什么是三次而不是两次？
- 防止已失效的连接请求到达服务器，导致服务器白白开启连接浪费资源
- 三次握手确保双方都确认了对方的收发能力

## TCP 四次挥手

```
Client → Server: FIN, seq=u
Server → Client: ACK, ack=u+1
Server → Client: FIN, seq=w
Client → Server: ACK, ack=w+1
```

为什么是四次？因为 TCP 是全双工的，每个方向需要单独关闭。
TIME_WAIT 状态等待 2MSL（通常 60s），确保最后的 ACK 到达对方。

## TCP 拥塞控制

四个阶段：
1. **慢启动**：cwnd 从 1 开始，每 RTT 翻倍（指数增长）
2. **拥塞避免**：cwnd 超过 ssthresh 后，每 RTT +1（线性增长）
3. **快重传**：收到 3 个重复 ACK，立即重传（不等超时）
4. **快恢复**：ssthresh = cwnd/2，cwnd = ssthresh，进入拥塞避免

## HTTP 协议演进

### HTTP/1.1
- 持久连接（Keep-Alive）
- 管道化（Pipelining）：有队头阻塞问题
- 文本协议，头部冗余大

### HTTP/2
- 二进制分帧
- 多路复用：一个 TCP 连接上并行多个请求/响应
- 头部压缩：HPACK 算法
- 服务器推送
- 仍有 TCP 层的队头阻塞

### HTTP/3（QUIC）
- 基于 UDP，彻底解决队头阻塞
- 0-RTT 建连（复用连接参数）
- 内置 TLS 1.3
- 连接迁移（切换网络不断连）

## HTTPS 与 TLS

### TLS 1.2 握手（2-RTT）
1. Client Hello（支持的密码套件、随机数）
2. Server Hello（选定套件、证书、随机数）
3. Client 验证证书 → 生成预主密钥 → 用服务器公钥加密发送
4. 双方用三个随机数生成会话密钥

### TLS 1.3（1-RTT）
- 精简密码套件（只保留 AEAD）
- 密钥交换前移，1-RTT 完成握手
- 支持 0-RTT（PSK 恢复）

## DNS 解析

```
浏览器缓存 → OS缓存 → hosts文件 → 本地DNS服务器
→ 根DNS(.com/.cn) → 顶级域DNS → 权威DNS → 返回IP
```

- 递归查询：客户端 → 本地 DNS（本地 DNS 负责全部）
- 迭代查询：本地 DNS 分别问根、顶级域、权威

### DNS 优化
- CDN：根据用户 IP 返回最近节点
- DNS 负载均衡：同一域名返回多个 IP
- DoH（DNS over HTTPS）：防劫持

## 网络性能优化

### CDN 原理
将静态资源缓存到全球边缘节点，用户就近访问：
- 智能 DNS 调度到最近 PoP
- 回源策略：缓存未命中时回源拉取
- 适用：图片、JS/CSS、视频

### 长连接与连接池
- HTTP Keep-Alive 复用 TCP 连接
- 数据库连接池避免频繁建连
- gRPC 基于 HTTP/2 多路复用

### 延迟优化
- 减少 RTT：就近部署、CDN
- 减少传输量：压缩（gzip/brotli）、图片优化（WebP）
- 预加载：DNS prefetch、preconnect、preload
