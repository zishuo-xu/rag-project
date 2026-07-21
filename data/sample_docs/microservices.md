# 微服务架构设计指南

## 什么是微服务

微服务架构是一种将应用程序构建为一组小型、独立部署的服务的设计方法。每个服务运行在自己的进程中，通过轻量级通信机制（通常是 HTTP/REST 或消息队列）进行交互。

与单体架构相比，微服务具有以下核心优势：
- **独立部署**：每个服务可以独立开发、测试和部署，不影响其他服务
- **技术异构**：不同服务可以使用不同的编程语言、数据库和框架
- **弹性伸缩**：可以针对高负载的服务单独扩容，而非整体扩容
- **故障隔离**：单个服务失败不会导致整个系统崩溃

## 服务拆分原则

### 按业务领域拆分（DDD）
使用领域驱动设计（Domain-Driven Design）中的限界上下文（Bounded Context）来划分服务边界。例如电商系统可拆分为：
- 用户服务（User Service）：注册、登录、权限管理
- 商品服务（Product Service）：商品CRUD、库存管理
- 订单服务（Order Service）：下单、支付、退款
- 物流服务（Logistics Service）：发货、追踪、签收

### 拆分粒度判断
服务不宜过大也不宜过小。判断标准：
1. 一个服务是否可以由一个小团队（2-pizza team）独立维护
2. 服务之间的通信频率是否过高（过高说明耦合严重）
3. 数据一致性要求是否允许最终一致性

## 核心基础设施

### 服务注册与发现
- **Consul**：HashiCorp 出品，支持健康检查、KV存储
- **Nacos**：阿里巴巴开源，集成配置中心
- **Eureka**：Netflix 出品，Spring Cloud 生态

### API 网关
统一入口，处理路由、鉴权、限流、日志：
- Kong（基于 Nginx/OpenResty）
- Spring Cloud Gateway
- AWS API Gateway

### 链路追踪
分布式系统中定位性能瓶颈：
- Jaeger：CNCF 毕业项目
- Zipkin：Twitter 开源
- SkyWalking：Apache 顶级项目，国产

### 配置中心
集中管理各环境配置：
- Nacos Config
- Apollo（携程开源）
- Spring Cloud Config

## 数据一致性

微服务最大的挑战是分布式事务。常见方案：

1. **Saga 模式**：将长事务拆为多个本地事务 + 补偿操作
2. **TCC（Try-Confirm-Cancel）**：业务层面实现两阶段提交
3. **事件驱动**：通过消息队列实现最终一致性（如 RocketMQ 事务消息）
4. **Seata**：阿里巴巴开源的分布式事务框架，支持 AT/TCC/Saga/XA 模式

## 容器化与编排

现代微服务通常运行在容器环境中：
- **Docker**：容器化打包，Dockerfile 定义镜像
- **Kubernetes（K8s）**：容器编排，自动伸缩、滚动更新、服务发现
- **Helm**：K8s 应用包管理器
- **Istio**：Service Mesh，处理服务间通信的 mTLS、流量管理、可观测性

## 监控与可观测性

三大支柱：
1. **Metrics（指标）**：Prometheus + Grafana，监控 QPS、延迟、错误率
2. **Logging（日志）**：ELK（Elasticsearch + Logstash + Kibana）或 Loki
3. **Tracing（追踪）**：分布式链路追踪，定位跨服务调用瓶颈

黄金指标（Four Golden Signals）：
- 延迟（Latency）
- 流量（Traffic）
- 错误率（Errors）
- 饱和度（Saturation）
