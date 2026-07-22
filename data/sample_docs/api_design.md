# API 设计原则与协议对比

## RESTful API 设计

### 核心约束
REST（Representational State Transfer）是一种架构风格，核心约束包括：
- 客户端-服务器分离：关注点分离
- 无状态：每个请求包含所有必要信息
- 可缓存：响应必须标明是否可缓存
- 统一接口：资源标识、表述、自描述消息、HATEOAS
- 分层系统：客户端不感知直接连接还是中间层

### URL 设计规范
```
GET    /api/v1/users          # 获取用户列表
GET    /api/v1/users/{id}     # 获取单个用户
POST   /api/v1/users          # 创建用户
PUT    /api/v1/users/{id}     # 全量更新
PATCH  /api/v1/users/{id}     # 部分更新
DELETE /api/v1/users/{id}     # 删除用户
```

设计原则：
- 使用名词复数（/users 而非 /user）
- 层级关系用嵌套（/users/{id}/orders）
- 过滤用查询参数（/users?status=active&page=1&size=20）
- 版本号放在 URL 或 Header（Accept: application/vnd.api.v2+json）

### HTTP 状态码规范
| 状态码 | 含义 | 使用场景 |
|--------|------|----------|
| 200 | OK | 请求成功 |
| 201 | Created | 资源创建成功 |
| 204 | No Content | 删除成功，无返回体 |
| 400 | Bad Request | 请求参数错误 |
| 401 | Unauthorized | 未认证（缺少/无效 Token） |
| 403 | Forbidden | 已认证但无权限 |
| 404 | Not Found | 资源不存在 |
| 409 | Conflict | 资源冲突（如重复创建） |
| 422 | Unprocessable Entity | 语义验证失败 |
| 429 | Too Many Requests | 触发限流 |
| 500 | Internal Server Error | 服务器内部错误 |

### 分页设计
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total_items": 156,
    "total_pages": 8
  }
}
```
- 游标分页（Cursor-based）：适合大数据量、实时流（如 Feed 流）
- 偏移分页（Offset-based）：适合后台管理、数据量可控

## GraphQL

### 核心概念
- Schema：定义类型系统（Query、Mutation、Subscription）
- Resolver：每个字段的数据获取逻辑
- 强类型：客户端精确获取所需字段，避免 Over-fetching

### 优势与劣势
| 优势 | 劣势 |
|------|------|
| 按需获取，减少网络传输 | 缓存复杂（无 HTTP 缓存语义） |
| 一次请求获取嵌套数据 | N+1 查询问题（需 DataLoader） |
| 强类型 Schema 即文档 | 学习曲线较陡 |
| 前端自主控制数据结构 | 权限控制粒度难（字段级） |

### 适用场景
- 移动端（带宽敏感）
- 复杂嵌套数据（如社交 Feed）
- 多端适配（Web/iOS/Android 需要不同字段）

## gRPC

### 核心特性
- 基于 HTTP/2：多路复用、头部压缩、双向流
- Protocol Buffers（protobuf）：二进制序列化，比 JSON 小 3-10 倍
- 强类型 IDL：.proto 文件定义接口，自动生成多语言代码
- 四种通信模式：Unary / Server Streaming / Client Streaming / Bidirectional

### proto 定义示例
```protobuf
syntax = "proto3";
service OrderService {
  rpc CreateOrder (CreateOrderRequest) returns (Order);
  rpc ListOrders (ListOrdersRequest) returns (stream Order);
}
message CreateOrderRequest {
  string user_id = 1;
  repeated OrderItem items = 2;
}
```

### gRPC vs REST
| 维度 | gRPC | REST |
|------|------|------|
| 协议 | HTTP/2 | HTTP/1.1（通常） |
| 序列化 | Protobuf（二进制） | JSON（文本） |
| 性能 | 高（小体积+多路复用） | 中 |
| 浏览器支持 | 需 gRPC-Web 代理 | 原生支持 |
| 调试 | 需专用工具 | curl 即可 |
| 适用 | 微服务内部通信 | 对外公开 API |

## API 认证与授权

### 认证方式
| 方式 | 原理 | 适用场景 |
|------|------|----------|
| API Key | 请求头携带固定密钥 | 服务间调用、简单场景 |
| JWT | 无状态 Token（Header.Payload.Signature） | 前后端分离、微服务 |
| OAuth 2.0 | 授权码/客户端凭证等流程 | 第三方授权、开放平台 |
| mTLS | 双向证书认证 | 零信任架构、金融 |

### JWT 结构
```
Header: {"alg": "RS256", "typ": "JWT"}
Payload: {"sub": "user123", "exp": 1700000000, "role": "admin"}
Signature: RS256(header + "." + payload, privateKey)
```
- Access Token：短期有效（15min），携带权限信息
- Refresh Token：长期有效（7d），用于刷新 Access Token
- 注意：JWT 无法主动失效（需配合黑名单或短过期时间）

## API 版本管理

### 策略对比
| 策略 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| URL 路径 | /v1/users, /v2/users | 直观、易路由 | URL 变化 |
| 查询参数 | /users?version=2 | URL 不变 | 不够显式 |
| Header | Accept: vnd.api.v2+json | 最 RESTful | 调试不便 |

### 版本演进原则
- 向后兼容：新增字段不影响旧客户端
- 废弃策略：提前通知 → 双版本并行 → 下线旧版本
- 破坏性变更才升大版本号

## API 网关

### 核心功能
- 路由：根据路径/Header 转发到后端服务
- 认证：统一验证 Token、API Key
- 限流：保护后端不被流量冲垮
- 日志：统一记录请求日志
- 协议转换：外部 REST → 内部 gRPC

### 主流方案
| 网关 | 特点 | 适用 |
|------|------|------|
| Kong | 插件生态丰富、Lua 扩展 | 通用场景 |
| APISIX | 高性能（Nginx + etcd） | 高吞吐场景 |
| Spring Cloud Gateway | Java 生态集成 | Spring Boot 项目 |
| Envoy | xDS 动态配置、Service Mesh | 云原生/K8s |
| AWS API Gateway | 全托管、按请求计费 | AWS 生态 |

## 幂等性设计

### 什么是幂等
同一请求执行一次和执行多次的效果相同。

### 各方法幂等性
- GET / PUT / DELETE：天然幂等
- POST：非幂等（需额外设计）

### POST 幂等方案
1. **Token 机制**：先获取幂等 Token，请求时携带，服务端去重
2. **唯一约束**：数据库唯一索引（如 order_no）
3. **状态机**：只允许合法状态转换（待支付→已支付，不可重复）
4. **乐观锁**：UPDATE ... WHERE version = old_version
