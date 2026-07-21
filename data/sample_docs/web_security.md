# Web 安全攻防实战

## OWASP Top 10（2021）

OWASP（开放 Web 应用安全项目）每几年发布最常见的 Web 安全风险排名。

### 1. 注入攻击（Injection）
- **SQL 注入**：用户输入拼接进 SQL 语句
  ```sql
  -- 攻击：输入 ' OR 1=1 --
  SELECT * FROM users WHERE name = '' OR 1=1 --' AND password = '...'
  ```
- 防御：参数化查询（Prepared Statement）、ORM、输入校验
- **XSS（跨站脚本）**：恶意 JS 注入页面
  - 存储型：恶意脚本存入数据库，所有访问者执行
  - 反射型：URL 参数中携带脚本，诱导点击
  - DOM 型：前端 JS 直接操作不可信数据
- 防御：输出编码（HTML Entity）、CSP 策略、HttpOnly Cookie

### 2. 认证失效（Broken Authentication）
- 暴力破解：限制登录频率、验证码、账户锁定
- Session 固定：登录后重新生成 Session ID
- JWT 安全：验证签名算法（防 alg:none）、设置过期时间、敏感操作二次验证
- 多因素认证（MFA）：密码 + TOTP / 短信 / 硬件密钥

### 3. 敏感数据泄露
- 传输加密：全站 HTTPS（TLS 1.2+）
- 存储加密：密码用 bcrypt/scrypt/Argon2（加盐慢哈希）
- 最小权限：数据库账户只给必要权限
- 日志脱敏：不记录密码、Token、身份证号

## CSRF（跨站请求伪造）

攻击者诱导已登录用户访问恶意页面，利用浏览器自动携带 Cookie 发起请求。

### 防御方案
1. **CSRF Token**：服务端生成随机 token，表单提交时验证
2. **SameSite Cookie**：`Set-Cookie: session=xxx; SameSite=Strict`
3. **验证 Referer/Origin**：检查请求来源
4. **双重 Cookie**：请求头中携带 Cookie 值（攻击者无法读取跨域 Cookie）

## 认证与授权

### OAuth 2.0 流程
```
用户 → 客户端 → 授权服务器（登录授权）
                ← 授权码（code）
客户端 → 授权服务器（code + client_secret）
        ← access_token + refresh_token
客户端 → 资源服务器（Bearer token）
        ← 受保护资源
```

### JWT 结构
```
Header.Payload.Signature
- Header: {"alg": "HS256", "typ": "JWT"}
- Payload: {"sub": "user123", "exp": 1700000000, "role": "admin"}
- Signature: HMACSHA256(base64(header) + "." + base64(payload), secret)
```

### RBAC vs ABAC
- **RBAC（基于角色）**：用户 → 角色 → 权限，简单直观
- **ABAC（基于属性）**：根据用户属性、资源属性、环境条件动态判断，灵活但复杂

## 安全 Headers

| Header | 作用 |
|--------|------|
| Content-Security-Policy | 限制资源加载来源，防 XSS |
| X-Frame-Options | 防止点击劫持（iframe 嵌套） |
| X-Content-Type-Options | 防止 MIME 嗅探 |
| Strict-Transport-Security | 强制 HTTPS（HSTS） |
| X-XSS-Protection | 浏览器 XSS 过滤器（已废弃，用 CSP 替代） |

## 密码学基础

### 对称加密
- AES-256-GCM：当前推荐，认证加密一体化
- 密钥管理：KMS（AWS KMS / 阿里云 KMS）

### 非对称加密
- RSA-2048：传统，较慢
- ECDSA / Ed25519：椭圆曲线，更快更安全
- 用途：数字签名、密钥交换、证书

### 哈希与签名
- SHA-256：完整性校验
- HMAC：消息认证码（带密钥的哈希）
- 数字签名：私钥签名 → 公钥验证（不可抵赖）

## 安全开发实践

### SDL（安全开发生命周期）
1. 需求阶段：威胁建模（STRIDE）
2. 设计阶段：安全架构评审
3. 开发阶段：安全编码规范、SAST 扫描
4. 测试阶段：DAST、渗透测试
5. 部署阶段：容器镜像扫描、WAF 配置
6. 运维阶段：漏洞响应、安全监控

### 常见工具
- **SAST**：SonarQube、Semgrep、CodeQL
- **DAST**：OWASP ZAP、Burp Suite
- **依赖扫描**：Snyk、Dependabot、Trivy
- **WAF**：Cloudflare、ModSecurity、阿里云 WAF
