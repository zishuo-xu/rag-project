# CI/CD 与 DevOps 工程实践

## 持续集成（CI）

持续集成（Continuous Integration）是开发人员频繁地将代码变更合并到主干，每次合并都通过自动化构建和测试验证的实践。

### 核心原则
- 频繁提交：每天至少一次合并到主干
- 自动化构建：提交即触发编译、测试、代码检查
- 快速反馈：构建失败立即通知，优先修复
- 主干开发：减少长生命周期分支，降低合并冲突

### CI 流水线阶段
```
代码提交 → 代码检查(Lint) → 单元测试 → 构建 → 集成测试 → 制品归档
```

| 阶段 | 工具 | 耗时目标 |
|------|------|----------|
| 代码检查 | ESLint / Pylint / SonarQube | < 1min |
| 单元测试 | Jest / Pytest / JUnit | < 5min |
| 构建 | Docker Build / Maven / Webpack | < 3min |
| 集成测试 | Testcontainers / Docker Compose | < 10min |
| 安全扫描 | Trivy / Snyk / CodeQL | < 3min |

## 持续交付与部署（CD）

### 持续交付（Continuous Delivery）
代码变更经过自动化测试后，随时可以部署到生产环境（手动触发部署）。

### 持续部署（Continuous Deployment）
在持续交付基础上，通过所有测试后自动部署到生产环境（无需人工干预）。

### 部署策略
| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 滚动更新 | 逐步替换旧 Pod/实例 | 零停机、资源利用高 | 短暂新旧版本共存 |
| 蓝绿部署 | 两套完整环境切换 | 快速回滚、隔离性好 | 资源成本翻倍 |
| 金丝雀发布 | 先放 5% 流量到新版本 | 风险可控 | 需要流量管理 |
| A/B 测试 | 按用户特征分流 | 数据驱动决策 | 实现复杂 |
| Feature Flag | 代码已部署但功能关闭 | 解耦部署和发布 | 代码复杂度增加 |

### 金丝雀发布流程
1. 部署新版本到少量实例（5%）
2. 导入少量流量（按权重或 Header 路由）
3. 监控核心指标（错误率、延迟、业务指标）
4. 指标正常则逐步扩大流量（5% → 20% → 50% → 100%）
5. 指标异常则自动回滚

## GitOps

### 核心理念
GitOps 以 Git 仓库作为基础设施的唯一事实来源（Single Source of Truth）：
- 声明式：用 YAML/HCL 描述期望状态
- 版本化：所有变更通过 Git 提交追踪
- 自动同步：控制器持续将实际状态收敛到期望状态
- 审计：每次变更都有 PR 审核记录

### ArgoCD 工作流
```
开发者 → Git Push → ArgoCD 检测变更 → Sync → Kubernetes 集群
                                              ↓
                                    健康检查 + 自动回滚
```

### ArgoCD vs Flux
| 维度 | ArgoCD | Flux |
|------|--------|------|
| UI | Web Dashboard | 无（CLI + K8s CRD） |
| 多集群 | 原生支持 | 需额外配置 |
| 应用模型 | Application CRD | Kustomization CRD |
| 通知 | 内置 Webhook/Slack | 需 Notification Controller |
| CNCF 状态 | 毕业项目 | 毕业项目 |

## 容器化最佳实践

### Dockerfile 优化
```dockerfile
# 多阶段构建示例（Python 应用）
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
USER nonroot
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

### 镜像安全
- 基础镜像：使用 slim/alpine/distroless 减小攻击面
- 非 root 运行：USER nonroot
- 漏洞扫描：Trivy / Grype 集成到 CI
- 镜像签名：Cosign + Sigstore 验证供应链安全
- 最小权限：只安装运行时必需依赖

### Kubernetes 资源管理
```yaml
resources:
  requests:        # 调度依据
    cpu: "250m"
    memory: "256Mi"
  limits:          # 硬上限
    cpu: "1000m"
    memory: "512Mi"
```
- requests 过低：节点超卖导致 OOM
- limits 过低：CPU Throttling 导致延迟升高
- 推荐：通过 VPA（Vertical Pod Autoscaler）自动推荐

## 基础设施即代码（IaC）

### Terraform
- 声明式：描述期望状态，自动计算变更计划
- 多云支持：AWS / GCP / Azure / 阿里云
- 状态管理：Remote State（S3 + DynamoDB 锁）
- 模块化：可复用的基础设施模块

### 核心工作流
```bash
terraform init      # 初始化 Provider
terraform plan      # 预览变更（不执行）
terraform apply     # 执行变更
terraform destroy   # 销毁资源
```

### Pulumi vs Terraform
| 维度 | Terraform | Pulumi |
|------|-----------|--------|
| 语言 | HCL（DSL） | Python/Go/TypeScript |
| 学习曲线 | 需学 HCL | 复用编程语言知识 |
| 状态管理 | 自带 | 自带（Pulumi Cloud） |
| 测试 | 有限 | 原生单元测试支持 |

## 测试金字塔与质量门禁

### 测试金字塔
```
        /  E2E  \        少量（慢、贵、脆弱）
       / 集成测试 \       适量（API 契约、数据库）
      /  单元测试   \     大量（快、便宜、稳定）
```

### 质量门禁（Quality Gate）
CI 流水线中的自动化检查，不通过则阻止合并/部署：
- 单元测试覆盖率 ≥ 80%
- 代码重复率 < 5%
- 无 Critical/Blocker 级别代码异味
- 安全漏洞扫描无 High/Critical
- API 契约测试通过（Pact / Schema 校验）

### 变更失败率（Change Failure Rate）
- DORA 指标之一：部署导致故障的比例
- Elite 团队：< 5%
- 改进方式：金丝雀发布、自动回滚、Feature Flag

## DORA 四大指标

| 指标 | 含义 | Elite 水平 |
|------|------|-----------|
| 部署频率 | 多久部署一次 | 按需/每天多次 |
| 变更前置时间 | 提交到生产的时间 | < 1小时 |
| 变更失败率 | 部署导致故障的比例 | < 5% |
| 恢复时间（MTTR） | 故障到恢复的时间 | < 1小时 |

这四个指标衡量团队的交付效率和稳定性，是 DevOps 成熟度的核心度量。
