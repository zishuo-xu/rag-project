# Docker 与 Kubernetes 实战

## Docker 核心概念

Docker 是一个容器化平台，将应用及其依赖打包为轻量级、可移植的容器。

### 镜像与容器
- **镜像（Image）**：只读模板，包含运行应用所需的代码、运行时、库、环境变量
- **容器（Container）**：镜像的运行实例，可以被创建、启动、停止、删除
- **Dockerfile**：定义镜像构建步骤的文本文件

### Dockerfile 最佳实践
```dockerfile
# 多阶段构建：减小最终镜像体积
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

### 网络模式
- **bridge**：默认，容器间通过 docker0 网桥通信
- **host**：共享宿主机网络栈，性能最好但端口冲突
- **overlay**：跨主机容器通信（Swarm/K8s 使用）
- **none**：完全隔离

### 数据持久化
- **Volume**：Docker 管理的持久化存储，推荐方式
- **Bind Mount**：挂载宿主机目录，开发时常用
- **tmpfs**：内存存储，容器停止后消失

## Kubernetes 架构

### 控制平面（Control Plane）
- **kube-apiserver**：集群入口，所有操作通过它
- **etcd**：分布式 KV 存储，保存集群状态
- **kube-scheduler**：决定 Pod 调度到哪个 Node
- **kube-controller-manager**：运行各种控制器（Deployment、ReplicaSet 等）

### 工作节点（Worker Node）
- **kubelet**：管理 Pod 生命周期
- **kube-proxy**：Service 的网络代理
- **容器运行时**：containerd / CRI-O

### 核心资源对象
| 对象 | 作用 |
|------|------|
| Pod | 最小调度单元，包含一个或多个容器 |
| Deployment | 声明式管理 Pod 副本，支持滚动更新 |
| Service | 稳定的网络入口（ClusterIP/NodePort/LoadBalancer） |
| Ingress | 七层路由，基于域名/路径转发 |
| ConfigMap/Secret | 配置和敏感数据管理 |
| PV/PVC | 持久化存储声明 |
| HPA | 水平自动伸缩 |

## 部署策略

### 滚动更新（Rolling Update）
```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
```
逐步替换旧 Pod，零停机。

### 蓝绿部署
两套完整环境，切换流量瞬间完成，回滚快，但资源翻倍。

### 金丝雀发布（Canary）
先将 5% 流量导到新版本，观察指标正常后逐步放量。
工具：Istio、Argo Rollouts、Flagger。

## 资源管理与调优

### 资源请求与限制
```yaml
resources:
  requests:
    cpu: "100m"
    memory: "128Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```
- requests：调度依据，保证最低资源
- limits：硬上限，超出被 OOMKill 或 CPU 节流

### 健康检查
- **livenessProbe**：容器是否存活，失败则重启
- **readinessProbe**：是否就绪接收流量，失败则从 Service 摘除
- **startupProbe**：慢启动应用专用，启动期间不触发 liveness

## 常用运维命令

```bash
# 查看 Pod 状态
kubectl get pods -n production -o wide

# 查看日志
kubectl logs -f deployment/api-server --tail=100

# 进入容器调试
kubectl exec -it pod-name -- /bin/sh

# 扩缩容
kubectl scale deployment/api-server --replicas=5

# 查看资源使用
kubectl top pods --sort-by=memory
```
