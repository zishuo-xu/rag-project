# 大语言模型（LLM）技术全景

## Transformer 架构

Transformer 是现代 LLM 的基础架构，2017 年由 Google 在 "Attention Is All You Need" 中提出。

### 核心组件
- **Self-Attention**：计算序列中每个 token 对其他 token 的注意力权重
- **Multi-Head Attention**：多组注意力并行，捕获不同子空间的关系
- **Feed-Forward Network**：两层全连接 + 激活函数（GELU/SwiGLU）
- **Layer Normalization**：稳定训练（Pre-LN vs Post-LN）
- **位置编码**：RoPE（旋转位置编码）成为主流

### 注意力机制公式
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- Q（Query）、K（Key）、V（Value）由输入线性变换得到
- 除以 √d_k 防止点积过大导致 softmax 梯度消失

### 模型规模
| 模型 | 参数量 | 层数 | 隐藏维度 |
|------|--------|------|----------|
| GPT-2 | 1.5B | 48 | 1600 |
| GPT-3 | 175B | 96 | 12288 |
| LLaMA-2 | 7B/13B/70B | 32/40/80 | 4096/5120/8192 |
| DeepSeek-V2 | 236B (MoE) | 60 | 5120 |

## 预训练与微调

### 预训练（Pre-training）
- 目标：Next Token Prediction（自回归语言建模）
- 数据：数万亿 token 的互联网文本
- 算力：数千 GPU × 数周（GPT-4 估计 >$100M）
- 优化：AdamW、学习率 warmup + cosine decay、混合精度训练

### 微调（Fine-tuning）
- **全参数微调**：更新所有权重，效果好但成本高（需要与预训练相当的显存）
- **LoRA（Low-Rank Adaptation）**：
  - 原理：将权重更新分解为两个低秩矩阵 ΔW = BA，其中 B∈R^(d×r)、A∈R^(r×k)，rank r 通常取 8~64
  - 训练时冻结原始权重 W，只训练 A 和 B 两个小矩阵
  - 优势：显存降低 60% 以上，训练速度快，可为不同任务保存不同 LoRA 适配器
  - 推理时可将 ΔW 合并回原始权重，不增加推理延迟
  - 应用：通常对 Attention 层的 Q/V 投影矩阵施加 LoRA
- **QLoRA**：4-bit 量化（NF4）+ LoRA，单卡可微调 70B 模型
- **Adapter**：在层间插入小型适配器模块

### RLHF / DPO
- **RLHF**：SFT → 奖励模型 → PPO 强化学习
- **DPO**：直接用偏好对优化，跳过奖励模型训练
- 目的：让模型输出更符合人类偏好（有帮助、诚实、无害）

## 推理优化

### KV Cache
自回归生成时，已计算的 Key/Value 缓存复用，避免重复计算。
- 显存占用：2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_bytes
- 优化：GQA（分组查询注意力）、MQA（多查询注意力）

### 量化
- **INT8**：精度损失小，速度提升 ~2x
- **INT4（GPTQ/AWQ）**：精度有损，速度提升 ~4x
- **FP8**：H100 原生支持，训练推理均可用

### 推理框架
| 框架 | 特点 |
|------|------|
| vLLM | PagedAttention，高吞吐 |
| TensorRT-LLM | NVIDIA 官方，极致优化 |
| llama.cpp | CPU/边缘设备，GGUF 格式 |
| TGI | HuggingFace 出品，生产就绪 |
| SGLang | 结构化生成优化 |

## RAG vs 微调

| 维度 | RAG | 微调 |
|------|-----|------|
| 知识更新 | 实时更新文档即可 | 需重新训练 |
| 成本 | 低（无需 GPU 训练） | 高（需算力和数据） |
| 幻觉控制 | 有来源可追溯 | 仍可能幻觉 |
| 风格适配 | 较弱 | 强（可学习特定语气） |
| 适用场景 | 知识密集型问答 | 风格/格式/领域适配 |

## Prompt Engineering

### 常用技巧
- **Few-shot**：提供示例引导格式
- **Chain-of-Thought（CoT）**：让模型逐步推理
- **ReAct**：推理 + 行动交替（工具调用）
- **Self-Consistency**：多次采样取多数投票

### 系统提示词设计
```
你是一个{角色}。
你的任务是{任务描述}。
约束：
1. {约束1}
2. {约束2}
输出格式：{格式要求}
```

## 多模态与 Agent

### 多模态模型
- 视觉：ViT 编码图像 → 投影层 → LLM 理解
- 代表：GPT-4V、LLaVA、Qwen-VL

### AI Agent
- 规划：将复杂任务分解为子步骤
- 工具调用：搜索、代码执行、API 调用
- 记忆：短期（上下文）+ 长期（向量数据库）
- 框架：LangChain、AutoGen、CrewAI
