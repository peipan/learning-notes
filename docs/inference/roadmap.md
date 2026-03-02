# 学习路线图

> Pan老师的 Agentic RL + 大模型推理优化 学习计划
> 创建时间：2026-03-02

## 总体规划

两条线并行，每条线分3个阶段，预计3-4个月完成核心内容。
建议每周两条线交替推进，保持新鲜感。

---

## 线路一：Agentic RL

### 阶段1：从DPO到Online RL（第1-3周）

你已经有SFT+DPO经验，这个阶段补上在线RL训练LLM的gap。

**核心概念：**
- PPO in LLM context（和你之前DRL的PPO区别在哪）
- GRPO（Group Relative Policy Optimization）— DeepSeek的核心方法
- RLOO（REINFORCE Leave-One-Out）
- Online vs Offline RL for LLM alignment

**必读论文：**
- [ ] [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — RLHF经典，快速过一遍
- [ ] [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — 你做过，重新从理论角度审视
- [ ] [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — 重点！GRPO + RL for reasoning
- [ ] [GRPO原始论文 - DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300) — GRPO方法细节
- [ ] [Online DPO / RLHF对比研究](https://arxiv.org/abs/2405.07863) — 理解online vs offline的trade-off

**实操：**
- [ ] 用 [TRL库](https://github.com/huggingface/trl) 跑通一个GRPO训练（小模型即可，如Qwen2.5-1.5B）
- [ ] 对比PPO vs DPO vs GRPO在同一任务上的效果

### 阶段2：Agent框架与Reward设计（第4-6周）

这是agentic RL的核心 — 怎么让模型学会用工具、多步推理。

**核心概念：**
- ReAct范式（Reasoning + Acting）
- Tool use / Function calling 的RL训练
- Process Reward Model (PRM) vs Outcome Reward Model (ORM)
- Multi-turn interaction的reward设计
- Verifiable rewards vs learned rewards

**必读论文：**
- [ ] [ReAct (Yao et al., 2023)](https://arxiv.org/abs/2210.03629) — agent推理范式
- [ ] [Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761) — 自学工具使用
- [ ] [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050) — PRM
- [ ] [OpenAI Process Reward Model](https://openai.com/research/improving-mathematical-reasoning-with-process-reward-models)
- [ ] [RAGEN (2025)](https://arxiv.org/abs/2504.20073) — Agentic RL训练框架
- [ ] [Search-R1 (2025)](https://arxiv.org/abs/2503.09516) — RL训练agent做搜索

**实操：**
- [ ] 实现一个简单的tool-use环境（如calculator + search）
- [ ] 用GRPO训练模型在该环境中学会调用工具
- [ ] 实现PRM并对比ORM效果

### 阶段3：前沿与深入（第7-10周）

**进阶主题：**
- [ ] Multi-agent RL — 多个LLM agent协作
- [ ] Self-play / Constitutional AI
- [ ] RL scaling laws for LLM
- [ ] Test-time compute scaling（inference-time RL）

**前沿论文（持续更新）：**
- [ ] [Scaling LLM Test-Time Compute (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)
- [ ] [STaR / Quiet-STaR](https://arxiv.org/abs/2203.14465) — 自我改进推理
- [ ] [STILL-ALIVE系列] — RL for long-horizon agent tasks

**项目：**
- [ ] 端到端实现一个agentic RL训练pipeline
- [ ] 写一篇技术博客总结agentic RL的核心方法论

---

## 线路二：大模型推理优化

### 阶段1：Serving工程 + Kernel入门（第1-3周）

Pan老师已了解 continuous batching、PagedAttention、RadixAttention 原理。
重点转向：实际部署调优 + CUDA/Triton 入门。

**核心目标：**
- 跑通生产级 serving pipeline（vLLM + SGLang）
- 入门 Triton 编程，写第一个 kernel
- 理解 GPU 内存层次和 roofline model

**必读材料：**
- [ ] [Efficient Transformers Survey (Tay et al., 2022)](https://arxiv.org/abs/2009.06732) — 快速过
- [ ] [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — 必读
- [ ] [FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691)
- [ ] [PagedAttention / vLLM (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180) — 重点
- [ ] [Orca: Continuous Batching (Yu et al., 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)

**实操：**
- [ ] 部署vLLM，阅读核心源码（scheduler、PagedAttention实现）
- [ ] 部署SGLang，阅读核心源码（RadixAttention、零开销调度器）
- [ ] 对比 vLLM vs SGLang 在不同场景下的性能差异
- [ ] 用benchmark工具对比不同batch策略的throughput和latency
- [ ] 手写一个简单的KV cache管理器

### 阶段2：量化与压缩（第4-6周）

**核心概念：**
- Weight-only quantization vs Weight-Activation quantization
- PTQ（Post-Training Quantization）vs QAT（Quantization-Aware Training）
- Calibration策略
- Mixed-precision推理

**必读论文：**
- [ ] [GPTQ (Frantar et al., 2023)](https://arxiv.org/abs/2210.17323) — weight-only PTQ经典
- [ ] [AWQ (Lin et al., 2024)](https://arxiv.org/abs/2306.00978) — activation-aware量化
- [ ] [SmoothQuant (Xiao et al., 2023)](https://arxiv.org/abs/2211.10438) — W8A8
- [ ] [QuIP# (Tseng et al., 2024)](https://arxiv.org/abs/2402.04396) — 2-bit量化
- [ ] [FP8 Training & Inference](https://arxiv.org/abs/2209.05433)

**实操：**
- [ ] 用AutoGPTQ/AutoAWQ量化一个模型，对比精度和速度
- [ ] 阅读GPTQ核心代码，理解Hessian-based量化
- [ ] 实现一个简单的per-channel vs per-group量化对比实验

### 阶段3：高级优化技术（第7-10周）

**核心概念：**
- Speculative Decoding
- KV Cache压缩（MLA、GQA、MQA、StreamingLLM）
- Prefill-Decode分离架构（PD disaggregation）
- Structured pruning / Distillation

**必读论文：**
- [ ] [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- [ ] [Medusa (Cai et al., 2024)](https://arxiv.org/abs/2401.10774) — parallel decoding heads
- [ ] [MLA - DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) — Multi-head Latent Attention
- [ ] [StreamingLLM (Xiao et al., 2024)](https://arxiv.org/abs/2309.17453) — 无限长度推理
- [ ] [DistServe / Splitwise](https://arxiv.org/abs/2401.09670) — PD分离
- [ ] [SGLang (Zheng et al., 2024)](https://arxiv.org/abs/2312.07104) — RadixAttention

**实操：**
- [ ] 实现speculative decoding（draft model + target model）
- [ ] 阅读SGLang源码，理解RadixAttention和prefix caching
- [ ] 对比GQA/MQA/MLA在不同场景下的性能

**项目：**
- [ ] 给vLLM或SGLang贡献一个PR（哪怕是小优化）
- [ ] 写一篇推理优化的系统性总结

---

## 学习节奏建议

| 时间 | Agentic RL | 推理优化 |
|------|-----------|---------|
| 周一三五 | 读论文 + 笔记 | — |
| 周二四六 | — | 读论文 + 代码 |
| 周日 | 实操项目 | 实操项目 |

灵活调整，核心是保持节奏，不要断。

## 学习资源

### 代码库
- [TRL](https://github.com/huggingface/trl) — RLHF/GRPO训练
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 分布式RLHF
- [vLLM](https://github.com/vllm-project/vllm) — 推理引擎
- [SGLang](https://github.com/sgl-project/sglang) — 推理框架
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA推理优化

### 博客/课程
- Hugging Face RLHF blog series
- Lilian Weng's blog（RL + LLM相关）
- 知乎/公众号上的中文解读（辅助理解）

---

> 这个路线图会随学习进展持续更新。
> 每完成一项打个 [x]，Pander会帮你跟踪进度 🐼
