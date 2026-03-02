# 为什么还需要推理优化工程师

> vLLM/SGLang 已经这么好了，还优化什么？
> 2026-03-02

---

## 核心观点

vLLM/SGLang 是通用框架，但你的场景不是通用的。

打个比方：vLLM/SGLang 就像 MySQL — 开箱即用，大部分场景够了。但当你的业务到了一定规模和特殊性，你就需要 DBA 去调参数、改索引、甚至改存储引擎。推理优化工程师就是 LLM serving 的 "DBA"。

框架解决的是"能跑起来"，推理优化工程师解决的是"跑得好、跑得快、跑得省"。框架给你 80 分，剩下的 20 分需要根据你的模型、场景、硬件去定制，而这 20 分往往决定了成本是 100 万还是 50 万。

改造的动机来自三个层面：

---

## 1. 你的模型不是标准模型

vLLM/SGLang 对主流架构（Llama、Qwen）支持很好，但现实中：

- 你的模型可能有自定义 attention（比如 DeepSeek 的 MLA，刚出来时没有框架支持）
- 你做了 SFT/DPO 后模型行为变了，decode 长度分布和通用模型不同，调度策略需要调整
- 你用 MoE 模型，expert 的激活模式和负载均衡需要针对你的数据分布优化
- 你加了 LoRA adapter，多 LoRA 并发时的调度和内存管理需要定制

**真实案例：** DeepSeek-V2 出来时用了 MLA（Multi-head Latent Attention），把 KV cache 压缩了 90%+。但当时 vLLM/SGLang 都不支持，SGLang 团队花了大量精力写专门的 MLA kernel 才跑起来，而且比 naive 实现快了 7 倍。

---

## 2. 你的场景有特殊约束

通用框架优化的是"平均情况"，但你的业务可能是：

### 延迟敏感型（比如实时对话）

- 用户能忍受的 TTFT（首 token 时间）< 200ms
- 通用调度器可能为了吞吐量牺牲延迟，你需要改调度优先级
- 可能需要定制 speculative decoding 的 draft model 选择策略

### 吞吐敏感型（比如批量处理、RL rollout）

- 你不在乎单个请求的延迟，只要总吞吐量最大
- 可以用更激进的 batching 策略
- RL 训练时 rollout 的推理有特殊模式 — 大量相似 prompt、需要多次采样

### 长上下文场景（比如文档分析）

- 128K context 下 KV cache 占用巨大，标准 PagedAttention 可能不够
- 需要 KV cache 量化（FP8 KV）、分层缓存（GPU→CPU→SSD）
- 可能需要 sparse attention 或 sliding window 来降低计算量

### 多模态场景

- 图片/视频的 prefill 计算量远大于文本
- encoder 和 decoder 的调度需要分开优化

---

## 3. 你的硬件不是标准配置

框架默认优化的是 A100/H100，但你可能：

- 用的是 A10、L40S 这种"性价比卡"，显存小、带宽低，需要更激进的量化和内存管理
- 多机多卡部署，跨节点通信成为瓶颈，需要优化 tensor parallel 的通信策略
- 用 AMD MI300X，kernel 需要适配 ROCm
- 混合部署：prefill 用大卡，decode 用小卡（PD 分离）

---

## 真实工作场景

用一个完整的案例串起来：

> 老板说：我们要上线一个 AI 客服，用 Qwen2.5-72B，要求 TTFT < 300ms，支持 1000 QPS，成本尽量低。

### 第一步：Profiling（找瓶颈）

```
先用 vLLM 跑 benchmark，发现：
- TTFT P99 = 800ms（超标）
- 吞吐量 = 400 QPS（不够）
- GPU 利用率只有 60%（有优化空间）

用 nsight systems 看 kernel 执行时间线
发现瓶颈：prefill 阶段的 attention 计算太慢
```

### 第二步：系统级优化

```
- 开启 chunked prefill，避免长 prompt 阻塞 decode
- 调整 max_num_seqs，找到 latency-throughput 的最优平衡点
- 开启 prefix caching（客服场景有大量相同的 system prompt）
- 尝试 PD 分离：prefill 和 decode 分到不同 GPU
```

### 第三步：模型级优化

```
- 72B 模型太大，尝试 FP8 量化
- 跑精度评估，发现 FP8 在客服场景精度损失可接受
- 量化后显存减半，可以用更少的卡 → 成本降低
- KV cache 也做 FP8 量化，进一步省显存
```

### 第四步：Kernel 级优化（如果前面还不够）

```
- 发现 attention kernel 在特定 seq_len 下性能不好
- 写一个针对这个 seq_len 分布优化的 Triton kernel
- 或者发现 RMSNorm + RoPE 可以 fuse 成一个 kernel，减少一次 HBM 读写
- 给 vLLM/SGLang 提 PR，或者维护内部 fork
```

### 第五步：持续监控和迭代

```
- 上线后监控 TTFT、TPS、GPU 利用率、错误率
- 流量模式变化时重新调优
- 新版本框架出来时评估是否升级
```

---

## 总结

| 层次 | 框架已解决 | 你需要做的 |
|------|-----------|-----------|
| 基础 serving | ✅ continuous batching、PagedAttention | 根据场景调参、选择框架 |
| 调度策略 | ✅ 通用调度器 | 针对你的流量模式定制优先级和抢占策略 |
| 内存管理 | ✅ 通用 KV cache 管理 | KV cache 量化、分层缓存、prefix sharing 策略 |
| 模型适配 | ✅ 主流架构 | 自定义 attention、MoE 负载均衡、多 LoRA 调度 |
| Kernel | ✅ 通用 kernel | 针对特定 shape/硬件写定制 kernel |
| 量化 | ✅ 基础量化支持 | 选择方案、calibration、精度评估、mixed-precision 策略 |
| 系统架构 | ✅ 单机部署 | PD 分离、多级缓存、跨节点通信优化 |
