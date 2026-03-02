# SGLang 部署指南 + 源码导读

> Pan老师的推理优化第一课（补充篇）
> 2026-03-02

---

## 为什么要学 SGLang

SGLang 和 vLLM 是目前两大主流推理框架，但设计哲学不同：

| 维度 | vLLM | SGLang |
|------|------|--------|
| 核心创新 | PagedAttention（内存管理） | RadixAttention（前缀缓存） |
| 调度器 | Python调度器 | 零开销CPU调度器 |
| 强项 | 通用性强、生态成熟 | 多轮对话/agent场景更快、structured output更强 |
| PD分离 | 支持 | 原生支持，更成熟 |
| RL集成 | 需要外部适配 | 原生RL集成（verl、AReaL等） |
| DeepSeek优化 | 支持 | 更早支持、优化更深（MLA、EP） |

简单说：vLLM 是"通用推理引擎"，SGLang 是"为复杂场景优化的推理引擎"。两个都得会。

---

## 一、快速部署

### 1. 安装

```bash
# 推荐用 uv
uv pip install "sglang[all]" --torch-backend=auto

# 或者 pip
pip install "sglang[all]"
```

### 2. 启动服务

```bash
# 启动 OpenAI 兼容服务（和 vLLM 一样的接口）
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

测试（和 vLLM 完全一样的 API）：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

### 3. 离线推理

```python
import sglang as sgl

llm = sgl.Engine(model_path="Qwen/Qwen2.5-7B-Instruct")

prompts = [
    "Explain RadixAttention in one paragraph.",
    "What is prefix caching?",
]
outputs = llm.generate(prompts)

for output in outputs:
    print(output["text"])

llm.shutdown()
```

### 4. SGLang 原生 API（这是它的特色）

SGLang 有自己的编程模型，特别适合多轮/分支/结构化生成：

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

state = multi_turn_qa.run(
    question_1="What is PagedAttention?",
    question_2="How does it compare to RadixAttention?",
)
print(state["answer_1"])
print(state["answer_2"])
```

这种写法的好处：多轮对话之间自动复用 KV cache（RadixAttention），不需要重新计算前缀。

### 5. 常用启动参数

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --tp 2                          # 张量并行
    --dp 2                          # 数据并行
    --mem-fraction-static 0.9       # 静态显存比例
    --chunked-prefill-size 8192     # chunked prefill大小
    --enable-torch-compile          # 开启torch.compile加速
    --quantization fp8              # 量化
    --disable-radix-cache           # 关闭RadixAttention（调试用）
```

---

## 二、源码导读

### 整体架构

SGLang 的架构分两层：前端（编程模型）和后端（Runtime）。

```
SGLang Frontend (sgl.function / OpenAI API)
    ↓
Router / Load Balancer
    ↓
Scheduler (零开销调度器)
    ↓
┌─────────────┬──────────────┐
│ Prefill     │ Decode       │  ← 可以PD分离部署
│ Worker      │ Worker       │
└─────────────┴──────────────┘
    ↓
RadixAttention + Paged KV Cache
    ↓
GPU Kernels (FlashInfer / Triton)
```

### 重点源码文件

#### 🔴 第一优先级：必读

**1. `python/sglang/srt/managers/scheduler.py` — 调度器**

SGLang 的调度器号称"零开销"，核心思路是把调度逻辑做到极致轻量。

关键点：
- 和 vLLM 的 scheduler 对比着看，理解设计差异
- 如何实现 continuous batching
- Prefill 和 Decode 的混合调度策略
- 请求优先级和抢占机制

**2. `python/sglang/srt/mem_cache/radix_cache.py` — RadixAttention**

这是 SGLang 最核心的创新。用 Radix Tree（基数树）管理 KV cache 的前缀共享。

核心思想：
```
用户A: "You are a helpful assistant. What is 1+1?"
用户B: "You are a helpful assistant. What is 2+2?"
                                    ↑
                    共享前缀的KV cache，不重复计算
```

关键理解：
- Radix Tree 的插入、查找、驱逐（eviction）
- 和 vLLM 的 prefix caching 对比：SGLang 是自动的、细粒度的
- 为什么这在多轮对话和 agent 场景下特别有效

**3. `python/sglang/srt/layers/attention/` — Attention 实现**

```
attention/
├── flashinfer_backend.py    # FlashInfer后端（SGLang默认）
├── triton_backend.py        # Triton后端
└── double_sparsity_backend.py  # 稀疏注意力
```

SGLang 默认用 FlashInfer 而不是 FlashAttention，理解为什么。

#### 🟡 第二优先级：深入理解

**4. `python/sglang/srt/managers/tp_worker.py` — Tensor Parallel Worker**

模型执行的核心，理解：
- 模型加载和分片
- 前向计算流程
- 和调度器的交互

**5. `python/sglang/srt/model_executor/` — 模型实现**

和 vLLM 类似，每个模型架构有对应实现。

**6. `python/sglang/srt/sampling/` — 采样**

SGLang 的 structured output（JSON schema、正则约束）实现很强，值得看。

#### 🟢 第三优先级：进阶特色

**7. `python/sglang/srt/speculative/` — Speculative Decoding**

**8. `python/sglang/srt/disaggregation/` — PD 分离**

Prefill-Decode 分离部署的实现，这是大规模部署的关键技术。

**9. `python/sglang/srt/managers/expert_distribution.py` — Expert Parallelism**

MoE 模型（如 DeepSeek-V3）的专家并行实现。

---

## 三、源码阅读路线

```
第1天：部署 SGLang，跑通 OpenAI API + 原生 API
  ↓
第2天：精读 radix_cache.py，理解 RadixAttention
       对比 vLLM 的 block_manager.py，写笔记
  ↓
第3天：精读 scheduler.py，对比 vLLM 的调度器
       理解"零开销"是怎么做到的
  ↓
第4天：看 attention backend，理解 FlashInfer 的使用
  ↓
第5天：跑 benchmark，对比 SGLang vs vLLM 在不同场景下的性能
       - 单轮 vs 多轮
       - 短prompt vs 长prompt
       - 高并发 vs 低并发
```

---

## 四、vLLM vs SGLang 对比学习建议

两个框架对照着看，效果最好：

| 模块 | vLLM | SGLang | 对比重点 |
|------|------|--------|---------|
| 内存管理 | block_manager.py | radix_cache.py | 页表 vs 基数树 |
| 调度器 | scheduler.py | scheduler.py | 调度策略差异 |
| Attention | FlashAttention | FlashInfer | kernel选择 |
| 前缀缓存 | 手动prefix caching | 自动RadixAttention | 粒度和自动化程度 |
| PD分离 | 支持 | 原生支持 | 架构设计差异 |

---

## 五、配套论文

1. **SGLang 论文**（和 radix_cache.py 对照）
   [Efficiently Programming Large Language Models using SGLang](https://arxiv.org/abs/2312.07104)

2. **FlashInfer 论文**（和 attention backend 对照）
   [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)

3. **DeepSeek-V2 MLA**（SGLang 对 MLA 的优化是亮点）
   [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

---

> 建议先跑通 vLLM，再跑 SGLang，然后对比着读源码。
> 有问题随时问 🐼
