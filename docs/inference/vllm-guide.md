# vLLM 部署指南 + 源码导读

> Pan老师的推理优化第一课
> 2026-03-02

---

## 一、快速部署

### 1. 环境准备

前提条件：
- Linux 系统
- Python 3.10 - 3.13
- NVIDIA GPU + CUDA driver（推荐 A100/4090/H100）

推荐用 uv（比 pip 快很多）：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv --python 3.12 --seed
source .venv/bin/activate

# 安装 vLLM（自动检测 CUDA 版本）
uv pip install vllm --torch-backend=auto
```

或者用 conda：

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

### 2. 离线推理（Offline Inference）

最简单的上手方式，理解 vLLM 的基本 API：

```python
from vllm import LLM, SamplingParams

# 加载模型（会自动下载）
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# 批量推理
prompts = [
    "Explain PagedAttention in one paragraph.",
    "What is continuous batching?",
    "How does KV cache work in transformers?",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text}\n")
```

### 3. 在线服务（OpenAI-Compatible Server）

这是生产环境最常用的方式：

```bash
# 启动服务（OpenAI兼容API）
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9
```

测试：

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

用 Python 客户端（和 OpenAI SDK 完全兼容）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### 4. 常用启动参数

```bash
vllm serve <model> \
    --tensor-parallel-size 2        # 张量并行（多卡）
    --pipeline-parallel-size 2      # 流水线并行
    --max-model-len 8192            # 最大序列长度
    --gpu-memory-utilization 0.9    # GPU显存利用率
    --quantization awq              # 量化方式（awq/gptq/fp8）
    --enable-prefix-caching          # 开启前缀缓存
    --max-num-seqs 256              # 最大并发序列数
    --dtype auto                    # 数据类型
    --enforce-eager                 # 关闭CUDA Graph（调试用）
```

### 5. 性能基准测试

vLLM 自带 benchmark 工具：

```bash
# 吞吐量测试
python -m vllm.entrypoints.openai.api_server &  # 先启动服务

# 用 benchmark 脚本
python benchmarks/benchmark_serving.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 1000 \
    --request-rate 10
```

---

## 二、源码导读

### 整体架构

vLLM 的核心是一个 **异步调度引擎**，架构可以简化为：

```
Client Request
    ↓
API Server (entrypoints/)
    ↓
LLMEngine / AsyncLLMEngine (engine/)
    ↓
Scheduler (core/scheduler.py)          ← 调度核心
    ↓
Model Runner (worker/model_runner.py)  ← 模型执行
    ↓
Attention Backend (attention/)         ← PagedAttention实现
    ↓
GPU Kernels (CUDA/Triton)
```

### 重点源码文件（按优先级排序）

#### 🔴 第一优先级：必读

**1. `vllm/core/scheduler.py` — 调度器**

这是 vLLM 的大脑。理解它就理解了 vLLM 为什么快。

核心职责：
- 决定每一步哪些请求参与计算（continuous batching）
- 管理 prefill 和 decode 请求的混合调度
- 控制 KV cache 的分配和回收（通过 block manager）

关键类和方法：
```
Scheduler
├── schedule()              # 每一步的调度决策，最核心的方法
├── _schedule_prefills()    # 调度新请求的prefill
├── _schedule_running()     # 调度正在decode的请求
├── _schedule_swapped()     # 调度被换出的请求
└── _can_append_slots()     # 检查是否有足够的KV cache空间
```

重点理解：
- `SchedulerOutputs` 结构 — 调度器每一步的输出
- Preemption策略 — 当显存不够时，哪些请求被换出
- Chunked prefill — 长prompt分块处理，避免阻塞decode

**2. `vllm/core/block_manager.py` — Block Manager（PagedAttention的内存管理）**

PagedAttention 的核心实现。把 KV cache 按 block 管理，类似操作系统的虚拟内存。

关键概念：
```
BlockManager
├── allocate()              # 为新序列分配物理block
├── append_slots()          # 为decode步骤追加block
├── can_allocate()          # 检查是否有空闲block
├── free()                  # 释放序列的block
├── swap_in() / swap_out()  # GPU↔CPU之间换入换出block
└── fork()                  # beam search时共享block（copy-on-write）
```

重点理解：
- Logical block → Physical block 的映射（和OS页表一样）
- Block 的引用计数和 copy-on-write 机制
- 为什么这比预分配连续内存高效得多

**3. `vllm/attention/` — Attention Backend**

```
attention/
├── backends/
│   ├── flash_attn.py       # FlashAttention后端
│   ├── flashinfer.py       # FlashInfer后端
│   └── xformers.py         # xFormers后端
├── layer.py                # Attention层的统一接口
└── selector.py             # 自动选择最优后端
```

重点理解：
- PagedAttention 的 kernel 如何读取非连续的 KV cache blocks
- Prefill（全量attention）vs Decode（增量attention）的不同计算路径

#### 🟡 第二优先级：深入理解

**4. `vllm/engine/` — 引擎层**

```
engine/
├── llm_engine.py           # 同步引擎（离线推理用）
├── async_llm_engine.py     # 异步引擎（在线服务用）
└── arg_utils.py            # 引擎参数定义
```

理解请求的完整生命周期：
`add_request() → schedule() → execute_model() → process_outputs() → 返回结果`

**5. `vllm/worker/model_runner.py` — 模型执行器**

负责实际的模型前向计算：
- 准备输入张量（input_ids, positions, attention metadata）
- 调用模型 forward
- 处理采样（sampling）

**6. `vllm/model_executor/models/` — 模型实现**

每个支持的模型架构都有对应实现，比如：
- `llama.py` — Llama系列
- `qwen2.py` — Qwen2系列

看一个就够了，理解 vLLM 如何把 HuggingFace 模型适配到自己的框架。

#### 🟢 第三优先级：进阶

**7. `vllm/spec_decode/` — Speculative Decoding**

```
spec_decode/
├── spec_decode_worker.py   # 投机解码的worker
├── draft_model_runner.py   # draft模型执行
└── batch_expansion.py      # 验证阶段的batch扩展
```

**8. `vllm/lora/` — LoRA 支持**

多 LoRA 并发服务的实现，理解如何在 serving 时动态切换 adapter。

---

## 三、源码阅读路线

建议按这个顺序读，每一步都跑一下对应的代码：

```
第1天：跑通部署，用 --enforce-eager 关掉CUDA Graph方便调试
  ↓
第2天：从 LLM.generate() 入口开始，跟踪一个请求的完整生命周期
  ↓
第3天：精读 scheduler.py，理解 continuous batching 的调度逻辑
  ↓
第4天：精读 block_manager.py，理解 PagedAttention 的内存管理
  ↓
第5天：看 attention backend，理解 paged attention kernel 的调用
  ↓
第6天：看 model_runner.py，理解输入准备和采样过程
  ↓
第7天：跑 benchmark，调参数，观察不同配置对性能的影响
```

### 调试技巧

```python
# 开启详细日志
export VLLM_LOGGING_LEVEL=DEBUG

# 关闭 CUDA Graph（方便断点调试）
vllm serve <model> --enforce-eager

# 用 py-spy 看性能瓶颈
pip install py-spy
py-spy top --pid <vllm_pid>

# 用 nsight systems 看 GPU kernel
nsys profile -o vllm_profile python -m vllm.entrypoints.openai.api_server ...
```

---

## 四、配套论文

按这个顺序读，和源码对照：

1. **PagedAttention 论文**（和 block_manager.py 对照）
   [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

2. **FlashAttention 论文**（和 attention backend 对照）
   [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

3. **Orca 论文**（和 scheduler.py 对照）
   [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

---

> 有问题随时问，我可以帮你解读具体的源码片段 🐼
