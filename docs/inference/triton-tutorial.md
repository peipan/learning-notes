# Triton 入门指南

> Pan老师的 GPU Kernel 开发第一课
> 2026-03-02

---

## 一、Triton 是什么

Triton 是 OpenAI 开源的 GPU 编程语言，定位在 CUDA 和 PyTorch 之间：

```
抽象层次（从高到低）：

PyTorch        → 最高层，自动调度，不控制 kernel
torch.compile  → 编译器自动融合，有限控制
Triton         → 写 kernel，但不用管 thread/warp/shared memory  ← 你在这里入门
CUDA           → 完全控制，但开发成本高
PTX/SASS       → 汇编级，极少需要
```

为什么先学 Triton 而不是 CUDA：
- 开发效率高 5-10 倍，性能通常能达到 CUDA 的 85-95%
- vLLM 和 SGLang 大量使用 Triton kernel
- FlashAttention 有 Triton 实现版本，方便学习
- 理解了 Triton 再学 CUDA 会更顺

---

## 二、GPU 编程核心概念（必须先懂）

在写第一行 Triton 代码之前，这些概念要清楚：

### 1. GPU 硬件模型

```
GPU
├── SM (Streaming Multiprocessor) × N个    ← A100有108个SM
│   ├── Warp (32个线程一组)                ← GPU调度的最小单位
│   ├── Shared Memory (SRAM)              ← 快，但小（A100: 192KB/SM）
│   └── Register File                     ← 最快，每个线程私有
├── L2 Cache                              ← 所有SM共享（A100: 40MB）
└── HBM (Global Memory / DRAM)            ← 大但慢（A100: 80GB, 2TB/s）
```

关键数字（A100）：
- HBM 带宽：2 TB/s
- L2 带宽：~4 TB/s  
- Shared Memory 带宽：~19 TB/s
- 计算能力：312 TFLOPS (FP16 Tensor Core)

### 2. Compute-bound vs Memory-bound

```
Arithmetic Intensity = FLOPs / Bytes Accessed

如果 AI > 硬件的 ops:byte ratio → compute-bound（计算是瓶颈）
如果 AI < 硬件的 ops:byte ratio → memory-bound（内存带宽是瓶颈）

A100 的 ops:byte ratio ≈ 312 TFLOPS / 2 TB/s ≈ 156 FLOPs/Byte
```

LLM 推理中：
- Prefill 阶段（大 batch GEMM）→ 通常 compute-bound
- Decode 阶段（batch=1 的 GEMV）→ 几乎总是 memory-bound
- Softmax、LayerNorm、RMSNorm → memory-bound（计算简单，数据搬运多）

这就是为什么 kernel fusion 对 memory-bound 操作特别有效 — 减少数据搬运次数。

### 3. Tiling（分块）

GPU kernel 优化的核心思想：把大矩阵切成小块，每块放进 SRAM 处理。

```
大矩阵 (M×N)
┌──────────────────┐
│ Block │ Block │...│  ← 每个 Block 由一个 program instance 处理
│ (BM×BN)│       │   │  ← Block 大小要适配 SRAM 容量
├────────┤       │   │
│ Block  │ Block │...│
└──────────────────┘
```

---

## 三、Triton 编程模型

### 核心概念

```python
@triton.jit
def my_kernel(..., BLOCK_SIZE: tl.constexpr):
    # 1. 确定当前 program 处理哪块数据
    pid = tl.program_id(axis=0)
    
    # 2. 计算这块数据的偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. 边界检查（mask）
    mask = offsets < n_elements
    
    # 4. 从 HBM 加载数据到 SRAM
    data = tl.load(ptr + offsets, mask=mask)
    
    # 5. 在 SRAM 中计算
    result = some_computation(data)
    
    # 6. 写回 HBM
    tl.store(out_ptr + offsets, result, mask=mask)
```

Triton 的关键抽象：
- 你写的是一个 **program**（不是 thread），每个 program 处理一个 block 的数据
- `tl.arange` 生成一组连续的索引（类似 numpy）
- `tl.load` / `tl.store` 处理内存访问，自动做 coalescing
- `BLOCK_SIZE` 是 `constexpr`，编译时确定，必须是 2 的幂

### 和 CUDA 的对比

| 概念 | CUDA | Triton |
|------|------|--------|
| 并行单位 | thread | program (处理一个 block) |
| 内存管理 | 手动管理 shared memory | 编译器自动管理 |
| 同步 | __syncthreads() | 不需要（编译器处理） |
| 向量化 | 手动 | 自动 |
| 调优 | 手动选 block/grid size | autotune 装饰器 |

---

## 四、实战练习（由浅入深）

### 练习1：Vector Add（Hello World）

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 program 的 ID
    pid = tl.program_id(axis=0)
    
    # 计算这个 program 负责的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 边界保护
    mask = offsets < n_elements
    
    # 加载 → 计算 → 存储
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    # grid: 需要多少个 program instance
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# 测试
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_triton = add(x, y)
output_torch = x + y
print(f"Max diff: {torch.max(torch.abs(output_torch - output_triton))}")
# 应该输出 0.0
```

**理解要点：**
- `triton.cdiv` = 向上取整除法，确保所有元素都被覆盖
- `grid` 决定启动多少个 program（类似 CUDA 的 grid size）
- `mask` 处理最后一个 block 可能不满的情况

### 练习2：Fused Softmax（第一个有实际意义的 kernel）

这是推理优化中最经典的例子 — 把 5 次 HBM 读写融合成 1 次。

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # 每个 program 处理若干行（persistent kernel 模式）
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # 计算当前行的起始指针
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        
        # 加载一整行到 SRAM（一次 HBM 读取）
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        # 全部在 SRAM 中计算（不再访问 HBM）
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        # 一次写回 HBM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 必须是2的幂
    
    num_warps = 8
    num_stages = 4
    
    y = torch.empty_like(x)
    
    # persistent kernel: 用较少的 program 处理所有行
    num_programs = min(n_rows, 128)  # 不超过128个program
    softmax_kernel[(num_programs, 1, 1)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y


# 测试
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), "Results don't match!"
print("Softmax kernel works correctly!")
```

**理解要点：**
- 为什么快：PyTorch 的 softmax 需要 5 次 HBM 读 + 3 次 HBM 写，fused 版本只需 1 读 1 写
- `tl.range` + `num_stages` = software pipelining（预取下一批数据）
- `other=-float('inf')` 处理 padding 位置，不影响 softmax 结果
- 限制：行宽必须 fit 进 SRAM（这就是为什么 FlashAttention 要用 tiling）

### 练习3：Matrix Multiplication（进阶，理解 tiling）

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # L2 cache 优化：super-grouping
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 计算 A 和 B 的 block 指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 累加器（在寄存器中）
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 沿 K 维度循环累加
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k[None, :] < (K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)  # 从 HBM 加载 A 的一个 tile
        k_mask_b = offs_k[:, None] < (K - k * BLOCK_SIZE_K)
        b = tl.load(b_ptrs, mask=k_mask_b, other=0.0)  # 从 HBM 加载 B 的一个 tile
        accumulator = tl.dot(a, b, accumulator)  # 在 SRAM 中做矩阵乘
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    # 写回结果
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

**理解要点：**
- `@triton.autotune` 自动搜索最优的 BLOCK_SIZE 组合
- `GROUP_SIZE_M` 的 super-grouping 优化 L2 cache 命中率（A100 上能提升 10%+）
- 沿 K 维度的循环就是经典的 tiling：每次加载一小块到 SRAM，累加到寄存器
- `tl.dot` 会自动使用 Tensor Core（如果硬件支持）

---

## 五、练习路线

```
Day 1: 跑通 vector add，理解 program/grid/mask 概念
  ↓
Day 2: 跑通 fused softmax，理解 kernel fusion 和 memory-bound 优化
  ↓
Day 3: 跑通 matmul，理解 tiling、L2 cache 优化、autotune
  ↓
Day 4: 自己写一个 fused RMSNorm kernel（LLM 中高频使用）
  ↓
Day 5: 自己写一个 fused RoPE kernel（位置编码）
  ↓
Day 6: 读 FlashAttention 的 Triton 实现，理解 online softmax + tiling
  ↓
Day 7: 尝试写一个简化版的 FlashAttention（单头、无 mask）
```

### Day 4 作业提示：Fused RMSNorm

```python
# RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
# 
# PyTorch 实现需要多次 HBM 读写
# Fused kernel 只需要 1 次读 x + 1 次读 weight + 1 次写 output
#
# 提示：结构和 softmax kernel 类似
# - 每个 program 处理一行
# - 加载一行到 SRAM
# - 计算 RMS = sqrt(mean(x^2) + eps)
# - 归一化并乘以 weight
# - 写回
```

---

## 六、进阶资源

### 必读
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/) — 从 01 到 06 全部做一遍
- [FlashAttention Triton 实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — LinkedIn 开源的 Triton kernel 集合，代码质量高，适合学习

### 推荐
- [GPU Mode](https://github.com/gpu-mode) — 社区驱动的 GPU 编程学习资源
- [PMPP 教材](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311) — 《Programming Massively Parallel Processors》经典教材
- [Triton Puzzles](https://github.com/srush/Triton-Puzzles) — Sasha Rush 的 Triton 练习题，非常好

### 从 Triton 到 CUDA
当你 Triton 写熟了，想进一步：
- 先读 CUTLASS 的文档，理解 GEMM 的分层 tiling（threadblock → warp → instruction）
- 再看 FlashAttention 的 CUDA 实现，对比 Triton 版本
- 最后尝试自己写 CUDA kernel

---

> 建议先把练习1-3跑通，有问题随时贴代码过来，我帮你 debug 🐼
