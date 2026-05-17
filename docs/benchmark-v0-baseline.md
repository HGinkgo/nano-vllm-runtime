# nano-vllm Baseline v0

## 项目预期

两个月内，目标交付 5 个东西：

1. 一个基于 `nano-vllm` 改出来的单机 GPU LLM inference runtime
2. 这个 runtime 至少包含 1 个核心改造方向：`scheduler / chunked prefill / KV cache / speculative decoding`
3. 一套 benchmark + profiling 工具，能评估 `throughput / latency / TTFT / ITL`
4. 基于 `mini-llm-engine` 手写并理解 2~3 个 CUDA 算子，如 `RMSNorm / RoPE / GEMV`
5. 一份能用于简历和面试叙事的项目总结，讲清楚设计、改造点、实验结果、性能分析

从 `nano-vllm-plus` 继承的内容：

- inference runtime 的整体骨架
- request / sequence / scheduler / model runner 这条主执行链路
- chunked prefill、continuous batching、KV cache/block management 这类框架级设计
- benchmark / profiling 驱动优化的方法
- speculative decoding 这类面向性能提升的系统级改造思路

从 `mini-llm-engine` 继承的内容：

- 对 LLM 单 token 推理流程的底层理解
- 对 attention、KV cache、prefill/decode 的执行细节理解
- C++/CUDA 手写算子的实现经验
- 对 `RMSNorm / RoPE / GEMV / INT8 GEMV` 等核心算子的性能直觉
- 从算子视角理解 runtime hotspot 的能力

项目定位：

做一个有 runtime 改造、有性能评测、有 CUDA 算子理解的个人 LLM inference 项目。

## 2026.04.21

## 1. 本次目标

目标：记录 `nano-vllm` 在当前机器上的第一版可复现 baseline。

日期：2026.04.21

当前阶段：跑通框架 / 建立 baseline / 源码初读

## 2. 实验环境

- 系统：Ubuntu 24.04.4 LTS on WSL2，kernel `6.6.87.2-microsoft-standard-WSL2`
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU
- 显存：6144 MiB
- CUDA：13.2（driver）；PyTorch build `cu128`，`torch.version.cuda = 12.8`
- Python：3.12.3
- PyTorch：2.9.0+cu128
- transformers：5.5.4
- flash-attn：2.8.3
- nano-vllm：0.2.0

## 3. 模型信息

- 模型名称：Qwen3-0.6B
- 精度设置：`bfloat16`
- tokenizer：`Qwen2Tokenizer`
- 最大上下文长度：`40960` tokens，以模型 `config.json` 中的 `max_position_embeddings` 为准
- 备注：`tokenizer.model_max_length = 131072`，但模型配置里的实际位置长度是 `40960`；当前示例从本地目录直接加载模型和 tokenizer

## 4. 运行配置

以下按当前 benchmark 脚本的设置填写。

- `tensor_parallel_size`：`1`
- `enforce_eager`：`False`
- `device`：`cuda:0`
- `dtype`：`bfloat16`
- `seed`：`0`

其他关键配置：

- `model path`：`/home/zte/huggingface/Qwen3-0.6B/`
- `max_model_len`：`4096`
- `max_num_batched_tokens`：`16384`
- `max_num_seqs`：`512`
- `gpu_memory_utilization`：`0.9`
- `kvcache_block_size`：`256`
- `temperature=0.6`
- `ignore_eos=True`
- `max_tokens`：每个请求随机采样，范围 `100~1024`
- `use_tqdm=False`

说明：

- `device=cuda:0` 表示单卡运行
- `dtype=bfloat16` 来自当前 Qwen3-0.6B 的模型精度
- `seed=0` 只固定了 Python `random`，用于保证 benchmark 中随机输入和随机输出长度可复现

## 5. Workload 定义

- 测试目标：当前脚本直接测的是吞吐（tokens/s）
- 输出指标：脚本输出 `Total / Time / Throughput`
- 缺失指标：没有直接统计单请求延迟，也没有采集显存占用
- 请求数：`256`
- 并发请求数：`256`
- `max_num_seqs`：`512`
- `max_num_batched_tokens`：`16384`
- 输入长度：每个请求随机采样 `100~1024` tokens
- 输出长度：每个请求随机采样 `100~1024` tokens
- prompt 构造方式：随机生成 token id 序列，不是真实文本，也不是固定模板
- warmup：是，计时前会先执行一次简单生成
- 重复次数：`1`

## 6. 运行命令

```bash
source /home/zte/venvs/torch/bin/activate
python bench.py
```

## 实验结果

| Exp ID | Requests | Input Len | Output Len | Throughput (tok/s) | Latency (s) | GPU Mem (GB) | Mode/Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 16 | 128 | 128 | 1279.74 | 1.60 | 5.63 | baseline-short, `enforce_eager=False`, `max_model_len=2048`, 峰值增量约 `4.30GB` |
| 2 | 32 | 256 | 256 | 1530.10 | 5.35 | 5.68 | throughput-scale-1, `enforce_eager=False`, `max_model_len=2048`, 峰值增量约 `4.48GB` |
| 3 | 64 | 256 | 256 | N/A | N/A | N/A | `IndexError: list index out of range`，不是 OOM，更像框架边界问题 |
| 4 | 16 | 1024 | 128 | 603.22 | 3.40 | 5.74 | prefill-stress, 长输入短输出, `enforce_eager=False`, 峰值增量约 `4.74GB` |
| 5 | 16 | 128 | 1024 | 1329.41 | 12.32 | 5.32 | decode-stress, 短输入长输出, `enforce_eager=False`, 峰值增量约 `4.32GB` |
| 6 | 32 | 100~1024 | 100~1024 | N/A | N/A | N/A | `IndexError: list index out of range`，混合随机 workload 不稳定 |
| 7 | 32 | 256 | 256 | 758.21 | 10.80 | 5.29 | eager-compare, `enforce_eager=True`, `max_model_len=2048`, 峰值增量约 `4.34GB` |

## Benchmark 小结

本次测试在 `RTX 3060 Laptop GPU 6GB` 上进行，目标是观察 `nano-vllm` 在单卡场景下的吞吐、耗时和显存表现，并初步了解不同 workload 对推理性能的影响。测试过程中固定使用单卡推理，重点对比了请求数、输入长度、输出长度以及 `enforce_eager` 配置对结果的影响。

从结果来看，当前这套环境下较稳的单卡 baseline 是：

- `Requests=32`
- `Input Len=256`
- `Output Len=256`
- `enforce_eager=False`
- `max_model_len=2048`

在这组配置下，中位数结果约为：

- `Throughput = 1530.10 tok/s`
- `Latency = 5.35 s`
- `GPU Mem = 5.68 GB`

从不同实验对比可以看到几个明显现象：

- `enforce_eager=False` 明显更适合做性能测试。在相同的 `32 x 256 x 256` 配置下，`False` 的吞吐约为 `1530 tok/s`，而 `True` 只有约 `758 tok/s`，性能差距接近 2 倍。
- 长输入比长输出更容易拖慢吞吐。`prefill-stress` 的吞吐只有约 `603 tok/s`，明显低于 `decode-stress` 的约 `1329 tok/s`，说明这套实现对 prefill 阶段更敏感。
- 显存占用整体已经比较接近 6GB 卡的上限。多数实验的峰值总占用都在 `5.3 ~ 5.7 GB` 左右，因此后续继续提高请求数或放大上下文长度时，稳定性和边界问题都会更明显。
- 部分高压力或混合 workload 配置没有正常完成。`Exp 3` 和 `Exp 6` 都出现了 `IndexError: list index out of range`，更像是当前 `nano-vllm` 实现里的调度或缓存边界问题，而不是单纯 OOM。

综合来看，`nano-vllm` 在这张 `3060 6GB` 上已经可以稳定完成基础吞吐测试，也能体现出 `prefill / decode / eager` 等关键行为差异。但如果继续提高 batch 或使用更复杂的 mixed workload，当前实现还存在一定稳定性问题。后续如果要做更深入的框架评测，建议优先排查 `Exp 3` 和 `Exp 6` 触发的边界错误。

## 今日总结

1. 已建立 `nano-vllm` 在当前 RTX 3060 6GB 环境上的第一版 baseline。
2. 当前阶段最直接关注的性能指标是 throughput 和总 latency，后续可再细化到 `TTFT / TPOT`。
3. 本次实验摸清了当前机器上的可运行边界，知道了哪些 workload 稳定、哪些配置会触发异常。
4. `enforce_eager` 配置会显著影响吞吐表现，说明执行模式是后续需要重点理解的因素。
5. 长输入短输出明显慢于短输入长输出，初步判断是长输入更重，因为 prefill 要对整段上下文做更重的 attention 计算；decode 借助 KV cache，每步只增量处理一个 token。
6. 高压力和混合 workload 下出现 `IndexError`，暂时更像框架边界问题，后续需要结合源码继续定位。

## 2026.04.28

待补充。
