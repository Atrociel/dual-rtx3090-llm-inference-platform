# Performance Tuning Guide

Methodology and parameter exploration for optimizing LLM inference on dual RTX 3090s.

## Tuning Philosophy

Consumer GPU inference tuning is a constrained optimization problem:

```
maximize(throughput, context_length)
subject to: VRAM ≤ 48 GB, PCIe bandwidth ≤ Gen5 x8 (~32 GB/s)
```

Every parameter adjustment trades one resource against another. The goal is finding a stable operating point, not the theoretical maximum.

## The Breakthrough: Pipeline Parallelism (PP=2)

The single biggest performance win in this deployment was switching from **Tensor Parallelism (TP=2) to Pipeline Parallelism (PP=2)**.

### Why PP beats TP on consumer hardware

| Aspect | TP=2 | PP=2 |
|--------|------|------|
| **Communication** | All-reduce every layer | Single activation transfer per forward pass |
| **All-reduce ops/step** | ~60 (per transformer layer) | 0 (zero — NCCL P2P only for pipeline flush) |
| **PCIe sensitivity** | High — every all-reduce crosses bifurcated link | Low — one tensor crosses the link per step |
| **Throughput** | 28 tok/s | 94–116 tok/s |
| **GPU memory** | ~20 GB each | ~16–17 GB each |

**Why it works**: The Qwen3.6-35B-A3B model has 44 transformer layers (estimated). With TP=2, each token requires **44 all-reduce operations** across the PCIe bus. With PP=2, GPU0 processes layers 1–22, passes activations once to GPU1, which processes layers 23–44 — **1 inter-GPU transfer per token**.

### PP negative: GPU underutilization at small batch sizes

PP is inherently less efficient at batch size 1 because the two GPUs work sequentially. With a batch of 4+ tokens, micro-batch pipelining kicks in and utilization improves. This is why `--max-num-seqs 4` is important for PP configs.

## Parameter Exploration

### GPU Memory Utilization

| Setting | VRAM Free (per GPU) | Max Context | Stability |
|---------|--------------------|-------------|-----------|
| 0.95 | ~1.2 GB | 262K | ❌ OOM on concurrent requests |
| 0.85 | ~3.6 GB | 262K | ⚠️ Stable with chunked prefill |
| 0.80 | ~4.8 GB | 262K | ✅ Stable, safety margin |
| 0.75 | ~6.0 GB | 262K | ✅ Recommended — extra headroom negligible overhead |
| 0.70 | ~7.2 GB | 131K | ❌ Wasted VRAM cuts context by half |

**Verdict**: 0.75 at 262K context. At shorter contexts (16–32K), 0.85 works fine and yields slightly more KV cache capacity for concurrent requests.

### Pipeline Parallelism vs Tensor Parallelism

| Strategy | Generation | Context | Notes |
|----------|-----------|---------|-------|
| **TP=2** | 28 tok/s | 16K | Old config — PCIe all-reduce bottleneck |
| **PP=2** (current) | 94–116 tok/s | 262K | 3.4× faster — avoids all-reduce entirely |
| PP=2 w/o flashinfer | ~70 tok/s | 262K | FlashInfer adds ~30% |
| PP=2 w/ enforce-eager | ~85 tok/s | 262K | CUDA graphs add ~10% |

**Verdict**: PP=2 with FlashInfer and CUDA graphs is the optimal configuration. No expert parallelism or data parallelism needed.

### Attention Backend

| Backend | Generation | Notes |
|---------|-----------|-------|
| **FlashInfer** | 94–116 tok/s | Fastest on Ampere, good with variable-length |
| SDP (default) | ~75 tok/s | PyTorch native, slightly slower |
| FlashAttention | ~80 tok/s | vLLM built-in, comparable to SDP |
| FlashInfer + chunked prefill | Same | Handles long-prompt interleaving |

**Verdict**: FlashInfer is a clear win. Install handled by `vllm add flashinfer` or installed via pip.

### Chunked Prefill + Prefix Caching

These two settings work together for interactive serving:

| Setting | Benefit | Cost |
|---------|---------|------|
| Chunked prefill (8K) | Long prompts don't block batch | ~2% overhead from chunk scheduling |
| Prefix caching | TTFT drops to near-zero for repeated prefixes | ~3% KV cache reservation overhead |
| Both | Interactive serving with 100K+ context prompts | Negligible |

**Verdict**: Enable both. The overhead is minimal and the benefits for agent workloads are substantial.

### FP8 KV Cache

| KV dtype | Memory efficiency | Accuracy |
|----------|------------------|----------|
| FP32 | 1× | Reference |
| FP16 | 2× | ~identical |
| **FP8** (this config) | **4×** | ~99.9% preserved |
| INT8 | 4× | ~99.5% preserved |

At FP8, the 262K context fits in ~870 MB per GPU instead of ~3.5 GB with FP16. This makes the full 262K context feasible on 48 GB total VRAM.

### Context Length

| Length | Use Case | Concurrent requests (@FP8 0.75) |
|--------|----------|--------------------------------|
| 4K | Simple chat | ~65 |
| 16K | Agent interactions | ~16 |
| 32K | Long agent context | ~8 |
| 131K | Document analysis | ~2 |
| **262K** (this config) | Maximum context | ~1–2 |

### Max Sequences (Concurrency)

| max-num-seqs | Generation (short) | Latency per request | Stability |
|-------------|-------------------|---------------------|-----------|
| 1 | ~95 tok/s | ~210 ms | ✅ Very stable |
| 2 | ~95 tok/s | ~210 ms | ✅ Stable |
| 4 | ~94 tok/s (shared) | ~210 ms | ✅ Stable with chunked prefill |
| 8 | ~90 tok/s (shared) | ~250 ms | ⚠️ Higher latency variance |

**Verdict**: `max-num-seqs=4` is the sweet spot. Chunked prefill prevents any single request from blocking the batch, and 4-way concurrency handles agent burst patterns well.

## Observed Bottlenecks

| Bottleneck | Impact | Mitigation |
|------------|--------|------------|
| PCIe bandwidth (Gen5 x8, no NVLink) | ~5% vs ideal PP | PP=2 avoids all-reduce; only 1 transfer/step |
| VRAM capacity | Hard cap at 48 GB | 4-bit quantization, FP8 KV cache, 0.75 util |
| MoE routing overhead | ~2% latency | FlashInfer handles variable-length well |
| NCCL init on PP | ~1s at startup | One-time cost, acceptable |

## Recommended Tuning Process

1. **Start with production config** in `deployment/start-vllm.sh`
2. **Benchmark** with the script in `benchmarks/benchmark-command.md`
3. **Adjust context length**: 262K is supported — use it if your workload needs it; drop to 32K for more concurrency
4. **Tune memory utilization**: Increase to 0.80–0.85 if running shorter contexts; decrease to 0.70–0.75 for 262K
5. **Document every change**: Record what changed, what broke, what worked

## Before/After Summary

| Metric | TP=2 (v1) | PP=2 (v2, current) | Δ |
|--------|-----------|-------------------|---|
| Generation throughput | 28 tok/s | **116 tok/s** | **+314%** |
| Prompt processing | 814 tok/s | **2,921 tok/s** | **+259%** |
| Max context length | 16,384 | **262,144** | **+1,500%** |
| GPU memory used | ~40 GB (83%) | ~33 GB (69%) | **-17%** |
| Tool calling | ❌ No | ✅ qwen3_coder | New |
| CUDA graphs | ❌ Disabled | ✅ Enabled | +10% |
