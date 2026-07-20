# Parallelism Strategy Comparison

## TP=2 vs PP=2 on Dual RTX 3090 (PCIe Gen5 x8 Bifurcation)

This document compares the two valid production configurations discovered during optimization. Each represents a different architectural approach to multi-GPU inference.

## The Hardware Constraint

| Factor | Detail |
|--------|--------|
| GPUs | 2× RTX 3090 24 GB |
| Interconnect | PCIe Gen5 x8 bifurcation (single x16 → two x8) |
| P2P | ❌ No NVLink, no direct peer access |
| Effective bandwidth | ~32 GB/s per direction (Gen5 x8 ≈ Gen4 x16) |

The absence of NVLink forces a communication bottleneck. Any strategy that requires frequent GPU-to-GPU synchronization pays a PCIe tax. The question is: *how much tax per strategy?*

## Architectural Comparison

```
TP=2 (Tensor Parallelism)
                          all-reduce
                     ┌─── every layer ──┐
                     ↓                  ↓
                ┌─────────┐       ┌─────────┐
                │  GPU 0  │◄─────►│  GPU 1  │
                │  half   │       │  half   │
                │weights  │       │weights  │
                └─────────┘       └─────────┘
                Communication: 60+ all-reduce ops per forward pass
                Bottleneck: Every layer waits for both GPUs to sync


PP=2 (Pipeline Parallelism)
                ┌──────────┐    P2P send    ┌──────────┐
                │  GPU 0   │───────────────►│  GPU 1   │
                │ layers   │                │ layers   │
                │  1–22    │                │  23–44   │
                └──────────┘                └──────────┘
                Communication: 1 activation transfer per forward pass
                Bottleneck: GPU idle during pipeline bubble (small batches)
```

### Communication Cost per Token

| Strategy | Inter-GPU transfers per token |
|----------|------------------------------|
| **TP=2** | ~60 all-reduce operations (each: GPU0→GPU1→GPU0, ~1 μs + PCIe latency) |
| **PP=2** | 1 P2P send (GPU0 → GPU1, single tensor) |

For the 44-layer Qwen model with TP=2, every single token requires **44 all-reduce syncs** across the shared PCIe Gen5 x8 link. PP=2 requires exactly **1 tensor transfer** per token.

## Benchmark Results

### Test Conditions

| Setting | TP=2 Config | PP=2 Config |
|---------|-------------|-------------|
| vLLM version | 0.22.0 | 0.22.0 / 0.23.0 |
| Engine | V0 (`VLLM_USE_V1=0`) | V1 (default) |
| Parallelism | `--tensor-parallel-size 2` | `--pipeline-parallel-size 2` |
| Quantization | AWQ (pack-quantized 4-bit) | AWQ (pack-quantized 4-bit) |
| KV cache dtype | FP16 | **FP8** |
| GPU memory util | 0.85 | **0.75** |
| Max context | 16,384 | **262,144** |
| Attention backend | Default (SDP) | **FlashInfer** |
| CUDA graphs | Disabled (`--enforce-eager`) | **Enabled** |
| Custom all-reduce | Disabled | N/A (PP doesn't use it) |
| Chunked prefill | No | **Yes** |
| Prefix caching | No | **Yes** |

### Performance

| Metric | TP=2 | PP=2 | Δ |
|--------|------|------|---|
| **Generation throughput** | 28.0 tok/s | **112.6 tok/s** | **+302%** |
| **Sustained generation (200 tok)** | 28.0 tok/s | **112.8 tok/s** | **+303%** |
| **Prompt processing (266 tok)** | ~700 tok/s | **1,962 tok/s** | **+180%** |
| **Prompt processing (2,058 tok)** | — | **5,158 tok/s** | — |
| **TTFT** | 54 ms | **49 ms** | **-9%** |
| **Concurrent throughput (2 req)** | ~48 tok/s | **199.5 tok/s** | **+316%** |
| **Max context length** | 16,384 | **262,144** | **+1,500%** |
| **GPU memory (per GPU)** | 20.1 GB / 19.9 GB | **16.7 GB / 17.4 GB** | **-14%** |
| **GPU power (per GPU)** | ~300W / ~300W | **228W / 203W** | **-27%** |
| **Variance (10 runs)** | <2% | **<2%** | Same |

## Analysis

### Why PP=2 Wins on This Hardware

**1. Communication pattern matters more than bandwidth**

TP=2's all-reduce per layer is pathological on bifurcated PCIe. Each of the 44 all-reduce operations per token requires:
- GPU0 → GPU1 (PCIe write)
- GPU1 computes reduction → GPU0 (PCIe write back)

That's 88 PCIe traversals per token for a 44-layer model. PP=2 sends the full activation tensor **once** — one PCIe traversal per token. The overall data volume might be similar, but the **latency overhead of 44 synchronizations** is the killer.

**2. FP8 KV cache frees VRAM for longer context**

FP8 halves KV cache memory vs FP16. Combined with the lower GPU memory utilization (0.75 vs 0.85), the PP=2 config uses **14% less VRAM** while supporting **16× longer context**.

**3. FlashInfer exploits Ampere's compute capabilities**

FlashInfer's attention kernels are optimized for Ampere's compute capability 8.6. Combined with the V1 engine's scheduler, this adds ~30% throughput over the default SDP backend.

**4. CUDA graphs add ~10%**

With PP=2 + FlashInfer, CUDA graph compilation succeeds (it segfaulted with TP=2 + AWQ). This recovers the ~10% throughput that `--enforce-eager` forfeited.

### When TP=2 Would Win

TP=2 has theoretical advantages that would materialize on different hardware:

| Scenario | Expected winner |
|----------|---------------|
| **NVLink-connected GPUs** (A100, H100, 4090 NVLink) | **TP=2** — low all-reduce latency |
| **Single GPU with enough VRAM** | Neither — no parallelism overhead |
| **Very small models** (<7B) | **TP=2** — lower pipeline bubble penalty |
| **Batch-heavy workloads** (>8 concurrent) | **TP=2** — better GPU utilization at scale |

## Engineering Value

This comparison demonstrates:

1. **Hardware-aware parallelism selection** — understanding that communication topology determines which parallel strategy performs best
2. **Measured optimization** — both configurations were benchmarked, not guessed
3. **Trade-off articulation** — PP=2 wins on this hardware for these reasons; TP=2 would win under different conditions
4. **Full-stack optimization** — not just parallelism, but KV cache format, attention backend, CUDA graphs, memory allocation, and context length all tuned together

## Config Files

| Config | File |
|--------|------|
| TP=2 (original) | `deployment/start-vllm-tp2.sh` |
| PP=2 (current) | `deployment/start-vllm.sh` |

Both scripts are annotated with the rationale for each flag.
