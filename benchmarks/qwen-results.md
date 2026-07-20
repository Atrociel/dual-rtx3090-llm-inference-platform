# Qwen3.6-35B-A3B — Benchmarks

Measured performance on dual RTX 3090s using vLLM.

## Current Configuration (PP=2)

| Setting | Value |
|---------|-------|
| **Model** | Qwen3.6-35B-A3B (pack-quantized 4-bit, compressed-tensors) |
| **Hardware** | 2× RTX 3090 24 GB (Z690 Taichi, PCIe Gen5 x8 bifurcation) |
| **vLLM version** | 0.22.0–0.23.0 (V1 engine) |
| **CUDA** | 12.4 |
| **Driver** | 550.xx |
| **Parallelism** | PP=2 (`--pipeline-parallel-size 2 --tensor-parallel-size 1`) |
| **KV cache** | FP8 |
| **Attention** | FlashInfer |
| **CUDA graphs** | Enabled |
| **GPU memory util** | 0.75 |

### Generation Throughput

| Metric | Value | Runs |
|--------|-------|------|
| **Average** | 117.3 tok/s single | 10 runs, 100-tok output |
| **Minimum** | 110.9 tok/s | Cold start excluded |
| **Maximum** | 119.1 tok/s | |
| **Variance** | <2% | |

### Sustained Continuous Load — 10 Minute Burn Test

Post-reboot, both GPUs at stock power limits (350W / 280W). 4 concurrent workers, continuous requests with no idle gaps.

| Metric | Value |
|--------|-------|
| **Total requests** | **1,108** — 0 errors |
| **Duration** | 605 seconds |
| **Aggregate throughput** | **200 tok/s** (all workers combined) |
| **Per-request throughput** | 49.9 tok/s avg (P50: 49.8 / P95: 52.3 / P99: 53.1) |
| **Latency avg / P95 / P99** | 2,182 / 2,227 / 2,263 ms |
| **GPU 0** | 72°C @ 248W, 100% util |
| **GPU 1** | 68°C @ 200W, 100% util |
| **System power** | ~488W |
| **Throughput/hr** | 719K tokens |
| **Operating cost** | **€0.08 per 1M tokens** |

**Note — GPU 1 hardware ceiling**: GPU 1 shows ~200W power draw under sustained load despite having no driver thermal history (counters reset on reboot). The SW thermal slowdown counter accumulates during load, indicating the driver is throttling based on GDDR6X memory junction temperature. This is common on RTX 3090s and does not affect aggregate throughput — PP=2 is saturated at GPU 0's pipeline stage.

### Single-Request Performance (Burst)

| Metric | Value |
|--------|-------|
| **Generation throughput** | **117.3 tok/s** (200-tok sustained) |
| **Prompt processing** | 1,962–5,158 tok/s (266 – 2K input) |
| **Time to first token (TTFT)** | 49 ms avg |
| **Concurrent (2 req)** | 199.5 tok/s aggregate (fair queuing, ~100 tok/s each) |

### Stability Verdict

Throughput variance <1.3% under sustained load. Thermal plateau reached within 2 min with no drift. The system is stable for 24/7 production inference. GPU 1 operates within its natural ~200W ceiling, which is already sufficient for the PP=2 config.

## Prompt Processing (Prefill)

| Input length | Tokens/s |
|-------------|----------|
| 266 tokens | 1,962 tok/s |
| 522 tokens | 2,998 tok/s |
| 1,034 tokens | 3,879 tok/s |
| 2,058 tokens | 5,158 tok/s |

Prefill scales with batch size — longer prompts achieve higher throughput as GPU parallelism fills.

### Concurrency

| Scenario | Throughput | Notes |
|----------|-----------|-------|
| Single request | 112.6 tok/s | Baseline |
| 2 concurrent requests | 199.5 tok/s | Aggregated, 1.00s wall time |
| Per-request (2 concurrent) | ~99.9 tok/s each | Fair queuing |

### Latency

| Metric | Value |
|--------|-------|
| **TTFT (avg)** | 49 ms |
| **Per-token generation** | ~8.9 ms |

### GPU Utilization

| GPU | Memory | Power | Utilization |
|-----|--------|-------|-------------|
| **GPU 0** (layers 1–22) | 16,698 MB / 24,576 MB (68%) | 227.5 W | Prefill-heavy |
| **GPU 1** (layers 23–44) | 17,367 MB / 24,576 MB (71%) | 203.4 W | Decode-heavy |

Total system power during inference: ~430W (both GPUs + CPU).

## Comparison: TP=2 vs PP=2

| Metric | TP=2 (Original) | PP=2 (Current) | Δ |
|--------|----------------|----------------|---|
| **Generation** | 28.0 tok/s | **112.6 tok/s** | **+302%** |
| **Sustained (200 tok)** | 28.0 tok/s | **112.8 tok/s** | **+303%** |
| **Prompt processing** | ~700 tok/s | **1,962–5,158 tok/s** | **+180–640%** |
| **TTFT** | 54 ms | **49 ms** | -9% |
| **2 concurrent** | ~48 tok/s | **199.5 tok/s** | **+316%** |
| **Max context** | 16,384 | **262,144** | **+1,500%** |
| **GPU memory** | ~20 GB each | **~17 GB each** | -14% |
| **GPU power** | ~600W total | **~430W total** | -27% |
| **Tool calling** | ❌ | ✅ | New |

### Why PP=2 Wins

The fundamental difference is **communication pattern**:

- **TP=2**: 44 all-reduce operations per token (one per transformer layer) across PCIe Gen5 x8. Each all-reduce requires GPU0→GPU1 write, reduction, GPU1→GPU0 write-back — 88 PCIe traversals per token.

- **PP=2**: 1 activation tensor transfer per token across PCIe. GPU0 processes layers 1–22, sends the result to GPU1 once, GPU1 processes layers 23–44. Only 2 PCIe traversals per token.

On hardware with NVLink, TP=2's communication penalty would be much smaller and it might outperform PP=2. On PCIe-connected consumer GPUs, PP=2 is decisively better.

## Reproducing

```bash
# Start server
./deployment/start-vllm.sh

# Run the comprehensive benchmark
python3 benchmarks/comprehensive-benchmark.py
```
