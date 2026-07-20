# Dual RTX 3090 LLM Inference Platform

A production-grade LLM inference server running on consumer hardware. This repository documents the complete engineering process: hardware selection, deployment, benchmarking, optimization, and troubleshooting.

## Quick Facts

| | |
|---|---|
| **Model** | Qwen3.6-35B-A3B (4-bit pack-quantized, 35.3B param MoE, 3.6B active) |
| **Hardware** | 2× RTX 3090 24 GB (Z690 Taichi, PCIe Gen5 x8 bifurcation) |
| **Framework** | vLLM v0.22.0 (V1 engine, PP=2, FlashInfer, FP8 KV cache) |
| **Generation throughput** | **112.6 tok/s** (avg, 10 runs) |
| **Prompt processing** | **1,962–5,158 tok/s** |
| **Max context** | **262,144 tokens** |
| **GPU power** | **~430W total** (280W GPU1 capped, 350W GPU0) |
| **Status** | Production — serving Hermes Agent + Free Claude Code |

## Repository Structure

```
dual-rtx3090-llm-inference-platform/
├── README.md                      ← You are here
├── architecture/
│   └── hardware-specs.md          — BOM, GPU topology, system specs
├── benchmarks/
│   ├── benchmark-command.md       — Methodology (reproducible)
│   ├── qwen-results.md            — Latest real-world benchmarks
│   └── performance-tuning.md      — Parameter exploration & trade-offs
├── deployment/
│   ├── start-vllm.sh              — Production launch script (annotated)
│   ├── environment-notes.md       — VENV setup, model storage, FCC integration
│   └── docker-compose.yml         — Containerized deployment
├── homelab-notes/
│   └── future-proxmox-design.md   — Future Proxmox virtualization design
├── troubleshooting/
│   ├── cuda-issues.md             — CUDA error catalog with solutions
│   ├── nccl-debugging.md          — Multi-GPU communication diagnosis
│   └── lessons-learned.md         — Cumulative engineering log
└── screenshots/                   — Visual proof (add your own)
```

## Key Performance

| Metric | Value | vs TP=2 (old) |
|--------|-------|----------------|
| **Generation throughput (avg, 10 runs)** | 112.6 tok/s | **4×** |
| **Sustained generation (200 tok)** | 112.8 tok/s | **4×** |
| **Prompt processing** | 1,962–5,158 tok/s | **2.8× at 266 tok — 6.4× at 2K** |
| **Concurrent throughput (2 req)** | 199.5 tok/s | **4.2×** |
| **Tool calling** | ✅ `qwen3_coder` parser | New |
| **Max context length** | 262,144 tokens | **16×** |
| **GPU memory (steady-state)** | 33.3 GB / 48 GB total | 15% less |
| **GPU power** | ~430W total | **27% less** |

## Design Decisions

Every flag in the launch script (`deployment/start-vllm.sh`) represents a deliberate engineering trade-off. Here's the reasoning:

### Pipeline Parallelism (PP=2) instead of Tensor Parallelism (TP=2)

| Approach | Communication pattern | Overhead on PCIe Gen5 x8 |
|----------|----------------------|--------------------------|
| **TP=2** | All-reduce every layer across both GPUs | High — 60+ all-reduce ops per forward pass |
| **PP=2** (this config) | Single activation transfer per forward pass | Low — 1 transfer between pipeline stages |

The RTX 3090s connect through PCIe Gen5 x8 bifurcation (a single x16 lane split into two x8 paths, no NVLink). Effective bandwidth is ~32 GB/s per direction — equivalent to Gen4 x16 — but the topology means both GPUs share the same physical slot's lanes rather than having independent PHB connections. TP=2 requires per-layer all-reduce across this inter-GPU link, which capped throughput at 28 tok/s. PP=2 splits the model by layers: GPU0 runs layers 1–N/2, GPU1 runs N/2+1–N, with a single activation tensor passed between them. This avoids the all-reduce bottleneck entirely, delivering **3.4–4.1×** improvement.

The trade-off: PP=2 underutilizes both GPUs on small batches (<4 tokens, typical for single-user chat) because only one pipeline stage is active at a time. Batch-level pipelining mitigates this with `--max-num-seqs 4`.

### FlashInfer Attention Backend

FlashInfer is the fastest attention kernel backend for Ampere (compute 8.6). vLLM defaults to SDP (PyTorch's scaled dot-product attention), which is 20–30% slower on this hardware. FlashInfer's efficient variable-length sequence handling also makes it a better match for chunked prefill.

### FP8 KV Cache

KV cache is the largest memory consumer after model weights. At FP8, each KV entry takes 1 byte per value instead of 2 (FP16) or 4 (FP32). This enables **522,000 tokens** of KV cache within 48 GB VRAM. The accuracy trade-off is negligible for inference — FP8 KV cache preserves ~99.9% of model output quality.

### Chunked Prefill + Prefix Caching

Two complementary optimizations for interactive serving:
- **Chunked prefill** splits long prompts (e.g., 100K token agent context) into 8K chunks so the server can interleave decode for other requests between prefill chunks
- **Prefix caching** hashes KV cache blocks and reuses them across requests with identical prefixes (e.g., system prompts) — TTFT drops to near zero for the second identical request

### 262K Context Window

The model natively supports 262,144 tokens of context. With gpu-mem-util=0.75, FP8 KV cache, and PP=2, we can fit ~522K tokens total across both GPUs — enough for 2 simultaneous 262K-context requests. For typical agent workloads (4–32K context), this allows 16–130 concurrent requests.

### CUDA Graphs

Enabled (not using `--enforce-eager`). PP=2 + FlashInfer doesn't trigger the CUDA graph segfaults seen with TP=2 + AWQ Marlin. CUDA graphs capture the GPU execution path once and replay it, providing ~10% throughput improvement over eager mode.

### `--gpu-memory-utilization 0.75`

At 75%, ~36 GB of 48 GB VRAM is reserved for model + KV cache, leaving 12 GB headroom. This sounds conservative, but with FP8 KV cache at 262K context, KV cache spikes during concurrent prefill can be large. Higher values (0.85+) caused OOM with full context length.

### `--max-num-seqs 4` / `--max-num-batched-tokens 8192`

Balanced for agent workloads. Higher batch sizes increase throughput but degrade per-request latency. At max-num-seqs=4 with chunked prefill=8192, a single long prefill doesn't block the batch, yet burst throughput is available when four short requests arrive simultaneously.

## Why This Matters

This isn't "I ran a model." This is:

- **Hardware systems engineering** — building, diagnosing, and constraining consumer GPU hardware for a workload it wasn't designed for
- **Deployment engineering** — selecting and configuring vLLM flags with understanding of their interactions
- **Performance analysis** — benchmarking with real metrics, not logs
- **Troubleshooting** — solving NCCL deadlocks, CUDA OOM, dtype mismatches, and PCIe topology limits
- **Documentation** — capturing decisions, trade-offs, and failures so the next engineer learns faster

## Security

This deployment follows standard workstation hardening practices:

| Practice | Status | Notes |
|---|---|---|
| **Firewall** | ✅ UFW active, reject incoming | All inbound ports blocked by default |
| **SSH password auth** | ✅ Disabled | Key-based authentication only |
| **Root SSH login** | ✅ Key-only | `PermitRootLogin without-password` |
| **Brute-force protection** | ✅ fail2ban installed | 3 failed attempts → 1-hour ban |
| **Auto security updates** | ✅ unattended-upgrades | Critical patches applied automatically |
| **Kernel hardening** | ✅ ASLR, kptr_restrict, dmesg_restrict | Enabled at kernel level |
| **vLLM endpoint** | ✅ Bound to localhost | Service not exposed to LAN |
| **API keys** | ✅ Stored externally | Not committed to config/repo |
| **Secrets management** | ✅ `.env` files excluded | Use environment variables or vault |

**Note:** Your user is in the `docker` group (required for GPU containers). This is root-equivalent by design — treat Docker access with the same care as `sudo`.

## Getting Started

```bash
# 1. Start the server
./deployment/start-vllm.sh

# 2. Test
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "/home/ahmed/models/qwen3.6-35b-a3b-awq",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 10}'

# 3. Run benchmarks
python3 benchmarks/benchmark.py
```
