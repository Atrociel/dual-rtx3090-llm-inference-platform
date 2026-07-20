#!/bin/bash
# ==============================================================
# start-vllm-tp2.sh — Original TP=2 launch script (preserved)
# Dual RTX 3090 · Qwen3.6-35B-A3B pack-quantized · 16K context
# ==============================================================
# This is the original configuration using tensor parallelism.
# Compared to PP=2 (start-vllm.sh), this config:
#   - Splits weights at the tensor level across GPUs
#   - Requires 60+ all-reduce ops per forward pass
#   - Is bottlenecked by PCIe chipset communication latency
#   - Achieved 28 tok/s generation vs 112 tok/s with PP=2
#
# Preserved for reproducibility and comparison benchmarking.
# See benchmarks/parallelism-comparison.md for full analysis.
# ==============================================================

cd ~
source ~/vllm-venv/bin/activate  # original T P=2 venv (Python 3.10 + vLLM 0.22.0 V0)

VLLM_USE_V1=0 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve ~/models/qwen3.6-35b-a3b-awq \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --host 0.0.0.0 \
  --port 8000

# ==============================================================
# Flag choices and rationale (original):
#
# --tensor-parallel-size 2
#   35B model exceeds single GPU VRAM. TP=2 splits each tensor
#   across both GPUs. Requires all-reduce at every layer.
#
# VLLM_USE_V1=0
#   Original vLLM V0 engine. V1 was unstable with AWQ + MoE on
#   Ampere at the time this was deployed.
#
# --dtype bfloat16
#   Explicit dtype for activation computations. Model weights are
#   pack-quantized int4 via compressed-tensors.
#
# --max-model-len 16384
#   VRAM-limited. FP16 KV cache at 16K uses most of available
#   memory after model weights.
#
# --max-num-seqs 2
#   Conservative for agent workload. Higher batch sizes increased
#   TTFT variance without meaningful throughput gain due to PCIe
#   all-reduce bottleneck.
#
# --gpu-memory-utilization 0.85
#   Allocates ~20 GB per GPU, leaving ~4 GB headroom for KV cache
#   and scheduler overhead.
#
# --enforce-eager
#   CUDA graphs caused segfaults with AWQ + MoE on V0 engine with
#   TP=2. This disables them, costing ~10% throughput.
#
# --disable-custom-all-reduce
#   Required on bifurcated PCIe (no NVLink). vLLM's custom all-reduce
#   kernel deadlocks without direct P2P access.
# ==============================================================
