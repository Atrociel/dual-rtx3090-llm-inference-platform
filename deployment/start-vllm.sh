#!/bin/bash
# ==============================================================
# start-vllm.sh — Production vLLM launch script
# Dual RTX 3090 · Qwen3.6-35B-A3B pack-quantized · 262K context
# ==============================================================
# This is the stable production configuration. Every flag here
# was chosen to address a real hardware constraint or software
# bug encountered during deployment.
# ==============================================================

cd ~
source ~/vllm/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
vllm serve ~/models/qwen3.6-35b-a3b-awq \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.75 \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 127.0.0.1 \
  --port 8000

# ==============================================================
# Flag choices and rationale:
#
# --pipeline-parallel-size 2 / --tensor-parallel-size 1
#    35B MoE model split by layers across GPUs rather than tensor sharding.
#    PP avoids the per-layer all-reduce bottleneck on PCIe chipset topology,
#    trading it for a single activation transfer between GPU0 and GPU1
#    per forward pass. This is ~5% faster than TP=2 on this hardware
#    and avoids the custom-all-reduce deadlock entirely.
#
# --dtype bfloat16
#    Explicit dtype avoids auto-detection mismatch. The model's
#    compressed-tensors weights are stored as int4, but activations
#    and intermediate computations must be bfloat16 for MoE routing
#    and attention. Without this, the engine would default to float16
#    and crash on fused MoE operations.
#
#    (No --quantization flag: vLLM auto-detects compressed-tensors
#    from the model config.json. Pack-quantized 4-bit with group_size=32.)
#    Internal: uses CompressedTensorsWNA16MarlinMoEMethod backend.
#
# --max-model-len 262144
#    Full native context length. KV cache is partitioned at ~522K tokens
#    total across both GPUs with gpu-mem-util=0.75 and fp8 KV cache.
#    Maximum concurrency at this context length: ~2 simultaneous requests.
#
# --max-num-seqs 4 / --max-num-batched-tokens 8192
#    Balanced for agent workloads with tool calling. Chunked prefill
#    splits long prompts into 8K token chunks so a 100K context prompt
#    doesn't block the batch for a single prefill cycle.
#
# --gpu-memory-utilization 0.75
#    Leaves 25% headroom (~6 GB per GPU) for KV cache spikes during
#    long-context prefill. With --enable-chunked-prefill and --enable-prefix-caching,
#    this is sufficient for interactive agent use. Values at 0.85+
#    caused OOM with full 262K context.
#
# --kv-cache-dtype fp8
#    Halves KV cache memory footprint vs fp16, enabling 522K tokens
#    of cache within 48 GB total VRAM. Slight accuracy tradeoff but
#    acceptable for inference workloads.
#
# --attention-backend flashinfer
#    Uses FlashInfer — the fastest attention kernel backend for Ampere.
#    Handles variable-length sequences efficiently with chunked prefill.
#
# --enable-chunked-prefill
#    Splits long prefills into 8K chunks. Prevents a single long prompt
#    from monopolizing GPU compute and delaying interleaved decode requests.
#    Critical for interactive agent serving where prompt length varies wildly.
#
# --enable-prefix-caching
#    Caches KV entries by hash. When the same system prompt prefix appears
#    across requests (common in agent frameworks), TTFT drops to near zero
#    for subsequent requests with the same prefix.
#
# --enable-auto-tool-choice --tool-call-parser qwen3_coder
#    Enables native function-calling support. The model can output
#    structured tool calls that vLLM parses into OpenAI-compatible
#    tool_calls format. Required for agent frameworks like Hermes.
#
# (No --enforce-eager — CUDA graphs work fine with PP=2 + flashinfer.)
# (No --disable-custom-all-reduce — PP mode uses NCCL P2P for single
#  activation transfer between pipeline stages, not all-reduce.)
# ==============================================================
