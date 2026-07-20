# CUDA Issues Log

A record of CUDA-related problems encountered during deployment and their solutions.

## Issue 1: AWQ + MoE dtype Mismatch

**Symptom**: vLLM engine crashes during model warm-up with errors like:
```
RuntimeError: expected scalar type BFloat16 but found Float
```
or silently fails on the first inference request.

**Root Cause**: vLLM v0.22.0's automatic dtype detection selects `torch.float16` as the default, but the AWQ quantization path and MoE routing operations expect `bfloat16`. The mismatch occurs during fused MoE kernel execution.

**Solution**: Explicitly set `--dtype bfloat16` in the launch command. This forces all engine operations to use bfloat16 precision consistently.

```bash
# ❌ Wrong — auto-detect picks float16
vllm serve ~/models/qwen3.6-35b-a3b-awq

# ✅ Correct — explicit bfloat16
vllm serve ~/models/qwen3.6-35b-a3b-awq --dtype bfloat16
```

## Issue 2: CUDA Out of Memory (OOM) During Long Context

**Symptom**: vLLM starts successfully but crashes during inference with long prompts or high concurrency:
```
CUDA out of memory. Tried to allocate ... MiB
```

**Root Cause**: KV cache grows proportionally to context length and batch size. At `gpu-memory-utilization=0.95`, only ~2.4 GB of headroom remains — insufficient for KV cache spikes during prefill.

**Solution**: Reduce `gpu-memory-utilization` to 0.85 (leaves ~7.2 GB headroom) and cap `max-model-len` at 32,768 for production.

**Progressive tuning**:
1. Start at 0.80 and 16K context
2. Increase context to 32K, keep 0.80
3. Slowly raise utilization: 0.80 → 0.82 → 0.85
4. Test with your actual workload before calling it stable

## Issue 3: CUDA 12.0 + Driver 595 Compatibility

**Symptom**: vLLM fails to load CUDA kernels with cryptic errors about missing symbols or incompatible versions.

**Root Cause**: The vLLM venv may have multiple CUDA stubs or conflicting CUDA paths. Specifically, `~/vllm-venv/lib/python3.12/site-packages/nvidia/cu13/` provides an alternative CUDA toolkit that can conflict with the system-installed CUDA 12.0.

**Solution**: Explicitly set CUDA environment variables when launching:

```bash
export CUDA_HOME=/home/ahmed/vllm-venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Alternatively, let vLLM auto-detect by **not** setting CUDA_HOME — the venv's CUDA package is self-contained.

## Issue 4: `trust_remote_code` Warning

**Symptom**: Warning during model loading about `trust_remote_code` being required:
```
Trusting remote code is required for this model
```

**Solution**: Add `--trust-remote-code` flag. This is required for Qwen models that include custom modeling code:

```bash
vllm serve ... --trust-remote-code
```

## Issue 5: FP8 Model Loading Failure

**Symptom**: The FP8 quantized model variant fails to load or produces garbled output.

**Root Cause**: The FP8 path requires specific CUDA compute capability (8.9+) or explicit kernel selection. RTX 3090s (compute capability 8.6) support FP8 but may need specific vLLM build flags.

**Solution**: Used a separate launch script (`~/launch-vllm-fp8.sh`) with `--dtype auto` to let vLLM select the appropriate kernel path for the FP8 model.

## Issue 6: Python Environment Path Confusion

**Symptom**: `pip install vllm` in `~/vllm-venv` (Python 3.12) works, but the system has multiple Python versions (3.11, 3.12) leading to activation issues.

**Root Cause**: The system's `pip` resolves to Python 3.12 while a separate `~/vllm/` venv exists with Python 3.11, creating confusion about which environment is active.

**Solution**: 
- Always use full path: `source ~/vllm-venv/bin/activate`
- Verify: `which python` and `python --version` before running vLLM
- Use the wrapper script `~/start-vllm.sh` which explicitly sources the correct environment

## Quick Reference — CUDA Diagnostics

```bash
# Check CUDA version
nvcc --version

# Check driver version
nvidia-smi | grep "Driver Version"

# Check GPU topology
nvidia-smi topo -m

# Monitor GPU memory in real-time
watch -n 1 nvidia-smi

# Check what CUDA libraries are loaded by vLLM
cat /proc/$(pgrep -f vllm | head -1)/maps 2>/dev/null | grep cuda
```
