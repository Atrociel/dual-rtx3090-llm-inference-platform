# Screenshots

Directory for visual documentation of the running system. Planned screenshots:

- [ ] `vllm-running.png` — vLLM server output showing startup logs, KV cache stats, model config
- [ ] `nvidia-smi.png` — `nvidia-smi` output showing both RTX 3090s during active inference (GPU utilization, memory, power draw)
- [ ] `benchmark-output.png` — Terminal session showing benchmark API calls with timing results
- [ ] `system-monitoring.png` — `htop` or `nvtop` showing system resource usage during inference
- [ ] `model-config.png` — Model config from the server API (`/v1/models`) showing available model and metadata

## How to Capture

```bash
# 1. vLLM running
# Capture the first ~40 lines of vLLM startup log showing KV cache and model config

# 2. nvidia-smi during inference
# Run this while the benchmark script is executing
nvidia-smi

# 3. Benchmark output
# Capture the Python benchmark script's output

# 4. System monitoring
htop  # or nvtop if installed

# 5. API model list
curl http://localhost:8000/v1/models | python3 -m json.tool
```

*Screenshots will be added after the initial portfolio push. The system is running and can produce them on demand.*
