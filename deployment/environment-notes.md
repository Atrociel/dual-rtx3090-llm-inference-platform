# Environment & Setup Notes

## Virtual Environment

The inference server runs from a dedicated Python virtual environment:

```
~/vllm/      ← Primary vLLM environment (Python 3.11)
```

### Setup

```bash
python3 -m venv ~/vllm-venv
source ~/vllm-venv/bin/activate
pip install --upgrade pip
pip install vllm
```

## Model Storage

Models are stored in `~/models/`:

```
~/models/
├── qwen3.6-35b-a3b-awq/     ← Symlink to HuggingFace cache
│   (→ ~/.cache/huggingface/hub/...)
├── qwen-coder/                ← Qwen coding model
├── skyreels/                  ← Video generation model
└── triposr/                   ← 3D reconstruction model
```

The primary inference model is **Qwen3.6-35B-A3B-AWQ**:
- **Source**: HuggingFace — `np-deploys/Qwen3.6-35B-A3B-AWQ-4bit`
- **Format**: AWQ (Activation-Aware Weight Quantization), 4-bit
- **Architecture**: Qwen3.5 MoeForConditionalGeneration
- **Parameters**: ~35B total, ~20B active (MoE)
- **File format**: safetensors (5 shards: model-00001 to model-00005)

## HuggingFace Setup

```bash
huggingface-cli login  # Required for gated models
```

## CUDA Configuration

- **CUDA 12.0** installed system-wide via `nvcc`
- **NVIDIA Driver 595.71.05** loaded
- **CUDA_HOME**: Set to vLLM venv's CUDA stubs when needed:

```bash
export CUDA_HOME=/home/ahmed/vllm/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

⚠️ Inconsistent CUDA paths have been a source of bugs — see `troubleshooting/cuda-issues.md`.

> **Note**: The venv path was corrected from `~/vllm-venv/` (which did not exist) to `~/vllm/` after verification. Always check the actual venv location with `pip show vllm | grep Location` before assuming the path.

## Launch Methods

### Primary: CLI wrapper (recommended)

```bash
~/start-vllm.sh
```

### Alternative: Python API server

```bash
cd ~
source ~/vllm-venv/bin/activate
python3 ~/vllm-server.py
```

The Python script (`~/vllm-server.py`) uses `vllm.entrypoints.openai.api_server` and logs to `~/vllm-server.log`.

## Client Integration

### Free Claude Code (FCC)

Located at `~/free-claude-code/`, configured in `.env`:

```
MODEL="vllm//home/ahmed/models/qwen3.6-35b-a3b-awq"
ANTHROPIC_AUTH_TOKEN=***
ENABLE_NETWORK_PROBE_MOCK=true
```

### Direct API Access

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/ahmed/models/qwen3.6-35b-a3b-awq",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

## Logging

- **Server log**: `~/vllm-server.log`
- **PID file**: `~/vllm.pid` or `~/vllm-fix.pid`
- **Ollama log** (if running): `~/ollama.log`

## Experimental Configurations

These have been tested but are not the primary running config:

| Variant | Config File | Notes |
|---------|------------|-------|
| FP8 | `~/launch-vllm-fp8.sh` | Port 8009, lower memory, faster prefill |
| High context | `~/start-vllm.sh` (tuned) | 32K+ context with chunked prefill |
| MTP speculative | `~/run-vllm-moe.sh` | Multi-Token Prediction (model supports MTP, config was unstable) |
| Docker | `deployment/docker-compose.yml` | Containerized deployment (not currently used) |
