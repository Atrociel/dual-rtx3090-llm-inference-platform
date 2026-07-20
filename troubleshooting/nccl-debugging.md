# NCCL & Multi-GPU Debugging

A record of NCCL (NVIDIA Collective Communications Library) issues on the dual RTX 3090 setup and their resolutions.

## Topology Background

The GPUs are connected via **PCIe chipset bifurcation** (no NVLink):

GPU0 ↔ PCIe Gen5 x8 ↔ Chipset Switch ↔ PCIe Gen5 x8 ↔ GPU1

This means:
- No direct GPU-to-GPU path (no NVLink, no PIX/PXB peer)
- All cross-GPU communication traverses the PCIe bus through the CPU
- Higher latency than NVLink (~5× penalty for all-reduce operations)
- NCCL's topology detection may incorrectly select suboptimal communication paths

## Issue 1: Custom All-Reduce Deadlock

**Symptom**: vLLM's V1 engine hangs during initialization or the first inference request. Process appears running but never produces output. No error message — just a freeze.

**Root Cause**: vLLM's custom all-reduce kernel (`--disable-custom-all-reduce` is off by default) assumes direct P2P connectivity between GPUs. On bifurcated PCIe topology (Gen5 x8, no NVLink), the P2P access tests may pass, but the custom kernel deadlocks under load because the shared x8 link cannot sustain the expected all-reduce throughput pattern.

**Solution**: Disable the custom all-reduce kernel:

```bash
vllm serve ... --disable-custom-all-reduce
```

This forces vLLM to use NCCL's native all-reduce implementation, which correctly handles bifurcated topologies. The performance penalty is approximately 5% for MoE inference workloads.

## Issue 2: NCCL P2P Disabled Fallback

**Symptom**: NCCL warnings about P2P being unavailable or falling back to CPU-mediated communication:
```
WARNING: NCCL P2P is disabled — using CPU socket as intermediary
```

**Root Cause**: NVIDIA's P2P (Peer-to-Peer) access requires:
1. NVLink connection, or
2. PCIe P2P support (available on some workstation chipsets but unreliable on consumer Z690)

The RTX 3090s on the Z690 chipset may pass the CUDA P2P access check (`cudaDeviceCanAccessPeer`) but fail under actual NCCL communication load.

**Solution**: When experiencing NCCL instability, explicitly disable P2P:

```bash
export NCCL_P2P_DISABLE=1
export VLLM_SKIP_P2P_CHECK=1
```

This forces NCCL to route all inter-GPU communication through CPU memory. While slower, it is **reliable** for inference workloads where per-token latency is not dominated by communication.

## Issue 3: NCCL IB (InfiniBand) Detection

**Symptom**: NCCL logs showing InfiniBand detection:
```
NCCL INFO NET/IB : IB device found: mlx5_0
```
...even though no InfiniBand hardware exists.

**Root Cause**: NCCL probes for InfiniBand devices on the system and may detect RDMA-capable NICs as IB devices. On this system, TCP/IP over RDMA (iWARP) or similar can trigger false detection.

**Solution**: Disable IB transport in NCCL:

```bash
export NCCL_IB_DISABLE=1
```

## Issue 4: PCIe Bandwidth Bottleneck

**Symptom**: Token generation is slower than expected. Profiling shows GPU utilization below 80% despite no obvious bottleneck.

**Root Cause**: The MoE model's expert routing requires frequent all-reduce operations. Each token requires routing to 2 of 128 experts per MoE layer, and the results need to be combined across GPUs. On PCIe Gen5 x8 bifurcation, this communication adds ~3–5 ms per token.

**Solution**: 
- This is a hardware limitation. Mitigations include:
  - Using tensor parallelism (TP=2) rather than expert parallelism (EP), which reduces communication frequency
  - Increasing batch size to amortize communication overhead across more tokens
  - Accepting the PCIe bandwidth as a fixed cost of consumer GPU inference

## NCCL Environment Reference

A consolidated set of NCCL flags for bifurcated dual GPU setups:

```bash
# Reliable config for bifurcated PCIe topology (Gen5 x8)
export NCCL_P2P_DISABLE=1           # Disable GPU P2P access
export NCCL_IB_DISABLE=1            # Disable InfiniBand detection
export NCCL_SOCKET_IFNAME=lo        # Force loopback socket communication
export VLLM_SKIP_P2P_CHECK=1        # Skip vLLM's P2P validation
export VLLM_USE_TRITON_MOE=1        # Use Triton MoE kernels

# Diagnosis commands
nvidia-smi topo -m                   # Check GPU topology
nvidia-smi nvlink -s                # Check NVLink status (none on 3090s)
nccl-tests/build/all_reduce_perf -b 8 -f 2 -g 2  # Benchmark NCCL perf
```

## Diagnosis Script

```bash
#!/bin/bash
# Diagnose multi-GPU communication setup
echo "=== GPU Topology ==="
nvidia-smi topo -m

echo ""
echo "=== NVLink Status ==="
nvidia-smi nvlink -s 2>/dev/null || echo "No NVLink available"

echo ""
echo "=== P2P Access Test ==="
python3 -c "
import torch
for i in range(2):
    for j in range(2):
        if i != j:
            can_access = torch.cuda.can_device_access_peer(i, j)
            print(f'GPU {i} → GPU {j} P2P: {can_access}')
"

echo ""
echo "=== NCCL Version ==="
python3 -c "
import torch
print(f'PyTorch NCCL version: {torch.cuda.nccl.version()}')
"
```
