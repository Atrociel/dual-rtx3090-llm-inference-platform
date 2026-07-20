# Hardware Specifications

Detailed specifications of the inference workstation.

## System Overview

| Component | Specification |
|-----------|---------------|
| **Motherboard** | ASRock Z690 Taichi |
| **CPU** | Intel Core i9-12900KS (16 cores / 24 threads, 5.5 GHz max) |
| **RAM** | 62 GB DDR4/DDR5 |
| **Storage** | NVMe SSD (model-loaded) |
| **OS** | Ubuntu 24.04.4 LTS (Noble) |
| **Kernel** | Linux 6.17.0-40-generic |
| **Hostname** | ahmed-Z690-Taichi |

## GPU Configuration

| | GPU 0 | GPU 1 |
|---|---|---|
| **Model** | NVIDIA GeForce RTX 3090 | NVIDIA GeForce RTX 3090 |
| **VRAM** | 24,576 MiB GDDR6X | 24,576 MiB GDDR6X |
| **Bus** | PCIe Gen5 x8 (bifurcation) | PCIe Gen5 x8 (bifurcation) |
| **Topology** | Shared slot bifurcation (single x16 → two x8) | Shared slot bifurcation (single x16 → two x8) |
| **NVLink** | Not present | Not present |
| **Cooling** | Air (open-air test bench) | Air (open-air test bench) |

### GPU Interconnect Topology

```
     PCIe Gen5 x16 slot
            │
      ┌─────┴─────┐
      │ bifurcation│
      │  (x8/x8)  │
      └─────┬─────┘
           / \
          /   \
     GPU0     GPU1
     (x8)     (x8)
```

Both GPUs share a single PCIe Gen5 x16 physical slot, split into two x8 lanes via motherboard bifurcation. There is no direct GPU-to-GPU bridge (NVLink). Tensor parallelism traffic traverses the PCIe bus through the chipset rather than through independent PHB connections. Effective bandwidth per GPU: ~32 GB/s (Gen5 x8 ≈ Gen4 x16).

## Software Versions

| Software | Version |
|----------|---------|
| **Ubuntu** | 24.04.4 LTS |
| **Linux Kernel** | 6.17.0-40-generic |
| **NVIDIA Driver** | 595.71.05 |
| **CUDA Toolkit** | 12.0 |
| **vLLM** | 0.22.0 (V1 engine) |
| **Python** | 3.12 |
| **PyTorch** | (shipped with vLLM) |

## Power & Thermal

| Component | Idle | Load (inference) |
|-----------|------|-------------------|
| **GPU 0** | ~46 W / 39°C | ~250–320 W / 70–80°C (capped at 350W) |
| **GPU 1** | ~39 W / 42°C | ~250–320 W / 70–80°C (capped at 280W — coil whine limit) |
| **CPU** | ~30 W | ~80–150 W |
| **System total** | ~150 W | ~650–800 W |

## Connected Systems

The inference server is accessed by:
- **Hermes Agent** — routes LLM calls to `localhost:8000`
- **Free Claude Code (FCC)** — configured as chat backend at port 8000
- **Direct HTTP clients** — any OpenAI-compatible library

## Bill of Materials

| Item | Quantity | Notes |
|------|----------|-------|
| NVIDIA RTX 3090 | 2 | 24 GB GDDR6X each |
| Intel i9-12900KS | 1 | LGA 1700 |
| Z690 motherboard | 1 | Dual ×16 PCIe slots |
| DDR4/DDR5 RAM | ~62 GB | Capacity verified by OS |
| NVMe SSD | 1+ | Model storage + OS |
| PSU | 1 | 1000W+ recommended for dual 3090s |
| Open-air chassis | 1 | Test bench / mining frame for airflow |
