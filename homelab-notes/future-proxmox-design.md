# Phase 3: Proxmox Virtualization & Homelab Design

**Status**: Planning phase — not yet implemented.

This document captures the design for evolving the bare-metal inference workstation into a self-hosted AI homelab platform using Proxmox VE as the hypervisor.

## Motivation

The current setup has limitations:
- Single-purpose physical machine — the GPU compute is idle when not serving inference
- No VM isolation — experiments risk destabilizing the production inference server
- No backup/storage architecture — model files and configuration live on a single disk
- No remote access pattern — LAN-only, no VPN or secure tunnel

Phase 3 addresses all of these by virtualizing the workload.

## Target Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Proxmox VE Host                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Inference    │  │ Agent/Hermes│  │ NAS + Backup │  │
│  │ VM          │  │ VM          │  │ VM          │  │
│  │ 2× RTX 3090 │  │ 4 vCPU      │  │ 8 vCPU      │  │
│  │ 32 GB RAM   │  │ 8 GB RAM    │  │ 16 GB RAM   │  │
│  │ 200 GB disk │  │ 50 GB disk  │  │ 2 TB+ disk  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │                │                │          │
│         └────────────────┼────────────────┘          │
│                          │                           │
│                    ┌─────▼──────┐                     │
│                    │  Proxmox   │                     │
│                    │  ZFS Pool  │                     │
│                    └────────────┘                     │
└──────────────────────────────────────────────────────┘
```

## VM Roles

### VM 1: Inference Server (GPU-passthrough)

| Resource | Allocation |
|----------|-----------|
| **vCPUs** | 8–12 cores (CPU pinning to physical cores) |
| **RAM** | 32–48 GB |
| **GPUs** | 2× RTX 3090 (PCIe passthrough via vfio-pci) |
| **Storage** | 200 GB NVMe (OS + model cache) |
| **OS** | Ubuntu 24.04 LTS |
| **Purpose** | vLLM server, model hosting, API endpoint |

**Notes**:
- GPU passthrough requires IOMMU groups to isolate the GPUs from the host
- Consumer GPUs (GeForce) in VMs require `hidden` hypervisor flag workaround for NVIDIA driver
- Model cache can be on a separate mount for easier snapshot/backup

### VM 2: Agent Runner

| Resource | Allocation |
|----------|-----------|
| **vCPUs** | 4 cores |
| **RAM** | 8 GB |
| **Storage** | 50 GB |
| **OS** | Ubuntu 24.04 LTS |
| **Purpose** | Hermes Agent, FCC CLI, development tools, job scheduling |

**Notes**:
- Lightweight — connects to Inference VM via internal Proxmox bridge
- Hosts cron jobs, agent scripts, monitoring
- Can be snapshotted before experimental changes

### VM 3: NAS & Backup Server

| Resource | Allocation |
|----------|-----------|
| **vCPUs** | 4–8 cores |
| **RAM** | 8–16 GB |
| **Storage** | 2 TB+ (ZFS pool, passed-through HDDs) |
| **OS** | TrueNAS SCALE or Ubuntu + ZFS |
| **Purpose** | Model storage, backup targets, NFS/SMB shares |

**Notes**:
- Serves model files to Inference VM over NFS
- Hosts backup targets for all VMs
- Can run additional lightweight services (Pi-hole, monitoring)

## Proxmox Host Configuration

### ZFS Storage Layout

```
Pool: zpool1
├── VM 1 (inference):     zvol/vm-100-disk-0   — 200 GB
├── VM 2 (agent):         zvol/vm-101-disk-0   — 50 GB
└── VM 3 (NAS):           zvol/vm-102-disk-0   — 2 TB
```

- `ashift=12` (4K sector alignment) for HDD-based arrays
- `compression=lz4` for model storage
- `atime=off` for performance

### GPU Passthrough Requirements

1. **Enable IOMMU** in BIOS and kernel:
   - BIOS: Enable VT-d (Intel)
   - Kernel: `intel_iommu=on iommu=pt`

2. **Isolate GPUs** from host via vfio-pci:
   ```bash
   # Find GPU PCI IDs
   lspci -nn | grep -i nvidia  # e.g., 01:00.0, 01:00.1, 02:00.0, 02:00.1
   
   # Add to kernel cmdline or /etc/modprobe.d/vfio.conf
   options vfio-pci ids=10de:2204,10de:1aef
   ```

3. **NVIDIA GeForce in VM workaround**:
   - Add `hypervisor=off,hidden=on` to VM CPU config
   - Required because GeForce drivers refuse to run in VMs
   - Not needed for Tesla/Quadro cards

### Networking

```
Proxmox Bridge (vmbr0)
├── WAN: host physical NIC → internet
├── VM 1: Inference Server — 10.0.0.10/24
├── VM 2: Agent Runner — 10.0.0.11/24
└── VM 3: NAS — 10.0.0.12/24
```

- Internal traffic between VMs stays on the bridge (no physical NIC bottleneck)
- NAT or reverse proxy for external access to inference API

## Migration Plan

### Phase 3a (Prerequisites)
1. Install Proxmox VE on dedicated boot drive
2. Configure ZFS pool on remaining storage
3. Verify IOMMU groups and GPU passthrough with a single GPU
4. Test VM boot with single GPU

### Phase 3b (Migration)
1. Create Inference VM, passthrough both GPUs
2. Install NVIDIA driver + vLLM in VM
3. Mount model storage from NAS VM or direct-attach
4. Validate inference performance matches bare-metal
5. Create Agent VM, deploy Hermes + FCC
6. Update all endpoint references to new VM IPs

### Phase 3c (Storage)
1. Set up NAS VM with ZFS pool
2. Migrate model files from local disk to NAS
3. Configure NFS export for Inference VM
4. Set up automated backups (Proxmox Backup Server or ZFS snapshots)

## Risks & Considerations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU passthrough degrades performance | 5–15% throughput loss | Accept — virtualization flexibility > peak perf |
| IOMMU group isolates audio function | Can't passthrough GPU without its audio device | Pass through the entire IOMMU group (GPU + audio) |
| GeForce VM driver block | VM won't load NVIDIA driver | Use hidden/off hypervisor flag |
| ZFS overhead on NVMe | ~5% IOPS loss | Acceptable for inference workloads |
| Complexity | Debugging is harder across VM boundaries | Maintain bare-metal revert path until fully validated |

## Current Status

- **Phase 1** (Build hardware) — ✅ Complete
- **Phase 2** (Deploy inference stack) — ✅ Complete
- **Phase 3a** (Proxmox prerequisites) — ❌ Not started
- **Phase 3b** (VM migration) — ❌ Not started
- **Phase 3c** (NAS + backup) — ❌ Not started
