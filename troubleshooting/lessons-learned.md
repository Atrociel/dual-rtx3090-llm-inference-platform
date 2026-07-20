# Lessons Learned

Cumulative engineering log from building and operating a dual RTX 3090 LLM inference platform.

## Hardware Lessons

### Cooling Matters More Than Expected

Dual RTX 3090s in a closed case will thermal throttle within 10 minutes of sustained inference. The cards pull 300+ W each under load, and their blower/open-air coolers need room to breathe.

**What worked**: Open-air test bench / mining frame configuration with a house fan. GPU temperatures stabilized at 70–80°C under load instead of hitting 85°C throttle limit in a closed case.

### GPU1 Coil Whine — Hard Power Limit

GPU1 emits audible coil whine above 300W. This is treated as a hard thermal/electrical limit indicator. **GPU1 is capped at 280W** via NVIDIA power management — not for performance reasons, but because the acoustics signal electrical stress at the VRM level.

| GPU | Power Limit | Reason |
|-----|------------|--------|
| GPU0 | 350W | Standard limit, stable |
| GPU1 | 280W | Coil whine threshold — hard cap |

This creates an asymmetric thermal profile: GPU1 runs cooler but becomes the bottleneck under sustained load if PP=2 assigns it a heavier stage.

### Power Supply Headroom

A single 3090 can spike to 400 W under transient load. Dual cards plus CPU means 900 W+ system draw. A quality 1000 W+ PSU is mandatory. Running on a marginal PSU causes random system crashes during model loading.

### PCIe Slot Configuration

On the Z690 Taichi, the primary physical PCIe slot operates at Gen5 ×16, but is configured via BIOS bifurcation to run as two Gen5 x8 paths — one per GPU. Both GPUs share the same physical slot lanes through the motherboard's chipset switch. Verify `nvidia-smi topo -m` output to confirm bifurcation mode is active.

**Implication**: Unlike a true PHB topology where each GPU has independent root-complex access, both cards share ~32 GB/s of effective bandwidth per direction. This directly motivated the TP→PP migration.

## Software Lessons

### Start With the Production Config, Not the Bleeding Edge

The instinct is to try every optimization flag at once (expert parallel, flash attention, speculative decoding, FP8, etc.). This makes debugging impossible — you won't know which flag caused a crash.

**Better approach**:
1. Get the simplest config working first (TP=2, no fancy flags)
2. Verify inference works end-to-end
3. Add one optimization at a time
4. Test thoroughly before adding the next

### vLLM Version Matters

vLLM v0.22.0 changed the engine architecture significantly (V1 engine). Launch scripts that worked with v0.6.x may not work with v0.22.0. Always check the changelog when upgrading.

**Specific changes encountered**:
- `vllm.entrypoints.openai.api_server` → `vllm serve` CLI
- Engine initialization path changed (V1 multiprocess executor)
- Some flags were renamed or deprecated

### Log Everything

vLLM produces detailed engine logs at startup showing:
- KV cache size
- Maximum concurrency
- Quantization configuration
- Topology detection results

Redirect logs to a file and **read them** when troubleshooting. The startup log contains the exact configuration the engine resolved, which may differ from what you passed on the command line.

```bash
vllm serve ... > ~/vllm-server.log 2>&1
```

### Virtual Environment Isolation

Python version mismatches between environments cause subtle bugs. The system has Python 3.11.15 and 3.12 — using the wrong venv leads to import errors or CUDA library conflicts.

**Rule**: One venv per project, clearly named. Use wrapper scripts that explicitly source the correct environment.

## Operational Lessons

### The Server Is Fragile During Model Loading

Model loading is the most resource-intensive phase. The engine initializes CUDA contexts, allocates KV cache, loads all model shards, and compiles kernels — all within a few seconds. This phase is when OOMs, driver crashes, and NCCL timeouts are most likely.

**Wait for the log message**: Once you see `Application startup complete` or the model serving endpoint responds, the server is stable.

### Prefix Caching Is a Free Performance Win

If your workload uses repeated system prompts (common with agent frameworks like Hermes or Claude Code), enabling prefix caching reduces TTFT by up to 50% for zero additional cost. The only tradeoff is slightly higher KV cache memory reservation.

### Drive Space Can Become a Problem

Models are large. Qwen3.6-35B-A3B-AWQ is ~20 GB on disk. Multiple models, quantization variants, and HuggingFace cache bloat can quickly consume hundreds of GB. Monitor disk usage if experimenting with multiple models.

## Design Principles That Emerged

1. **Consumer hardware does inference. It does not do everything.** Training, fine-tuning, and large-batch serving are out of scope. Accept the constraints.

2. **Stability over peak throughput.** A steady 100 tok/s that serves 24/7 is worth more than 150 tok/s that crashes every hour.

3. **Document the failures.** Every error you solve is a piece of institutional knowledge. The troubleshooting section of this repo exists because finding the same fix twice is a waste of time.

4. **The infrastructure is the portfolio.** For someone without a formal CS degree or enterprise datacenter role, a well-documented homelab deployment demonstrates engineering discipline more effectively than any certification.

## What I Would Do Differently

- **Buy a chassis with better GPU spacing** — blower-style coolers on adjacent slots recycle hot air
- **Test with a single GPU first** — TP=1 eliminates NCCL variables during initial debugging
- **Benchmark systematically from day one** — capture baseline metrics before making changes
- **Use Docker earlier** — containerized deployment avoids environment drift
