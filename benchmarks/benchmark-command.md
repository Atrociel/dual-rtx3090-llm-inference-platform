# Benchmark Methodology

## How These Numbers Were Measured

The benchmarks use vLLM's OpenAI-compatible API endpoint with direct HTTP requests. This measures real-world latency as experienced by client applications (Hermes Agent, Free Claude Code), not synthetic throughput.

## Prerequisites

- vLLM server running with the desired config (`deployment/start-vllm.sh` for PP=2, `deployment/start-vllm-tp2.sh` for TP=2)
- Server responds on `http://localhost:8000/v1/completions`

## Quick Benchmark

The comprehensive benchmark suite:

```bash
source ~/vllm/bin/activate
python3 benchmarks/comprehensive-benchmark.py
```

This measures:
- Generation throughput (10 runs, 100-tok output)
- Prompt processing speed (266 → 2,058 token inputs)
- Concurrent throughput (2 simultaneous requests)
- GPU utilization and power
- Sustained generation (200-tok output)
- Time to First Token

## Manual Benchmark Script

Save as `benchmark.py` for isolated tests:

```python
import time, json, urllib.request, threading

MODEL = '/home/ahmed/models/qwen3.6-35b-a3b-awq'
BASE = 'http://localhost:8000'

def chat(prompt, max_tokens=100):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }).encode()
    req = urllib.request.Request(f'{BASE}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    t1 = time.perf_counter()
    data = json.loads(resp.read())
    usage = data.get('usage', {})
    out = usage.get('completion_tokens', 0)
    return out / (t1 - t0) if t1 > t0 else 0

# Generation benchmark
results = []
for i in range(5):
    t = chat("What is the capital of France?", 100)
    results.append(t)
    print(f"Run {i+1}: {t:.1f} tok/s")
print(f"Avg: {sum(results)/len(results):.1f} tok/s")
```

## Comparison Benchmarking

To compare TP=2 vs PP=2:

```bash
# 1. Start TP=2 server (separate terminal)
./deployment/start-vllm-tp2.sh

# 2. Run benchmarks
python3 benchmarks/comprehensive-benchmark.py > results-tp2.txt

# 3. Stop TP=2 server, start PP=2 server
pkill -f "vllm serve"
./deployment/start-vllm.sh

# 4. Run same benchmarks
python3 benchmarks/comprehensive-benchmark.py > results-pp2.txt

# 5. Compare
diff results-tp2.txt results-pp2.txt
```
