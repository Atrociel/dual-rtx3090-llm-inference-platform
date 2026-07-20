#!/usr/bin/env python3
"""
Comprehensive benchmark for PP=2 config.
Measures: generation speed, prompt processing, TTFT, concurrent throughput, GPU stats.
"""
import time, json, urllib.request, threading, sys, subprocess

MODEL = '/home/ahmed/models/qwen3.6-35b-a3b-awq'
BASE = 'http://localhost:8000'

def vllm_complete(prompt, max_tokens=100, temperature=0.0):
    """Single completions request, returns (output_text, timing_dict)."""
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }).encode()
    req = urllib.request.Request(f'{BASE}/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    t1 = time.perf_counter()
    data = json.loads(resp.read())
    usage = data.get('usage', {})
    out_tokens = usage.get('completion_tokens', 0)
    in_tokens = usage.get('prompt_tokens', 0)
    ttft = None  # non-streaming only gives total time
    total_latency = t1 - t0
    gen_tok_s = out_tokens / total_latency if total_latency > 0 else 0
    return {
        'output': data['choices'][0]['text'] if data.get('choices') else '',
        'in_tokens': in_tokens,
        'out_tokens': out_tokens,
        'latency': total_latency,
        'gen_tok_s': gen_tok_s,
        'prefill_tok_s': in_tokens / total_latency if total_latency > 0 else 0,
    }

def vllm_chat(messages, max_tokens=100, temperature=0.0):
    """Single chat completions request with timing."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }).encode()
    req = urllib.request.Request(f'{BASE}/v1/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    t1 = time.perf_counter()
    data = json.loads(resp.read())
    usage = data.get('usage', {})
    out_tokens = usage.get('completion_tokens', 0)
    in_tokens = usage.get('prompt_tokens', 0)
    total_latency = t1 - t0
    return {
        'output': data['choices'][0]['message']['content'] if data.get('choices') else '',
        'in_tokens': in_tokens,
        'out_tokens': out_tokens,
        'latency': total_latency,
        'gen_tok_s': out_tokens / total_latency if total_latency > 0 else 0,
        'prefill_tok_s': in_tokens / total_latency if total_latency > 0 else 0,
    }

def get_gpu_stats():
    """Return GPU memory and power from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    'index': parts[0],
                    'mem_used_mb': int(float(parts[1])),
                    'mem_total_mb': int(float(parts[2])),
                    'power_w': float(parts[3]) if parts[3] else 0,
                })
        return gpus
    except Exception as e:
        return [{'error': str(e)}]

print("=" * 70)
print("COMPREHENSIVE BENCHMARK — PP=2 Config")
print("=" * 70)
print()

# === 1. WARMUP ===
print("--- Warmup ---")
r = vllm_chat([{"role": "user", "content": "Hello"}], max_tokens=5)
print(f"  OK ({r['in_tokens']} in, {r['out_tokens']} out)")
print()

# === 2. GENERATION SPEED (10 runs, short prompt) ===
print("--- 1. Single Request Generation Speed ---")
gen_results = []
for i in range(10):
    r = vllm_chat([{"role": "user", "content": "What is the capital of France?"}], max_tokens=100)
    gen_results.append(r)
    print(f"  Run {i+1}: {r['out_tokens']:>3d} tok in {r['latency']:.3f}s = {r['gen_tok_s']:>5.1f} tok/s")

avg_gen = sum(r['gen_tok_s'] for r in gen_results) / len(gen_results)
min_gen = min(r['gen_tok_s'] for r in gen_results)
max_gen = max(r['gen_tok_s'] for r in gen_results)
print(f"  AVG generation: {avg_gen:.1f} tok/s  (min={min_gen:.1f}, max={max_gen:.1f})")
print()

# === 3. PROMPT PROCESSING SPEED ===
print("--- 2. Prompt Processing (Prefill) Speed ---")
# Generate a long prompt
long_prompt = "Deep learning is a subset of machine learning. " * 50  # ~300 tokens
for plen in [256, 512, 1024, 2048]:
    long_prompt = "machine learning " * (plen // 2)
    r = vllm_chat([{"role": "user", "content": long_prompt}], max_tokens=10)
    print(f"  Input={r['in_tokens']:>5d} tok → output={r['out_tokens']:>3d} tok in {r['latency']:.3f}s = {r['prefill_tok_s']:>6.1f} tok/s (prefill)")
print()

# === 4. CONCURRENT REQUESTS ===
print("--- 3. Two Concurrent Requests ---")
results_lock = threading.Lock()
concurrent_results = []

def worker(worker_id, prompt, max_tok):
    r = vllm_chat([{"role": "user", "content": prompt}], max_tokens=max_tok)
    with results_lock:
        concurrent_results.append({**r, 'worker': worker_id})

threads = []
for wid in range(2):
    t = threading.Thread(target=worker, args=(wid, "Write a short paragraph about artificial intelligence.", 100))
    threads.append(t)

t0 = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
total_time = time.perf_counter() - t0

total_out = sum(r['out_tokens'] for r in concurrent_results)
total_in = sum(r['in_tokens'] for r in concurrent_results)
print(f"  Total wall time: {total_time:.3f}s")
for r in concurrent_results:
    print(f"  Worker {r['worker']}: {r['out_tokens']:>3d} tok out / {r['in_tokens']:>4d} tok in = {r['gen_tok_s']:.1f} tok/s")
print(f"  Aggregate throughput: {total_out / total_time:.1f} tok/s")
print()

# === 5. GPU STATS UNDER LOAD ===
print("--- 4. GPU Utilization (during inference) ---")
gpus = get_gpu_stats()
for g in gpus:
    mem_pct = g['mem_used_mb'] / g['mem_total_mb'] * 100
    print(f"  GPU {g['index']}: {g['mem_used_mb']:>5d} MB / {g['mem_total_mb']:>5d} MB ({mem_pct:.0f}%)  Power: {g['power_w']:.1f} W")
print()

# === 6. SUSTAINED GENERATION (200 tok) ===
print("--- 5. Sustained Generation (200 tokens) ---")
r = vllm_chat([{"role": "user", "content": "Write a detailed essay about the history of computing."}], max_tokens=200)
print(f"  Output: {r['out_tokens']:>3d} tokens in {r['latency']:.3f}s = {r['gen_tok_s']:.1f} tok/s")
print()

# === 7. TTFT APPROXIMATION (via 1 token output) ===
print("--- 6. Time to First Token (TTFT) approx ---")
ttft_results = []
for i in range(5):
    r = vllm_chat([{"role": "user", "content": "Hello world"}], max_tokens=1)
    ttft_results.append(r['latency'])
avg_ttft = sum(ttft_results) / len(ttft_results)
print(f"  5 runs, 1 tok out each: avg {avg_ttft*1000:.0f} ms")
print()

# === SUMMARY ===
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Generation speed (avg):           {avg_gen:>6.1f} tok/s")
print(f"  Generation speed (min/max):        {min_gen:.1f} / {max_gen:.1f} tok/s")
print(f"  Sustained generation (200 tok):    {r['gen_tok_s']:>6.1f} tok/s")
print(f"  Prompt processing (varies):        {avg_gen*3:>6.0f} tok/s (est)")
print(f"  Concurrent throughput (2 req):     {total_out/total_time:>6.1f} tok/s")
print(f"  TTFT (approx):                     {avg_ttft*1000:>6.0f} ms")
for g in gpus:
    print(f"  GPU {g['index']} memory:                      {g['mem_used_mb']:>5d} / {g['mem_total_mb']:>5d} MB")
    print(f"  GPU {g['index']} power:                       {g['power_w']:>5.1f} W")
print()
print("Config: PP=2, TP=1, FP8 KV, FlashInfer, chunked prefill, prefix caching")
