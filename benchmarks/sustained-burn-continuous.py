#!/usr/bin/env python3
"""Continuous sustained load test — keeps both GPUs busy 100% of the time."""
import json, time, threading, urllib.request, sys
from datetime import datetime

URL = "http://localhost:8000/v1/completions"
MODEL = "/home/ahmed/models/qwen3.6-35b-a3b-awq"
DURATION = 600  # 10 minutes
CONCURRENCY = 4
OUTPUT = "/home/ahmed/dual-rtx3090-llm-inference-platform/benchmarks/sustained-burn-results-uncapped.json"

PROMPT = "Write a detailed essay about the evolution of transformer architectures in deep learning, covering attention mechanisms, multi-head attention, positional encodings, and the transition from encoder-decoder to decoder-only models."

results = []
errors = 0
lock = threading.Lock()
start_time = time.time()
stop_event = threading.Event()

def send_request(worker_id):
    global errors
    while not stop_event.is_set():
        t0 = time.time()
        try:
            body = json.dumps({
                "model": MODEL,
                "prompt": PROMPT,
                "max_tokens": 200,
                "temperature": 0
            }).encode()
            req = urllib.request.Request(URL, data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            t1 = time.time()
            text = data["choices"][0]["text"]
            tok_count = len(text.split())
            latency_ms = (t1 - t0) * 1000
            tok_s = tok_count / (t1 - t0) if (t1 - t0) > 0 else 0

            with lock:
                results.append({
                    "worker": worker_id,
                    "elapsed": round(time.time() - start_time, 2),
                    "tok_s": round(tok_s, 2),
                    "latency_ms": round(latency_ms, 2),
                    "tok_count": tok_count,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            with lock:
                errors += 1
                if errors <= 5:
                    print(f"  [W{worker_id}] Error: {e}")

threads = []
for i in range(CONCURRENCY):
    t = threading.Thread(target=send_request, args=(i,), daemon=True)
    t.start()
    threads.append(t)

print(f"=== Continuous Sustained Burn Test (After Reboot) ===")
print(f"Concurrency: {CONCURRENCY} | Duration: {DURATION}s | Output: {OUTPUT}")
print()

try:
    while time.time() - start_time < DURATION:
        elapsed = int(time.time() - start_time)
        with lock:
            recent = [r for r in results if r["elapsed"] > elapsed - 10]
            avg_tok = sum(r["tok_s"] for r in recent) / len(recent) if recent else 0
            total = len(results)
        import subprocess
        nvidia = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpu_line = nvidia.stdout.strip().replace("\n", " | ")
        print(f"[{elapsed}s] reqs={total} err={errors} avg_tok/s={avg_tok:.1f} | GPU: {gpu_line}")
        time.sleep(10)
except KeyboardInterrupt:
    pass
finally:
    stop_event.set()
    for t in threads:
        t.join(timeout=2)

total_duration = time.time() - start_time
import subprocess
nvidia = subprocess.run(
    ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True, timeout=5
)

output_data = {
    "test_params": {
        "duration_s": DURATION, "concurrency": CONCURRENCY,
        "model": MODEL, "state": "after_reboot_uncapped"
    },
    "summary": {
        "total_requests": len(results), "errors": errors,
        "total_duration_s": round(total_duration, 1),
        "avg_tok_s": round(sum(r["tok_s"] for r in results) / len(results), 2) if results else 0,
        "min_tok_s": round(min(r["tok_s"] for r in results), 2) if results else 0,
        "max_tok_s": round(max(r["tok_s"] for r in results), 2) if results else 0,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1) if results else 0,
        "final_gpu": nvidia.stdout.strip()
    },
    "results": results
}

with open(OUTPUT, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\n=== Complete: {len(results)} requests over {total_duration:.0f}s ===")
print(f"Results saved to {OUTPUT}")
print(f"Avg tok/s: {output_data['summary']['avg_tok_s']}")
print(f"Avg latency: {output_data['summary']['avg_latency_ms']}ms")
print(f"Final GPU: {output_data['summary']['final_gpu']}")
