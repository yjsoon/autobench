#!/usr/bin/env python3
"""OpenAI-compatible serving-benchmark load generator (zero third-party deps).

Drives any engine that exposes an OpenAI /v1/chat/completions endpoint
(llama-server, vLLM, SGLang, TRT-LLM) against the real ShareGPT workload and
reports system throughput at a fixed concurrency.

Metrics (all server-reported token counts, via stream_options.include_usage):
  prefill_toks = total prompt tokens / benchmark wall-clock   (input throughput)
  decode_toks  = total completion tokens / benchmark wall-clock (output throughput)
  plus median TTFT and median TPOT for the Notes.

Runtime cap (policy): stops at --num-prompts entries OR --max-seconds wall-clock,
whichever is shorter. Prints a one-line `RESULT ...` summary + a JSON blob.

Usage:
  bench-serving.py --base-url http://localhost:8080 --model M \
      --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
"""
import argparse, json, queue, statistics, sys, threading, time, urllib.request


def load_sharegpt(path, n):
    """First human turn of the first n conversations that have one."""
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human" and turn.get("value", "").strip():
                prompts.append(turn["value"])
                break
        if len(prompts) >= n:
            break
    return prompts


def one_request(base_url, model, prompt, max_tokens, timeout):
    """Stream one chat completion; return (ttft, end_dt, prompt_toks, completion_toks) or None."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    ttft = None
    prompt_toks = completion_toks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("error"):
                # Server streamed an error mid-response (e.g. llama.cpp's harmony
                # parser rejecting gpt-oss). Count as a failed request, not a 0-token
                # success — otherwise throughput silently reads as 0.
                raise RuntimeError(f"server error chunk: {str(chunk['error'])[:120]}")
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {}) or {}
                # First token = first content OR reasoning_content (reasoning models
                # like SmolLM3/Nemotron stream reasoning_content before content).
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft = time.perf_counter() - start
            if chunk.get("usage"):
                prompt_toks = chunk["usage"].get("prompt_tokens", 0)
                completion_toks = chunk["usage"].get("completion_tokens", 0)
    end = time.perf_counter() - start
    if ttft is None:
        ttft = end
    return ttft, end, prompt_toks, completion_toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--max-seconds", type=float, default=900)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--request-timeout", type=float, default=600)
    args = ap.parse_args()

    prompts = load_sharegpt(args.dataset, args.num_prompts)
    print(f"==> loaded {len(prompts)} ShareGPT prompts; "
          f"concurrency={args.concurrency} cap={args.num_prompts}/{args.max_seconds}s",
          file=sys.stderr)

    work = queue.Queue()
    for p in prompts:
        work.put(p)
    results, errors = [], [0]
    lock = threading.Lock()
    deadline = time.perf_counter() + args.max_seconds
    bench_start = time.perf_counter()

    def worker():
        while time.perf_counter() < deadline:
            try:
                prompt = work.get_nowait()
            except queue.Empty:
                return
            try:
                r = one_request(args.base_url, args.model, prompt,
                                args.max_tokens, args.request_timeout)
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001 — count and continue under load
                with lock:
                    errors[0] += 1
                    if errors[0] <= 3:
                        print(f"    [err] {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    duration = time.perf_counter() - bench_start

    if not results:
        print("RESULT error=no_successful_requests "
              f"errors={errors[0]}")
        sys.exit(1)

    ttfts = [r[0] for r in results]
    prompt_toks = sum(r[2] for r in results)
    completion_toks = sum(r[3] for r in results)
    tpots = [(r[1] - r[0]) / max(r[3] - 1, 1) for r in results if r[3] > 1]

    prefill_toks = prompt_toks / duration
    decode_toks = completion_toks / duration
    out = {
        "completed": len(results),
        "errors": errors[0],
        "duration_s": round(duration, 1),
        "concurrency": args.concurrency,
        "hit_time_cap": (time.perf_counter() >= deadline) and not work.empty(),
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "prefill_toks": round(prefill_toks, 2),
        "decode_toks": round(decode_toks, 2),
        "ttft_median_ms": round(statistics.median(ttfts) * 1000, 1),
        "tpot_median_ms": round(statistics.median(tpots) * 1000, 1) if tpots else None,
        "req_throughput": round(len(results) / duration, 3),
    }
    print(f"RESULT prefill_toks={out['prefill_toks']} decode_toks={out['decode_toks']} "
          f"completed={out['completed']} errors={out['errors']} "
          f"duration_s={out['duration_s']} hit_time_cap={out['hit_time_cap']}")
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
