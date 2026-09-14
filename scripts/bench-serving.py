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


def one_request(base_url, model, prompt, max_tokens, timeout, chat_template_kwargs):
    """Stream one chat completion; return (ttft, end_dt, prompt_toks, completion_toks) or None."""
    body_obj = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat_template_kwargs is not None:
        body_obj["chat_template_kwargs"] = chat_template_kwargs
    body = json.dumps(body_obj).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    ttft = None
    prompt_toks = completion_toks = 0
    usage_seen = False
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
                # First token = first content token; leading role/empty/reasoning chunks are skipped.
                if ttft is None and delta.get("content"):
                    ttft = time.perf_counter() - start
            if chunk.get("usage"):
                prompt_toks = chunk["usage"].get("prompt_tokens", 0)
                completion_toks = chunk["usage"].get("completion_tokens", 0)
                usage_seen = True
    end = time.perf_counter() - start
    if ttft is None:
        ttft = end
    if not usage_seen:
        raise RuntimeError("missing server usage metrics in streamed response")
    return ttft, end, prompt_toks, completion_toks


def percentile(values, p):
    """Linearly interpolated percentile, stable for the small sample sets here."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


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
    ap.add_argument(
        "--chat-template-kwargs",
        default='{"enable_thinking": false}',
        help="JSON object passed to the OpenAI-compatible endpoint; null omits it",
    )
    ap.add_argument("--output", help="write the aggregate JSON result to this path")
    ap.add_argument("--samples-output", help="write per-request samples to this path")
    ap.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    ap.add_argument("--slo-tpot-ms", type=float, default=50.0)
    args = ap.parse_args()

    if args.chat_template_kwargs.lower() == "null":
        chat_template_kwargs = None
    else:
        try:
            chat_template_kwargs = json.loads(args.chat_template_kwargs)
        except json.JSONDecodeError as exc:
            ap.error(f"--chat-template-kwargs must be JSON: {exc}")
        if not isinstance(chat_template_kwargs, dict):
            ap.error("--chat-template-kwargs must decode to a JSON object or null")

    prompts = load_sharegpt(args.dataset, args.num_prompts)
    print(f"==> loaded {len(prompts)} ShareGPT prompts; "
          f"concurrency={args.concurrency} cap={args.num_prompts}/{args.max_seconds}s",
          file=sys.stderr)

    work = queue.Queue()
    for p in prompts:
        work.put(p)
    results, errors, error_examples = [], [0], []
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
                                args.max_tokens, args.request_timeout, chat_template_kwargs)
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001 — count and continue under load
                with lock:
                    errors[0] += 1
                    if errors[0] <= 3:
                        error_examples.append({"type": type(e).__name__, "message": str(e)})
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
        out = {
            "completed": 0,
            "errors": errors[0],
            "duration_s": round(duration, 1),
            "concurrency": args.concurrency,
            "hit_time_cap": (time.perf_counter() >= deadline) and not work.empty(),
            "error_examples": error_examples,
            "slo_ttft_ms": args.slo_ttft_ms,
            "slo_tpot_ms": args.slo_tpot_ms,
        }
        if args.output:
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
                f.write("\n")
        print("RESULT error=no_successful_requests "
              f"errors={errors[0]} JSON={json.dumps(out)}")
        sys.exit(1)

    ttfts = [r[0] for r in results]
    prompt_toks = sum(r[2] for r in results)
    completion_toks = sum(r[3] for r in results)
    tpot_all = [(r[1] - r[0]) / max(r[3] - 1, 1) for r in results]
    tpots = [v for v, r in zip(tpot_all, results) if r[3] > 1]

    prefill_toks = prompt_toks / duration
    decode_toks = completion_toks / duration
    hit_time_cap = (time.perf_counter() >= deadline) and not work.empty()
    ttft_ms = [v * 1000 for v in ttfts]
    tpot_ms = [v * 1000 for v in tpots]
    goodput = 100.0 * sum(
        1 for ttft, tpot in zip(ttft_ms, tpot_all)
        if ttft <= args.slo_ttft_ms and tpot * 1000 <= args.slo_tpot_ms
    ) / len(results)

    out = {
        "completed": len(results),
        "errors": errors[0],
        "duration_s": round(duration, 1),
        "concurrency": args.concurrency,
        "hit_time_cap": hit_time_cap,
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "prefill_toks": round(prefill_toks, 2),
        "decode_toks": round(decode_toks, 2),
        "ttft_median_ms": round(statistics.median(ttfts) * 1000, 1),
        "tpot_median_ms": round(statistics.median(tpots) * 1000, 1) if tpots else None,
        "req_throughput": round(len(results) / duration, 3),
        "ttft_p95_ms": round(percentile(ttft_ms, 95), 1),
        "ttft_p99_ms": round(percentile(ttft_ms, 99), 1),
        "tpot_p95_ms": round(percentile(tpot_ms, 95), 1) if tpot_ms else None,
        "goodput_pct": round(goodput, 1),
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_tpot_ms": args.slo_tpot_ms,
        "error_examples": error_examples,
    }
    samples = [
        {
            "ttft_ms": round(r[0] * 1000, 3),
            "end_ms": round(r[1] * 1000, 3),
            "tpot_ms": round((r[1] - r[0]) * 1000 / max(r[3] - 1, 1), 3),
            "prompt_tokens": r[2],
            "completion_tokens": r[3],
        }
        for r in results
    ]
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
    if args.samples_output:
        with open(args.samples_output, "w") as f:
            json.dump({"concurrency": args.concurrency, "samples": samples}, f, indent=2)
            f.write("\n")
    print(f"RESULT prefill_toks={out['prefill_toks']} decode_toks={out['decode_toks']} "
          f"completed={out['completed']} errors={out['errors']} "
          f"duration_s={out['duration_s']} hit_time_cap={out['hit_time_cap']}")
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
