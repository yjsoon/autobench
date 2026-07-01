---
title: Qwen3.6-35B-A3B · vLLM-ultimate (AEON) · NVFP4 + DFlash · conc 4
model: nvidia/Qwen3.6-35B-A3B-NVFP4
company: Alibaba
family: Qwen
params: 35B / 3B (MoE, hybrid GDN+full-attn) + DFlash external drafter
engine: vLLM (aeon-vllm-ultimate custom container, v0.23.0+aeon.sm121a.dflash)
speculative: DFlash (z-lab/Qwen3.6-35B-A3B-DFlash @31977fbe small-page rev, num_speculative_tokens 11)
quant: NVFP4 (modelopt_mixed — W4A16_NVFP4 experts + FP8 GDN gates)
quant_rationale: conc-4 fine-grained point of the DFlash money-chart line (EXPERIMENTS.md #13/#16). With conc-2 it traces the shallow-loss plateau between the 1→2 crossover and the conc-16/32 collapse. Same one-boot protocol as the published c1/8/32 series (official checkpoint, small-page drafter, max-num-seqs 64, ctx 40960). SAFETY — untrusted third-party image; NO credentials, weights + drafter READ-ONLY, port loopback-only.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 40960
modalities: [text, image, video]
mm_served: false
concurrency: 4
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-4]
status: done
prefill_toks: 193.92
decode_toks: 213.81
mem_gb: 110.56
mem_source: system MemAvailable delta (10s sampling) over the one-boot conc-2/4/16 sweep — vLLM static KV (util 0.85) + DFlash drafter
spec_acceptance: mean acceptance length ~3.8–4.1 of 11 drafted · avg draft acceptance ~25–28% · per-position ~0.76–0.78 (pos0) decaying to ~0.06–0.10 (pos10)
measured_on: 2026-07-01
completed_at: 2026-07-01 13:33 +0800
engine_image: ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix@sha256:be9e05a11da6e72607ab6f3e960993b253b673af0727005122a3266129a518e3
run_command: |
  # UNTRUSTED image — NO creds; weights + drafter READ-ONLY; loopback port. ONE boot (max-num-seqs 64, ctx 40960,
  # small-page drafter @31977fbe) sweeping client conc 2/4/16 — matches the published c1/8/32 protocol.
  # (Same docker run as the -c2 page; only the client --concurrency differs.)
  python3 scripts/bench-serving.py --base-url http://127.0.0.1:8000 --model official \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json --concurrency 4 --num-prompts 1000 --max-seconds 600 --max-tokens 256
  # 504/1000 prompts (hit 600 s cap), 0 errors.
---

**conc-4 point of the Qwen3.6-35B-A3B NVFP4 + DFlash line — mid-plateau, still behind MTP.** Official
`nvidia/Qwen3.6-35B-A3B-NVFP4` on the AEON image, DFlash n=11 via the small-page drafter, one-boot sweep.
**0 errors.**

- **Result (conc 4):** prefill 193.92 / decode **213.81** tok/s aggregate; 504/1000 prompts (hit the 600 s
  cap), **0 errors**. **vs MTP conc-4 (232.43): −8.0%.**
- **The shallow-loss plateau.** After the 1→2 sign flip, DFlash sits ~−7-8% behind MTP across conc 2/4/8
  (−8.4% / −8.0% / −6.6%) — a roughly flat modest loss — before the batch-fill collapse at conc-16 (−20.6%)
  and conc-32 (−25.9%). So the DFlash penalty is not monotone-steep; it's flat-then-cliff.
- **Why — wasted draft compute.** DFlash drafts **11** at **~25–28% acceptance / accept-len ~3.8–4.1** (~7
  wasted forward passes/step); MTP drafts 3 at ~67% / accept-len 3.0. On this compute-bound MoE the extra
  draft work already dominates by conc-2 and worsens as streams divide the GPU. Acceptance flat vs conc
  (workload-driven), as in the published series.
- One server lifetime for c2/c4/c16 → mem is the single 110.56 GB reservation. TPOT 0.0 = `qwen3`
  reasoning-parser client artifact.
- Series: [`c1/8/32` (main)](qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash) ·
  [`-c2`](qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash-c2) ·
  [`-c16`](qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash-c16). Matched MTP:
  [`-mtp-c4`](qwen3-6-35b-a3b-nvfp4-vllm-mtp-c4).
