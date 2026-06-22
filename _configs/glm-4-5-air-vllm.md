---
title: GLM-4.5-Air · vLLM · (quant TBD)
model: zai-org/GLM-4.5-Air
company: Zhipu AI
family: GLM
params: 106B / 12B (MoE)
engine: vLLM
quant: TBD
quant_rationale: A human should pick the quant — the two fitting options trade official-but-tight against community-but-comfortable (see Notes).
source_repo: zai-org/GLM-4.5-Air
download_url: https://huggingface.co/zai-org/GLM-4.5-Air
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [glm-4.5-air, Zhipu AI, GLM, 41-130B, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # blocked — pick a quant first (see Notes), then fill in and run.
---

**Blocked — needs a human to specify the quant before running.** Added 2026-06-22.

GLM-4.5-Air (106B total / 12B active MoE) is the **only GLM that realistically fits a single Spark**
for the text benchmark — the bigger GLMs are out of reach (GLM-4.6 is a ~355B MoE → too large even at
4-bit; GLM-4.6V is vision-first → wrong methodology, like phi-4-reasoning-vision). It's the natural
substitute for the blocked `glm-4-7-flash` slot. But the two fitting quants trade off in ways that
warrant a human call:

| Quant | Size | Fit on 121 GB | Source | Caveat |
|---|---|---|---|---|
| `zai-org/GLM-4.5-Air-FP8` | ~106 GB | **tight** — needs `--gpu-memory-utilization ~0.95` + modest ctx, may OOM | **official** (Zhipu, compressed-tensors) | quality-safe, but little KV headroom |
| `cpatonn/GLM-4.5-Air-AWQ-4bit` | ~55 GB | comfortable | **community** (cpatonn) | not a top-trusted quantizer (same tier concern as the Devstral AWQ) |

**Decision needed:** which quant to benchmark —
- **FP8** (official, accept the tight-fit / OOM risk and reduced context), or
- **AWQ-4bit** (comfortable fit, but accept the community-quantizer source per the "trusted repo"
  policy), or
- wait for an official 4-bit (e.g. a RedHatAI `quantized.w4a16` — none exists yet).

Once a quant is chosen, set `quant`/`source_repo`/`download_url`, add the quant tag, flip to `pending`,
and run via `scripts/bench-vllm-serving.sh`.
