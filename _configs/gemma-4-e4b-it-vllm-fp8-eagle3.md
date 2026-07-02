---
title: Gemma 4 E4B · vLLM · FP8 + EAGLE3
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: vLLM
speculative: EAGLE3 (RedHatAI speculator convention)
quant: FP8
quant_rationale: Would stack leon-se's FP8-Dynamic base (the done FP8 base/MTP quant) with a RedHatAI EAGLE3 speculator, mirroring the 26B-A4B/31B EAGLE3 rows — but no E4B EAGLE3 head exists to draft with (see below).
source_repo: leon-se/gemma-4-E4B-it-FP8-Dynamic
download_url: https://huggingface.co/leon-se/gemma-4-E4B-it-FP8-Dynamic
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-e4b, Google, Gemma, FP8, ≤4B, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-07-02
completed_at:
run_command: |
  # BLOCKED on draft sourcing, NOT the harness — no EAGLE3 head exists for Gemma-4 E4B.
  # RedHatAI publishes gemma-4 EAGLE3 speculators ONLY for 26B-A4B and 31B
  # (RedHatAI/gemma-4-{26B-A4B,31B}-it-speculator.eagle3, both 200); the E4B name
  # RedHatAI/gemma-4-E4B-it-speculator.eagle3 is 404, and an HF search across ALL
  # authors (2026-07-02) returned NO E4B eagle3 head from anyone. So there is nothing
  # to pass as the EAGLE3 draft model. Intended command if a head existed:
  scripts/bench-vllm-serving.sh leon-se/gemma-4-E4B-it-FP8-Dynamic 65536 32 1000 900 256 \
    --speculative-config '{"method":"eagle3","model":"<no-such-E4B-eagle3-head>","num_speculative_tokens":3}'
---

## BLOCKED — no EAGLE3 head exists for Gemma-4 E4B (draft-sourcing block, not a harness/image wall)

The E4B FP8 row already has **base (869.7)** and **+MTP (1261.5)**; only the EAGLE3 cell was open.
Filling it needs an EAGLE3 speculator trained against the Gemma-4 E4B target — and **none exists**.

- **RedHatAI ships gemma-4 EAGLE3 heads for only two sizes:** `RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`
  and `RedHatAI/gemma-4-31B-it-speculator.eagle3` (both verified HTTP 200, both already benchmarked here).
  The inferred E4B name `RedHatAI/gemma-4-E4B-it-speculator.eagle3` returns **404** (as does the 12B).
- **No E4B EAGLE3 head from any other author.** An HF model-search on 2026-07-02
  (`gemma-4-E4B eagle3` / `gemma-4-E4B speculator`) returned **zero** results — unlike the 12B, which at
  least has third-party heads (`deepseek-ai/eagle3_gemma4_12b_ttt7`, `BCCard/...`).
- **So this is blocked on sourcing/training a draft, not on the harness or an image wall** — the FP8
  target itself serves fine (see the base/MTP rows). Revisit if RedHatAI (or a trusted lab) publishes an
  E4B EAGLE3 speculator. The **MTP** path already covers E4B spec-decode on vLLM (1261.5 tok/s).
