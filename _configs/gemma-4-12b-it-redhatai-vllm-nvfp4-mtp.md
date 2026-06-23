---
title: Gemma 4 12B · vLLM · NVFP4 + MTP · RedHatAI
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: NVFP4
quant_rationale: RedHatAI/gemma-4-12B-it-NVFP4 base + Google's official `google/gemma-4-12B-it-assistant` drafter, using vLLM's native Gemma-4 Unified MTP path. vLLM's current docs explicitly route Gemma-4 assistant checkpoints through `method=mtp`, not generic draft-model speculation.
source_repo: RedHatAI/gemma-4-12B-it-NVFP4
download_url: https://huggingface.co/RedHatAI/gemma-4-12B-it-NVFP4
context: 65536
modalities: [text, image, audio, video]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, NVFP4, 5-15B, conc-32]
status: pending
run_command: |
  # vLLM docs (June 2026) say Gemma-4 Unified assistant checkpoints use the native MTP path:
  # `--speculative-config {"method":"mtp","model":"google/gemma-4-12B-it-assistant",...}`.
  # Repo convention uses 3 speculative tokens for the throughput benchmark sweep.
  scripts/bench-vllm-serving.sh RedHatAI/gemma-4-12B-it-NVFP4 65536 32 1000 900 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":3}'
---

**Queued — the official Google assistant on the trusted Red Hat AI NVFP4 base.**

- **Why this spec method:** current vLLM docs say Gemma-4 assistant checkpoints for both
  `Gemma4ForConditionalGeneration` and `Gemma4UnifiedForConditionalGeneration` use the **Gemma-4 MTP
  path**, not generic draft-model or EAGLE handling.
- **No trusted 12B EAGLE sibling found:** I searched Hugging Face and did not find a trusted
  12B Gemma EAGLE speculator analogous to RedHatAI's 26B-A4B and 31B EAGLE repos. So the reliable
  spec axis here is the **official Google assistant**, not a guessed EAGLE draft.
- **Watch the image choice when this runs:** the repo notes already document that some Gemma-4 NVFP4
  paths still prefer older `cu130-nightly` images while Gemma-4 MTP support is better on newer vLLM
  builds. This queue stub uses the current wrapper default first; if it hits the known NVFP4/MTP image
  split, record that explicitly on the result page.
