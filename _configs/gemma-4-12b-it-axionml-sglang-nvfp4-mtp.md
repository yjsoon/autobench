---
title: Gemma 4 12B · SGLang · NVFP4 + MTP · AxionML
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: SGLang
speculative: MTP (NEXTN + Google assistant drafter)
quant: NVFP4
quant_rationale: AxionML/Gemma-4-12B-NVFP4 base + Google's official `google/gemma-4-12B-it-assistant` drafter, using the exact SGLang NEXTN recipe that AxionML documents for this quantized target. This is the best-supported spec-decode path I found for the Axion checkpoint.
source_repo: AxionML/Gemma-4-12B-NVFP4
download_url: https://huggingface.co/AxionML/Gemma-4-12B-NVFP4
context: 65536
modalities: [text, image, audio, video]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, NVFP4, 5-15B, conc-32]
status: pending
run_command: |
  # AxionML's documented speculative recipe: SGLang NEXTN + the official Google assistant drafter.
  # Verify the exact SGLang image/tag first; the card requires a Gemma-4-capable branch with
  # ModelOpt FP4 support and uses the Triton attention backend for this path.
  SGLANG_IMAGE=<gemma4-modelopt-capable-sglang-image> \
    scripts/bench-sglang-serving.sh AxionML/Gemma-4-12B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
    --attention-backend triton \
    --speculative-algorithm NEXTN \
    --speculative-draft-model-path google/gemma-4-12B-it-assistant \
    --speculative-draft-model-quantization unquant \
    --speculative-num-steps 5 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1 \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --mem-fraction-static 0.85
---

**Queued — the documented speculative path for AxionML's 12B NVFP4.**

- **Why this spec method:** AxionML's own model card explicitly documents **NEXTN + the official
  Google assistant drafter** for this exact quantized target. That is the strongest current source for
  a 12B speculative recipe on this checkpoint.
- **No trusted 12B EAGLE repo found:** I searched Hugging Face and did **not** find a trusted
  `gemma-4-12B-it-speculator.eagle3` style repo comparable to the RedHatAI 26B/31B Gemma speculators.
  So I am not queueing a 12B EAGLE sibling on guesswork.
- **Run-time caveat to verify:** the card notes an older SGLang bug around Gemma-4 KV/projection
  scales on quantized targets; it claims the fix is in the required branch. So the first run should
  confirm the exact image/tag before treating this path as production-ready.

