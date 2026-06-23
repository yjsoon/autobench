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
status: done
prefill_toks: 461.64
decode_toks: 399.82
mem_gb: 107.50
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV reservation (mem-fraction-static 0.85) + assistant drafter
spec_acceptance: 38-47% accept rate · mean acceptance length ~2.9-3.3 · FROZEN_KV_MTP promoted from NEXTN
measured_on: 2026-06-24
completed_at: 2026-06-24 00:08 +0800
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed@sha256:ca580c17cf5f9d2e268f4153d977e3cd46528feb2c62a4de8683a05d08da3cf2
run_command: |
  # AxionML's documented speculative recipe also needs the same branch overlay as the base run.
  # On current SGLang this assistant path is auto-promoted from NEXTN to FROZEN_KV_MTP.
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
  SGLANG_BRANCH=gemma4-modelopt-ptq \
  TRANSFORMERS_REF=1423d22f7a3b62e8c70ad67b58ec25cd9b675897 \
    scripts/bench-sglang-gemma4-modelopt-serving.sh AxionML/Gemma-4-12B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
    --attention-backend triton \
    --speculative-algorithm NEXTN \
    --speculative-draft-model-path google/gemma-4-12B-it-assistant \
    --speculative-draft-model-quantization unquant \
    --speculative-num-steps 5 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1 \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --mem-fraction-static 0.85
  # 1000/1000 prompts, 0 errors, 583.8 s. TTFT median 253.3 ms, TPOT median 74.4 ms.
---

**DONE — the documented assistant path works, but at conc-32 it barely improves the Axion base.**

- **Result (conc 32):** prefill **461.64** tok/s, decode **399.82** tok/s aggregate; **1000/1000, 0
  errors** in **583.8 s**. Median **TTFT 253.3 ms**, median **TPOT 74.4 ms**, request throughput
  **1.713 req/s**.
- **Speedup vs the Axion base is only marginal:** decode rises from **386.56 -> 399.82 tok/s**, just
  **+3.4%**. Prefill also moves only slightly (**446.48 -> 461.64 tok/s**). So this assistant path is
  functionally correct but not a meaningful throughput accelerator on the full 32-way ShareGPT sweep.
- **Acceptance was decent, so the missing speedup is not a draft mismatch:** the overlay logged
  **mean acceptance length ~2.9-3.3** with **accept rate ~38-47%** across the steady-state windows.
  That is lower than the "~70%+" ideal, but still high enough that a healthy overlap-capable serving
  path would usually show more benefit than this.
- **Why the gain stayed small:** SGLang detected `Gemma4AssistantForCausalLM` and **promoted
  `NEXTN -> FROZEN_KV_MTP`** automatically. In the same step it reset **`max_running_requests` to 48**
  and explicitly **disabled the overlap scheduler** for this speculative mode. The logs say this
  directly: *"Overlap scheduler is disabled when using Frozen-KV MTP speculative decoding (spec v2 is
  not supported yet)."* That scheduler loss is the most plausible reason the good acceptance metrics do
  not turn into a material end-to-end win at conc-32.
- **Takeaway:** on this Axion/SGLang branch, the Google assistant path is now **servable**, but it is
  not competitive with the **RedHatAI vLLM + assistant** path (**782.37 decode tok/s**) and barely
  beats the Axion base at this batch size.
