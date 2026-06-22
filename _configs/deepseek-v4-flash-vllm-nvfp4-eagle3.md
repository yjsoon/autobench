---
title: DeepSeek V4-Flash · vLLM · NVFP4 + EAGLE3.1
model: deepseek-ai/DeepSeek-V4-Flash
company: DeepSeek
family: DeepSeek
params: 158B (MoE)
engine: vLLM
speculative: EAGLE3.1
quant: NVFP4
quant_rationale: NVIDIA's NVFP4 (~79 GB) is the only build of this 158B MoE that fits one Spark — FP8 is ~160 GB. Base nvidia/DeepSeek-V4-Flash-NVFP4 + the ManiacLabs EAGLE3.1 draft (hidden sizes match: 4096). Blocked on engine support — see Notes.
source_repo: nvidia/DeepSeek-V4-Flash-NVFP4
download_url: https://huggingface.co/ManiacLabs/DeepSeek-V4-Flash-EAGLE3.1
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [deepseek-v4-flash, DeepSeek, NVFP4, 130B+, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # blocked — deepseek_v4 arch not in the stock engines (see Notes). Intended once an engine supports it:
  #   vllm serve nvidia/DeepSeek-V4-Flash-NVFP4 --max-model-len 65536 --gpu-memory-utilization 0.9 \
  #     --max-num-seqs 32 --speculative-config \
  #     '{"model":"ManiacLabs/DeepSeek-V4-Flash-EAGLE3.1","method":"eagle3","num_speculative_tokens":3}'
---

**Blocked — the pieces fit, but no stock engine on the Spark supports the `deepseek_v4` architecture
yet.** Investigated 2026-06-22 at the user's request.

**What lines up (the run *would* work once an engine supports the arch):**
- **Base fits at NVFP4.** DeepSeek-V4-Flash is a 158B MoE (256 experts, `DeepseekV4ForCausalLM`). FP8 is
  ~160 GB (over the 121 GB ceiling); **`nvidia/DeepSeek-V4-Flash-NVFP4` is ~79 GB** — NVIDIA-official,
  fits with room for KV. (The model-list stub's AWQ-Int4 was the right idea; NVFP4 is the better,
  trusted, fitting build.)
- **The EAGLE3.1 draft is compatible.** `ManiacLabs/DeepSeek-V4-Flash-EAGLE3.1` is a standard
  `LlamaForCausalLMEagle3` head whose **`target_hidden_size` (4096) matches the base's `hidden_size`
  (4096)** — so vLLM's `eagle3` method should drive it. (Community org, 159 dl/mo — would carry a
  source-tier caveat, but it's wirable.)

**The hard blocker — engine arch support:**
- **vLLM cu130-nightly:** ships `deepseek_v2` + `deepseek_eagle3`/`deepseek_mtp`, but **no
  `deepseek_v4` model** and nothing for `DeepseekV4ForCausalLM` in the registry → the base won't load.
- **SGLang `:spark`** (the user asked specifically): ships `deepseek_v2` + `deepseek_nextn` + an EAGLE3
  worker, but **also no `deepseek_v4`** → same wall. So **neither stock engine can serve the base**,
  EAGLE3.1 or not.
- A transformers bump won't help — this is a missing *engine model implementation*, not just a config
  the tokenizer can't parse.

**Paths to unblock (need a decision):**
1. **Newer engine build** — a bleeding-edge vLLM (main) or SGLang nightly that has added `deepseek_v4`.
   Not the documented Spark images; ARM64/GB10 compatibility unverified, so this is itself an
   experiment.
2. **llama.cpp via GGUF** — many DeepSeek-V4-Flash GGUFs exist (incl. Spark-targeted
   `0xSero/DeepSeek-V4-Flash-Spark-GGUF`), so a recent llama.cpp likely supports `deepseek_v4`. That
   would run the **base**, and a **native-MTP** GGUF draft exists — but **not** this specific EAGLE3.1
   safetensors head (no GGUF eagle3 draft for it). So that's "V4-Flash + MTP on llama.cpp", a different
   spec config, not the requested one.
3. **Wait** for `deepseek_v4` to land in the stock vLLM/SGLang Spark images, then run exactly as the
   commented `run_command` above.
