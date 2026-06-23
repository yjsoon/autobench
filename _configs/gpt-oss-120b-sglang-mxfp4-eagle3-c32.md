---
title: gpt-oss-120b · SGLang · MXFP4 + EAGLE3 · conc 32
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: SGLang
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss native MXFP4 base + SGLang's own LMSYS/SpecForge EAGLE3 draft (lmsys/EAGLE3-gpt-oss-120b-bf16). conc-32 throughput point — directly comparable to base SGLang (decode 140 tok/s) and vLLM+EAGLE3 (138). Tests whether the conc-1 spec-decode win survives heavy batching on a 120B model.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe, conc-32]
status: pending
run_command: |
  # lmsysorg/sglang:spark; harmony tiktoken encodings mounted; LMSYS/SpecForge EAGLE3 draft.
  docker run -d --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/tiktoken_encodings:/tiktoken_encodings \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-120b --host 0.0.0.0 --port 30000 \
    --context-length 65536 --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  python3 scripts/bench-serving.py --base-url http://localhost:30000 \
    --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The throughput regime — does the conc-1 EAGLE3 win survive 32-way batching?** SGLang's LMSYS/SpecForge
draft (`lmsys/EAGLE3-gpt-oss-120b-bf16`) on gpt-oss-120b, conc 32. Companion to the
[conc-1] run (decode 40.56 tok/s, accept-len ~2.4) — compare against base SGLang (decode **140.3**) and
vLLM+EAGLE3 (decode **138.5**, which *lost* ~45% vs its base at this concurrency).

<!-- results pending -->
