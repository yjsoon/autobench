---
title: gpt-oss-120b · vLLM · MXFP4 + EAGLE3
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: vLLM
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss MXFP4 base + NVIDIA's throughput-tuned EAGLE3 draft head (nvidia/gpt-oss-120b-Eagle3-throughput) — the spec-decode dimension on the gpt-oss-120b headliner. gpt-oss is text-only, so it should dodge vLLM's multimodal draft-model block that complicated Gemma.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe]
status: done
prefill_toks: 169.8
decode_toks: 138.45
mem_gb: 113.63
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-22
completed_at: 2026-06-22 17:42 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, NVIDIA EAGLE3-throughput draft.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"model":"nvidia/gpt-oss-120b-Eagle3-throughput","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**A clear negative result: EAGLE3 *hurt* gpt-oss-120b at concurrency 32 — the opposite of the Gemma
spec-decode wins, and an important lesson about where speculation pays off.** gpt-oss-120b MXFP4 +
NVIDIA's throughput-tuned EAGLE3 head, on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **513/1000, 76 errors**, **hit the 15-min cap**. Loaded +
  CUDA-graph captured in **511 s**.
- **It went the wrong way vs the base:**

  | gpt-oss-120b config | prefill | decode | completed | TTFT med |
  |---|---|---|---|---|
  | vLLM (base) | 278.8 | **252.8** | 788 | 29.6 s |
  | **vLLM + EAGLE3** | 169.8 | **138.5** | 513 | 53.2 s |
  | SGLang (base) | 187.7 | 140.3 | 560 | — |

  Decode **dropped 253 → 138** (−45%) and completions fell 788 → 513 with EAGLE3 on. The draft head's
  extra memory (reservation 113.6 vs 107.9 GB) and per-step draft+verify compute **compete with an
  already-saturated batch**: at conc 32 a **120B** model leaves the GB10 with no idle compute for
  speculation to exploit, so the draft is pure overhead and TTFT balloons to 53 s.
- **Why Gemma won and this lost — the size × concurrency interaction.** vLLM-EAGLE3 gave the *26–31B*
  Gemmas **+41–59%** at the same conc 32, because those smaller models leave headroom on the GB10 for
  the draft to use. gpt-oss-120b is ~4× larger and already compute-bound at conc 32, so the same
  technique inverts. **Speculative decoding helps when there's spare compute (small model and/or low
  concurrency); for a large model under heavy batching it's counterproductive.** It would very likely
  help gpt-oss-120b at low concurrency (a single-user latency deployment) — the regime EAGLE3 is built
  for — just not in this 32-way throughput test. (Lowering `num_speculative_tokens` might claw some
  back, but it won't beat the no-spec base here.)
- **Harmony caveat unchanged:** still the harmony chat path (TPOT 0.0 not meaningful; 76 errors from
  256-token truncation — fewer than the base's 212 only because far fewer requests completed).
- **Takeaway:** for gpt-oss-120b throughput on one Spark, run **base vLLM** (253) — not EAGLE3. Save
  EAGLE3 for latency-bound single-stream use.
