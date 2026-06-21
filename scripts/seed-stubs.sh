#!/usr/bin/env bash
# Seed pending stub config pages for every runnable (non-🔴) model in notes/MODELLIST.md.
# One canonical config per model (best-suited engine + planned quant). Re-runnable: skips
# files that already exist so it never clobbers measured results.
#
# Tag taxonomy (exactly five categories): lab · family · quant · size-bucket · [Spark recipe]
# Fields per row (delimiter ^): slug ^ name ^ model ^ lab ^ family ^ params ^ engine ^ quant ^ context ^ modalities ^ spark(0/1) ^ bucket ^ note
set -euo pipefail
cd "$(dirname "$0")/.."

ROWS='
gpt-oss-20b-vllm-mxfp4^gpt-oss-20b^openai/gpt-oss-20b^OpenAI^gpt-oss^21B / 3.6B (MoE)^vLLM^MXFP4^131072^text^1^16-40B^Fast, Apache-2.0, native MXFP4.
nemotron-3-super-120b-a12b-vllm-nvfp4^Nemotron-3 Super 120B^nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4^NVIDIA^Nemotron^123B / 12B (MoE)^vLLM^NVFP4^131072^text^1^41-130B^NVIDIA-native NVFP4 showcase at the 120B class (~62 GB). FP8/BF16 also published.
nemotron-3-nano-omni-30b-a3b-vllm-nvfp4^Nemotron-3 Nano-Omni 30B-A3B^nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4^NVIDIA^Nemotron^33B / 3B (MoE)^vLLM^NVFP4^131072^text, image^1^16-40B^Best Spark showcase: ~56 tok/s decode, 7417 prefill (vLLM). Omni — verify audio/video support at download.
nemotron-3-elastic-30b-a3b-vllm-nvfp4^Nemotron-3 Elastic 30B-A3B^nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4^NVIDIA^Nemotron^30B / 3B (MoE)^vLLM^NVFP4^131072^text^1^16-40B^Elastic-width MoE; NVFP4-native.
nemotron-3-nano-4b-llamacpp-q4_k_m^Nemotron-3 Nano-4B^nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF^NVIDIA^Nemotron^4B (dense)^llama.cpp^Q4_K_M^131072^text^1^≤4B^NVIDIA-native tiny; NVFP4/FP8/GGUF published.
nemotron-terminal-32b-vllm-fp8^Nemotron-Terminal-32B^nvidia/Nemotron-Terminal-32B^NVIDIA^Nemotron^33B (dense)^vLLM^FP8^131072^text^0^16-40B^NVIDIA agentic/terminal-tuned (Feb 2026).
deepseek-v4-flash-vllm-awq^DeepSeek V4-Flash^deepseek-ai/DeepSeek-V4-Flash^DeepSeek^DeepSeek^158B (MoE)^vLLM^AWQ-Int4^131072^text^0^>130B^~79 GB @4-bit — the 128 GB ceiling stress test.
qwen3-5-122b-a10b-vllm-gptq^Qwen3.5-122B-A10B^Qwen/Qwen3.5-122B-A10B-GPTQ-Int4^Alibaba^Qwen^122B / 10B (MoE)^vLLM^GPTQ-Int4^131072^text^0^41-130B^~61 GB; A10B active → fast decode. Int4 published by Qwen.
mistral-small-4-119b-vllm-nvfp4^Mistral Small 4 119B^mistralai/Mistral-Small-4-119B-2603^Mistral AI^Mistral^119B (dense-ish)^vLLM^NVFP4^131072^text^0^41-130B^NVFP4 variant exists; dense → slower decode.
devstral-2-123b-vllm-awq^Devstral-2 123B^mistralai/Devstral-2-123B-Instruct-2512^Mistral AI^Devstral^123B (MoE)^vLLM^AWQ-Int4^131072^text^0^41-130B^Mistral flagship coder; replaces Codestral.
llama-4-scout-17b-16e-vllm-awq^Llama 4 Scout 17B-16E^meta-llama/Llama-4-Scout-17B-16E-Instruct^Meta^Llama^109B / 17B (MoE)^vLLM^AWQ-Int4^1048576^text^0^41-130B^~55 GB @4-bit; 10M-ctx claim → long-context test.
qwen3-6-35b-a3b-vllm-fp8^Qwen3.6-35B-A3B^Qwen/Qwen3.6-35B-A3B^Alibaba^Qwen^36B / 3B (MoE)^vLLM^FP8^131072^text^0^16-40B^Newest mid Qwen MoE; very fast. FP8 variant published.
gemma-4-31b-it-llamacpp-q4_k_m^Gemma 4 31B^google/gemma-4-31B-it^Google^Gemma^33B (dense)^llama.cpp^Q4_K_M^131072^text, image^0^16-40B^Official QAT w4a16 + GGUF. Multimodal (text+image).
gemma-4-26b-a4b-it-vllm-w4a16^Gemma 4 26B-A4B^google/gemma-4-26B-A4B-it^Google^Gemma^26B / 4B (MoE)^vLLM^W4A16^131072^text, image^0^16-40B^MoE Gemma 4; multimodal; QAT quants.
glm-4-7-flash-vllm-fp8^GLM-4.7-Flash^zai-org/GLM-4.7-Flash^Zhipu AI^GLM^31B (MoE)^vLLM^FP8^131072^text^0^16-40B^The GLM that fits comfortably; agentic, MIT.
qwen3-6-27b-vllm-fp8^Qwen3.6-27B^Qwen/Qwen3.6-27B^Alibaba^Qwen^27.8B (dense)^vLLM^FP8^131072^text^0^16-40B^Newest mid dense Qwen; FP8 variant published.
granite-4-1-30b-vllm-fp8^Granite 4.1 30B^ibm-granite/granite-4.1-30b^IBM^Granite^28.9B (MoE)^vLLM^FP8^131072^text^0^16-40B^New IBM family; GGUF + FP8 official. Apache-2.0.
granite-switch-4-1-30b-vllm-fp8^Granite-switch 4.1 30B^ibm-granite/granite-switch-4.1-30b-preview^IBM^Granite^32B (MoE)^vLLM^FP8^131072^text^0^16-40B^IBM MoE "switch" preview.
devstral-small-2-24b-llamacpp-q4_k_m^Devstral-Small-2 24B^mistralai/Devstral-Small-2-24B-Instruct-2512^Mistral AI^Devstral^24B (dense)^llama.cpp^Q4_K_M^131072^text^0^16-40B^Efficient coder; replaces Codestral 22B.
phi-4-reasoning-plus-llamacpp-q4_k_m^Phi-4-reasoning-plus^microsoft/Phi-4-reasoning-plus^Microsoft^Phi^14B (dense)^llama.cpp^Q4_K_M^32768^text^0^5-15B^Best-in-class SLM reasoning; Phi-4 gen current (no Phi-5).
phi-4-reasoning-vision-15b-vllm-fp8^Phi-4-reasoning-vision 15B^microsoft/Phi-4-reasoning-vision-15B^Microsoft^Phi^15B (dense)^vLLM^FP8^131072^text, image^0^5-15B^Multimodal reasoning (Jan 2026).
gemma-4-12b-it-llamacpp-q4_k_m^Gemma 4 12B^google/gemma-4-12B-it^Google^Gemma^12B (dense)^llama.cpp^Q4_K_M^131072^text, image^0^5-15B^Multimodal; QAT w4a16/GGUF official.
granite-4-1-8b-llamacpp-q4_k_m^Granite 4.1 8B^ibm-granite/granite-4.1-8b^IBM^Granite^8B (dense)^llama.cpp^Q4_K_M^131072^text^0^5-15B^GGUF + FP8 official.
qwen3-coder-next-vllm-fp8^Qwen3-Coder-Next^Qwen/Qwen3-Coder-Next^Alibaba^Qwen^79.7B (MoE)^vLLM^FP8^262144^text^0^41-130B^Best local coder line; FP8 + GGUF official. ~40 GB @4-bit.
qwen3-coder-30b-a3b-vllm-fp8^Qwen3-Coder-30B-A3B^Qwen/Qwen3-Coder-30B-A3B-Instruct^Alibaba^Qwen^30.5B / 3B (MoE)^vLLM^FP8^262144^text^0^16-40B^MoE = very fast; recommended Spark coder.
qwen2-5-coder-32b-llamacpp-q4_k_m^Qwen2.5-Coder 32B^Qwen/Qwen2.5-Coder-32B-Instruct^Alibaba^Qwen^32B (dense)^llama.cpp^Q4_K_M^131072^text^0^16-40B^Strong dense coding baseline.
gemma-4-e4b-it-llamacpp-q4_k_m^Gemma 4 E4B^google/gemma-4-E4B-it-qat-w4a16-ct^Google^Gemma^~4B (edge)^llama.cpp^Q4_K_M^131072^text, image^0^≤4B^Edge ("E") Gemma 4; QAT, mobile-ready.
phi-4-mini-reasoning-llamacpp-q4_k_m^Phi-4-mini-reasoning^microsoft/Phi-4-mini-reasoning^Microsoft^Phi^3.8B (dense)^llama.cpp^Q4_K_M^131072^text^0^≤4B^Best sub-4B reasoner; ~3.5 GB @Q4. Smoke-test candidate.
granite-4-1-3b-llamacpp-q4_k_m^Granite 4.1 3B^ibm-granite/granite-4.1-3b^IBM^Granite^3B (dense)^llama.cpp^Q4_K_M^131072^text^0^≤4B^GGUF + FP8 official.
ministral-3-3b-llamacpp-q4_k_m^Ministral-3-3B^mistralai/Ministral-3-3B-Instruct-2512^Mistral AI^Ministral^3.4B (dense)^llama.cpp^Q4_K_M^131072^text^0^≤4B^Edge SLM; ONNX/GGUF variants.
ministral-3-14b-reasoning-llamacpp-q4_k_m^Ministral-3-14B-Reasoning^mistralai/Ministral-3-14B-Reasoning-2512^Mistral AI^Ministral^14B (dense)^llama.cpp^Q4_K_M^131072^text^0^5-15B^Small reasoner; GGUF published.
llama-3-3-70b-llamacpp-q4_k_m^Llama 3.3 70B^meta-llama/Llama-3.3-70B-Instruct^Meta^Llama^70B (dense)^llama.cpp^Q4_K_M^131072^text^0^41-130B^Dense baseline; ~5–8 tok/s on Spark — capability test only.
llama-3-1-8b-llamacpp-q4_k_m^Llama 3.1 8B^meta-llama/Llama-3.1-8B-Instruct^Meta^Llama^8B (dense)^llama.cpp^Q4_K_M^131072^text^0^5-15B^Ubiquitous small baseline.
deepseek-v2-lite-chat-llamacpp-q4_k_m^DeepSeek-V2-Lite-Chat^deepseek-ai/DeepSeek-V2-Lite-Chat^DeepSeek^DeepSeek^15.7B / 2.4B (MoE)^llama.cpp^Q4_K_M^163840^text^0^16-40B^Small DeepSeek MoE (2.4B active) — fast, comfortable fit. GGUF quant from a trusted repo at run time.
deepseek-coder-v2-lite-instruct-llamacpp-q4_k_m^DeepSeek-Coder-V2-Lite-Instruct^deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct^DeepSeek^DeepSeek^15.7B / 2.4B (MoE)^llama.cpp^Q4_K_M^163840^text^0^16-40B^DeepSeek coding MoE; 2.4B active → very fast. Classic local coder.
deepseek-r1-distill-qwen-32b-llamacpp-q4_k_m^DeepSeek-R1-Distill-Qwen-32B^deepseek-ai/DeepSeek-R1-Distill-Qwen-32B^DeepSeek^DeepSeek^32B (dense)^llama.cpp^Q4_K_M^131072^text^0^16-40B^Popular R1 reasoning distill (Qwen2.5-32B base).
deepseek-r1-distill-qwen-14b-llamacpp-q4_k_m^DeepSeek-R1-Distill-Qwen-14B^deepseek-ai/DeepSeek-R1-Distill-Qwen-14B^DeepSeek^DeepSeek^14B (dense)^llama.cpp^Q4_K_M^131072^text^0^5-15B^R1 reasoning distill, mid-small.
deepseek-r1-distill-llama-70b-llamacpp-q4_k_m^DeepSeek-R1-Distill-Llama-70B^deepseek-ai/DeepSeek-R1-Distill-Llama-70B^DeepSeek^DeepSeek^70B (dense)^llama.cpp^Q4_K_M^131072^text^0^41-130B^R1 distill on Llama-70B; dense 70B → ~5–8 tok/s, capability test.
deepseek-r1-0528-qwen3-8b-llamacpp-q4_k_m^DeepSeek-R1-0528-Qwen3-8B^deepseek-ai/DeepSeek-R1-0528-Qwen3-8B^DeepSeek^DeepSeek^8B (dense)^llama.cpp^Q4_K_M^131072^text^0^5-15B^Newer R1-0528 distill on a Qwen3-8B base.
'

created=0; skipped=0
while IFS='^' read -r slug name model lab family params engine quant context modalities spark bucket note; do
  [ -z "${slug// }" ] && continue
  file="_configs/${slug}.md"
  if [ -e "$file" ]; then echo "skip (exists): $file"; skipped=$((skipped+1)); continue; fi

  tags="$lab, $family, $quant, $bucket"
  [ "$spark" = "1" ] && tags="$tags, Spark recipe"

  # Why this quant (generic per-format rationale; the running Opus agent refines + may add quants).
  case "$quant" in
    NVFP4)     why="Blackwell-native FP4 — hardware-accelerated on the GB10; first choice for NVIDIA models." ;;
    MXFP4)     why="gpt-oss's native FP4 format; FP4-accelerated with the CUTLASS kernels." ;;
    GPTQ-Int4) why="4-bit to fit one Spark; official GPTQ-Int4 weights published by the lab." ;;
    AWQ-Int4)  why="4-bit to fit one Spark; AWQ preserves quality well at Int4." ;;
    FP8)       why="Near-BF16 quality at half the bytes; official FP8 weights published." ;;
    W4A16)     why="Official QAT w4a16 — quality-preserving 4-bit weights." ;;
    Q4_K_M)    why="GGUF Q4_K_M — widest llama.cpp coverage, strong size/quality balance." ;;
    Q8_0)      why="GGUF Q8_0 — near-lossless reference point." ;;
    *)         why="Planned quant; rationale to confirm at download time." ;;
  esac
  url="https://huggingface.co/${model}"

  cat > "$file" <<EOF
---
title: ${name} · ${engine} · ${quant}
model: ${model}
company: ${lab}
family: ${family}
params: ${params}
engine: ${engine}
quant: ${quant}
quant_rationale: ${why}
source_repo: ${model}
download_url: ${url}
context: ${context}
modalities: [${modalities}]
mm_served: true
tags: [${tags}]

status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned configuration — filled in by the run when benchmarked
---

${note}

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from \`notes/MODELLIST.md\` are still being confirmed).
EOF
  echo "wrote $file"; created=$((created+1))
done <<< "$ROWS"

echo "---"; echo "created=$created skipped=$skipped"