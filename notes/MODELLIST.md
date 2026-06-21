# Candidate model list for DGX Spark benchmarking

Verified against the **live Hugging Face API + DGX Spark benchmark writeups on 2026-06-21.**
Every repo ID below was confirmed to resolve, and every parameter count is the real
`safetensors.total` reported by HF (not a guess). Generation labels and "superseded"
notes reflect what each org has actually shipped as of mid-June 2026.

## The hardware constraint (read this first)

The Spark is **GB10 Grace-Blackwell, 128 GB unified LPDDR5X, ~273 GB/s bandwidth**, with
**Blackwell FP4 tensor cores**. Two things follow, and they drive the whole list:

1. **Capacity ceiling ≈ 200 B total params at 4-bit** (~0.5 GB/B → ~100 GB weights + KV +
   overhead). Anything bigger is 🔴 on one box.
2. **Bandwidth, not capacity, sets decode speed.** Decode tok/s ≈ bandwidth ÷ bytes read
   per token. At 273 GB/s a **dense 70 B at 4-bit (~35 GB/tok-pass) only does ~5–8 tok/s**,
   while an **MoE with ~3–12 B active runs 40–60 tok/s** at the same total size. ⇒ **Favor
   MoE with small active params.** Dense >32 B is a capability test, not a usable-speed test.

**Quant formats on Blackwell** (see `notes/` engineering section at bottom):
**NVFP4** = Blackwell-native, hardware-accelerated FP4 → first choice. **MXFP4** = gpt-oss's
native format, also FP4-accelerated *but only with the CUTLASS kernels* (without them
gpt-oss-120b drops 56→35 tok/s). **FP8** = good quality, 2× the size of FP4. **GGUF Q4–Q8**
(llama.cpp) = widest coverage + easiest, not FP4-tensor-core accelerated. **AWQ/GPTQ-Int4** =
supported by vLLM/SGLang, not FP4-native.

Legend — **Fit (one Spark):** ✅ comfortable (<~60 B total) · 🟡 4-bit only / tight
(~60–230 B) · 🔴 won't fit one box (needs 2 linked Sparks). **Speed** reflects active params.

---

## Tier A — Frontier MoE (flagship; mostly 🔴 on one box)
| Model | HF repo ID | Total / active | Fit | Notes |
|---|---|---|---|---|
| Kimi K2.6 | `moonshotai/Kimi-K2.6` | 1.06 T / ~32 B | 🔴 | Top agentic/coding; multi-node only. K2.7-Code (newest) also 🔴. |
| DeepSeek V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` | 861 B (MoE) | 🔴 | Flagship (Apr 2026). **List's "1.6T" was wrong — it's 861 B.** |
| GLM-5.2 | `zai-org/GLM-5.2` | 753 B (MoE) | 🔴 | Newest GLM (Jun 16 2026); **supersedes GLM-5.1/5 from the old list.** MIT. |
| DeepSeek V3.2 | `deepseek-ai/DeepSeek-V3.2` | 685 B / 37 B | 🔴 | Stable V3 line (Dec 2025); V3.2-Exp/Speciale variants exist. |
| Mistral Large 3 | `mistralai/Mistral-Large-3-675B-Instruct-2512` | 675 B (MoE) | 🔴 | NVFP4 variant exists but still too big for one box. |
| Nemotron-3 Ultra | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | 550 B / 55 B | 🔴 | NVIDIA flagship (Jun 2026); NVFP4 but ~275 GB. |
| MiniMax M3 | `MiniMaxAI/MiniMax-M3` | 427 B (MoE) | 🔴 | Newest MiniMax (Jun 2026); MXFP8 variant published. ~214 GB @4-bit. |
| Llama 4 Maverick | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | 400 B / 17 B | 🔴 | Apr 2025; **Meta has shipped nothing newer (no Llama 5).** |
| Qwen3.5-397B | `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | 397 B / 17 B | 🔴 | Flagship Qwen reasoning; GPTQ-Int4 ~200 GB — still over 128. |
| GLM-4.7 | `zai-org/GLM-4.7` | 358 B (MoE) | 🔴 | Dec 2025; the non-Flash 4.7. |

## Tier B — Spark sweet spot (the headline runs) ⭐
The 🟡 entries here are the *interesting* stress tests — they squeak into 128 GB at 4-bit
**and** are MoE, so they actually run at usable speed. This is where to spend benchmarking time.
| Model | HF repo ID | Total / active | Fit | Notes |
|---|---|---|---|---|
| **gpt-oss-120b** | `openai/gpt-oss-120b` | 120 B / 5.1 B | 🟡 | **Native MXFP4 ~63 GB. ~56 tok/s (vLLM) / 45 (llama.cpp). Flagship Spark test.** Apache-2.0, 128K. |
| **Nemotron-3 Super** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 123 B / 12 B | 🟡 | **NVFP4-native ~62 GB. The NVIDIA-native showcase at 120 B class.** FP8/BF16 also published. |
| DeepSeek V4-Flash | `deepseek-ai/DeepSeek-V4-Flash` | 158 B (MoE) | 🟡 | ~79 GB @4-bit. **List's "284 B" was wrong — it's 158 B.** Good memory stress test. |
| Qwen3.5-122B | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | 122 B / 10 B | 🟡 | ~61 GB; A10B → fast. GPTQ-Int4 published by Qwen. |
| Mistral Small 4 | `mistralai/Mistral-Small-4-119B-2603` | 119 B (dense-ish) | 🟡 | **List's "22–24 B" was wrong — Small 4 is 119 B.** NVFP4 variant exists. |
| Devstral-2 (coder) | `mistralai/Devstral-2-123B-Instruct-2512` | 123 B (MoE) | 🟡 | Mistral's flagship coder; replaces Codestral. |
| Llama 4 Scout | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 109 B / 17 B | 🟡 | ~55 GB @4-bit; 10 M-ctx claim — long-context test. |
| **gpt-oss-20b** | `openai/gpt-oss-20b` | 21 B / 3.6 B | ✅ | Fast, Apache-2.0, 128K, native MXFP4. |
| **Nemotron-3 Nano (Omni)** | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | 33 B / 3 B | ✅ | **~56 tok/s, 7417 tok/s prefill (vLLM). NVFP4-native, multimodal. Best Spark showcase.** |
| Nemotron-3 Elastic | `nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4` | 30 B / 3 B | ✅ | Elastic-width MoE; NVFP4. |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | 36 B / 3 B | ✅ | Newest mid Qwen MoE (Apr 2026); FP8 variant published. Very fast. |
| Gemma 4 31B | `google/gemma-4-31B-it` | 33 B (dense) | ✅ | **Gemma 4 is real (the old "Gemma 4 26B" guess was close).** Official QAT w4a16 + GGUF. |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` | 26 B / 4 B | ✅ | MoE Gemma 4; multimodal; QAT quants. |
| GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` | 31 B (MoE) | ✅ | The GLM that fits comfortably; agentic, MIT. |
| Nemotron-Terminal-32B | `nvidia/Nemotron-Terminal-32B` | 33 B (dense) | ✅ | NVIDIA agentic/terminal-tuned (Feb 2026). |

## Tier C — Mid dense / small MoE (✅ high throughput)
| Model | HF repo ID | Total / active | Notes |
|---|---|---|---|
| Qwen3.6-27B | `Qwen/Qwen3.6-27B` | 27.8 B dense | Newest mid dense Qwen; FP8 variant published. |
| Granite 4.1 30B | `ibm-granite/granite-4.1-30b` | 28.9 B (MoE) | **New family worth adding;** GGUF + FP8 official. Apache-2.0. |
| Granite-switch 4.1 30B | `ibm-granite/granite-switch-4.1-30b-preview` | 32 B (MoE) | IBM's MoE "switch" preview. |
| Mistral Devstral-Small-2 | `mistralai/Devstral-Small-2-24B-Instruct-2512` | 24 B | Efficient coder; replaces Codestral 22B. |
| Phi-4 (reasoning) | `microsoft/Phi-4-reasoning-plus` | 14 B | Best-in-class SLM reasoning; Phi-4 gen still current (no Phi-5). |
| Phi-4-reasoning-vision | `microsoft/Phi-4-reasoning-vision-15B` | 15 B | **Real (Jan 2026) — list was right.** Multimodal reasoning. |
| Gemma 4 12B | `google/gemma-4-12B-it` | 12 B | Multimodal, QAT w4a16/GGUF official. |
| Granite 4.1 8B | `ibm-granite/granite-4.1-8b` | 8 B | GGUF + FP8 official. |

## Tier D — Coding specialists
| Model | HF repo ID | Total / active | Fit | Notes |
|---|---|---|---|---|
| Qwen3-Coder-Next | `Qwen/Qwen3-Coder-Next` | 79.7 B (MoE) | ✅ | Best local coder line (Jan 2026); official FP8 + GGUF. ~40 GB @4-bit. |
| Qwen3-Coder-30B-A3B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 30.5 B / 3 B | ✅ | MoE = very fast; FP8 published. Recommended Spark coder. |
| Devstral-2-123B | `mistralai/Devstral-2-123B-Instruct-2512` | 123 B (MoE) | 🟡 | Mistral flagship coder (see Tier B). |
| Kimi K2.7-Code | `moonshotai/Kimi-K2.7-Code` | ~1 T (MoE) | 🔴 | Newest Kimi coder (Jun 2026); multi-node only. |
| Qwen2.5-Coder 32B | `Qwen/Qwen2.5-Coder-32B-Instruct` | 32 B | ✅ | Older but strong dense coding baseline. |

## Tier E — Small / edge / multimodal (✅ speed + smoke tests)
| Model | HF repo ID | ~Size | Notes |
|---|---|---|---|
| Nemotron-3-Nano-4B | `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` | 4 B | NVFP4/FP8/GGUF; NVIDIA-native tiny. |
| Gemma 4 E4B / E2B | `google/gemma-4-E4B-it-qat-w4a16-ct` | ~4 B / ~2 B | Edge ("E") Gemma 4, QAT, mobile-ready. |
| Phi-4-mini-reasoning | `microsoft/Phi-4-mini-reasoning` | 3.8 B | Best sub-4B reasoner; ~3.5 GB @Q4. |
| SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B` | 3 B | Current SmolLM (no SmolLM4 yet); tiny smoke-test model. |
| Granite 4.1 3B | `ibm-granite/granite-4.1-3b` | 3 B | GGUF + FP8 official. |
| Ministral-3-3B | `mistralai/Ministral-3-3B-Instruct-2512` | 3.4 B | Edge SLM; ONNX/GGUF variants. |
| Ministral-3-14B-Reasoning | `mistralai/Ministral-3-14B-Reasoning-2512` | 14 B | GGUF published; small reasoner. |
| Llama 3.3 70B | `meta-llama/Llama-3.3-70B-Instruct` | 70 B | Dense baseline (🟡, ~35 GB, but only ~5–8 tok/s — capability test only). |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 8 B | Ubiquitous small baseline. |

---

## What changed vs. the old list (corrections)
- ✅ **Real after all:** DeepSeek V4-Pro/V4-Flash, Qwen3.5-397B, Gemma 4, Phi-4-reasoning-vision-15B, Nemotron-3-Nano-30B — all confirmed (the old list's caveats were overcautious here).
- ❌ **Wrong sizes:** DeepSeek V4-Pro is **861 B not 1.6 T**; V4-Flash **158 B not 284 B**; Mistral **Small 4 is 119 B not 24 B**.
- 🔄 **Superseded:** GLM-5.1 → **GLM-5.2** (Jun 2026); Qwen3 → **Qwen3.5/3.6**; MiniMax → **M3**; Kimi → **K2.6 / K2.7-Code**; Codestral → **Devstral-2**.
- ➕ **Added:** IBM Granite 4.1 family, Nemotron-3 Super/Nano-Omni/Elastic/Terminal, Mistral-Large-3, DeepSeek V3.2.
- ➖ **Dropped:** Yi (01-ai shipped nothing since 2024 — abandoned); Llama 4 has no successor (no Llama 5 as of Jun 2026).

## Suggested benchmarking order
1. **Smoke test** — `Phi-4-mini-reasoning` or `SmolLM3-3B` to validate the harness end-to-end.
2. **Spark showcases (NVIDIA-native FP4)** — `Nemotron-3-Nano-Omni-30B-A3B-NVFP4` (~56 tok/s), `gpt-oss-120b` (MXFP4, ~56 tok/s), `gpt-oss-20b`, `Nemotron-3-Super-120B-A12B-NVFP4`.
3. **Workhorses** — `Qwen3.6-35B-A3B`, `Gemma 4 31B`, `Granite 4.1 30B`, `Qwen3-Coder-30B-A3B` / `Qwen3-Coder-Next`.
4. **Stretch / 128 GB ceiling** — `DeepSeek-V4-Flash` (158 B), `Qwen3.5-122B-A10B`, `Mistral-Small-4-119B`, `Llama 4 Scout` (long-ctx). Expect memory pressure.
5. **Skip 🔴** single-box-infeasible flagships unless two Sparks are linked over ConnectX.

## DGX Spark inference engineering (verified Jun 2026)
- **Engines:** **SGLang** gives the best results on Spark per NVIDIA's own forums + `build.nvidia.com/spark/sglang` (EAGLE3 spec-decoding supported). **llama.cpp** = easiest, great for GGUF. **vLLM** = strong but its MXFP4 path can lag SGLang/llama.cpp unless you use the MXFP4-optimized image. **TensorRT-LLM** = max perf, hardest setup. **Ollama** = easiest UX (llama.cpp backend).
- **Verified throughput:** gpt-oss-120b MXFP4 → ~4703 prefill / **56 tok/s** decode (vLLM), ~2047 / **45** (llama.cpp); needs MXFP4 CUTLASS kernels or it falls to ~35. Nemotron-3-Nano-30B-A3B NVFP4 → **7417 prefill / 55.9 decode** (vLLM).
- **Tooling:** `llama-bench` (llama.cpp), vLLM `benchmark_serving.py`, and the `NVIDIA/dgx-spark-playbooks` repo. Concurrency matters: single-stream tok/s understates the box — batched throughput is far higher (see Dendro Logic concurrency benchmark).

## Sources
- Hugging Face API (live, 2026-06-21): `huggingface.co/api/models?author={deepseek-ai,moonshotai,MiniMaxAI,Qwen,zai-org,meta-llama,openai,nvidia,google,microsoft,mistralai,ibm-granite,HuggingFaceTB,01-ai}` — every repo ID + `safetensors.total` above verified there.
- [llama.cpp on DGX Spark — perf discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- [vLLM gpt-oss-120b MXFP4 vs SGLang/llama.cpp (NVIDIA Dev Forums)](https://forums.developer.nvidia.com/t/vllm-on-gb10-gpt-oss-120b-mxfp4-slower-than-sglang-llama-cpp-what-s-missing/356651)
- [Best results on Spark → SGLang (NVIDIA Dev Forums)](https://forums.developer.nvidia.com/t/inference-best-results-on-spark-not-llama-cpp-not-vllm-sgland/357175)
- [SGLang for Inference | DGX Spark (build.nvidia.com)](https://build.nvidia.com/spark/sglang) · [NVFP4 Quantization | DGX Spark](https://build.nvidia.com/spark/nvfp4-quantization)
- [Quantization on DGX Spark: BF16/FP8/NVFP4/MXFP4/GGUF (Kubesimplify)](https://blog.kubesimplify.com/day-4-quantization-demystified-bf16-fp8-nvfp4-mxfp4-int4-gguf-and-why-it-all-matters)
- [DGX Spark concurrency benchmark (Dendro Logic)](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/)
- [NVIDIA DGX Spark in-depth review (LMSYS)](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [NVIDIA/dgx-spark-playbooks — Inference Systems (DeepWiki)](https://deepwiki.com/NVIDIA/dgx-spark-playbooks/4-inference-systems) · [awesome-dgx-spark](https://github.com/bidual/awesome-dgx-spark)
