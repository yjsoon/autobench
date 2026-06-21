# Candidate model list for DGX Spark benchmarking

Rough shortlist of ~40 of the most popular / talked-about models (mid-2026), assembled
from web research. Ordered roughly by tier. The Spark has **128 GB unified memory**, so the
practical ceiling is ~200B params at aggressive 4-bit quant (a 120B-class model ≈ 60–70 GB
at 4-bit, leaving room for context + overhead).

> ⚠️ Caveats: some 2026 names below come from forward-looking blog/leaderboard posts and may
> be unreleased, renamed, or gated. **Verify the exact HF repo ID, license, and quant
> availability at download time.** "Fit" column assumes a single Spark unless noted.

Legend — **Fit:** ✅ comfortable · 🟡 needs 4-bit/aggressive quant · 🔴 too big for one Spark
(needs 2 linked Sparks or won't run). **Engine** = best-suited backend(s) of {llama.cpp, vLLM, TRT-LLM}.

---

## Tier A — Frontier MoE (flagship; mostly 🔴 / 🟡 on one box)
| # | Model | ~Size (total/active) | Fit | Engine | Notes |
|---|---|---|---|---|---|
| 1 | DeepSeek V4-Pro | 1.6T / 49B (MoE) | 🔴 | vLLM | Top open-weight leaderboard; won't fit one Spark. |
| 2 | DeepSeek V4-Flash | 284B / 13B (MoE) | 🟡 | vLLM | 1M ctx; 4-bit may squeak in — good stress test. |
| 3 | Kimi K2.6 (Thinking) | ~1T (MoE) | 🔴 | vLLM | Strong agentic/coding; multi-node only. |
| 4 | GLM-5.1 | ~355B class | 🟡/🔴 | vLLM/TRT-LLM | Best all-round agentic coder; 200K ctx, MIT. |
| 5 | Qwen3.5-397B (Reasoning) | 397B (MoE) | 🔴 | vLLM | Flagship Qwen reasoning. |
| 6 | MiniMax M3 | large MoE | 🔴 | vLLM | 1M ctx + native multimodal (Jun 2026). |
| 7 | Llama 4 Maverick | ~400B (MoE) | 🔴 | vLLM/TRT-LLM | Needs serious multi-GPU. |
| 8 | DeepSeek-R1 (orig) | 671B / 37B (MoE) | 🔴 | vLLM | Classic reasoning baseline. |

## Tier B — Spark sweet spot (30B–120B; the headline runs) ⭐
| # | Model | ~Size | Fit | Engine | Notes |
|---|---|---|---|---|---|
| 9 | **gpt-oss-120b** | 116.8B / 5.1B (MoE, MXFP4) | 🟡 | vLLM/TRT-LLM | ~56–76 tok/s reported on Spark. Apache-2.0, 128K. Flagship test. |
| 10 | **gpt-oss-20b** | 20.9B / 3.6B (MoE) | ✅ | vLLM/llama.cpp | Fast, Apache-2.0, 128K. |
| 11 | Qwen3-235B | 235B (MoE) | 🟡 | vLLM | 4-bit; pushes memory. |
| 12 | Qwen3-80B / 3.5-122B | 80–122B | 🟡 | vLLM | ~40–45 tok/s reported on Spark. |
| 13 | Llama 4 Scout | 109B / 17B (MoE) | 🟡 | vLLM/TRT-LLM | 10M ctx claim — great context test. |
| 14 | Nemotron-3-Nano-30B-A3B (NVFP4) | 30B / 3B (MoE) | ✅ | vLLM/TRT-LLM | ~56 tok/s, 7400 tok/s prefill — NVIDIA-native, great showcase. |
| 15 | Llama 3.1 70B Instruct | 70B | 🟡 | vLLM/llama.cpp | Workhorse dense 70B baseline. |
| 16 | Llama 3.3 70B Instruct | 70B | 🟡 | vLLM/llama.cpp | Improved 70B; very popular. |
| 17 | Qwen2.5 72B Instruct | 72B | 🟡 | vLLM/llama.cpp | Strong multilingual baseline. |
| 18 | Mistral Large 2 | 123B | 🟡 | vLLM | Dense flagship from Mistral. |
| 19 | Gemma 3 27B | 27B | ✅ | llama.cpp/vLLM | Text+image, 128K, 140+ langs. |
| 20 | Gemma 4 26B-A4B | 26B (MoE) | ✅ | vLLM | Newer MoE Gemma. |

## Tier C — Mid dense (12B–34B; high throughput) 
| # | Model | ~Size | Fit | Engine | Notes |
|---|---|---|---|---|---|
| 21 | Qwen2.5 32B Instruct | 32B | ✅ | all | Reliable mid baseline. |
| 22 | Qwen3 32B | 32B | ✅ | all | Thinking/non-thinking modes. |
| 23 | Qwen3.6-27B / 35B-A3B | 27–35B | ✅ | vLLM | Newer Qwen mid tier. |
| 24 | Mistral Small 3 / Small 4 | 22–24B | ✅ | all | Efficient production pick. |
| 25 | Gemma 3 12B | 12B | ✅ | llama.cpp | Text+image, 128K. |
| 26 | Phi-4 (14B) | 14B | ✅ | all | Best-in-class SLM benchmarks; 12 GB-class. |
| 27 | Phi-4-reasoning-vision-15B | 15B | ✅ | vLLM | Adds multimodal reasoning. |
| 28 | Yi-1.5 34B | 34B | ✅ | all | Popular dense alt. |
| 29 | DeepSeek-V2-Lite / V2 | 16B–236B(MoE) | ✅/🟡 | vLLM | Lite is small & fast. |

## Tier D — Coding specialists
| # | Model | ~Size | Fit | Engine | Notes |
|---|---|---|---|---|---|
| 30 | **Qwen3-Coder-Next** | ~30B class | ✅ | vLLM/llama.cpp | Best local coder 2026; 256K ctx, SWE-bench 58.7%. |
| 31 | Qwen3-Coder 30B-A3B Instruct (AWQ) | 30B / 3B (MoE) | ✅ | vLLM | Recommended Spark coder; MoE = very fast. |
| 32 | Qwen2.5-Coder 32B | 32B | ✅ | all | Strong local coding baseline. |
| 33 | DeepSeek-Coder-V2 | 16B–236B (MoE) | ✅/🟡 | vLLM | Classic coding MoE. |
| 34 | Codestral 22B | 22B | ✅ | all | Fast; Mistral non-prod license. |
| 35 | GLM-4.x (coder tier) | 9B–32B | ✅ | all | Agentic coding, MIT. |

## Tier E — Small / edge / multimodal (<12B; speed + vision)
| # | Model | ~Size | Fit | Engine | Notes |
|---|---|---|---|---|---|
| 36 | Llama 3.1 8B / 3.3 8B Instruct | 8B | ✅ | all | Best all-round small baseline. |
| 37 | Qwen3 8B | 8B | ✅ | all | Great default starter. |
| 38 | Gemma 3 4B | 4B | ✅ | llama.cpp | Best small multimodal/vision pick. |
| 39 | Phi-4-mini (3.8B) | 3.8B | ✅ | all | Best sub-4B reasoner; ~3.5 GB at Q4. |
| 40 | Ministral-3-3B (+vision) | 3.4B | ✅ | all | Edge multimodal SLM. |
| 41 | SmolLM3 (3B) | 3B | ✅ | all | Beats Llama-3.2-3B; tiny smoke-test model. |
| 42 | Mistral 7B Instruct v0.3 | 7B | ✅ | all | Ubiquitous 7B baseline. |

---

## Suggested benchmarking order
1. **Smoke test** with a tiny model (#39 Phi-4-mini or #41 SmolLM3) to validate the harness end-to-end.
2. **Spark showcases** next: #14 Nemotron-Nano-30B (NVFP4), #9 gpt-oss-120b (MXFP4), #10 gpt-oss-20b — these are the NVIDIA-native, best-documented Spark runs.
3. **Workhorses:** #16 Llama 3.3 70B, #17 Qwen2.5 72B, #19 Gemma 3 27B, #30/31 coders.
4. **Stretch / stress:** #2 DeepSeek V4-Flash, #11 Qwen3-235B, #13 Llama 4 Scout (long-ctx) — expect 🟡 memory pressure; good for finding the 128 GB ceiling.
5. Skip 🔴 single-box-infeasible flagships unless two Sparks are linked.

## Sources
- [NVIDIA DGX Spark: Best Local LLM Hardware 2026 (explainx.ai)](https://www.explainx.ai/blog/nvidia-dgx-spark-local-llm-best-setup-2026)
- [Four Inference Engines, One Box: DGX Spark (Medium/Hannecke)](https://medium.com/@michael.hannecke/four-inference-engines-one-box-when-to-use-which-on-the-dgx-spark-6b32a53db768)
- [llama.cpp performance on DGX Spark (GitHub disc. #16578)](https://github.com/ggml-org/llama.cpp/discussions/16578)
- [DGX Spark concurrency benchmark (Dendro Logic)](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/)
- [Best Open Source/Open-Weight LLMs to run locally 2026 (HF/daya-shankar)](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally)
- [Best Open-Source LLMs 2026: coding/local/agentic (HF/daya-shankar)](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Open-Source AI Landscape April 2026 — Gemma/Qwen/Llama (digitalapplied)](https://www.digitalapplied.com/blog/open-source-ai-landscape-april-2026-gemma-qwen-llama)
- [gpt-oss-120b vs Llama 4 Maverick (Artificial Analysis)](https://artificialanalysis.ai/models/comparisons/gpt-oss-120b-vs-llama-4-maverick)
- [Best Local Coding Models 2026 (promptquorum)](https://www.promptquorum.com/power-local-llm/best-local-coding-models-2026)
- [Best Small Language Models 2026 (Local AI Master)](https://localaimaster.com/blog/small-language-models-guide-2026)
- [Best Open-Source Ollama Models June 2026 (promptquorum)](https://www.promptquorum.com/local-llms/top-open-source-models-ollama)
