---
name: update-model-list
description: Refresh notes/MODELLIST.md with current, verified open-weight LLMs to benchmark on the DGX Spark. Use when the model list is stale, when checking for new model-family releases, or when verifying repo IDs / sizes / quants before a download. Produces a source-verified MODELLIST.md.
---

# Update the DGX Spark model list

Goal: turn a stale/speculative shortlist into a **source-verified** list of modern models +
quants that actually fit and run on a **DGX Spark (GB10, 128 GB unified, ~273 GB/s, Blackwell
FP4)**. The deliverable is a rewritten `notes/MODELLIST.md`.

## The efficient method (learned the hard way)

**Do NOT fan out a dozen web-search subagents per model family.** That hits API rate limits
and several agents die mid-run. Instead, work **inline and sequentially**, and use the
**Hugging Face API as ground truth** — it's faster, exact, and non-hallucinating. Reserve
WebSearch for the few things the API can't tell you (Spark throughput, engine/quant guidance).

### 0. Auth
`HF_TOKEN` lives in `.env` (gitignored). Load it: `set -a; source .env; set +a`.
The token only matters for gated repos; listing + `safetensors.total` work unauthenticated too.

### 1. List newest repos per org (identity = ground truth)
Sort by `createdAt` descending — the top of each org tells you the current generation and
instantly exposes hallucinated/renamed entries in the old list.

```bash
for org in deepseek-ai moonshotai MiniMaxAI Qwen zai-org meta-llama openai \
           nvidia google microsoft mistralai HuggingFaceTB 01-ai ibm-granite; do
  echo "== $org =="
  curl -s "https://huggingface.co/api/models?author=${org}&sort=createdAt&direction=-1&limit=18" \
  | python3 -c "import sys,json;[print(m['id'],'|',m.get('createdAt','')[:10]) for m in json.load(sys.stdin)]"
done
```
Use `&search=<substr>` to dig into a family (e.g. `search=Nemotron-3`, `search=Coder`).

### 2. Get EXACT parameter counts (don't trust names or memory)
`safetensors.total` is the real number. Pass the repo via argv so the shell doesn't mangle the
Python format string:

```bash
curl -s -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/models/$r" \
| python3 -c "import sys,json;r=sys.argv[1];d=json.load(sys.stdin);st=d.get('safetensors') or {};print('%-46s total=%s gated=%s'%(r,st.get('total'),d.get('gated')))" "$r"
```
Active params for MoE are usually encoded in the repo name (`-A3B`, `-A12B`, `-A55B`).

### 3. Two or three targeted WebSearches (only what the API can't give)
- Spark throughput: `DGX Spark GB10 <model> tok/s vLLM llama.cpp benchmark`
- Quant/engine guidance: `DGX Spark Blackwell NVFP4 vs MXFP4 SGLang best inference engine`
Trustworthy sources: `build.nvidia.com/spark/*`, NVIDIA Dev Forums, `NVIDIA/dgx-spark-playbooks`,
llama.cpp discussion #16578, LMSYS review, Dendro Logic concurrency benchmark.

### 4. Compute Spark fit + expected speed (pure arithmetic)
- 4-bit weights ≈ **0.5 GB per billion total params**. Capacity ceiling ≈ **200 B total**.
- ✅ <~60 B total · 🟡 ~60–230 B (4-bit only) · 🔴 >~230 B (needs 2 linked Sparks).
- **Speed is bandwidth-bound:** decode tok/s ≈ 273 GB/s ÷ (bytes read per token). A dense 70 B
  @4-bit ≈ 5–8 tok/s; an MoE with 3–12 B active ≈ 40–60 tok/s. **Favor small-active MoE.**
- Quant priority on Blackwell: **NVFP4** (native FP4) > **MXFP4** (gpt-oss; needs CUTLASS kernels)
  > **FP8** (2× size) > **GGUF Q4–Q8** (easiest, not FP4-accelerated) ≈ **AWQ/GPTQ-Int4**.

## What to produce (the deliverable)

Rewrite `notes/MODELLIST.md` with:
1. **Hardware-constraint preamble** — capacity ceiling + bandwidth/MoE point + quant-format ranking.
2. **Tiered tables (A frontier 🔴 → E small ✅)**, each row = real **HF repo ID**, total/active
   params (verified `safetensors.total`), Fit, and a one-line note. Star ⭐ the Spark sweet spot.
3. A **"What changed / corrections"** section: hallucinated-vs-real, wrong sizes, superseded
   generations, added/dropped families. This is the highest-value part — it's the audit trail.
4. A **suggested benchmarking order** (smoke test → NVIDIA-native FP4 showcases → workhorses →
   128 GB stretch → skip 🔴).
5. A short **inference-engineering section** (engine verdict, verified tok/s, tooling).
6. **Sources** — list the HF API query and every cited URL. Date-stamp the whole file.

## Rules
- Every repo ID must resolve on HF; every param count must be the API's `safetensors.total`.
- Never carry a number from memory or the old list without re-verifying — that's how the
  speculative entries got there. When unsure, fetch.
- Note the verification date prominently; this list goes stale in weeks.
