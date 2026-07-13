#!/usr/bin/env python3
"""Generate strix-halo llama.cpp config pages from a batch-sweep results dir.

Reads the per-model serving logs + llama-bench + summary.tsv and emits one
_configs/*.md per (model, concurrency), with Spark cross-links. Idempotent:
overwrites its own output files. Data is the source of truth; prose angles
live in MODELS below.
"""
import re, sys, pathlib

BATCH = sys.argv[1] if len(sys.argv) > 1 else "results/batch-20260713-225032"
ROOT = pathlib.Path(__file__).resolve().parent.parent
BATCHDIR = ROOT / BATCH
OUT = ROOT / "_configs"
IMAGE = "ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41"

# Per-model metadata. spark = (slug, c32_decode) of the Spark llama.cpp Q4_K_M page
# for a same-engine/quant head-to-head, or None if the Spark only ran other engines.
MODELS = {
 "smollm3-3b-q4km": dict(
    slug="smollm3-3b", title="SmolLM3 3B", model="HuggingFaceTB/SmolLM3-3B",
    company="HuggingFaceTB", family="SmolLM3", params="3B", size="≤4B",
    repo="ggml-org/SmolLM3-3B-GGUF", modalities="[text]",
    spark=("smollm3-3b-llamacpp-q4_k_m", 653.55),
    angle="The fastest model of the sweep and the small-end anchor — a 3B that tops the decode chart."),
 "llama-3p1-8b-q4km": dict(
    slug="llama-3-1-8b", title="Llama 3.1 8B", model="meta-llama/Llama-3.1-8B-Instruct",
    company="Meta", family="Llama", params="8B", size="5-15B",
    repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", modalities="[text]",
    spark=("llama-3-1-8b-llamacpp-q4_k_m", 365.22),
    angle="The classic 8B baseline — the strongest concurrency scaler here (956/1000 completed at c32)."),
 "phi-4-reasoning-plus-q4km": dict(
    slug="phi-4-reasoning-plus", title="Phi-4-reasoning-plus", model="microsoft/Phi-4-reasoning-plus",
    company="Microsoft", family="Phi", params="14B", size="5-15B",
    repo="unsloth/Phi-4-reasoning-plus-GGUF", modalities="[text]",
    spark=("phi-4-reasoning-plus-llamacpp-q4_k_m", 230.99),
    angle="A 14B reasoning model — prefill-heavy (fast prompt ingest), decode more modest."),
 "deepseek-r1-distill-qwen-14b-q4km": dict(
    slug="deepseek-r1-distill-qwen-14b", title="DeepSeek-R1-Distill-Qwen-14B",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    company="DeepSeek", family="DeepSeek-R1-Distill", params="14B", size="5-15B",
    repo="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF", modalities="[text]",
    spark=("deepseek-r1-distill-qwen-14b-llamacpp-q4_k_m", 243.75),
    angle="The middle rung of the R1-distill dense ladder (8B→14B→32B) — clean scaling."),
 "gemma-4-12b-q4km": dict(
    slug="gemma-4-12b", title="Gemma 4 12B", model="google/gemma-4-12B-it",
    company="Google", family="Gemma", params="12B (dense)", size="5-15B",
    repo="unsloth/gemma-4-12b-it-GGUF", modalities="[text, image]",
    spark=("gemma-4-12b-it-llamacpp-q4_k_m", 195.25),
    angle="The cleanest cross-machine control: same engine, same Q4_K_M, same model as a Spark llama.cpp run."),
 "qwen3p6-27b-q4km": dict(
    slug="qwen3.6-27b", title="Qwen3.6 27B", model="Qwen/Qwen3.6-27B",
    company="Alibaba", family="Qwen", params="27B (dense)", size="16-40B",
    repo="unsloth/Qwen3.6-27B-GGUF", modalities="[text, image, video]",
    spark=None, spark_note="Spark ran Qwen3.6-27B only on vLLM/SGLang (NVFP4/FP8), not llama.cpp — no same-engine baseline; see the [model tag](../tags/model/) for the Spark configs.",
    angle="The dense-vs-MoE contrast: at 27B dense it decodes ~11 tok/s at c1, vs ~74 for the 30B-A3B MoE — active-parameter count, not total, sets single-stream speed."),
 "qwen3-coder-30b-a3b-q4km": dict(
    slug="qwen3-coder-30b-a3b", title="Qwen3-Coder 30B-A3B", model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    company="Alibaba", family="Qwen", params="30B / 3B (MoE)", size="16-40B",
    repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", modalities="[text]",
    spark=None, spark_note="Spark ran Qwen3-Coder-30B-A3B on vLLM FP8 (decode 295.82) and the DDTree harness, not llama.cpp — no same-engine baseline.",
    angle="The coding MoE (3B active) — fastest mid-tier decode of the sweep and directly relevant to the local OpenCode setup."),
 "gemma-4-31b-q4km": dict(
    slug="gemma-4-31b", title="Gemma 4 31B", model="google/gemma-4-31B-it",
    company="Google", family="Gemma", params="31B (dense)", size="16-40B",
    repo="unsloth/gemma-4-31B-it-GGUF", modalities="[text, image]",
    spark=("gemma-4-31b-it-llamacpp-q4_k_m", 78.45),
    angle="The pool workout: dense 31B at c32 pushed VRAM to 83.7 GiB — the first run to seriously use the 96 GiB pool (KV grows fast for a dense 31B at 32 slots)."),
 "deepseek-r1-distill-qwen-32b-q4km": dict(
    slug="deepseek-r1-distill-qwen-32b", title="DeepSeek-R1-Distill-Qwen-32B",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    company="DeepSeek", family="DeepSeek-R1-Distill", params="32B", size="16-40B",
    repo="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF", modalities="[text]",
    spark=("deepseek-r1-distill-qwen-32b-llamacpp-q4_k_m", 117.59),
    angle="The slow end of the dense ladder — a 32B dense is memory-bandwidth-bound on the iGPU (10.1 tok/s single-stream)."),
}

def load_manifest():
    """label -> gguf relpath, from models-midsmall.tsv"""
    m = {}
    for line in (ROOT / "scripts" / "models-midsmall.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"): continue
        c = line.split("\t")
        if len(c) >= 2: m[c[1]] = c[0]
    return m

MANIFEST = load_manifest()

def parse_result(log):
    t = pathlib.Path(log).read_text() if pathlib.Path(log).exists() else ""
    m = re.search(r"^RESULT prefill_toks=([\d.]+) decode_toks=([\d.]+) completed=(\d+) errors=(\d+) duration_s=([\d.]+) hit_time_cap=(\w+)", t, re.M)
    return dict(prefill=float(m[1]), decode=float(m[2]), completed=int(m[3]), errors=int(m[4]), cap=m[6]) if m else None

def parse_bench(log):
    t = pathlib.Path(log).read_text() if pathlib.Path(log).exists() else ""
    pp = re.search(r"pp512.*?([\d.]+) ± ([\d.]+)", t)
    tg = re.search(r"tg128.*?([\d.]+) ± ([\d.]+)", t)
    return (float(pp[1]) if pp else None, float(tg[1]) if tg else None)

def summary_rows(model_dir):
    """label -> (vram_peak_gb, vram_delta_gb, finished_at)"""
    rows = {}
    f = model_dir / "summary.tsv"
    if not f.exists(): return rows
    for line in f.read_text().splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 9:
            rows[c[3]] = (c[7], c[8], c[1])
    return rows

for label, meta in MODELS.items():
    md = BATCHDIR / label
    bench_pp, bench_tg = parse_bench(next(md.glob("*llamabench*.log")))
    rows = summary_rows(md)
    results = {c: parse_result(md / f"{label}-serving-c{c}.log") for c in (1, 8, 32)}
    if any(v is None for v in results.values()):
        print(f"WARN {label}: missing a serving result, skipping"); continue
    for conc in (1, 8, 32):
        r = results[conc]
        vp, vd, fin = rows.get(f"{label}-serving-c{conc}", ("?", "?", "2026-07-14"))
        fslug = meta['slug'].replace(".", "-")   # dash-form for filenames/links; dot stays in the tag
        slug = f"{fslug}-llamacpp-q4_k_m-strix-c{conc}"
        others = [f"[`-c{c}`]({fslug}-llamacpp-q4_k_m-strix-c{c})" for c in (1, 8, 32) if c != conc]
        # Spark comparison line
        if meta.get("spark"):
            sp_slug, sp32 = meta["spark"]
            if conc == 32:
                pct = round(100 * r["decode"] / sp32)
                spark_line = (f"- **vs Spark** ([`{sp_slug}`]({sp_slug}), llama.cpp CUDA Q4_K_M c32 decode {sp32}): "
                              f"Strix Halo Vulkan reaches **{pct}%** on the identical engine/quant/workload.")
            else:
                spark_line = f"- **Spark same-engine baseline:** [`{sp_slug}`]({sp_slug}) (llama.cpp CUDA Q4_K_M, conc 32: decode {sp32})."
        else:
            spark_line = f"- {meta['spark_note']}"
        cap = "hit the 900 s cap" if r["cap"] == "True" else f"finished under the cap"
        headline = meta["angle"] if conc == 1 else {
            8: "The throughput case — aggregate decode with continuous batching.",
            32: "Peak concurrency — where the slot-split (2048 tokens/request) starts costing errors.",
        }[conc]
        body = f"""---
title: {meta['title']} · llama.cpp · Q4_K_M · Strix Halo · conc {conc}
model: {meta['model']}
company: {meta['company']}
family: {meta['family']}
params: {meta['params']}
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from {meta['repo'].split('/')[0]} — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: {meta['repo']}
download_url: https://huggingface.co/{meta['repo']}
context: 65536
modalities: {meta['modalities']}
mm_served: false
concurrency: {conc}
tags: [{meta['slug']}, {meta['company']}, {meta['family']}, Q4_K_M, {meta['size']}, conc-{conc}, strix-halo]
status: done
prefill_toks: {r['prefill']}
decode_toks: {r['decode']}
mem_gb: {vd}
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak {vp} GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: {vp}
vram_delta_gb: {vd}
measured_on: 2026-07-14
completed_at: {fin}
engine_image: {IMAGE}
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel {conc} -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh {MANIFEST[label]} 65536 {conc} 1000 900 256 99
  # {r['completed']}/1000 prompts ({cap}), {r['errors']} errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**{headline}**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`{BATCH}/`).

- **Result (conc {conc}):** prefill {r['prefill']} / decode **{r['decode']}** tok/s{' aggregate' if conc>1 else ''};
  {r['completed']}/1000 prompts ({cap}), **{r['errors']} errors**{' (slot-split — 2048 tokens/request)' if conc==32 and r['errors'] else ''}.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill {bench_pp} / decode **{bench_tg}** tok/s.
{spark_line}
- **Memory:** VRAM pool delta **{vd} GiB** (peak {vp} GiB, co-resident with the OpenCode server).
- Sweep siblings: {' · '.join(others)}. Evidence: `{BATCH}/{label}/`.
"""
        (OUT / f"{slug}.md").write_text(body)
        print(f"wrote {slug}.md  (c{conc}: {r['decode']} tok/s, vram {vp})")
print("done")
