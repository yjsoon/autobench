# AGENTS.md — source of truth for the Strix Halo fork

Operational notes for the agent running benchmarks here. CLAUDE.md just points at this file.

**autobench (Strix Halo fork)**: a harness + results website for benchmarking open-weight LLMs on
a single **AMD Strix Halo** box. Forked from
[gauravmm/autobench](https://github.com/gauravmm/autobench) (NVIDIA DGX Spark, kept as the
`upstream` remote) — the existing `_configs/` pages are **Spark results, kept as the comparison
baseline**; new runs on this machine get new config pages. Live state = the `_configs/` pages +
homepage listing; no separate journal.

- Companion docs: `notes/BENCHMARKING.md` (methodology: dataset, run caps, spec-decode — image
  policy and digest map there are Spark-era; this fork's images live in the wrapper scripts) ·
  `notes/INCOMPATIBILITIES.md` (Spark-era engine/model walls, still useful for model-side quirks).
- The wider research context (ROCm/Vulkan findings on this machine) lives in the parent repo's
  diary: `../logs/`.

## State as of 2026-07-03

- Scripts ported to this machine and smoke-tested; ShareGPT V3 dataset downloaded to
  `benchmark_data/` (gitignored).
- **First Strix Halo config pages recorded (2026-07-06):** the 2026-07-04 overnight llama.cpp
  Vulkan sweep (Gemma 4 E4B + Qwen3.6-35B-A3B Q4_K_M, conc 1/8/32) lives in
  `_configs/*-strix-c*.md`, tagged `strix-halo`, browsable at `/tags/strix-halo/`
  (`tags-strix-halo.md`). Raw evidence: `results/overnight-20260704-070027/` (committed;
  `results/` is otherwise gitignored — `git add -f` curated run dirs). Headline: Gemma E4B c32
  decode **314.6 tok/s** ≈ 72% of the Spark CUDA number on the identical workload.
- **Local commits may be unpushed** (`git log origin/main..main`) — pushing to `main` needs
  the user's approval; ask, don't force it.
- **gpt-oss-20b MXFP4 benchmarked (2026-07-06):** the Spark's harmony chat-parser blocker is
  fixed in llama.cpp b9859 — `_configs/gpt-oss-20b-llamacpp-mxfp4-strix-c*.md` are done
  (c1 decode 68.5 tok/s beats the Spark vLLM c1's 45.6; c32 is ~34% of Spark vLLM batch).
  `scripts/sweep-gguf.sh <gguf> <label>` now runs the standard sweep for any single GGUF.
- Not yet done: SGLang image not built; vLLM untested; GitHub Pages not enabled on the fork.

## The machine

- GMKtec EVO X2, **AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S iGPU (gfx1151)**, x86_64,
  Arch Linux. 32 CPU threads, ~780 GB free on /home, internet.
- **Memory split (BIOS UMA):** 96 GiB allocated to the GPU as VRAM
  (`/sys/class/drm/card*/device/mem_info_vram_total`), ~32 GiB left as system RAM. The 96 GiB
  VRAM pool is the model-fit ceiling; 32 GiB is the ceiling for anything CPU-resident.
- Verify GPU: `vulkaninfo --summary` → `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`. User is in
  `render`, `video`, `docker` groups; `/dev/kfd` + `/dev/dri/renderD128` accessible.
- **No CUDA.** GPU paths are **Vulkan (RADV — primary, most mature)** and **ROCm (gfx1151 —
  officially unsupported, works via community builds)**. Containers get the GPU via
  `--device /dev/dri --device /dev/kfd`, NOT `--gpus all`. Use x86_64 image tags, not aarch64.
- **sudo is password-gated** — can't run non-interactively; hand the user the exact command.
- **Creds in `.env`** (gitignored): `HF_TOKEN` (gated HF). Load with `set -a; source .env; set +a`.
  Never echo or commit them.
- **LM Studio** is installed on the host (Vulkan runtime), GGUFs under `~/.lmstudio/models`
  (the default `MODELS_DIR` for the llama.cpp wrappers). CLI at `~/.lmstudio/bin/lms`.

## Measuring memory — record BOTH numbers

Headline mem = **system `MemAvailable` delta** from idle baseline, 10 s sampling (keeps numbers
comparable with upstream's Spark methodology). BUT on this box Vulkan/ROCm allocations land in
the 96 GiB VRAM pool and may barely move `MemAvailable` (the smoke test moved it only 0.26 GiB
for a 5 GB model). So also sample **`/sys/class/drm/card*/device/mem_info_vram_used`** before and
during the run, record both on the config page, and say which one moved. The cgroup
`memory.peak` probe in `bench-llamacpp.sh` found no file on this Arch/docker setup — don't
trust it until fixed; the sysfs VRAM delta is the meaningful GPU-side number.

## Engines & how to run each

All wrappers live in `scripts/` and end with a `RESULT`/`MEM` summary. Env-overridable:
`MODELS_DIR`, `LLAMACPP_IMAGE`, `SGLANG_IMAGE`, `VLLM_IMAGE`, `LMSTUDIO_URL`.

- **llama.cpp (Vulkan container — primary, verified).**
  - Synthetic: `scripts/bench-llamacpp.sh <gguf-path-under-MODELS_DIR> [pp] [tg] [ngl]`
  - Serving: `scripts/bench-llamacpp-serving.sh <gguf-path> [ctx] [conc] [num_prompts] [max_s] [max_tok] [ngl] [extra llama-server args]`
  - Image `ghcr.io/ggml-org/llama.cpp:full-vulkan` (already pulled).
- **LM Studio (host Vulkan — zero setup).** Start the server (`~/.lmstudio/bin/lms server start`,
  then `lms load <model>` with full GPU offload, or via the GUI), check the id at
  `http://localhost:1234/v1/models`, then:
  `scripts/bench-lmstudio-serving.sh <model-id> [conc] [num_prompts] [max_s] [max_tok]`.
  Load the model BEFORE the script grabs its memory baseline, and note that the baseline
  therefore excludes the model itself.
- **SGLang (ROCm — needs one-time image build).** Build `strix-halo-sglang:dev` from
  https://github.com/JeremiahM37/strix-halo-sglang first. Then:
  `scripts/bench-sglang-serving.sh <hf-model> [ctx] [conc] ... --mem-fraction-static 0.5 --attention-backend triton --disable-cuda-graph`.
  Prefer AWQ quants; BF16 dense models crawl (GTT-bound). GPTQ-on-MoE doesn't work (NVIDIA-only).
  Keep the tunableop cache mount — without it single-stream throughput halves.
- **vLLM (ROCm — experimental, expect breakage).** `scripts/bench-vllm-serving.sh`, default
  `rocm/vllm:latest`; may need `HSA_OVERRIDE_GFX_VERSION=11.5.1` exported, or a custom gfx1151
  build (https://blog.epheo.eu/notes/strix-halo/index.html). Get llama.cpp + SGLang results first.
- **TensorRT-LLM and NIM are NVIDIA-only — mark such configs `status: blocked`,** don't attempt.
- Quants: sweep GGUF Q4_K_M/Q8, AWQ, MXFP4, BF16 × context × concurrency. **NVFP4 is Blackwell
  hardware format — substitute GGUF/AWQ and note the quant difference** when comparing with
  upstream Spark configs. Verify exact HF repo IDs at download time.

## Task queue (in order)

1. Push the pending commits (user-approved), enable GitHub Pages (Settings → Pages → Source:
   GitHub Actions) so the site deploys to https://yjsoon.github.io/autobench.
2. **First headline run:** LM Studio + `qwen3.6-35b-a3b` (already on disk, Q4_K_M) at conc 1 via
   `bench-lmstudio-serving.sh` → compare against the Spark's **74.7 decode tok/s**
   (`_configs/qwen3-6-35b-a3b-nvfp4-vllm-c1.md`; note NVFP4-vs-Q4_K_M quant gap). Record as a
   new config page.
3. llama-bench quant/model sweep of the on-disk GGUFs; gpt-oss-20b MXFP4 GGUF is a good
   Spark-overlap candidate to download next.
4. Build the SGLang image; rerun the headliners at conc 1/8/32.
5. vLLM ROCm attempt, results or a documented failure either way.

## Recording results

- One model at a time: download → run across engine/quant/context configs → record **full run
  command · context window · tok/s (prefill + decode) · peak memory (both counters)**. The notes
  ARE the deliverable — over-note.
- `scripts/new-config.sh` scaffolds a page in `_configs/`; fill
  `prefill_toks/decode_toks/mem_gb/mem_source/run_command/completed_at/engine_image`, set
  `status: done`. Tag new runs so they're distinguishable from the Spark baseline (add a
  `strix-halo` tag; keep the existing 7-category taxonomy; engine is a field, not a tag).
- `engine_image` = pinned `repo:tag@sha256:<digest>` — capture with
  `docker inspect --format '{{index .RepoDigests 0}}' <tag>` right after the run.
- **Read `date "+%Y-%m-%d %H:%M %z"` immediately before writing any `completed_at`.**
- Website: Ruby only in Docker — `./serve.sh` to preview, `./build.sh` before pushing to catch
  Liquid errors. Commit after every benchmark; ask the user before pushing `main`.

## Benchmark monitoring

- When checking long-running engine startup or benchmark jobs, prefer **narrow status checks** over
  dumping whole logs into the context.
- First check the smallest signal that answers the question:
  - endpoint readiness such as `curl -fsS http://localhost:<port>/health`
  - container/process liveness via `docker ps`, `docker inspect`, or `docker stats --no-stream`
  - targeted log grep such as `docker logs ... | rg "Application startup complete|ERROR|Traceback|Killed"`
- Only read broader log output when a narrow check shows an actual failure or when the expected
  marker cannot be found with targeted grep.
- For startup progress, grep for specific readiness or failure markers instead of ingesting the
  whole container log.
