# CLAUDE.md — orientation for a future agent

**autobench (Strix Halo fork)**: a harness + results website for benchmarking open-weight LLMs on
a single **AMD Strix Halo** box. Forked from gauravmm/autobench (NVIDIA DGX Spark, kept as the
`upstream` remote) — the existing `_configs/` pages are **Spark results, kept as the comparison
baseline**; new runs on this machine get new config pages. Live state = the `_configs/` pages +
homepage listing; no separate journal.

- **The model list IS the `_configs/` collection** — one stub page per model × engine × quant is the
  source of truth for what to benchmark.
- **Companion docs:** `notes/BENCHMARKING.md` (how to run/measure: wrappers, dataset, run caps,
  spec-decode, image policy, digests) · `notes/INCOMPATIBILITIES.md` (engine/model walls + fixes —
  **check before debugging a failing launch**).

## Mission

- One model at a time: download → run across engine / quant / context configs → record
  **full run command · context window · tok/s (prefill + decode) · peak memory**. The notes ARE the
  deliverable — over-note.
- **Run by an Opus-class agent** — quant choice, sourcing trusted HF repos, and fixing imprecise
  model-list entries (wrong/renamed/gated repos, ambiguous sizes) need judgment; don't downgrade.

## The machine

- GMKtec EVO X2, **AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S iGPU (gfx1151)**, x86_64,
  Arch Linux. 32 CPU threads, ~780 GB free on /home, internet.
- **Memory split (BIOS UMA):** 96 GiB allocated to the GPU as VRAM
  (`/sys/class/drm/card*/device/mem_info_vram_total`), ~32 GiB left as system RAM
  (`free -h` shows ~30 Gi). The 96 GiB VRAM pool is the model-fit ceiling; the 32 GiB system
  side is the ceiling for anything CPU-resident.
- Verify GPU: `vulkaninfo --summary` → `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`.
  User is in `render`, `video`, and `docker` groups; `/dev/kfd` + `/dev/dri/renderD128` accessible.
- **No CUDA.** GPU paths are **Vulkan (RADV — primary, most mature)** and **ROCm (gfx1151 —
  officially unsupported, works via community builds)**. Containers get the GPU via
  `--device /dev/dri --device /dev/kfd`, NOT `--gpus all`. Use x86_64 image tags, not aarch64.
- **sudo is password-gated** — can't run non-interactively; hand the user the exact command.
- **Creds in `.env`** (gitignored): `HF_TOKEN` (gated HF). Load with
  `set -a; source .env; set +a`. Never echo or commit them.
- **Measuring memory:** unified memory again, but split differently from the Spark. Headline mem =
  **system `MemAvailable` delta** from idle baseline, 10 s sampling (comparable with upstream's
  Spark numbers), cross-checked against `/sys/class/drm/card*/device/mem_info_vram_used` for the
  GPU-side allocation. Note that Vulkan/ROCm allocations land in the 96 GiB VRAM pool and may
  barely move `MemAvailable` — record BOTH and note which moved.
- **LM Studio** is installed on the host with a Vulkan runtime and local GGUFs under
  `~/.lmstudio/models` (the default `MODELS_DIR` for the llama.cpp wrappers).

## Engines

- **llama.cpp (Vulkan container — primary) · LM Studio (host, Vulkan) · SGLang (ROCm, community
  image `strix-halo-sglang:dev`) · vLLM (ROCm — experimental on gfx1151)**.
  TensorRT-LLM and NIM are NVIDIA-only: **not available on this fork** — mark such configs blocked.
- Per model, sweep quant (GGUF Q4_K_M/Q8; AWQ/MXFP4/BF16) × context × concurrency.
  **NVFP4 is Blackwell hardware format — substitute GGUF/AWQ and note the quant difference**
  when comparing against upstream Spark configs.
- Verify exact HF repo IDs at download time — the model list has unverified 2026 names.
- Wrappers, per-model engine selection, spec-decode → `notes/BENCHMARKING.md` (image policy and
  digest map there are upstream/Spark-era; this fork's images are in the wrapper scripts). Every
  known engine/model wall + fix → `notes/INCOMPATIBILITIES.md` (ditto — Spark-era, still useful
  for model-side quirks).

## Website (Jekyll)

- **Ruby lives only in Docker — never install it on the host.** `./serve.sh` = live preview;
  `./build.sh` = static build into `_site` (run before pushing to catch Liquid errors).
- `_configs/` — the collection, one page per config. Front matter: `model/company/family/params/
  engine/speculative/quant/quant_rationale/source_repo/download_url/context/modalities/tags` + results
  (`prefill_toks/decode_toks/mem_gb/mem_source/run_command/completed_at/engine_image`) +
  `status: pending | blocked | done`. `engine_image` = pinned `repo:tag@sha256:<digest>` (map in
  `notes/BENCHMARKING.md`). Homepage sorts done (newest) → pending → blocked.
- `_archive/configs/` — `git mv` a page here to hide it without losing data (excluded in `_config.yml`).
- **Tag taxonomy = 7 categories:** concurrency (`conc-N`) · model (per-model slug, FIRST in `tags:`,
  identical across a model's configs) · lab · family · quant · size-bucket
  (`≤4B`/`5-15B`/`16-40B`/`41-130B`/`130B+`) · `Spark recipe`. Engine is a field, NOT a tag.
- `scripts/new-config.sh` generates a page; `scripts/seed-stubs.sh` applies the taxonomy. Keep `tags*.md`
  pure-Liquid (Pages-safe).
- **Deploy:** push `main` → GitHub Actions → Pages (https://yjsoon.github.io/autobench once enabled
  in Settings → Pages → Source: GitHub Actions). Remote is HTTPS via `gh` auth; `git push` just
  works. **Commit/push after every new benchmark.**

## Working style

- User moves fast; keep momentum, validate before pushing.
- **Read `date "+%Y-%m-%d %H:%M %z"` immediately before writing any `completed_at`** — never reuse an
  earlier timestamp (a run drifts it by minutes).
- Aggressive autocompact (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=40`) — rely on this file + `_configs/` pages
  as durable memory, not in-context history.
- No journal: results → per-config pages; decisions/methodology → here + the two `notes/` docs.
