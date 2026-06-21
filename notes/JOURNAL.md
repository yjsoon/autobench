---
layout: default
title: Journal
permalink: /journal/
---

# Autobench — DGX Spark model benchmarking journal

Running log of the auto-benchmark effort. Newest entries at the bottom of the Journal.
Goal: given a list of models, one at a time → download → run in various configs →
record essential attributes. We especially capture, for every run:
**full run command · context window · tok/s (prefill + decode) · memory used.**

---

## System inventory (captured 2026-06-21)

| Item | Value |
|---|---|
| Host | `heruli` — NVIDIA DGX Spark |
| Arch / OS | aarch64 / Ubuntu 24.04, kernel `6.17.0-1021-nvidia` |
| GPU | GB10 Grace-Blackwell (PCI VGA dev `2e12`) |
| Unified memory | 121 GB total (shared CPU/GPU) |
| CPU | 20 cores |
| Disk | 3.7 TB on `/` (nvme0n1p2), ~3.5 TB free |
| CUDA toolkit | 13.0 (`nvcc` works) |
| Docker | 29.2.1 + nvidia-container-toolkit (`nvidia-ctk`) installed |
| Internet | OK (huggingface.co reachable, HTTP 200) |

### Engine + runtime decisions
- **Engines to benchmark:** llama.cpp, vLLM, TensorRT-LLM.
- **Runtime:** Docker, using NVIDIA NGC containers (NVIDIA's recommended path on Spark).

---

## 🚧 BLOCKERS — ✅ both CLEARED 2026-06-21

GPU driver now loads (`nvidia-smi` shows **GPU 0: NVIDIA GB10**) and the user is in the
`docker` group. The fixes that were applied are kept below for reference.

### 1. GPU driver not loaded — kernel/module version mismatch
`nvidia-smi` fails ("couldn't communicate with the NVIDIA driver"); no `/dev/nvidia*` nodes.
- Running kernel: `6.17.0-1021-nvidia`
- Installed nvidia kernel module only built for `6.17.0-1014-nvidia` → no `nvidia.ko` for
  the running kernel → `modprobe nvidia` → "Module nvidia not found".
- **Fix (matching pkg confirmed available in apt cache):**
  ```bash
  sudo apt-get install -y linux-modules-nvidia-580-open-6.17.0-1021-nvidia
  sudo modprobe nvidia      # or reboot
  nvidia-smi                # verify GPU shows up
  ```

### 2. Not in `docker` group (needed since all engines run as containers)
```bash
sudo usermod -aG docker $USER   # then log out / back in (or `newgrp docker`)
```

Until both are cleared, no GPU-accelerated run is possible.

---

## Open questions
- **Model list?** Not yet provided. Need: model IDs/repos, and which precisions/quants
  to test per model (e.g. GGUF Q4_K_M/Q8 for llama.cpp; FP8/AWQ/BF16 for vLLM/TRT-LLM).
- **Configs to sweep per model?** Proposed axes: context length, batch/concurrency,
  quantization, engine. Confirm which matter.
- **Pass/fail criteria?** e.g. must fit in 121 GB, min tok/s threshold, must reach N ctx.

---

## Journal

### 2026-06-21 — orientation
- Surveyed the box (table above). Found the driver blocker (#1) and docker-group blocker (#2).
- Confirmed the matching kernel-module package is installable and internet works.
- Decided engines (llama.cpp/vLLM/TRT-LLM) + Docker/NGC runtime with the user.
- Next: user clears blockers → I pull NGC images, build the run+measure harness, get model list.

### 2026-06-21 — model list + website scaffold
- `notes/MODELLIST.md`: ~42 candidate models (fork-researched), tiered with Spark fit ratings
  and a smoke-test-first order. Exact HF repo IDs to be verified at download time.
- Stood up a **Jekyll site** (Ruby kept entirely in Docker — `Dockerfile.site` + `serve.sh`/`build.sh`):
  - `_configs/` collection = one page per model×engine×quant config; tagged by family/company/engine/quant.
  - `index.md` (listing), `tags.md` (pure-Liquid tag browser, no plugins), `scripts/new-config.sh` generator.
  - `.github/workflows/jekyll.yml` builds + deploys to GitHub Pages on push to `main`.
  - This journal now also publishes at **`/journal/`**.
- ⏳ **Build not yet validated** — `serve.sh`/`build.sh` need Docker access (blocker #2). Run `./build.sh`
  once in the docker group to confirm the Liquid compiles before first push.
- Before first deploy: set `url`/`baseurl` in `_config.yml`, enable Pages → Source: GitHub Actions.

### 2026-06-21 — first real runs: SmolLM3-3B smoke test (llama.cpp) ✅
- Engine image: `ghcr.io/ggml-org/llama.cpp:full-cuda` (ARM64). **Gotcha:** dispatcher entrypoint —
  invoke llama-bench as `--bench` (not `llama-bench`). All 99 layers offload to the GB10 (CUDA).
- GGUFs from `ggml-org/SmolLM3-3B-GGUF`. Wrapper: `scripts/bench-llamacpp.sh`.
- Numbers (pp512/tg128): **Q4_K_M 7214/105.7 tok/s @ 2.94 GiB**; **Q8_0 6391/70.6 tok/s @ 4.18 GiB**.
  Decode tracks bytes-per-weight (bandwidth bound); prefill stays high (compute bound).
- **Memory-measurement learnings** (see methodology): nvidia-smi = N/A on GB10; cgroup/docker-stats
  undercounts unified GPU memory; system `MemAvailable` delta is the trustworthy number.
- Workflow validated end-to-end. Next: scale up (gpt-oss-20b/Nemotron-Nano) and bring up vLLM/TRT-LLM.

---

## Measurement methodology

- **Peak memory** (`mem_gb`, `mem_source`): ⚠️ Two gotchas on the GB10, both found by measuring:
  1. `nvidia-smi` reports **N/A** for all GPU memory (unified with system RAM).
  2. The container's cgroup / `docker stats` **undercounts** GPU memory — CUDA's unified
     allocations don't all land in the memory cgroup (observed: `docker stats` ~0.6 GiB while
     the system actually consumed ~2.9 GiB for a 1.78 GiB Q4 model).
  → **Primary metric = system `MemAvailable` delta from an idle baseline**, sampled every **10 s**
  (take the max drop). The box is dedicated, so whole-system delta ≈ the model's footprint.
  Keep `docker stats` as a CPU-side secondary; also note engine self-reports (vLLM KV-cache size,
  llama.cpp model+KV buffers). Record which source the headline number used.
- **Input modalities** (`modalities`, `mm_served`): detect what the *model* accepts from the HF
  repo — `pipeline_tag`, `vision_config`/`audio_config`, `preprocessor_config.json` / AutoProcessor —
  mapped to {text, image, audio, video}. `mm_served: false` flags runs where the engine serves a
  multimodal model text-only (e.g. no `mmproj` for llama.cpp, no `--limit-mm-per-prompt` for vLLM).

## Results

Engine: llama.cpp `full-cuda` build `063d9c156 (9744)`, `-ngl 99`, llama-bench `-p 512 -n 128`.
Mem = system `MemAvailable` delta (cgroup/nvidia-smi unusable — see methodology).

| Date | Model | Quant/Precision | Engine | Ctx | Prefill tok/s | Decode tok/s | Mem | Full command | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-21 | SmolLM3-3B | Q4_K_M | llama.cpp | pp512/tg128 | 7214.5 | 105.7 | 2.94 GiB | `--bench -m SmolLM3-Q4_K_M.gguf -p 512 -n 128 -ngl 99` | smoke test ✅ |
| 2026-06-21 | SmolLM3-3B | Q8_0 | llama.cpp | pp512/tg128 | 6391.1 | 70.6 | 4.18 GiB | `--bench -m SmolLM3-Q8_0.gguf -p 512 -n 128 -ngl 99` | decode bw-bound vs Q4 |
