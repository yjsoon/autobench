# CLAUDE.md — orientation for a future agent

This repo is **autobench**: a harness + results website for benchmarking open-weight LLMs
on a single **NVIDIA DGX Spark**. Read this first; live state lives in the `_configs/` pages
(per-run results) and the homepage listing — there is no separate journal.

## The mission

Given a list of models, **one at a time**: download → run across engine / quant / context
configurations → record the attributes that matter, above all the
**full run command · context window · tok/s (prefill + decode) · peak memory**.
Take notes the entire time. The notes ARE the deliverable.

**Experiments are conducted by an Opus-class agent.** Choosing between quants, sourcing trusted
HF repos, and recovering from imprecise/unverified model-list entries (wrong repo IDs, renamed or
gated models, ambiguous sizes) all require judgment — do not downgrade this run loop to a smaller model.

## The machine (`heruli`)

- DGX Spark, **GB10 Grace-Blackwell**, **ARM64 (aarch64)**, Ubuntu 24.04, CUDA toolkit 13.
- **121 GB unified memory** (shared CPU/GPU) — this is the hard ceiling that decides fit.
- 20 cores, ~3.5 TB free on `/`. Internet works.
- Verify GPU with `nvidia-smi` (should show `GPU 0: NVIDIA GB10`).

### Environment gotchas (learned the hard way)
- **sudo is password-gated** — you cannot run sudo non-interactively. If a step needs it,
  hand the user the exact command. (Driver fix + docker-group add were both done this way.)
- The GPU driver once broke on a **kernel/module version mismatch** (running kernel newer
  than the installed `linux-modules-nvidia-580-open`). If `nvidia-smi` fails again, check
  `uname -r` vs the installed `linux-modules-nvidia-580-open-*` package; fix with
  `sudo apt-get install -y linux-modules-nvidia-580-open-$(uname -r)` then `sudo modprobe nvidia`
  (or reboot). Docker-group fix was `sudo usermod -aG docker $USER`.
- The user is in the `docker` group; **docker works without sudo**. All engines run as
  **NVIDIA NGC containers** (the decided runtime). Use `nvidia-ctk` / `--gpus all`.
- ARM64 matters: many images/wheels are x86-only. Pick `arm64`/`sbsa` image tags.
- **`nvidia-smi` reports N/A for all GPU memory** (unified with system RAM), and cgroup/`docker stats`
  *undercounts* it — CUDA's unified allocations skip the memory cgroup (saw 0.6 GiB cgroup vs 2.9 GiB
  real for a 1.78 GiB model). Headline memory = **system `MemAvailable` delta** from an idle baseline,
  10 s sampling. NOT nvidia-smi, NOT cgroup.

## Engines to benchmark (decided)

**llama.cpp**, **vLLM**, **TensorRT-LLM** — all via NGC containers. Per model, sweep
quant/precision (e.g. GGUF Q4_K_M/Q8; FP8/AWQ/MXFP4/NVFP4/BF16), context length, and
concurrency. Verify exact HF repo IDs at download time — the model list has unverified 2026 names.

- **llama.cpp image gotcha:** `ghcr.io/ggml-org/llama.cpp:full-cuda` (ARM64) uses a dispatcher
  entrypoint — run llama-bench via `--bench`, the server via `--server` (not the bare binary names).
  Reusable wrapper: `scripts/bench-llamacpp.sh` (runs `--bench`, samples memory at 10 s, parses tok/s).

## Benchmarking policy

- **Run order:** generally do the **headliners first** (the `Spark recipe` / best-documented
  showcases — gpt-oss, the Nemotron NVFP4 models), then work through the rest **small → big**, so
  cheap fast runs validate each engine/path before the heavy giants.
- **Record anything interesting in the per-config Notes** (the markdown body of the `_configs/*.md`
  page, rendered as the "Notes" section). When a run surprises you — a context length that fits or
  OOMs, the memory cliff where it stops fitting, an unusually good/bad quant, throughput that defies
  the bandwidth estimate, a format/dispatcher gotcha, multimodal quirks — write it there. The user
  reviews these manually, so prefer over-noting to losing the observation.
- **Extra quants welcome.** Beyond the one stub config per model, add more quants whenever useful —
  but only from a **trusted HF repo** (the model's own org, or a well-known quantizer like
  ggml-org / unsloth / the lab's official quant). For every config record **`quant_rationale`**
  (why this quant) and **`download_url`** + `source_repo` (its HF page).
- **When unsure, BLOCK — don't guess.** If you're not confident a quant is safe/legit to run
  (sketchy/unofficial repo, unclear license, can't confirm the format fits), set
  **`status: blocked`** instead of running it; the user reviews blocked items later.
- **Status values:** `pending` → `blocked` (needs human review) / `done`. Listing sorts
  done (newest first) → pending → blocked (last).
- **Compound dimensions — that's the point.** Wherever a model supports it, benchmark the
  cross-product as separate configs: engine × quant × speculative-decoding (e.g. gpt-oss-20b →
  vLLM, SGLang, SGLang+EAGLE3, TRT-LLM). More comparable data points is the goal, not one config
  per model.
- **Engine selection per model.** When loading a new model to benchmark, first check current
  sources (NVIDIA Dev Forums, `build.nvidia.com/spark`, the engine repos/discussions) for which
  serving stack is claimed **highest-performance for THAT model on GB10/Spark**. If one is the
  obvious winner, use it; if it's close, **benchmark the top two**. Briefly note the decision + why
  (with a source) on the config page. The stub's engine/quant is only a starting guess — override it
  per this check. (This is what surfaces SGLang where it actually wins, rather than defaulting to vLLM.)
- **TensorRT-LLM build time.** TRT-LLM compiles a per-model engine before it can serve — **record
  that compilation / engine-build time on the model page.** It's a real cost of the path (often the
  reason to prefer SGLang/vLLM) and belongs in the comparison.
- **NIM: blocked for now.** Create NIM configs as `status: blocked` — the NGC NIM containers are
  gated and no API key is available yet. Revisit when access is provided.
- **Workload dataset:** benchmark against the real dataset in **`./benchmark_data/`** —
  `ShareGPT_V3_unfiltered_cleaned_split.json` (ShareGPT V3, 94k conversations, ~642 MB,
  **gitignored — never commit it**), not synthetic fixed-length tokens. Feed it to each engine's
  serving benchmark: vLLM `benchmark_serving.py --dataset-path …`, SGLang
  `bench_serving --dataset-name sharegpt --dataset-path …`, TRT-LLM `trtllm-bench --dataset …`.
  llama.cpp's `llama-bench` takes no dataset, so for llama.cpp run `llama-server` + a serving
  benchmark against it (or derive representative input/output lengths from ShareGPT). Always record
  concurrency. (This supersedes the synthetic pp512/tg128 used for the SmolLM3 smoke test.)
- **Speculative decoding:** where a model ships an **MTP** module (e.g. DeepSeek) or an
  **EAGLE/EAGLE3/Medusa** draft is available for the engine, benchmark a **separate config with
  speculation enabled IN ADDITION to the base (non-spec) config** — never replace it. Record the
  method in a `speculative:` field (e.g. `EAGLE3`, `MTP`) and the decode-tok/s speedup vs the base
  run in Notes. SGLang (EAGLE3) and vLLM/TRT-LLM (MTP) are the usual paths. The `speculative:` value
  is **folded into the Engine display** (e.g. `SGLang + EAGLE3`) on the page and listing — it is NOT
  a separate table column.
- **Benchmark measures throughput, not accuracy** (tok/s + concurrency), per the speed-only scope.

## What exists vs. what's left

- ✅ **Website** (this repo) — see below. Builds clean; deploys via CI.
- ✅ `notes/MODELLIST.md` — ~42 candidate models, tiered by Spark fit, smoke-test-first order.
- ❌ **The actual run-and-measure harness is NOT built yet.** Next big task: per model,
  pull it, launch each engine container, run a fixed prompt, parse prefill/decode tok/s +
  ctx + peak mem, write a `_configs/*.md` page (use `scripts/new-config.sh`) and set
  `status: done` + `completed_at`, tear down, move on. Smoke-test on a tiny model first
  (SmolLM3-3B / Phi-4-mini) before the big runs. SmolLM3-3B is already done (Q4_K_M + Q8_0).

## The website (Jekyll)

**Ruby lives only in Docker — never install Ruby on the host.**

| Command | What |
|---|---|
| `./serve.sh` | Live-reload preview at http://localhost:4000 |
| `./build.sh` | One-off static build into `./_site` (run before pushing to catch Liquid errors) |

Structure:
- `_configs/` — the collection; **one markdown page per configuration** (model × engine ×
  quant). Front matter: `model/company/family/params/engine/speculative/quant/quant_rationale/
  source_repo/download_url/context/modalities/tags` plus results (`prefill_toks/decode_toks/mem_gb/
  mem_source/run_command/completed_at`) and `status: pending | blocked | done`.
  Layout: `_layouts/config.html`.
- `index.md` — project explainer + live config listing (homepage).
- `tags.md` — pure-Liquid tag browser (no plugins, so it works on GitHub Pages). Don't add
  plugins that aren't Pages-safe unless CI builds with `ruby/setup-ruby` (it does — so 4.x +
  any plugin is fine; just keep tags.md plugin-free as-is).
- **Tag taxonomy = exactly 5 categories**, nothing else: lab · family · quant · size-bucket
  (`≤4B`/`5-15B`/`16-40B`/`41-130B`/`130B+`, by total params) · `Spark recipe` (native DGX Spark
  support). Engine is a field/column, NOT a tag. `scripts/seed-stubs.sh` applies this when
  generating stub pages — keep new pages consistent.
- Done configs carry `completed_at` (date+time); shown as "Completed" on the page and in the listing.
  The homepage sorts done (newest first) → pending → blocked (last).
- `scripts/new-config.sh` — generates a config page from CLI flags (the harness should call this).

### Deploy
- Push to `main` → `.github/workflows/jekyll.yml` builds with `ruby/setup-ruby` + `jekyll build`
  and deploys to **GitHub Pages**. Live at **https://gauravmm.github.io/autobench**.
- `_config.yml` has `url`/`baseurl` set for that project path. Pages source = GitHub Actions (enabled).
- **Gotcha:** `Gemfile.lock` must list `x86_64-linux` (CI runner) AND `aarch64-linux-gnu`
  (this host), else `setup-ruby` fails. Regenerate platforms with
  `docker run --rm -v "$PWD":/site autobench-site bundle lock --add-platform x86_64-linux ruby`.

### Git / auth
- Remote is **SSH**: `git@github.com:gauravmm/autobench.git`, authenticated by a repo
  **deploy key** at `.ssh-key/id_ed25519` (gitignored, no passphrase). `core.sshCommand` is
  configured to use it — normal `git push` just works.
- Commit/push after every new benchmark.

## Working style notes
- The user moves fast and issues rapid directives; keep momentum, validate before pushing.
- `run.sh` launches Claude with `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=20` — an aggressive
  autocompact threshold that **compacts context earlier** to keep the working context small
  during long benchmarking sessions. Expect frequent compaction; rely on this file + the
  `_configs/` pages as durable memory rather than in-context history.
- No separate journal/narrative (dropped by request): results live in the per-config pages,
  decisions/environment/methodology live here. Update those, not a chronological log.
