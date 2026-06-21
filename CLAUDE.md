# CLAUDE.md — orientation for a future agent

This repo is **autobench**: a harness + results website for benchmarking open-weight LLMs
on a single **NVIDIA DGX Spark**. Read this first, then `notes/JOURNAL.md` for live state.

## The mission

Given a list of models, **one at a time**: download → run across engine / quant / context
configurations → record the attributes that matter, above all the
**full run command · context window · tok/s (prefill + decode) · peak memory**.
Take notes the entire time. The notes ARE the deliverable.

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
  `uname -r` vs the installed `linux-modules-nvidia-580-open-*` package — see JOURNAL blocker #1.
- The user is in the `docker` group; **docker works without sudo**. All engines run as
  **NVIDIA NGC containers** (the decided runtime). Use `nvidia-ctk` / `--gpus all`.
- ARM64 matters: many images/wheels are x86-only. Pick `arm64`/`sbsa` image tags.
- **`nvidia-smi` reports N/A for all GPU memory** (unified with system RAM). Measure peak via the
  container's cgroup-v2 `memory.peak` + system `MemAvailable` delta — NOT nvidia-smi. Sampler: 10 s.

## Engines to benchmark (decided)

**llama.cpp**, **vLLM**, **TensorRT-LLM** — all via NGC containers. Per model, sweep
quant/precision (e.g. GGUF Q4_K_M/Q8; FP8/AWQ/MXFP4/NVFP4/BF16), context length, and
concurrency. Verify exact HF repo IDs at download time — the model list has unverified 2026 names.

## What exists vs. what's left

- ✅ **Website** (this repo) — see below. Builds clean; deploys via CI.
- ✅ `notes/MODELLIST.md` — ~42 candidate models, tiered by Spark fit, smoke-test-first order.
- ❌ **The actual run-and-measure harness is NOT built yet.** Next big task: per model,
  pull it, launch each engine container, run a fixed prompt, parse prefill/decode tok/s +
  ctx + peak mem, write a `_configs/*.md` page (use `scripts/new-config.sh`), append a row
  to the JOURNAL results table, tear down, move on. Smoke-test on a tiny model first
  (SmolLM3-3B / Phi-4-mini) before the big runs.

## The website (Jekyll)

**Ruby lives only in Docker — never install Ruby on the host.**

| Command | What |
|---|---|
| `./serve.sh` | Live-reload preview at http://localhost:4000 |
| `./build.sh` | One-off static build into `./_site` (run before pushing to catch Liquid errors) |

Structure:
- `_configs/` — the collection; **one markdown page per configuration** (model × engine ×
  quant). Front matter carries `model/company/family/params/engine/quant/context/tags` plus
  results (`prefill_toks/decode_toks/mem_gb/run_command/status`). Layout: `_layouts/config.html`.
- `index.md` — project explainer + live config listing (homepage).
- `tags.md` — pure-Liquid tag browser (no plugins, so it works on GitHub Pages). Don't add
  plugins that aren't Pages-safe unless CI builds with `ruby/setup-ruby` (it does — so 4.x +
  any plugin is fine; just keep tags.md plugin-free as-is).
- **Tag taxonomy = exactly 5 categories**, nothing else: lab · family · quant · size-bucket
  (`≤4B`/`5-15B`/`16-40B`/`41-130B`/`>130B`, by total params) · `Spark recipe` (native DGX Spark
  support). Engine is a field/column, NOT a tag. `scripts/seed-stubs.sh` applies this when
  generating stub pages — keep new pages consistent.
- Done configs carry `completed_at` (date+time); shown as "Completed" on the page and in the listing.
- `notes/JOURNAL.md` — the journal; force-included via `_config.yml` `include:` and rendered
  at **`/journal/`** even though `notes/` is otherwise excluded.
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
- Commit/push only when the user asks. Branch is `main`.

## Working style notes
- The user moves fast and issues rapid directives; keep momentum, validate before pushing.
- `run.sh` launches Claude with `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=20` — an aggressive
  autocompact threshold that **compacts context earlier** to keep the working context small
  during long benchmarking sessions. Expect frequent compaction; rely on JOURNAL.md +
  this file as durable memory rather than in-context history.
- Keep `notes/JOURNAL.md` updated as you go — it's the single source of truth for state.
