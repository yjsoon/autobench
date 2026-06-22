# CLAUDE.md — orientation for a future agent

This repo is **autobench**: a harness + results website for benchmarking open-weight LLMs
on a single **NVIDIA DGX Spark**. Read this first; live state lives in the `_configs/` pages
(per-run results) and the homepage listing — there is no separate journal.

**The model list IS the `_configs/` collection.** One stub page per model × engine × quant is
the source of truth for what to benchmark — there is no separate model-list file. (The old
`notes/MODELLIST.md` was removed; don't recreate or rely on it.)

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
- **Credentials live in `.env`** (gitignored): `NIM_KEY` (NGC `nvapi-…`, for `nvcr.io` + NIM)
  and `HF_TOKEN` (for gated HF downloads). Load with `set -a; source .env; set +a`. Never echo or
  commit the values. `docker login nvcr.io` was run with `NIM_KEY` (succeeds).
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
  Reusable wrappers: `scripts/bench-llamacpp.sh` (synthetic `--bench`, superseded) and
  `scripts/bench-llamacpp-serving.sh` (launches `--server` + drives `scripts/bench-serving.py`
  against ShareGPT — the current path). `bench-serving.py` is engine-agnostic (any OpenAI
  `/v1/chat/completions` endpoint: llama-server, vLLM, SGLang, TRT-LLM) and counts a streamed
  `{"error":...}` chunk as a failed request (not a 0-token success).
- **Engine wrappers (all drive `bench-serving.py`):** `scripts/bench-sglang-serving.sh`
  (`lmsysorg/sglang:spark`, port 30000) and `scripts/bench-vllm-serving.sh`
  (**`vllm/vllm-openai:cu130-nightly`** — the image NVIDIA/vLLM document for DGX Spark, port 8000,
  `--gpu-memory-utilization 0.85 --max-num-seqs=<conc>`). Both pre-reserve a static KV fraction, so
  their `mem_gb` is a reservation, not the model footprint — dig the resident breakdown out of the
  engine logs for the Notes. **vLLM is the documented path for the Nemotron NVFP4 headliners**
  (`--trust-remote-code --reasoning-parser nemotron_v3`); the cu130-nightly serves
  Nemotron-3-Super-120B-A12B-NVFP4 fine (an *older* NGC vLLM container rejected its
  `quant_algo: MIXED_PRECISION` against a hardcoded whitelist — use cu130-nightly, not NGC, for the
  NVFP4 Nemotrons).
- **Mistral native-format + MLA gotcha (Mistral-Small-4-119B-NVFP4):** the official NVFP4 repo
  (`mistralai/Mistral-Small-4-119B-2603-NVFP4`) ships **Mistral native format** (`params.json`,
  `consolidated-*.safetensors`, `tekken.json` — no `config.json`), so vLLM needs
  `--tokenizer-mode mistral --config-format mistral --load-format mistral`. The model is an **MLA**
  (DeepSeek-V2-style) + Pixtral-vision arch, and vLLM's **Triton MLA decode kernel fails to compile
  on GB10** (`triton_decode_attention` → `Cannot make_shape_compatible: incompatible dimensions
  256 and 512`) — every request kills EngineCore. Fix: **`VLLM_MLA_DISABLE=1`** (use standard
  attention, bypassing the broken kernel) **+ `--gpu-memory-utilization 0.90`** (MLA-disabled
  materializes full KV, so 0.85 under-sizes the pool for 65536 ctx). `FLASHINFER_MLA` override is
  *ignored* (falls back to Triton MLA) — disabling MLA is the only path that works on this build.
- **gpt-oss + llama.cpp = blocked (harmony):** llama-server's OpenAI chat endpoint can't parse
  gpt-oss's **harmony** channel output on build `b9744` — every `/v1/chat/completions` request 500s
  with "does not match the expected peg-native format" (raw `/completion` works, `--jinja` /
  `--reasoning-format none` don't help). Benchmark gpt-oss on **vLLM/SGLang** (its recommended
  Spark engines) instead. Other (non-harmony) models serve fine on llama.cpp.
- **gpt-oss + vLLM = harmony tiktoken-vocab egress gotcha:** at startup vLLM's `OpenAIServingResponses`
  loads the **harmony** encoding, whose rust `openai_harmony` lib downloads `o200k_base.tiktoken`
  (sha256 `446a9538…`) from `$TIKTOKEN_ENCODINGS_BASE` (default = openaipublic blob). The HF *model*
  download already used the run's **one** allowed external connection, so this **second** fetch is
  egress-capped → `HarmonyError: failed to download or load vocab file` and the server never comes up.
  Fix (baked into `scripts/bench-vllm-serving.sh`): pre-seed the vocab at
  `~/models/tiktoken_cache/o200k_base.tiktoken`, mount it `-v …:/vocab:ro`, and set
  **`TIKTOKEN_ENCODINGS_BASE=/vocab`** (a **plain path**, NOT `file:///vocab` — the rust file:// parser
  fails with "No such file or directory"). No second egress then needed. Same applies to any
  harmony/gpt-oss model on SGLang.
- **Gemma-4 MTP on vLLM = BLOCKED on GB10 (heterogeneous-head + TRITON_ATTN), despite the recipe.**
  vLLM *does* now ship native gemma-4 MTP (`--speculative-config '{"method":"mtp","model":"google/gemma-4-E<size>-it-assistant","num_speculative_tokens":N}'`)
  — so the old "vLLM rejects gemma spec-decode" claim is only **half** true. Two walls, in order: (1) the
  drafter's `gemma4_assistant` arch needs **transformers ≥ ~5.12**; stock `cu130-nightly` (vLLM
  0.19.2rc1 / tf 5.6.0) rejects it at config-validation. Built **`scripts/Dockerfile.vllm-022-tf`**
  (`FROM autobench-vllm-cu130-022` = vLLM **0.22.0** + `pip install -U transformers` → **5.12.1**, image
  tag **`autobench-vllm-022-tf`**, pass via `VLLM_IMAGE=`) to clear it. (2) Past that, EngineCore dies in
  KV-cache profiling: `triton_attn.py get_num_attention_heads_from_layers` asserts uniform `num_heads`
  per attention group, but Gemma-4 (E4B) has **mixed `{8,4}` heads** — the same heterogeneous-head
  property (head_dim 256/512) that *force-pins* it to TRITON_ATTN. With MTP's draft layer the builder
  groups differing head counts → `AssertionError`. **Not flag-fixable** (backend is forced;
  `num_speculative_tokens=1`/`enforce_eager` don't change grouping); **independent of base quant** (kills
  FP8+MTP and NVFP4+MTP alike). **Working Gemma-4 MTP path = llama.cpp** (`--spec-type draft-mtp`, unsloth
  merged-GGUF drafter; E-series also needs `-fa off`). See the `gemma-4-e4b-it-vllm-*-mtp` config pages.
- **NVFP4 (W4A4) is a genuine GB10 fast-path even for tiny models** — cosmicproc's `gemma-4-E4B-it-NVFP4`
  (ModelOpt, `FlashInferCutlassNvFp4LinearKernel`) beat FP8 on the same E4B by ~23% decode (the
  benchmark's fastest decode). Worth trying NVFP4 alongside FP8 wherever a trusted NVFP4 exists. (Trust:
  cosmicproc is an *individual* uploader — normally a BLOCK per the trusted-repo policy, run here on the
  user's explicit request; 61k dl/mo + a clean error-free result corroborated it.)
- **Transient EngineCore spawn deadlock on first vLLM launch — kill & retry.** Seen twice: a fresh
  `vllm serve` hangs at init (log frozen at `Enabled custom fusions: act_quant`, **no** EngineCore
  startup line, container ~0.2% CPU / weights never loaded). It's a **spawn race, not an
  incompatibility** — a clean `docker rm -f` + relaunch loads fine. Distinguish from a *real* failure
  (which prints a Python traceback / non-zero exit and reproduces every time): the deadlock is silent and
  clears on retry. Don't mark a config blocked on the first hang.

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
- **NIM: now available.** An NGC key is in `.env` as `NIM_KEY` (verified 2026-06-21: `nvapi-…`,
  `docker login nvcr.io -u '$oauthtoken' --password-stdin` → Login Succeeded). NIM configs may now
  be **run** (not auto-blocked) — pull `nvcr.io/nim/...` images and pass the key as `NGC_API_KEY`.
  Only block a specific NIM config if that model isn't in the NIM catalog / can't be served.
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
  - **ALWAYS capture the draft acceptance rate for spec runs.** The serving wrappers print a
    `==> SPEC-METRICS (acceptance):` block (grepped from the engine container's log **before** teardown)
    whenever a speculative flag is present — vLLM logs mean acceptance length / per-position acceptance,
    llama.cpp logs `n_draft`/`n_accept`, SGLang logs accept length. **Record the acceptance rate (and
    mean accepted tokens/step) on the config page** alongside the speedup — it's what explains *why* a
    spec run helped or hurt (e.g. gpt-oss-120b EAGLE3 went net-negative at conc 32: low acceptance +
    compute-saturated batch). If the grep catches nothing, note the metric wasn't emitted rather than
    omitting it silently.
  - **Cross-check the acceptance rate against the published/expected value, and flag any gap.** Before
    accepting a spec result, look up what the model's drafter is *supposed* to achieve (the lab/unsloth
    blog, vLLM recipe, the EAGLE3/MTP paper) and compare. Rules of thumb: well-matched **MTP/EAGLE3
    drafts land ~70–85% draft acceptance** (often >80% on *coding* workloads; **ShareGPT general chat
    runs lower**, e.g. Qwen3.6-27B MTP measured **67%** here vs ~80%+ reported on code), **mean
    acceptance length ≈ 3.0** for `num_speculative_tokens=3`; off-the-shelf separate draft models
    usually <50%. **Acceptance is workload-driven, NOT strongly concurrency-driven** — it should stay
    roughly constant across conc 1/8/32; what changes with concurrency is whether the *speedup*
    materializes (bigger at low batch). So if a measured acceptance is **far from expectation**
    (e.g. an MTP head at 30%, or acceptance that swings wildly with concurrency), that's a **red flag**
    — likely a misconfigured method name, a base/draft mismatch, or a quant/format problem — and should
    be investigated and noted on the config page, not silently recorded.
- **Benchmark measures throughput, not accuracy** (tok/s + concurrency), per the speed-only scope.
- **Per-run runtime cap:** target **~1000 entries or 15 minutes of wall-clock, whichever is
  shorter**, for each config's serving benchmark. Cap the ShareGPT prompt count (`--num-prompts`
  / equivalent) at ~1000 and stop a run that crosses 15 min — enough to get a stable tok/s without
  letting any single config dominate the session. Note in the config if a run hit the time cap
  before the entry cap (signals a slow path worth flagging).

## What exists vs. what's left

- ✅ **Website** (this repo) — see below. Builds clean; deploys via CI.
- ✅ **The candidate models** live as stub `_configs/*.md` pages (the model list — see top of file).
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
- `_archive/configs/` — **retired config pages that should no longer render.** To pull a model out of
  the site without losing its data (or to delete clutter), `git mv` its `_configs/*.md` here; `_archive/`
  is underscore-prefixed AND listed in `_config.yml` `exclude`, so Jekyll ignores it. (Done = preserved
  but hidden. True deletion is `git rm`.)
- **`tags-model.md` / `tags-family.md` fold low-count groups into a trailing "Other" section** (via the
  Liquid `concat` filter — accumulate single-config groups into one array, render multi-config groups in
  their own `<h2>`, then one "Other" `<h2>` at the end). Thresholds differ on purpose: model folds only
  true one-offs (`group.size == 1`); family folds `<= 3`. Keep both pure-Liquid (Pages-safe).
- `index.md` — project explainer + live config listing (homepage).
- `tags.md` — pure-Liquid tag browser (no plugins, so it works on GitHub Pages). Don't add
  plugins that aren't Pages-safe unless CI builds with `ruby/setup-ruby` (it does — so 4.x +
  any plugin is fine; just keep tags.md plugin-free as-is).
- **Tag taxonomy = 7 categories**, nothing else: **concurrency** (a `conc-N` tag from the
  `concurrency:` field, e.g. `conc-32` — the parallel-serve count; lets low-conc variants like
  `conc-1` group separately) · **model** (a per-model slug, e.g. `gemma-4-31b`,
  derived from the canonical `model:` field — basename, lowercased, with `-it`/`-instruct`/format/quant/
  date suffixes stripped; **every run of one model shares it**, grouping all its engine/quant/spec
  variants) · lab · family · quant · size-bucket (`≤4B`/`5-15B`/`16-40B`/`41-130B`/`130B+`, by total
  params) · `Spark recipe` (native DGX Spark support). Engine is a field/column, NOT a tag. Keep the
  `model:` field identical across all configs of one model so the model tag groups them. Put the model
  slug FIRST in the `tags:` list. `scripts/seed-stubs.sh` applies the taxonomy when generating stubs —
  keep new pages consistent.
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
- **Check the system time immediately before writing any timestamp into a document.** Run
  `date "+%Y-%m-%d %H:%M %z"` as part of the same step that writes a config's `completed_at` (or any
  dated field) — never reuse a timestamp captured earlier in the pipeline. Results land minutes after
  the last `date` read (downloads, model load, the up-to-15-min run), so a stale value drifts.
- `run.sh` launches Claude with `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=20` — an aggressive
  autocompact threshold that **compacts context earlier** to keep the working context small
  during long benchmarking sessions. Expect frequent compaction; rely on this file + the
  `_configs/` pages as durable memory rather than in-context history.
- No separate journal/narrative (dropped by request): results live in the per-config pages,
  decisions/environment/methodology live here. Update those, not a chronological log.
