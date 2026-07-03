# Autobench (Strix Halo fork)

Auto-benchmark harness + results site for running LLMs on an **AMD Strix Halo** box
(GMKtec EVO X2, Ryzen AI Max+ 395, Radeon 8060S, 96 GiB UMA VRAM).

Forked from [gauravmm/autobench](https://github.com/gauravmm/autobench), which benchmarks the
**NVIDIA DGX Spark** — the existing `_configs/` results are Spark numbers, kept as the comparison
baseline. The scripts here are ported to Vulkan (llama.cpp, LM Studio) and ROCm (SGLang, vLLM);
see `CLAUDE.md` for the machine notes.

For each model we download it, run it across engines/quants/contexts, and record the
**full run command, context window, and tok/s (prefill + decode)**. Results are published
as a Jekyll site — one page per configuration, tagged by model family, company, engine, quant.

## Layout

| Path | Purpose |
|---|---|
| `CLAUDE.md` | Durable notes: environment, decisions, measurement methodology, gotchas. |
| `notes/MODELLIST.md` | Candidate models to benchmark. |
| `index.md` | Project explainer + live configuration listing (site homepage). |
| `_configs/` | One markdown page per benchmark configuration (the site collection). |
| `_layouts/`, `index.md`, `tags.md`, `assets/` | Jekyll site (config spec sheets + tag browser). |
| `scripts/new-config.sh` | Scaffold a new config page (used by the harness). |
| `scripts/seed-stubs.sh` | Seed pending stub pages for all runnable models in `MODELLIST.md`. |
| `scripts/bench-llamacpp.sh` | Run llama-bench (Vulkan container) + 10s memory sampling for one GGUF. |
| `scripts/bench-lmstudio-serving.sh` | Drive an already-running LM Studio server with the ShareGPT workload. |
| `Dockerfile.site`, `serve.sh`, `build.sh` | Dockerized Ruby/Jekyll — **no Ruby on the host**. |
| `.github/workflows/jekyll.yml` | Builds & deploys the site to GitHub Pages on push to `main`. |

## Local preview (Ruby stays in Docker)

```bash
./serve.sh     # builds the toolchain image, serves http://localhost:4000 with live reload
./build.sh     # one-off static build into ./_site
```

Requires Docker access (be in the `docker` group — see `NOTES.md` blocker #2).

## Publishing

Push to `main`; the Actions workflow builds with Jekyll and deploys to GitHub Pages.
Before the first deploy, set `url`/`baseurl` in `_config.yml` and enable
**Settings → Pages → Source: GitHub Actions** on the repo.

## Adding a configuration

```bash
scripts/new-config.sh --model openai/gpt-oss-20b --company OpenAI --family gpt-oss \
  --params 20.9B --engine vLLM --quant MXFP4 --context 131072 \
  --tags "gpt-oss,OpenAI,vLLM,MXFP4,MoE,20B"
```

Then run the benchmark and fill in `prefill_toks`, `decode_toks`, `mem_gb`, `run_command`,
and set `status: done`.
