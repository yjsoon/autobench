# Speculative-decoding post — experiments still needed

Benchmark runs to fill the data gaps behind
`posts/2026-07-01-speculative-decoders-are-all-you-need.md`. Each item is self-contained for handoff.

**Conventions (read first):**

- Follow `notes/BENCHMARKING.md` (wrappers, ShareGPT V3 dataset, run/time caps, `engine_image` digest
  capture) and check `notes/INCOMPATIBILITIES.md` before debugging a launch.
- Every spec run: **capture `spec_acceptance`** (avg draft acceptance + mean accept-len + per-position)
  and cross-check against the published expectation; flag a wild miss.
- One config page per run under `_configs/`; set `status`, fill results, `git mv` dead runs to `_archive/`.
- **Read `date "+%Y-%m-%d %H:%M %z"` immediately before writing `completed_at`.**
- After each new run, regenerate the affected chart: `./posts/assets/make_plots.py` and update the CSV's
  `source_url` column to the new config page.

Priority: **P0** = unblocks a chart/claim already in the post · **P1** = closes a comparison gap ·
**P2** = nice-to-have / watch-list.

---

## P0 — Complete the DFlash line on the money chart

The concurrency chart (`posts/assets/plots/mtp_vs_dflash_35b.svg`) currently carries a footnote:
DFlash landed only at conc-1/2/4/16 **and at ctx 40960 vs MTP/base's 65536**. The base and MTP lines are
complete (1→32); DFlash is the ragged one.

- **Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4` · **engine:** vLLM (aeon-ultimate DFlash container, pinned
  `@31977fbe` small-page drafter rev, `num_speculative_tokens 11`) · **quant:** NVFP4.
- **Runs:** conc **8** and **32** (the missing endpoints) — plus **re-run 1/2/4/16 at ctx 65536** so the
  whole line is matched to base/MTP. Mixed context windows are a confound, not just at conc-1.
- **Config slugs:** `qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash-c8`, `-c32`; overwrite the existing
  1/2/4/16 pages' `context: 65536`.
- **Unblocks:** removes the chart footnote and the "4→16 interpolated" caveat — a fully-sampled,
  matched base-vs-MTP-vs-DFlash curve.

## P1 — gpt-oss-20b: pair the EAGLE3 curve with a complete base

EAGLE3 has the full ladder (1→32); the no-spec base is missing conc-1 and conc-8, so the
"EAGLE3 is *inverted* — a loss at low batch that only turns positive at high batch" claim (§ performance
trade-off) has no gap-free base to plot against.

- **Model:** `openai/gpt-oss-20b` · **engine:** vLLM · **quant:** MXFP4 · no speculation.
- **Runs:** conc **1** and **8**.
- **Config slugs:** `gpt-oss-20b-vllm-mxfp4-c1`, `-c8` (match the existing 2/4/16/32 base pages).
- **Unblocks:** a fully-paired base-vs-EAGLE3 concurrency chart showing the low-batch pathology cleanly.

## P1 — Qwen3.6-27B: complete the MTP ladder for a sibling curve

27B NVFP4 **base** is fully sampled (1→32) but **MTP** has only 1/8/32. The "MTP on the 27B decays
smoothly (+81% → +63% → +46%)" fingerprint claim would be a second complete base-vs-MTP curve alongside
the 35B chart if the intermediate points existed.

- **Model:** `Qwen/Qwen3.6-27B` · **engine:** vLLM · **quant:** NVFP4 · spec: MTP.
- **Runs:** conc **2, 4, 16**.
- **Config slugs:** `qwen3-6-27b-nvfp4-vllm-mtp-c2`, `-c4`, `-c16`.
- **Unblocks:** a 27B companion to the 35B money chart; firms up the "smooth decay" claim.

## P2 — Per-position acceptance for DFlash n=11

The acceptance bar-chart TODO (§ Agreement improves performance) wants the DFlash-n11 decay tail
(0.84 → ~0.05) beside the MTP triple (0.84/0.66/0.51). MTP's per-position triple is logged and reproduces;
confirm the **DFlash per-position vector** is captured in `spec_acceptance` on the 35B DFlash pages (it may
only have mean accept-len). If absent, re-parse the run logs or add a short re-run with per-position logging.

- **Unblocks:** the `TODO(graphic): per-position acceptance bar chart` — currently prose-only.

## P2 — Matched-cap recheck provenance (watch-list)

The dropped § performance-trade-off table (+2.9% / −11.2% / −25.9% DFlash-vs-MTP) came from a 600 s-cap
**matched MTP recheck** that was never persisted as config pages — which is why headline conc-1 reads +8.5%
here, not +2.9%. The table is gone, so this is low priority, but if that head-to-head returns, **store the
recheck runs as `_configs/` pages** so every number in the post traces to a page (per `posts/CLAUDE.md`).
