# Speculator post — experiments still needed

Backlog of benchmark runs to fill the data gaps behind
`posts/2026-07-01-speculators-are-all-you-need.md`. Each item is self-contained for handoff.

**Conventions (read first):**
- Follow `notes/BENCHMARKING.md` (wrappers, ShareGPT dataset, ~1000-prompt / 15-min cap, `engine_image`
  digest capture) and check `notes/INCOMPATIBILITIES.md` before debugging a launch.
- Every spec run: **capture `spec_acceptance`** (avg draft acceptance + mean accept-len + per-position)
  and **cross-check vs the published expectation** — a wild miss is a red flag, note it.
- One config page per run under `_configs/`; set `status`, fill results, `git mv` to `_archive/` if dead.
- **Read `date "+%Y-%m-%d %H:%M %z"` immediately before writing `completed_at`.**

Priority: **P0** = unblocks a headline claim in the post · **P1** = closes a comparison gap · **P2** = nice-to-have / watch-list.

---

## P0 — DDTree on the Spark (the post's one un-measured method)

The whole "## The future" section is currently un-benchmarked. This is the missing data point.

1. **DDTree vs MTP vs single-line DFlash — Qwen3.6-35B-A3B, concurrency sweep.**
   - Target `Qwen/Qwen3.6-35B-A3B` (NVFP4), drafter `z-lab/Qwen3.6-35B-A3B-DFlash` — the *same* drafter
     already used in `qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash`.
   - DDTree is **not in vLLM/SGLang** (only SGLang discussion #24605), so use the research path:
     `z-lab/dflash` + the DDTree tree code, or `CaDDTree` (github.com/ZhangShuai1230/CaDDTree) /
     Tencent `AngelSlim`. Expect a PyTorch harness, not a serving container — record setup/build cost.
   - Measure accept-len + decode tok/s at **conc 1 / 8 / 32**. Compare head-to-head against the existing
     `-mtp` (541 tok/s @32) and `-ultimate-dflash` (401 @32) rows.
   - **Question to answer:** does the tree recover DFlash's under-load losses (§3c: +8.5%@c1 → −26%@c32)?
   - Watch: drafter revision / KV-page unify wall (same as DFlash — may need the small-page pin) and the
     ShareGPT-vs-coding acceptance gap. New page: `qwen3-6-35b-a3b-nvfp4-ddtree-c{1,8,32}`.

2. **DDTree — Qwen3.6-27B** with `z-lab/Qwen3.6-27B-DFlash`, same protocol, so the tree result exists for
   both the dense 27B and the MoE 35B (parallels the MTP pair already in the sweep). New:
   `qwen3-6-27b-nvfp4-ddtree-c{1,8,32}`.

3. **DDTree on a coding workload (HumanEval / code dataset).** The paper's 8.22× is coding-only; our runs
   are ShareGPT. Run one Qwen3.6 DDTree config on a code dataset to see if acceptance jumps toward the
   published number — validates the §1c/§3b "workload-driven" claim with the *tree* method specifically.

---

## P1 — close comparison gaps in the existing tables

4. **Matched base runs at conc-1 and conc-8** for the MTP pairs. Several MTP configs (`*-mtp-c1`,
   `*-mtp-c8`) have **no matched non-spec base**, so the post can only quote speedups at conc-32.
   Run bases for: Qwen3.6-27B NVFP4 & FP8, Qwen3.6-35B-A3B NVFP4 & FP8, at conc-1 and conc-8. Lets us
   draw the full MTP concurrency-crossover curve, not just the DFlash one.

5. **gpt-oss-120b EAGLE3 on vLLM with the LMSYS draft.** Today vLLM used the NVIDIA throughput draft
   (−45%) and SGLang used the LMSYS/SpecForge draft (+22%) — draft *and* engine both differ, so the
   "draft match dominates" claim (§3b) is confounded. Run **vLLM + `lmsys/EAGLE3-gpt-oss-120b-bf16`** (or
   whichever LMSYS draft vLLM accepts) at conc 1/8/32 to isolate draft-vs-engine. This is the cleanest
   possible test of the post's single most persuasive point.

6. **Gemma-4 E4B llama.cpp MTP — matched `-fa off` base.** The current −36% (`gemma-4-e4b-it-llamacpp-mtp`
   vs `-q4_k_m`) is **confounded**: MTP forced `-fa off`, the base ran `-fa on`. Re-run the base with
   `-fa off` so the MTP delta is apples-to-apples. Until then the post flags it as not-a-real-regression;
   this makes it a real number. Update the existing `gemma-4-e4b-it-llamacpp-mtp` Notes.

7. **gpt-oss EAGLE3 on a coding dataset.** Notes explicitly say re-test on code before claiming any EAGLE3
   win for reasoning models. Run gpt-oss-20b (and 120b w/ LMSYS draft) EAGLE3 on HumanEval-style prompts;
   expect acceptance to rise from the ShareGPT ~5–30% toward the ~70% band. Confirms the workload thesis.

---

## P0.5 — fine-grained concurrency (conc 2 / 4 / 16) for the headline cases

The sweep is measured at conc **1 / 8 / 32** only, so every "wins single-stream, loses under load" claim
is drawn through *three* points. The whole §1b/§2d/§3c argument is really a **crossover-location** claim —
"somewhere between conc-1 and conc-8 the speedup goes negative." Conc **2/4/16** pin that crossover and
confirm monotonicity, turning the money chart from a 3-dot sketch into a curve. Only run these for the
cases the post actually leans on. Keep everything else identical; vary only concurrency
(`--max-num-seqs`). Acceptance should stay ~flat across conc (workload-driven) — if it drifts, that's a
red flag worth noting, not smoothing over.

13. **Qwen3.6-35B-A3B — MTP vs DFlash @ conc 2/4/16 (THE money chart).** DFlash goes **+8.5% (c1) →
    −6.6% (c8) → −26% (c32)** vs MTP. The sign flip lives in **1→8**, so **conc-2 and conc-4 locate the
    exact crossover**; **conc-16** traces the c8→c32 collapse. Run both `qwen3-6-35b-a3b-nvfp4-vllm-mtp`
    and `qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash` at 2/4/16 (add DDTree once P0 #1 lands → three lines).
    New pages: `*-mtp-c{2,4,16}`, `*-ultimate-dflash-c{2,4,16}`. **Highest-leverage item here** — it's the
    post's central figure.

14. **MTP speedup-decay curve — base vs MTP @ conc 2/4/16.** MTP's win shrinks as the batch fills and
    spare compute vanishes; today we only have the conc-32 delta for most pairs. Run base + MTP at 2/4/16
    for the two cleanest headliners — **Qwen3.6-27B NVFP4** (`-nvfp4-vllm` / `-nvfp4-vllm-mtp`) and
    **Qwen3.6-35B-A3B NVFP4** — to draw the full "speedup vs concurrency" curve. Pairs with P1 #4 (which
    adds the c1/c8 bases); together they give base+MTP at conc 1/2/4/8/16/32.

15. **gpt-oss-20b EAGLE3 — is the +28%@c32 really an artifact? @ conc 2/4/16.** The post claims the
    conc-32 win is a scheduling/prefill effect, not acceptance (which *degrades* ~30%@c1 → ~5%@c8). If
    that's right, decode-speedup vs base should be **negative or flat through c2/c4/c16 and only spike at
    c32** — a non-monotonic signature that proves the artifact. If instead it rises smoothly, the "artifact"
    framing is wrong and needs rewriting. Run base + eagle3 (`gpt-oss-20b-vllm-mxfp4` / `-eagle3`) at
    2/4/16. Low cost (small fast model), high evidentiary value for §3b.

16. **(contingent on P0) DDTree crossover @ conc 2/4/16.** Once #1 exists, add the intermediate points so
    the DDTree line has the same resolution as MTP/DFlash on the money chart — the key question is whether
    the tree pushes its own crossover *rightward* (stays positive to higher concurrency than single-line
    DFlash).

---

## P2 — retest-when-unblocked / watch-list

8. **Gemma-4 E4B NVFP4 + MTP** — currently BLOCKED by the image standoff (nightly-aarch64 runs MTP but
   regresses NVFP4 load; cu130-nightly loads NVFP4 but too old for the drafter arch). **Retest when a
   single vLLM image (0.23+) both loads Gemma-4 NVFP4 and recognizes the drafter arch.** Would complete
   the E4B quant×spec grid. Page exists: `gemma-4-e4b-it-vllm-nvfp4-mtp`.

9. **Qwen3.6-35B-A3B NVFP4 on SGLang (base + MTP)** — BLOCKED by the GatedDeltaNet 32-wide gate not being
   FP8-block-128 tileable. Retest if SGLang fixes GDN gate quant validation, or if a trusted alt-NVFP4
   packing (non-block-FP8 gates, like unsloth's 27B) appears for the 35B-A3B. Pages exist.

10. **DeepSeek-V4-Flash EAGLE3.1** — BLOCKED by the 168 GB fit wall (MIXED_PRECISION NVFP4 ≈ FP8-sized).
    Engine support already landed. **Watch for a true ~80 GB 4-bit build** (none is NVFP4 today); if one
    appears, this becomes the only DeepSeek MTP/EAGLE datapoint on Spark. Page: `deepseek-v4-flash-vllm-nvfp4-eagle3`.

11. **CaDDTree (cost-aware budget selection)** vs plain DDTree on a Qwen3.6 target — the paper claims
    CaDDTree matches oracle-budget DDTree. Only worth it after P0 lands and if DDTree shows a Spark win.

12. **DDTree for Gemma-4 — needs a drafter first.** No block-diffusion DFlash drafter exists for Gemma-4
    (its speculators are EAGLE3 heads). Out of scope for benchmarking; log here so the gap is explicit —
    would require training a Gemma-4 DFlash drafter before any DDTree run is possible.

---

## Graphics these experiments feed (see TODO(graphic) markers in the post)

- **§1b money chart** — decode tok/s vs conc {1,8,32}, lines for MTP / DFlash / **DDTree** (needs #1, #4).
- **§3b draft-match bars** — gpt-oss-120b −45% vs +22%, cleaned up by isolating draft-vs-engine (needs #5).
- **§2b per-position acceptance** — MTP-n3 vs DFlash-n11 vs **DDTree tree** (needs #1).
