# On Speculative Decoders

**TL;DR:** run native MTP. On one DGX Spark the best configs we measured are [Qwen3.6-35B-A3B NVFP4 + MTP on vLLM](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-35b-a3b) (541 tok/s aggregate decode at conc-32) and [Gemma-4-E4B FP8 + MTP on vLLM](https://gauravmm.github.io/autobench/tags/model/#gemma-4-e4b) (1262 tok/s — the fastest run in the whole sweep).

A lot of ink and bits get spilled speeding up *datacenter-scale* inference (batching, disaggregation, giant KV pools). Speculative decoding is the rare trick that is particularly effective at speeding up the *small, local, low-concurrency* case — the single RTX or Apple Silicon chip on your lap.

A **speculative decoder** (a.k.a. draft model, drafter, or speculator) is a small, cheap predictor that *guesses the next few tokens* the big model is about to produce. The big ("target") model then performs its decode pass, accepting guesses that match; the first disagreement is where normal decoding resumes. When the guesses are good, you get several tokens for the price of one.

This means that tokens where the speculative decoder agrees with the target model are nearly free, but there's a base cost to run the speculative decoder. In the rest of this article we'll explore this tradeoff and ground this in real data.

Speculative decoders work well when the next token is easy to guess. Something like this, with one obvious continuation, works well:

> The quick brown fox jumps *over the lazy dog.*

Something like this does not:

> I saw her duck *under the branch.* \
> I saw her duck *waddle away.*
>

- TODO(graphic): schematic — target model + speculative decoder both reading the same KV cache; speculative decoder emits k candidate tokens; target verifies in one pass; accepted prefix in green, first reject in red.

## The Options

Speculative decoding is a fast-moving field, so keep an eye out for new versions.

As of right now, there are three flavours that are common (and one emerging).

- **[MTP (multi-token prediction)](https://arxiv.org/abs/2404.19737):** extra prediction heads baked into the model. (DeepSeek, Qwen3.6, Gemma-4, etc.) The lightest-weight alternative.
- **[EAGLE3](https://arxiv.org/abs/2503.01840):** a *separate*, small draft head grafted into the model, reads activations at multiple levels to make its predictions. Quality depends entirely on *which* draft you load.
- **[DFlash](https://github.com/z-lab/dflash):** an external diffusion-based drafter that speculates *many* tokens per step (11 in our main configs, up to 16). High fixed cost, with the chance for huge speedups.
- **[DDTree](https://liranringel.github.io/ddtree/) (emerging):** DFlash with a tree instead of a single draft line. Only runs in a research harness so far (no serving engine yet), but it already turns block-diffusion from a single-stream loss into a win.

TODO: Some table showing the speedups with different models.

The numbers below come from 154 benchmark configs run on this box (138 completed; 68 of them speculative-decoding runs across MTP, EAGLE3, DFlash, and DDTree), plus 8 archived configs.

TODO: A short two sentences describing the setup. See the repo (link to it via public URL -- this may be published elsewhere)

> One box, ShareGPT V3, decode tok/s aggregate. "base" = matched non-spec run, same model/engine/quant/concurrency.

## The performance trade-off

The drafter runs first and proposes a short continuation — 3 tokens for MTP, 5-16 for DFlash depending on the config. This imposes a **fixed cost**, and buys a **variable speedup** depending on how many tokens are accepted. This only pays off when our spare compute and acceptance rates are both good enough.

Once the GPU is saturated, the drafter must compete with real requests for compute. Native MTP drafters are nearly-free and win at every measured concurrency. It's the heavy external drafters (DFlash) that *spend spare compute to buy low-concurrency throughput* — and a busy server has none to spend.

![Decode throughput vs concurrency for Qwen3.6-35B-A3B NVFP4 on vLLM, log-log, three lines: no-spec base, MTP, and DFlash. MTP leads at every concurrency (541 tok/s at conc-32); DFlash beats base only at low batch and collapses back toward the baseline by conc-16 (344 vs base 332).](assets/plots/mtp_vs_dflash_35b.svg)

Even with spare compute, the heavy drafter barely beats the built-in MTP; once the batch saturates the GPU it loses badly, even slipping below the no-drafter baseline (TODO: Check this). The trade-off curve belongs to the *drafter's cost*, not to speculation itself.

As with everything in LLMs, the exact tradeoff curve is a fingerprint of the method, not a universal law. Later on we'll discuss other trends that confound this simple rule.

### Agreement improves performance

In one target forward pass, the model computes the probability of each speculated token in parallel. Where the target agrees, we **pretend the token was there all along**; at the first disagreement we discard the rest and let the target generate that token normally, then re-speculate.

- **MTP** ≈ 3.0 ([Qwen3.6-27B](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-27b), [35B-A3B](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-35b-a3b)) — that's ~2 of 3 drafted tokens accepted plus the one free "bonus" token from the verify pass. The efficient end of the spectrum, though not lossless: about a third of drafted tokens are still thrown away.
- **EAGLE3** ≈ 2.0-2.4 of 3 ([Gemma-4](https://gauravmm.github.io/autobench/tags/model/#gemma-4-31b); [gpt-oss](https://gauravmm.github.io/autobench/tags/model/#gpt-oss-120b) only with a workload-matched draft — bad draft/engine combos measured as low as ~1.05) — a separate head, so lower than native MTP.
- **DFlash** ≈ 3.2-4.4 of **11** ([Qwen3.6-35B-A3B](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-35b-a3b)) — collapses after a few positions, leading to much waste.

- TODO(graphic): per-position acceptance bar chart (0.84 / 0.66 / 0.51 for MTP-n3 vs the long DFlash-n11 tail decaying to ~0.05). Shows *why* short high-acceptance drafts beat long ones.

### Drafters are brittle

Drafters are being judged by the target model's token choices, and are fed by the target model's own KV cache. That's what makes the draft cheap, but also fragile. Even if the drafter produces a plausible next token, it is only accepted if it is the same next token that the target model would have made.

This means that the effectiveness depends on the drafter model, the exact training and quantization of the target model, the workload, and the serving software. Change any one and the win can evaporate or the launch can fail outright.

With gpt-oss-120b on vLLM and an identical ShareGPT workload, NVIDIA's throughput-tuned EAGLE3 draft slows decoding by **45%** (acceptance ~9% at conc-1, falling to ~1.5% at conc-8 — the lowest we measured anywhere); swap in the LMSYS/SpecForge draft and the same setup is roughly neutral (**−2.4%**, ~29% acceptance). Same model, same engine, same workload — the draft alone moves the result by 43 points. (A footnote on acceptance numbers: SGLang reports ~55-60% for the same LMSYS draft that vLLM logs at ~29% — the engines *define* acceptance differently, so never compare acceptance across engines.)

## The Models

TODO: Change this to instead show different model families, highlighting performance differences.

### 3a. MTP — the model drafting for itself (the clean win)

- Native head, no separate model, acceptance ~66-70% (accept-len ~3.0 incl. the bonus token) and **flat across concurrency** (workload-driven, as expected).
- Consistent double-digit wins on vLLM at conc-32:

  | model · quant | base → MTP | speedup |
  |---|---|---|
  | Qwen3.6-27B · FP8 | 154.7 → 240.9 | **+56%** |
  | Qwen3.6-27B · NVFP4 | 187.7 → 274.1 | **+46%** |
  | Qwen3.6-35B-A3B · FP8 | 286.0 → 407.9 | **+43%** |
  | Qwen3.6-35B-A3B · NVFP4 | 430.8 → 541.3 | **+26%** (fastest of the big-MoE runs) |
  | Gemma-4-12B · NVFP4 | 503.8 → 782.4 | **+55%** |
  | Gemma-4-E4B · FP8 | 869.7 → 1261.5 | **+45%** (fastest decode in the whole sweep) |

- **Slower quant, bigger draft win.** Notice each model gains *more* from MTP on its FP8 build than on its faster NVFP4 build (27B: +56% vs +46%; 35B-A3B: +43% vs +26%). Quant sets the base speed (NVFP4 > FP8 > BF16), but a slower target leaves more idle bandwidth for the drafter to fill — so the speculative decoder's *relative* payoff runs inversely to the base speed. (Two model pairs, so treat this as a pattern to watch, not a law.) The flip side: speculation doesn't close a quant gap — 35B-A3B NVFP4 *without* a speculative decoder (430.8) still out-decodes FP8 *with* MTP (407.9). Pick the fast quant first, then add the speculative decoder.

- **Engine matters:** the *same* Qwen3.6-27B NVFP4 MTP is **+46% on vLLM but only +10.5% on SGLang** — there, SGLang's hybrid scheduler runs the GDN/mamba-layer bookkeeping alongside the NEXTN drafter, and that overhead eats most of the spec win. Gemma-4-12B on SGLang lands at **+3.4%** for a different documented reason: SGLang disables its overlap scheduler on the Frozen-KV MTP path. In absolute terms the gap is a full **2×**: identical Gemma-4-12B NVFP4 assistant-MTP runs **782 tok/s on vLLM vs 400 on SGLang**.
- **Batch saturation matters:** fast small models on llama.cpp barely gain (Gemma-4-12B Q4 **+3.5%**), while the slow big one gains more (Gemma-4-31B Q4 **+18.5%**) — more expensive forward pass = more to amortize. (Caveat: llama.cpp serves the MTP head through its generic `--model-draft` path — real MTP weights, but classic draft-then-verify mechanics rather than vLLM's fused verify, which is part of why its gains are so much smaller.)
- **Acceptance scales with model size** (llama.cpp, conc-1): E4B ~2.88 → 12B ~3.21 → 31B ~3.41 accept-len — three points from one engine and one family, so directional only.
- TODO(graphic): grouped bar — base vs MTP decode tok/s for the four vLLM headliners above.
- TODO(graphic): the vLLM-vs-SGLang same-model comparison (+46% vs +10.5%) as a two-bar callout — "the engine, not the method."

### 3b. EAGLE3 — a separate draft head (draft choice is everything)

- Lower accept-len than MTP (~2.0-2.4 of 3) because it's a bolt-on head, but still wins big *when the draft matches the workload*.

  | model · engine · draft | base → EAGLE3 | speedup | note |
  |---|---|---|---|
  | Gemma-4-31B · vLLM NVFP4 | 167.0 → 264.7 | **+59%** | biggest EAGLE3 win; dense benefits most |
  | Gemma-4-26B-A4B · vLLM NVFP4 | 384.1 → 541.0 | **+41%** | |
  | gpt-oss-20b · vLLM MXFP4 | 535.3 → 686.5 | **+28%** (c32) | but a **−29% / −33%** *loss* at c2 / c4 → see below |
  | gpt-oss-120b · SGLang · LMSYS draft | 140.3 → 171.9 | **+22%** | first 120b spec win here (mixed engine images; direction solid, % approximate) |
  | gpt-oss-120b · vLLM · LMSYS draft | 252.8 → 246.7 | **−2.4%** | same engine as below, better draft → neutral |
  | gpt-oss-120b · vLLM · NVIDIA draft | 252.8 → 138.5 | **−45%** | wrong draft, saturated model |

- **The headline lesson is the draft — and speculative decoding can't fix a bad config.** The last three rows are the *same model, same workload*. Holding the engine fixed (vLLM) and swapping only the draft — NVIDIA throughput-tuned (~9% accept) → LMSYS/SpecForge (~29% accept) — rescues **−45% to roughly neutral**. SGLang turns the good draft into **+22%**, but only relative to its own much slower base: SGLang *with* the best draft (171.9) is still ~32% below vLLM with **no speculation at all** (252.8). The fastest gpt-oss-120b setup we measured is vLLM, no spec. Get the engine and quant right first; then use the draft the community has validated for your engine.
- **Engine brittleness can masquerade as a concurrency effect.** gpt-oss-20b EAGLE3 *loses* at c2/c4 (−29% / −33%) with acceptance collapsing to ~5%, then wins **+28%** by c32 (~44% accept). Acceptance is supposed to be workload-driven, not batch-driven — and our controls (same prompts at both concurrencies; prefix caching off) didn't remove the collapse, so this looks like a low-batch vLLM EAGLE3 pathology (suspected CUDA-graph padding), not a property of the draft. Either way, the "+28%" is a high-batch number, not an everywhere number.
- Dense models (Gemma-4-31B, +59%) out-gain MoE (26B-A4B, +41%) here — consistent with a heavier per-token forward pass giving speculation more to hide behind, but it's one pair, not a rule.
- TODO(graphic): the two *vLLM* gpt-oss-120b bars (−45% NVIDIA draft vs −2.4% LMSYS draft) side by side — same engine, the draft alone flips the sign; the single most persuasive "brittleness" visual.

### 3c. DFlash — many-token drafting (high ceiling, high fixed cost)

- Drafts 5-16 tokens depending on config (11 in our main runs) at ~20-45% acceptance, front-loaded and decaying — the aggressive end of the trade-off.
- **The only setup where DFlash was the best available config:** Qwen3.5-122B-A10B int4 (no MTP head to compete against), conc-8: **85.5 → 107.4 = +26%** (accept ~44%, len ~3.2-of-5) — and only after pinning the *older small-page* drafter revision so its KV spec would unify.
- **It does beat the plain base at low batch** — Qwen3.6-35B-A3B DFlash vs its *no-spec* base is **+36% / +30% / +23%** at conc-1 / 2 / 4, fading to +3.5% by conc-16. DFlash isn't a loser against nothing; it's a loser against *MTP*.
- **Against native MTP it loses under load** (Qwen3.6-35B-A3B table in §1b): a noise-level +2.9% at conc-1, −25.9% at conc-32.
- **Verdict from the notes:** DFlash *works* on this box (the earlier "architecturally blocked" claim was refuted by measurement) — it's just not worth it for a mixed-concurrency gateway. Keep MTP: draft-efficient, no external drafter, no forbidden revision, no untrusted image.
- TODO(graphic): the n=11 per-position acceptance decay curve (the "wasted compute" picture) — pairs with 2b.

## So... what should I do?

For a general chat gateway on one Spark, native MTP is the default winner for the Qwen3.6 / Gemma-4 families — it won at *every* concurrency we measured, +26-56% even with the batch full. EAGLE3 is competitive only with a validated, workload-matched draft; DFlash is a single-stream / latency-critical special case.

Two rules fall out of the data. First, **the drafter's cost decides the curve**: a near-free drafter (native MTP, ~67% acceptance) wins everywhere, while a heavy one (DFlash, ~3.7-of-11) buys low-concurrency latency with spare parallelism and gives it back under load. Second, **speculative decoding can't fix a bad config**: 35B-A3B NVFP4 with no speculative decoder out-decodes FP8 with MTP (430.8 vs 407.9), and vLLM with no speculative decoder out-decodes SGLang with the best draft (252.8 vs 171.9). Pick the right engine and quant first — the speculative decoder is a multiplier, not a rescue.

### The future — DDTree (draft *trees*, not draft *lines*)

DFlash bets everything on *one* long draft line, and we saw that line's acceptance decay almost to nothing by the tail (§3c) — most of the drafted compute wasted. [DDTree](https://liranringel.github.io/ddtree/) (Block Diffusion Draft Trees, arXiv [2604.12989](https://arxiv.org/abs/2604.12989), Ringel & Romano) spends that *same* verify budget differently: instead of collapsing the drafter's per-position distributions into one path, it builds a **tree** of likely continuations (best-first heap) and the target verifies the *whole tree* in a single pass via **tree attention**. "One long bet that usually breaks early" becomes "many short bets, keep the best."

**We measured it.** DDTree isn't in vLLM or SGLang, so it only runs in the paper's PyTorch harness (batch-1, bf16). Our own Qwen3.6 targets are hybrid GatedDeltaNet, which the harness can't roll back for spec verification (blocked) — so we ran the harness's own validated proxy, Qwen3-Coder-30B-A3B (same-size 30B-A3B MoE, standard attention), on identical prompts:

| workload | method | decode tok/s | accept-len | vs base |
|---|---|--:|--:|--:|
| chat (mt-bench) | autoregressive | 18.52 | 1.00 | — |
| | DFlash (single line) | 16.95 | 2.25 | **−8.4%** |
| | **DDTree (budget 64)** | **20.75** | 3.22 | **+12.1%** |
| | DDTree (budget 256) | 17.32 | 3.69 | −6.5% |
| code (HumanEval) | autoregressive | 17.66 | 1.00 | — |
| | DFlash (single line) | 47.87 | 7.96 | **2.71×** |
| | **DDTree (budget 64)** | **49.34** | 9.74 | **2.79×** |
| | DDTree (budget 256) | 41.30 | 10.50 | 2.34× |

- **The tree recovers DFlash's loss** — §3c's open question, answered. On chat, single-line DFlash is a *net loss* at batch-1 (−8.4%: a 2.25-of-16 accept-len doesn't repay the draft+verify). Rebuilt as a tree, the *same* draft becomes a **+12.1% win** — a +22% swing. On this box the block-diffusion draft only pays off *as a tree*.
- **Bigger tree isn't better — there's a budget optimum.** Budget 256 has the *highest* acceptance (3.69 / 10.50) but is *slower* than budget 64 both times: past ~64 nodes the extra acceptance costs more to verify than it saves.
- **The tree earns its keep on HARD workloads.** DDTree beats the single line by **+22% on chat** but only **+3.1% on code** — once DFlash already accepts ~8-of-16 (templated code) there's little headroom left; on high-entropy chat the extra branches matter. And the workload effect dwarfs the method: chat→code lifts accept-len ~3.5× and flips spec from a batch-1 loss to a 2.7-2.8× win.

**Two honest caveats.** The paper reports **8.22× lossless** on HumanEval; we see 2.7-2.8× — but that is our number on a **single DGX Spark (GB10)**, not the authors' datacenter GPUs, and it is smaller because this is batch-1, unquantized bf16, with a tree-attention mask that forces the slow torch-SDPA verify kernel (flash-attn can't express it), over only 12 problems. The *shape* transfers (tree > line, budget optimum, workload-driven); the absolute multiple doesn't. And **concurrency is still open** — the harness doesn't batch, so whether the tree still pays once the batch saturates the GB10 (the trade-off question above) is unmeasured. Serving-engine support is the blocker: DDTree is not merged into vLLM or SGLang (only [SGLang discussion #24605](https://github.com/sgl-project/sglang/discussions/24605)); a batched run needs an implementation like [CaDDTree](https://github.com/ZhangShuai1230/CaDDTree) to land in an engine first.

- **Gemma-4 and our Qwen3.6 hybrids can't play — yet.** DDTree needs a *block-diffusion* drafter; Gemma-4 has none (its speculative decoders are EAGLE3 heads, e.g. `RedHatAI/gemma-4-31B-it-speculator.eagle3`), and Qwen3.6-27B/35B-A3B are hybrid-GDN, which the harness can't roll back. The proxy above is standard-attention Qwen3-Coder.
- TODO(graphic): draft-*line* (DFlash, one decaying path) vs draft-*tree* (DDTree, branching, verified in one pass) side-by-side schematic — the single clearest "what changed" visual.
