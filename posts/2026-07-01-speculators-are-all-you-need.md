# On Speculators

TODO: Include a TL;DR: Use this Qwen 3.6 config or this Gemma 4 config. (with links to the fastest MoE versions of each)

TODO: Speculators to Speculative Decoders

A lot of ink and bits get spilled speeding up *datacenter-scale* inference (batching, disaggregation, giant KV pools). Speculative decoding is the rare trick that is particularly effective at speeding up the *small, local, low-concurrency* case — the single RTX or Apple Silicon chip on your lap.

A **speculator** (a.k.a. draft model, speculative decoder) is a small, cheap predictor that *guesses the next few tokens* the big model is about to produce. The big ("target") model then performs its decode pass, accepting guesses that match; the first disagreement is where normal decoding resumes. When the guesses are good, you get several tokens for the price of one.

This means that tokens where the speculator agrees with the target model are nearly free, but there's a base cost to run the speculator. In the rest of this article we'll explore this tradeoff and ground this in real data.

Speculators work well when the next token is easy to guess. Something like this, with one obvious continuation, works well:

> The quick brown fox jumps *over the lazy dog.*

Something like this does not:

> I saw her duck *under the branch.* \
> I saw her duck *waddle away.*
>

- TODO(graphic): schematic — target model + speculator both reading the same KV cache; speculator emits k candidate tokens; target verifies in one pass; accepted prefix in green, first reject in red.

## The Options

Speculator research is a fast-moving field, so keep an eye out for new versions.

As of right now, there are three flavours that are common (and one emerging).

- **[MTP (multi-token prediction)](https://arxiv.org/abs/2404.19737):** extra prediction heads baked into the model. (DeepSeek, Qwen3.6, Gemma-4.)
- **[EAGLE3](https://arxiv.org/abs/2503.01840):** a *separate*, small draft head grafted into the model, reads activations at multiple levels to make its predictions. Quality depends entirely on *which* draft you load.
- **[DFlash](https://github.com/z-lab/dflash):** an external diffusion-based drafter that speculates *many* tokens (11-12) per step — high ceiling, high fixed cost.
- **[DDTree](https://liranringel.github.io/ddtree/) (emerging):** DFlash with a tree instead of a single draft line. Only runs in a research harness so far (no serving engine yet), but it already turns block-diffusion from a single-stream loss into a win — see the last section.

TODO: Some table showing the speedups with different models.

We ran an extensive (~number, including the _archive) benchmarks... (give some stats)

TODO: A short two sentences describing the setup. See the repo (link to it via public URL -- this may be published elsewhere)

> One box, ShareGPT V3, decode tok/s aggregate. "base" = matched non-spec run, same model/engine/quant/concurrency.

## The performance trade-off

The drafter runs first and proposes a short continuation of about 3 tokens for MTP, and maybe 5 for DFlash. This imposes a **fixed cost**, and buys a **variable speedup** depending on how many tokens are accepted. This only pays off when our spare compute and acceptance rates are both good enough.

Once the GPU is saturated, the drafter must compete with real requests for compute, which slows the whole system down. This is the mechanism behind the (ref: chart) result. One way to think about this is that speculation is a way to *spend spare parallelism to cut latency* — and a busy server has none to spend.

TODO: graphic: line chart — x = concurrency (1/8/32), y = decode tok/s, two lines (MTP vs DFlash) for Qwen3.6-35B-A3B, crossover annotated. **This is the money chart for 1b.**

  | concurrency | DFlash vs MTP decode |
  |---|---|
  | 1 (spare compute) | **+8.5%** |
  | 8 | −6.6% |
  | 32 (saturated) | **−25.9%** |

When model sizes are small and concurrency is low, speculative decoding has enough spare compute to help. Once the batch saturates the GPU, the *same* speculator loses. Heavier drafters (like DFlash) work better at lower compute than the built-in MTP.

TODO: Trim

But "wins at low batch, loses at high" is only the *average* story — the exact shape of the speedup-vs-concurrency curve is a fingerprint of the method, not a universal law. In our sweep MTP on the 27B decays smoothly (+81% → +63% → +46% going from 1 to 8 to 32 concurrent requests); MTP on the 35B-A3B is non-monotonic, peaking around +42% at concurrency 2; the gpt-oss-20b EAGLE3 head is *inverted* — a loss at low batch that only turns positive once the batch is large enough to hide its poor acceptance; and DFlash follows the textbook wins-low-loses-high curve. Before you trust any headline speedup, ask *at what concurrency*.

### Agreement improves performance

In one target forward pass, the model computes the probability of each speculated token in parallel. Where the target agrees, we **pretend the token was there all along**; at the first disagreement we discard the rest and let the target generate that token normally, then re-speculate.

TODO: Link each to the tags-model page (use the full base url gauravmm.github.io/autobench/)

- **MTP** ≈ 3.0 of 3 (Qwen3.6), i.e. near-lossless drafting — almost no wasted passes.
- **EAGLE3** ≈ 2.0-2.4 of 3 (Gemma/gpt-oss) — a separate head, so lower than native MTP.
- **DFlash** ≈ 3.2-4.4 of **11** — collapses after a few positions, leading to much waste.

- TODO(graphic): per-position acceptance bar chart (0.84 / 0.66 / 0.51 for MTP-n3 vs the long DFlash-n11 tail decaying to ~0.05). Shows *why* short high-acceptance drafts beat long ones.

### Drafters are brittle

Drafters are being judged by the target model's token choices, and are fed by the target model's own KV cache. That's what makes the draft cheap, but also fragile. Even if the drafter produces a plausible next token, it is only accepted if it is the same next token that the target model would have made.

This means that the effectiveness depends on the drafter model, the exact training and quantization of the target model, the workload, and the serving software. Change any one and the win can evaporate or the launch can fail outright.

With gpt-oss-120b, an EAGLE3 drafter, and identical ShareGPT workload, the NVIDIA drafter slows down inference by **45%** (~9% acceptance); the LMSYS/SpecForge draft speeds up inference by **+22%** (~60% acceptance). Same model, different draft, opposite sign.

## The Models

TODO: Change this to instead show different model families, highlighting performance differences.
TODO: Where possible, find evidence for the points above in the data below.

### 3a. MTP — the model drafting for itself (the clean win)

- Native head, no separate model, acceptance ~66-70% / ~3.0-of-3 and **flat across concurrency** (workload-driven, as expected).
- Consistent double-digit wins on vLLM at conc-32:

  | model · quant | base → MTP | speedup |
  |---|---|---|
  | Qwen3.6-27B · FP8 | 154.7 → 240.9 | **+56%** |
  | Qwen3.6-27B · NVFP4 | 187.7 → 274.1 | **+46%** |
  | Qwen3.6-35B-A3B · FP8 | 286.0 → 407.9 | **+43%** |
  | Qwen3.6-35B-A3B · NVFP4 | 430.8 → 541.3 | **+26%** (fastest of the big-MoE runs) |
  | Gemma-4-12B · NVFP4 | 503.8 → 782.4 | **+55%** |
  | Gemma-4-E4B · FP8 | 869.7 → 1261.5 | **+45%** (fastest decode in the whole sweep) |

- **Slower quant, bigger draft win.** Notice each model gains *more* from MTP on its FP8 build than on its faster NVFP4 build (27B: +56% vs +46%; 35B-A3B: +43% vs +26%). Quant sets the base speed (NVFP4 > FP8 > BF16), but a slower target leaves more idle bandwidth for the drafter to fill — so the speculator's *relative* payoff runs inversely to the base speed.

- **Engine matters:** the *same* Qwen3.6-27B NVFP4 MTP is **+46% on vLLM but only +10.5% on SGLang** — SGLang's overlap scheduler is disabled for the MTP/NEXTN path, so scheduler overhead eats the draft win. (Gemma-4-12B SGLang "Frozen-KV MTP" is the same story: **+3.4%**.) In absolute terms the gap is a full **2×**: identical Gemma-4-12B NVFP4 assistant-MTP runs **782 tok/s on vLLM vs 400 on SGLang**.
- **Batch saturation matters:** fast small models on llama.cpp barely gain (Gemma-4-12B Q4 **+3.5%**), while the slow big one gains more (Gemma-4-31B Q4 **+18.5%**) — more expensive forward pass = more to amortize.
- **Acceptance scales with model size** (llama.cpp, conc-1): E4B ~2.76 → 12B ~3.21 → 31B ~3.41 accept-len.
- TODO(graphic): grouped bar — base vs MTP decode tok/s for the four vLLM headliners above.
- TODO(graphic): the vLLM-vs-SGLang same-model comparison (+46% vs +10.5%) as a two-bar callout — "the engine, not the method."

### 3b. EAGLE3 — a separate draft head (draft choice is everything)

- Lower accept-len than MTP (~2.0–2.4 of 3) because it's a bolt-on head, but still wins big *when the draft matches the workload*.

  | model · engine · draft | base → EAGLE3 | speedup | note |
  |---|---|---|---|
  | Gemma-4-31B · vLLM NVFP4 | 167.0 → 264.7 | **+59%** | biggest EAGLE3 win; dense benefits most |
  | Gemma-4-26B-A4B · vLLM NVFP4 | 384.1 → 541.0 | **+41%** | |
  | gpt-oss-20b · vLLM MXFP4 | 535.3 → 686.5 | **+28%** (c32) | but a **−29% / −33%** *loss* at c2 / c4 → see below |
  | gpt-oss-120b · SGLang · LMSYS draft | 140.3 → 171.9 | **+22%** | first 120b spec win here |
  | gpt-oss-120b · vLLM · LMSYS draft | 252.8 → 246.7 | **−2.4%** | same engine as below, better draft → neutral |
  | gpt-oss-120b · vLLM · NVIDIA draft | 252.8 → 138.5 | **−45%** | wrong draft, saturated model |

- **The headline lesson is the draft, not the engine.** The last three rows are the *same model, same workload*. Holding the engine fixed (vLLM) and swapping only the draft — NVIDIA throughput-tuned (~9% accept) → LMSYS/SpecForge (~29% accept) — rescues **−45% to roughly neutral**. Then the right *engine* (SGLang) on the good draft turns neutral into **+22%**. Draft choice moves the sign; engine tunes the magnitude. **Always use the model's own recommended draft.**
- **A drafter can lose at low batch and win at high.** gpt-oss-20b EAGLE3 *loses* at c2/c4 (−29% / −33%, acceptance collapsing to ~5%) and only turns positive by c16 (~44% accept). The "+28%" is a high-batch number, not an everywhere number.
- Dense models (Gemma-4-31B, +59%) out-gain MoE (26B-A4B, +41%): a heavier per-token forward pass gives speculation more to hide behind.
- TODO(graphic): the two gpt-oss-120b bars (−45% vs +22%) side by side, labelled by *draft*, not engine — the single most persuasive "brittleness" visual.

### 3c. DFlash — many-token drafting (high ceiling, high fixed cost)

- Drafts 11–12 tokens at ~16–40% acceptance, front-loaded and decaying — the aggressive end of the trade-off.
- **Only clean win:** Qwen3.5-122B-A10B int4, conc-8: **85.5 → 107.4 = +26%** (accept ~44%, len ~3.2-of-5) — and only after pinning the *older small-page* drafter revision so its KV spec would unify.
- **It does beat the plain base at low batch** — Qwen3.6-35B-A3B DFlash vs its *no-spec* base is **+36% / +30% / +23%** at conc-1 / 2 / 4, fading to +3.5% by conc-16. DFlash isn't a loser against nothing; it's a loser against *MTP*.
- **Against native MTP it loses under load** (Qwen3.6-35B-A3B table in §1b): +8.5% at conc-1, −26% at conc-32. The custom-container AEON-27B DFlash (184 tok/s @ conc-32) doesn't even match native-MTP-on-stock-vLLM (303).
- **Verdict from the notes:** DFlash *works* on this box (the earlier "architecturally blocked" claim was refuted by measurement) — it's just not worth it for a mixed-concurrency gateway. Keep MTP: draft-efficient, no external drafter, no forbidden revision, no untrusted image.
- TODO(graphic): the n=11 per-position acceptance decay curve (the "wasted compute" picture) — pairs with 2b.

## So... what should I do?

TODO: Turn into prose

For a general chat gateway on one Spark, native MTP is the default winner.
for the Qwen3.6 / Gemma-4 families; EAGLE3 is competitive *only with the engine's own draft*; DFlash is a single-stream / latency-critical special case.

Remember that **speculation buys latency with spare parallelism.** It only pays when you have spare compute and good acceptance rates.

- small model or low concurrency (spare lanes) ✓
- high, workload-matched acceptance ✓ (native MTP ≈ 3-of-3 > EAGLE3 ≈ 2-of-3 > DFlash ≈ 3.7-of-11)

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
- **The tree earns its keep on HARD workloads.** DDTree beats the single line by **+22% on chat** but only **+3.1% on code** — once DFlash already accepts ~8-of-16 (templated code) there's little headroom left; on high-entropy chat the extra branches matter. And the workload effect dwarfs the method: chat→code lifts accept-len ~3.5× and flips spec from a batch-1 loss to a 2.7–2.8× win.

**Two honest caveats.** The paper reports **8.22× lossless** on HumanEval; we see 2.7–2.8× — but that is our number on a **single DGX Spark (GB10)**, not the authors' datacenter GPUs, and it is smaller because this is batch-1, unquantized bf16, with a tree-attention mask that forces the slow torch-SDPA verify kernel (flash-attn can't express it), over only 12 problems. The *shape* transfers (tree > line, budget optimum, workload-driven); the absolute multiple doesn't. And **concurrency is still open** — the harness doesn't batch, so whether the tree still pays once the batch saturates the GB10 (the trade-off question above) is unmeasured. Serving-engine support is the blocker: DDTree is not merged into vLLM or SGLang (only [SGLang discussion #24605](https://github.com/sgl-project/sglang/discussions/24605)); a batched run needs the [CaDDTree](https://github.com/ZhangShuai1230/CaDDTree) / [Tencent AngelSlim](https://github.com/Tencent/AngelSlim) implementations to land in an engine first.

- **Gemma-4 and our Qwen3.6 hybrids can't play — yet.** DDTree needs a *block-diffusion* drafter; Gemma-4 has none (its speculators are EAGLE3 heads, e.g. `thoughtworks/Gemma-4-31B-Eagle3`), and Qwen3.6-27B/35B-A3B are hybrid-GDN, which the harness can't roll back. The proxy above is standard-attention Qwen3-Coder.
- TODO(graphic): draft-*line* (DFlash, one decaying path) vs draft-*tree* (DDTree, branching, verified in one pass) side-by-side schematic — the single clearest "what changed" visual.
