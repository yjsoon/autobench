# On Speculators

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
- **[DDTree](https://liranringel.github.io/ddtree/) (emerging):** DFlash with a tree. Not in a serving engine yet (research code only).

TODO: Some table showing the speedups with different models.

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

Concrete failures from autobench:

1. With gpt-oss-120b, an EAGLE3 drafter, and identical ShareGPT workload, the NVIDIA drafter slows down inference by **45%** (~9% acceptance); the LMSYS/SpecForge draft speeds up inference by **+22%** (~60% acceptance). Same model, different draft, opposite sign.
2. With Gemma-4 E4B, NVFP4, and an MTP head, no single container image works: the one that loads NVFP4 is too old for the drafter architecture, and the newer one runs MTP but regresses NVFP4 loading. The config is **blocked** on mutually-exclusive images.
3. Acceptance is workload-driven: EAGLE3/MTP land ~70–85% on **coding** but lower on **general chat**. gpt-oss-20b EAGLE3's conc-32 "+28%" is a **scheduling/prefill artifact** — acceptance actually *degrades* with concurrency (~30%→~5%), so it doesn't generalize down the sweep.

## 3. Evidence — the three speculators, measured

TODO: A short two sentences describing the setup. See the repo (link to it via public URL -- this may be published elsewhere)
> One box, ShareGPT V3, decode tok/s aggregate. "base" = matched non-spec run, same model/engine/quant/concurrency.

### 3a. MTP — the model drafting for itself (the clean win)

- Native head, no separate model, acceptance ~66-70% / ~3.0-of-3 and **flat across concurrency** (workload-driven, as expected).
- Consistent double-digit wins on vLLM at conc-32:

  | model · quant | base → MTP | speedup |
  |---|---|---|
  | Qwen3.6-27B · FP8 | 154.7 → 240.9 | **+56%** |
  | Qwen3.6-27B · NVFP4 | 187.7 → 274.1 | **+46%** |
  | Qwen3.6-35B-A3B · NVFP4 | 430.8 → 541.3 | **+26%** (fastest decode in the whole sweep) |
  | Gemma-4-12B · NVFP4 | 503.8 → 782.4 | **+55%** |

- **Engine matters:** the *same* Qwen3.6-27B NVFP4 MTP is **+46% on vLLM but only +10.5% on SGLang** — SGLang's overlap scheduler is disabled for the MTP/NEXTN path, so scheduler overhead eats the draft win. (Gemma-4-12B SGLang "Frozen-KV MTP" is the same story: **+3.4%**.)
- **Batch saturation matters:** fast small models on llama.cpp barely gain (Gemma-4-12B Q4 **+3.5%**), while the slow big one gains more (Gemma-4-31B Q4 **+18.5%**) — more expensive forward pass = more to amortize.
- **Acceptance scales with model size** (llama.cpp, conc-1): E4B ~2.76 → 12B ~3.21 → 31B ~3.41 accept-len.
- TODO(graphic): grouped bar — base vs MTP decode tok/s for the four vLLM headliners above.
- TODO(graphic): the vLLM-vs-SGLang same-model comparison (+46% vs +10.5%) as a two-bar callout — "the engine, not the method."

### 3b. EAGLE3 — a separate draft head (draft choice is everything)

- Lower accept-len than MTP (~2.0–2.4 of 3) because it's a bolt-on head, but still wins big *when the draft matches the workload*.

  | model · engine | base → EAGLE3 | speedup | note |
  |---|---|---|---|
  | Gemma-4-31B · vLLM NVFP4 | 167.0 → 264.7 | **+59%** | biggest EAGLE3 win; dense benefits most |
  | Gemma-4-26B-A4B · vLLM NVFP4 | 384.1 → 541.0 | **+41%** | |
  | gpt-oss-20b · vLLM MXFP4 | 535.3 → 686.5 | **+28%** | but accept degrades w/ conc → prefill artifact |
  | gpt-oss-120b · SGLang (LMSYS draft) | 140.3 → 171.9 | **+22%** | first 120b spec win here |
  | gpt-oss-120b · vLLM (NVIDIA draft) | 252.8 → 138.5 | **−45%** | wrong draft, saturated model |

- **The headline lesson** — the last two rows are the *same model on the same workload*: the LMSYS/SpecForge draft (its own recipe) accepts ~60% and wins; the NVIDIA throughput-tuned draft accepts ~9% and is dead weight. **Always use the engine's own recommended draft.**
- Dense models (Gemma-4-31B, +59%) out-gain MoE (26B-A4B, +41%): a heavier per-token forward pass gives speculation more to hide behind.
- TODO(graphic): the two gpt-oss-120b bars (−45% vs +22%) side by side, labelled by *draft*, not engine — the single most persuasive "brittleness" visual.

### 3c. DFlash — many-token drafting (high ceiling, high fixed cost)

- Drafts 11–12 tokens at ~16–40% acceptance, front-loaded and decaying — the aggressive end of the trade-off.
- **Only clean win:** Qwen3.5-122B-A10B int4, conc-8: **85.5 → 107.4 = +26%** (accept ~44%, len ~3.2-of-5) — and only after pinning the *older small-page* drafter revision so its KV spec would unify.
- **Against native MTP it loses under load** (Qwen3.6-35B-A3B table in §1b): +8.5% at conc-1, −26% at conc-32. The custom-container AEON-27B DFlash (184 tok/s @ conc-32) doesn't even match native-MTP-on-stock-vLLM (303).
- **Verdict from the notes:** DFlash *works* on this box (the earlier "architecturally blocked" claim was refuted by measurement) — it's just not worth it for a mixed-concurrency gateway. Keep MTP: draft-efficient, no external drafter, no forbidden revision, no untrusted image.
- TODO(graphic): the n=11 per-position acceptance decay curve (the "wasted compute" picture) — pairs with 2b.

---

## Cross-cutting takeaway

- **Speculation buys latency with spare parallelism.** It pays when *spare compute × acceptance* beats the fixed drafter+verify cost:
  - small model or low concurrency (spare lanes) ✓
  - high, workload-matched acceptance ✓ (native MTP ≈ 3-of-3 > EAGLE3 ≈ 2-of-3 > DFlash ≈ 3.7-of-11)
  - a heavy per-token forward pass to hide behind (dense > MoE; big > small) ✓
- **For a general chat gateway on one Spark, native MTP is the default winner** for the Qwen3.6 / Gemma-4 families; EAGLE3 is competitive *only with the engine's own draft*; DFlash is a single-stream / latency-critical special case.
- TODO(graphic): 2×2 or quadrant — axes "spare compute (concurrency)" × "acceptance," plotting where each method lands and where speculation stops paying.

---

## The future — DDTree (draft *trees*, not draft *lines*)

> Not benchmarked here yet (no Spark runs — research code, not a serving stack). Included as the direction of travel, and the natural fix for DFlash's weakness (§3c).

- **What it is:** [DDTree](https://liranringel.github.io/ddtree/) (Block Diffusion Draft Trees, arXiv [2604.12989](https://arxiv.org/abs/2604.12989), Ringel & Romano) is the tree-structured successor to DFlash — same block-diffusion drafter lineage as the DFlash configs already in this sweep.
- **The key idea:** instead of collapsing the drafter's per-position distributions into *one* long guess path (vanilla DFlash), DDTree builds a **tree** of likely continuations (best-first heap) and the target verifies the *whole tree* in a single pass via **tree attention**.
- **Why it directly attacks §3c's problem:** DFlash's pain was a single 11-token line whose acceptance decayed to ~0.05 by the tail — ~7 wasted forward passes per step. A tree spends that *same* verify budget hedging across multiple branches, so far more of the drafted compute lands on an accepted token. It turns "one long bet that usually breaks early" into "many short bets, keep the best."
- **Reported result:** DFlash + DDTree = **8.22× lossless** speedup on Qwen3-Coder-30B-A3B-Instruct (HumanEval), beating EAGLE-3. **Lossless** = the target keeps its own decoding rule, so output distribution is unchanged.
- **Read it against our data with two caveats:**
  - **Workload:** 8.22× is **HumanEval (coding)** — the high-acceptance regime. Our ShareGPT (general chat) runs sit lower across the board, so expect a smaller tree win here too (the §1c/§3b workload lesson).
  - **Concurrency:** the reported win is single-stream. The §1b/§2d question stands unanswered — does a draft *tree* still pay once the batch saturates the GB10, or does tree attention's extra verify width just compete with real requests? That's exactly the experiment worth running.
- **It's runnable on *this* box — no new drafter needed.** DDTree assembles a tree from an existing DFlash drafter's per-position outputs, so it reuses **the exact z-lab drafters this sweep already runs**: [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) and [`z-lab/Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash). The tree is a *decoding-time change*, not a retrain.
  - **…but not in a serving container yet.** DDTree is **not merged into vLLM or SGLang** (only [SGLang discussion #24605](https://github.com/sgl-project/sglang/discussions/24605)) — the AEON `aeon-vllm-ultimate` fork we use for DFlash doesn't expose it. A Spark run means the research PyTorch path: [z-lab/dflash](https://github.com/z-lab/dflash) + the DDTree tree code, or the [CaDDTree](https://github.com/ZhangShuai1230/CaDDTree) / [Tencent AngelSlim](https://github.com/Tencent/AngelSlim) implementations. (There's an Apple-Silicon [ddtree-mlx](https://github.com/humanrouter/ddtree-mlx) port too — wrong hardware for us.)
  - **Gemma-4 can't play — yet.** DDTree needs a *block-diffusion* drafter; Gemma-4 has none (its speculators are EAGLE3 heads, e.g. `thoughtworks/Gemma-4-31B-Eagle3`). No Gemma-4 DFlash drafter = no Gemma-4 DDTree until someone trains one.
- **TODO(benchmark):** stand up DDTree on the Spark against `Qwen/Qwen3.6-35B-A3B` (drafter `z-lab/Qwen3.6-35B-A3B-DFlash`, already used in the DFlash configs) — measure accept-len and decode tok/s at conc 1/8/32, and put it head-to-head with native MTP *and* single-line DFlash on the §1b concurrency-crossover chart. That closes the loop: does the tree recover DFlash's under-load losses (§3c)?
- TODO(graphic): draft-*line* (DFlash, one decaying path) vs draft-*tree* (DDTree, branching, verified in one pass) side-by-side schematic — the single clearest "what changed" visual.

---

## External post

1. **The strange duality between speculators and tokenization** — TODO(prose): both are bets about "what usually comes next"; a tokenizer compresses frequent sequences into one symbol ahead of time, a speculator predicts frequent continuations at runtime. Where do they overlap / trade off?

---

### Appendix — blocked configs (brittleness receipts)

- `gemma-4-e4b-it-vllm-nvfp4-mtp` — image standoff (NVFP4 load vs. drafter arch).
- `qwen3-6-35b-a3b-nvfp4-sglang` / `-mtp` — GDN gate not FP8-block-128 tileable.
- `deepseek-v4-flash-vllm-nvfp4-eagle3` / `-awq` — 168 GB > 121 GB fit wall.
- `ornith-1-0-35b-aeon-vllm-nvfp4-dflash-blocked` — hybrid + draft KV can't unify (with correction: small-page drafter pin boots on sibling 35B checkpoints).
- `gemma-4-e4b-it-llamacpp-mtp` — *not* a real MTP regression: forced `-fa off` confounds the −36%; needs an `-fa off` base to compare.
