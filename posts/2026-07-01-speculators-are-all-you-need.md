# On Speculators

A lot of ink and bits get spilled speeding up *datacenter-scale* inference (batching, disaggregation, giant KV pools). Speculative decoding is the rare trick that is particularly effective at speeding up the *small, local, low-concurrency* case — the single RTX or Apple Silicon chip on your lap.

---

A **speculator** (a.k.a. draft model, speculative decoder) is a small, cheap predictor that *guesses the next few tokens* the big model is about to produce. The big ("target") model then performs its decode pass, accepting guesses that match; the first disagreement is where normal decoding resumes.

This means that tokens where the speculator agrees with the target model are nearly free, but there's a base cost to run the speculator. In the rest of this article we'll explore this tradeoff and ground this in real data.

---

- TODO(prose): find an example where the speculator guesses right for a long run (boilerplate/code) vs falls over immediately (a surprising fact/name) — motivates "variable speedup."

Speculators work well when the next token is easy to guess.

```
TODO: A nice example where the tokens are obvious.
```

```
I saw her duck behind the rock.
I saw her duck in the pond.

- 
- Net effect: decode goes from *one expensive forward pass per token* to *one expensive pass per accepted run of tokens*. When the guesses are good, you get several tokens for the price of one.
- Three flavours appear in this dataset — worth naming up front because they behave very differently:
  - **MTP (multi-token prediction):** the checkpoint ships *its own* extra prediction head(s). No separate model, no distribution mismatch. (DeepSeek, Qwen3.6, Gemma-4.)
  - **EAGLE3:** a *separate*, small draft head trained to mimic the target. Quality depends entirely on *which* draft you load.
  - **DFlash:** an external drafter that speculates *many* tokens (11–12) per step — high ceiling, high fixed cost.
- TODO(graphic): schematic — target model + speculator both reading the same KV cache; speculator emits k candidate tokens; target verifies in one pass; accepted prefix in green, first reject in red. (Anchors 1a + 2a/2b/2c.)

### 1b. Performance is a trade-off: fixed cost vs. variable speedup

- Every speculative step pays a **fixed cost** (run the drafter, then verify its guesses) to buy a **variable speedup** (however many tokens got accepted).
- The trade only pays off when **(spare compute) × (draft acceptance)** clears the fixed cost. Two levers:
  - **Spare compute** — plentiful at small model size and/or low concurrency; gone once the batch saturates the GPU. This is why the *same* speculator wins single-stream and loses under load.
  - **Acceptance** — how often the guess is right. MTP's ~66% / ~3-of-3 tokens is structurally efficient; DFlash's ~25% / ~3.7-of-11 wastes ~7 forward passes per step.
- **The trade-off made literal** — Qwen3.6-35B-A3B, DFlash vs. the model's own MTP head:

  | concurrency | DFlash vs MTP decode |
  |---|---|
  | 1 (spare compute) | **+8.5%** |
  | 8 | −6.6% |
  | 32 (saturated) | **−25.9%** |

  Same model, same drafter — the sign of the speedup flips with load. That *is* the fixed-cost/variable-speedup curve.
- TODO(graphic): line chart — x = concurrency (1/8/32), y = decode tok/s, two lines (MTP vs DFlash) for Qwen3.6-35B-A3B, crossover annotated. **This is the money chart for 1b.**

### 1c. Speculators are brittle

- They depend on **(1) the model**, **(2) its exact training + quantization details**, and **(3) the data/workload** — change any one and the win can evaporate or the launch can fail outright. Concrete failures from the sweep:
- **(1) Model / architecture walls:**
  - *Hybrid-attention KV can't absorb a draft's KV spec.* Qwen3.5-122B-A10B and Ornith-1.0-35B (Gated-DeltaNet + full-attn) unify their two KV specs fine — until a DFlash draft adds a *third*, tripping `assert page_size_bytes == max_page_size`. The 122B only ran after pinning a *specific older* drafter revision (small-page: 4 KV-heads, no sliding window); the "correct" newer drafter *asserts on this box*.
  - *A 32-wide GDN gate isn't FP8-block-128 tileable* → Qwen3.6-35B-A3B NVFP4 is **blocked on SGLang** (base *and* MTP) before speculation is even reached. (Use vLLM.)
- **(2) Training/quant details:**
  - *Draft match dominates everything.* gpt-oss-120b, EAGLE3, identical ShareGPT workload: the NVIDIA *throughput*-tuned draft is **−45%** (accepts ~9%); the LMSYS/SpecForge draft is **+22%** (accepts ~60%). Same model, different draft, opposite sign.
  - *Quant × image standoff.* Gemma-4 E4B NVFP4+MTP is **blocked** by mutually-exclusive images — one loads NVFP4 but is too old for the drafter arch; the newer one runs MTP but regresses NVFP4 loading.
  - *Fit wall.* DeepSeek-V4-Flash's EAGLE3.1 path is real in the engine now, but the only NVFP4 repo is 168 GB (FP8-sized MIXED_PRECISION) — **can't load on one 121 GB Spark** regardless of the speculator.
- **(3) Data / workload:**
  - *Acceptance is workload-driven.* EAGLE3/MTP land ~70–85% on **coding**, lower on **general chat**. gpt-oss EAGLE3's conc-32 "+28%" on gpt-oss-20b is a **scheduling/prefill artifact** — acceptance actually *degrades* with concurrency (~30%→~5%), so it doesn't generalize down the sweep.
- TODO(graphic): a small "wall of walls" table/figure — 5 brittleness failure modes (KV-unify assert · GDN FP8-block · draft mismatch · image standoff · fit wall), each with the one-line symptom. Reinforces 1c.

---

## 2. How do they work?

### 2a. They *speculate* by guessing the rollout of the next few words

- The drafter runs first and proposes a short candidate continuation (k tokens; k≈3 for MTP, ~5 for the 122B DFlash, 11–12 for the aggressive DFlash configs).
- Bigger k = higher ceiling *if* accepted, but more wasted compute when rejected — see the acceptance-length numbers below.

### 2b. Verify in the decode phase; accept on agreement, fall back on disagreement

- In one target forward pass, the model computes the probability of each speculated token in parallel. Where the target agrees, we **pretend the token was there all along**; at the first disagreement we discard the rest and let the target generate that token normally, then re-speculate.
- So a step yields *1 + (accepted draft tokens)* real tokens. **Mean acceptance length** is the headline efficiency number:
  - **MTP** ≈ 3.0 of 3 (Qwen3.6), i.e. near-lossless drafting — almost no wasted passes.
  - **EAGLE3** ≈ 2.0–2.4 of 3 (Gemma/gpt-oss) — a separate head, so lower than native MTP.
  - **DFlash** ≈ 3.2–4.4 of **11** — front-loaded acceptance that collapses by the later positions ⇒ many wasted passes.
- TODO(graphic): per-position acceptance bar chart (0.84 / 0.66 / 0.51 for MTP-n3 vs the long DFlash-n11 tail decaying to ~0.05). Shows *why* short high-acceptance drafts beat long ones.

### 2c. They connect to the primary model's already-computed KV cache

- The drafter doesn't run a separate context — it **reads the target model's existing KV cache** and offers its guess against that state. That's what makes the draft cheap (no re-encoding the prompt) and also what makes hybrid-attention models fragile: the draft's KV spec has to *unify* with the target's page layout (the assert wall in 1c).

### 2d. The parallelizability limitation

- Verification is one big parallel pass — great when the GPU has spare lanes. But **once concurrency saturates the device, there are no spare lanes left**: the drafter's extra compute now competes with real requests instead of filling idle capacity.
- This is the mechanism behind every "wins at conc-1, loses at conc-32" result. Speculation is a way to *spend spare parallelism to cut latency* — and a busy server has none to spend.
- TODO(graphic): reuse/emphasize the 1b concurrency-crossover chart, or a small "GPU lanes" cartoon (idle lanes filled by drafter at conc-1; fully packed at conc-32).

---

## 3. Evidence — the three speculators, measured

> One box, ShareGPT V3, decode tok/s aggregate. "base" = matched non-spec run, same model/engine/quant/concurrency.

### 3a. MTP — the model drafting for itself (the clean win)

- Native head, no separate model, acceptance ~66–70% / ~3.0-of-3 and **flat across concurrency** (workload-driven, as expected).
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
