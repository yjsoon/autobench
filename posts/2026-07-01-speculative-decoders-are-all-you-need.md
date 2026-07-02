# On Speculative Decoders

---

**TL;DR:** run native MTP where the model has it, EAGLE3 where it doesn't. On one DGX Spark the fastest config we measured is [Gemma-4-26B-A4B NVFP4 + MTP](https://gauravmm.github.io/autobench/tags/model/#gemma-4-26b-a4b) at **692.1 tok/s**, with [Qwen3.6-35B-A3B NVFP4 + MTP](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-35b-a3b) close behind at **541.3 tok/s**

---

A lot of ink and bits get spilled speeding up *datacenter-scale* inference (batching, disaggregation, giant KV pools). Speculative decoding is the rare trick that is particularly effective at speeding up the *small, local, low-concurrency* case — the single RTX or Apple Silicon chip on your lap.

A **speculative decoder** (a.k.a. draft model, drafter, or speculator) is a small, cheap predictor that *guesses the next few tokens* the big model is about to produce. The big ("target") model then performs its decode pass, accepting guesses that match; the first disagreement is where normal decoding resumes. When the guesses are good, you get several tokens for the price of one.

This means that tokens where the speculative decoder agrees with the target model are nearly free, but there's a base cost to run the speculative decoder. In the rest of this article we'll explore this tradeoff and ground this in real data.

Speculative decoders work well when the next token is easy to guess. Something like this, with one obvious continuation, works well:

<link rel="stylesheet" href="assets/token-stream.css">

<p class="token-stream fork">
<span class="ctx"><span class="tok c0">The</span><span class="tok c1">quick</span><span class="tok c2">brown</span><span class="tok c3">fox</span></span>
<span class="guess"><span class="ell">…</span></span>
<span class="guess"><span class="ell">…</span> <span class="tok c4">jumps</span><span class="tok c5">over</span><span class="tok c0">the</span><span class="tok c1">lazy</span><span class="tok c2">dog</span><span class="tok c3">.</span></span>
</p>

Something like this does not:

<p class="token-stream fork">
<span class="ctx"><span class="tok c0">I</span><span class="tok c1">saw</span><span class="tok c2">her</span><span class="tok c3">duck</span></span>
<span class="guess"><span class="ell">…</span></span>
<span class="guess"><span class="ell">…</span> <span class="tok c4">under</span><span class="tok c5">the</span><span class="tok c0">branch</span><span class="tok c1">.</span></span>
<span class="guess"><span class="ell">…</span> <span class="tok c4">waddle</span><span class="tok c5">away</span><span class="tok c0">.</span></span>
</p>

> TODO(graphic): schematic — target model + speculative decoder both reading the same KV cache; speculative decoder emits k candidate tokens; target verifies in one pass; accepted prefix in green, first reject in red.

## The Options

Speculative decoding is a fast-moving field, so keep an eye out for new versions.

As of right now, there are three flavours that are common (and one emerging).

- **[MTP (multi-token prediction)](https://arxiv.org/abs/2404.19737):** extra prediction heads baked into the model. (DeepSeek, Qwen3.6, Gemma-4, etc.) The lightest-weight alternative.
- **[EAGLE3](https://arxiv.org/abs/2503.01840):** a *separate*, small draft head grafted into the model, reads activations at multiple levels to make its predictions. Quality depends entirely on *which* draft you load.
- **[DFlash](https://github.com/z-lab/dflash):** an external diffusion-based drafter that speculates many (up to 16) tokens per step. High fixed cost, with the chance for huge speedups.
- **[DDTree](https://liranringel.github.io/ddtree/) (emerging):** DFlash with a tree instead of a single draft line. Amazingly quick when it works.

> TODO: Update 138 after the next round of benchmarks come back.

The numbers below come from 138 benchmark configs run on an NVIDIA DGX Spark. These benchmarks were run semi-autonomously by an Opus 4.8 agent over about a week. Full results are [on the autobench website](https://gauravmm.github.io/autobench/).

## The performance trade-off

The drafter runs first and proposes a short continuation — 3 tokens for MTP, 5-16 for DFlash depending on the config. This imposes a **fixed cost**, and buys a **variable speedup** depending on how many tokens are accepted. This only pays off when our spare compute and acceptance rates are both good enough.

### Rule 1 — Drafters trade compute for speed {#drafters-trade-compute-for-speed}

Once the GPU is saturated, the drafter must compete with real requests for compute. Native MTP drafters are almost free and provide fantastic tradeoff. It's the heavy external DFlash drafters that *spend spare compute to buy low-concurrency throughput* — and a busy server has none to spend.

![Decode throughput vs concurrency for Qwen3.6-35B-A3B NVFP4 on vLLM, log-log, three lines: no-spec base, MTP, and DFlash. At conc-1 the three cluster (base 75, MTP 94, DFlash 100); MTP then leads at every concurrency (541 tok/s at conc-32); DFlash beats base only at low batch and dips below the no-spec baseline by conc-32 (407 vs base 431).](assets/plots/mtp_vs_dflash_35b.svg)
{: #fig-concurrency-crossover}

**Figure 1 — The concurrency crossover.** Decode tok/s vs concurrency for Qwen3.6-35B-A3B NVFP4 on vLLM: MTP leads at every batch size, while the heavy DFlash drafter beats the no-spec baseline only at low concurrency and slips below it by conc-32.
{: .figcaption}

Even with spare compute, the heavy DFlash drafter barely beats the built-in MTP; once the batch saturates the GPU it loses badly, even slipping below the no-drafter baseline at conc-32 (407 vs 431). The trade-off curve belongs to the *drafter's cost*, not to speculation itself.

As with everything in LLMs, the exact tradeoff curve is a fingerprint of the method, not a universal law. Later on we'll discuss other trends that confound this simple rule.

### Rule 2 — Agreement is critical to performance {#agreement-is-critical-to-performance}

In one target forward pass, the model computes the probability of each speculated token in parallel. Where the target agrees, we **pretend the token was there all along**; at the first disagreement we discard the rest and let the target generate that token normally, then re-speculate.

- **MTP** ≈ 3.0 of 4([Qwen3.6-27B](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-27b), [35B-A3B](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-35b-a3b)) — that's ~2 of 3 drafted tokens accepted plus the one free "bonus" token from the verify pass. Very efficient.
- **EAGLE3** ≈ 2.0-2.4 of 4 ([Gemma-4](https://gauravmm.github.io/autobench/tags/model/#gemma-4-31b); [gpt-oss](https://gauravmm.github.io/autobench/tags/model/#gpt-oss-120b) only with a workload-matched draft.
- **DFlash** ≈ 3.2-4.4 of 11 ([Qwen3.6-35B-A3B](https://gauravmm.github.io/autobench/tags/model/#qwen3-6-35b-a3b)) collapses after a few positions, leading to much waste.
- **DDTree** ≈ **9.7 of 16 on code**, as low as 3.2 of 16 on chat ([Qwen3-Coder-30B-A3B](https://gauravmm.github.io/autobench/tags/model/#qwen3-coder-30b-a3b)) — DFlash's block draft rebuilt as a *tree* of candidate continuations, verified together so more of each block survives. Full numbers in the  DDTree section below.

### Rule 3 — Drafters are brittle {#drafters-are-brittle}

Drafters are being judged by the target model's token choices, and are fed by the target model's own KV cache. That's what makes the draft cheap, but also fragile. Even if the drafter produces a plausible next token, it is only accepted if it is the *same next token that the target model would have made*.

This means that the effectiveness depends on:

1. the drafter model,
2. the exact training and quantization of the target model,
3. the workload, and
4. the serving software.

Change any one and the win can evaporate or the launch can fail outright.

The starkest case: **gpt-oss-120b on vLLM, one ShareGPT workload — swap only the EAGLE3 draft**, and the result moves 43 points.

| draft | decode tok/s | Δ | acceptance |
|---|--:|--:|--:|
| none (baseline) | 252.8 | 0% | — |
| NVIDIA EAGLE3 | 138.5 | **−45%** | 1.5~9% |
| LMSYS / SpecForge EAGLE3 | 246.7 | −2.4% | ~29% |

**Table 1 — The draft is the whole story.** gpt-oss-120b on vLLM, one ShareGPT workload; swap only the EAGLE3 draft and decode moves 43 points — from −45% (NVIDIA draft) to −2.4% (LMSYS/SpecForge).
{: .figcaption}

Same model, same engine, same workload — the draft alone is the whole difference. Before picking a drafter, test it in the exact configuration you will be using it.

### Rule 4 — Slower target, bigger relative win {#slower-target-bigger-relative-win}

The costlier the target's forward pass, the more idle bandwidth the drafter hides behind — so the *relative* speedup runs inversely to base speed. Two knobs move that cost: **quant** (a slower FP8 base gains more than the same model on faster NVFP4) and **architecture** (a dense model gains more than a comparable MoE, whose lighter per-token pass leaves less to amortize). Sort the four Qwen3.6 MTP runs from slowest base to fastest and the relative win falls straight down the table:

| model · quant | base → MTP | Δ |
|---|---|--:|
| Qwen3.6-27B · FP8 | 154.7 → 240.9 | **+56%** |
| Qwen3.6-27B · NVFP4 | 187.7 → 274.1 | **+46%** |
| Qwen3.6-35B-A3B · FP8 | 286.0 → 407.9 | **+43%** |
| Qwen3.6-35B-A3B · NVFP4 | 430.8 → 541.3 | **+26%** |

**Table 2 — Slower base, bigger relative win.** Four Qwen3.6 MTP runs at conc-32 on vLLM, sorted slowest base to fastest; the MTP speedup falls monotonically from +56% to +26%.
{: .figcaption}

Each model gains more on its slower FP8 quant than its faster NVFP4 quant, and the 27B dense architecture gains more than the 35B-A3B MoE architecture.

The flip side (rule 5): NVFP4 *without* a speculator (430.8) still out-decodes FP8 *with* MTP (407.9) — pick the fast quant first, then add the drafter.

### Rule 5 — Speculation can't rescue a bad config {#speculation-cant-rescue-a-bad-config}

A speculative decoder is a multiplier, not a fix. In both of these, the plainer setup with **no speculation at all** out-decodes a fancier one running its best available draft (decode tok/s):

| model | best config, no drafter | tok/s | worse config + best drafter | tok/s |
|---|---|--:|---|--:|
| Qwen3.6-35B-A3B | `NVFP4` quant | [**430.8**](https://gauravmm.github.io/autobench/configs/qwen3-6-35b-a3b-nvfp4-vllm/) | `FP8` quant + MTP | [407.9](https://gauravmm.github.io/autobench/configs/qwen3-6-35b-a3b-vllm-fp8-mtp/) |
| gpt-oss-120b `MXFP4` | vLLM engine | [**252.8**](https://gauravmm.github.io/autobench/configs/gpt-oss-120b-vllm-mxfp4/) | SGLang engine + LMSYS | [171.9](https://gauravmm.github.io/autobench/configs/gpt-oss-120b-sglang-mxfp4-eagle3-c32/) |

**Table 3 — Speculation can't rescue a bad config.** In both pairs the plainer no-drafter setup out-decodes a fancier one running its best available draft.
{: .figcaption}

Get the quant and engine right *first*; speculation compounds a good setup, it can't paper over a bad one.

## The Models

Which method you even *get* is largely decided by the family — MTP only exists where the lab baked in a head, EAGLE3 only where someone has trained a draft for that exact model. So the walk below goes family by family: Qwen3.6 and Gemma-4 with native MTP, then gpt-oss with EAGLE3 only.

![Grouped bar chart of decode tok/s, base vs +MTP at conc-32 on vLLM, for eight Qwen3.6 and Gemma-4 configs; MTP adds +26% to +94%, peaking at Gemma-4-E4B FP8 at 1262 tok/s.](assets/plots/base_vs_mtp.svg)
{: #fig-base-vs-mtp}

**Figure 2 — Native MTP across the family.** Decode tok/s, base vs +MTP at conc-32 on vLLM, across eight Qwen3.6 and Gemma-4 configs; MTP adds +26% to +94%.
{: .figcaption}

### Qwen3.6 — native MTP {#qwen36-native-mtp}

This family runs a close second on the board: the **[35B-A3B MoE on NVFP4 + MTP hits 541.3 tok/s](https://gauravmm.github.io/autobench/configs/qwen3-6-35b-a3b-nvfp4-vllm-mtp/)** — bettered only by Gemma-4-26B-A4B's native MTP (below). The decode is so quick because it lands on the right side of each of our five rules.

**[Rule 1 — Drafters trade compute for speed](#drafters-trade-compute-for-speed).** Native MTP is nearly free — it wins at *every* concurrency (the MTP line in [Figure 1](#fig-concurrency-crossover)), with none of the spare-compute tax that makes heavy DFlash fade under load.

**[Rule 2 — Agreement](#agreement-is-critical-to-performance).** High acceptance — ~66%, ~3.0 of 4 including the free bonus token.

**[Rule 3 — Drafters are brittle](#drafters-are-brittle).** Robust here by construction: the MTP head ships with the model, so draft and target are matched.

**[Rule 4 — Slower target, bigger relative win](#slower-target-bigger-relative-win).** A light MoE pass on fast NVFP4 leaves little to amortize, so MTP adds "only" **[+26%](https://gauravmm.github.io/autobench/configs/qwen3-6-35b-a3b-nvfp4-vllm-mtp/)** — the small end of the curve.

**[Rule 5 — Speculation can't rescue a bad config](#speculation-cant-rescue-a-bad-config).** That +26% rides on the fastest quant-and-engine we measured, so the absolute number lands near the very top of the board — second only to Gemma-4-26B-A4B + MTP. Speculation compounds a good config; here it compounds one of the best.

One interesting discovery we made is that minor engine details can greatly affect performance ([Rule 3](#drafters-are-brittle)). On the dense 27B NVFP4 + MTP, the **[+46% gain on vLLM](https://gauravmm.github.io/autobench/configs/qwen3-6-27b-nvfp4-vllm-mtp/)** is only **[+10.5% on SGLang](https://gauravmm.github.io/autobench/configs/qwen3-6-27b-nvfp4-sglang-mtp/)**. This seems to be due to scheduling decisions in the engine.

### Gemma-4 26B-A4B NVFP4 + MTP

**Gemma 4 26B-A4B NVFP4 + MTP [tops the board at 692.1 tok/s](https://gauravmm.github.io/autobench/configs/gemma-4-26b-a4b-it-vllm-nvfp4-mtp/)**, ahead of Qwen.

Gemma-4 is the only family here with *both* a native assistant-MTP path and grafted EAGLE3 heads, so it exercises the widest spread of the rules. Because the same model also carries a grafted **[EAGLE3 head (541.0)](https://gauravmm.github.io/autobench/configs/gemma-4-26b-a4b-it-vllm-nvfp4-eagle3/)**, this is the one place we can put the two drafters head-to-head — and native MTP wins.

We compare MTP and EAGLE3 drafters, and find that MTP wins the two head-to-head rows outright:

| model · quant | base | → MTP | Δ | → EAGLE3 | Δ |
|---|--:|--:|--:|--:|--:|
| E4B · FP8 | 869.7 | **[1261.5](https://gauravmm.github.io/autobench/configs/gemma-4-e4b-it-vllm-fp8-mtp/)** | +45% | [—](https://gauravmm.github.io/autobench/configs/gemma-4-e4b-it-vllm-fp8-eagle3/) | — |
| 12B · NVFP4 | 503.8 | **[782.4](https://gauravmm.github.io/autobench/configs/gemma-4-12b-it-redhatai-vllm-nvfp4-mtp/)** | +55% | [—](https://gauravmm.github.io/autobench/configs/gemma-4-12b-it-redhatai-vllm-nvfp4-eagle3/) | — |
| 26B-A4B · NVFP4 | 384.1 | **[692.1](https://gauravmm.github.io/autobench/configs/gemma-4-26b-a4b-it-vllm-nvfp4-mtp/)** | +80% | [541.0](https://gauravmm.github.io/autobench/configs/gemma-4-26b-a4b-it-vllm-nvfp4-eagle3/) | +41% |
| 31B · NVFP4 | 167.0 | **[323.5](https://gauravmm.github.io/autobench/configs/gemma-4-31b-it-vllm-nvfp4-mtp/)** | +94% | [264.7](https://gauravmm.github.io/autobench/configs/gemma-4-31b-it-vllm-nvfp4-eagle3/) | +59% |

**Table 4 — MTP vs EAGLE3 across Gemma-4.** Decode tok/s at conc-32 on vLLM, base vs each drafter. MTP wins both head-to-head rows; the two small models have no usable EAGLE3 head (dashes link to why).
{: .figcaption}

Because it hands us both drafters across four sizes, Gemma-4 is the cleanest illustration of three of our rules.

**[Rule 2 — Agreement](#agreement-is-critical-to-performance).** Native MTP posts a higher accept-len (~2.7-2.8 of 3, ~55-65% draft acceptance) than EAGLE3 (~2.0-2.4 of 3), and that decides the head-to-head: MTP beats EAGLE3 by **[+28%](https://gauravmm.github.io/autobench/configs/gemma-4-26b-a4b-it-vllm-nvfp4-mtp/)** on 26B-A4B (692.1 vs 541.0) and **[+22%](https://gauravmm.github.io/autobench/configs/gemma-4-31b-it-vllm-nvfp4-mtp/)** on 31B (323.5 vs 264.7).

**[Rule 3 — Drafters are brittle](#drafters-are-brittle),** and the engine bites hardest. Hold the model, quant, and MTP drafter fixed at Gemma-4-12B, swap only the engine, and the win swings from huge to nil:

| engine · quant | base → MTP | Δ | why |
|---|--:|--:|---|
| vLLM · NVFP4 | 503.8 → **[782.4](https://gauravmm.github.io/autobench/configs/gemma-4-12b-it-redhatai-vllm-nvfp4-mtp/)** | **+55%** | overlap scheduler on ✅ |
| SGLang · NVFP4 | 386.6 → **[399.8](https://gauravmm.github.io/autobench/configs/gemma-4-12b-it-axionml-sglang-nvfp4-mtp/)** | +3.4% | overlap scheduler off ❌ |
| llama.cpp · Q4 | 195.3 → **[202.2](https://gauravmm.github.io/autobench/configs/gemma-4-12b-it-llamacpp-mtp/)** | +3.5% | overlap scheduler off ❌ |

**Table 5 — Same drafter, three engines.** Gemma-4-12B NVFP4 + MTP held fixed, only the engine changes; vLLM's overlap scheduler turns a +55% win into +3–4% on SGLang and llama.cpp.
{: .figcaption}

The overlap scheduler runs work concurrently instead of sequentially, allowing the drafter (and CPU) overhead to be hidden. This is not available on llama.cpp, and disabled for this model under the current SGLang, hence the poor performance gain.

**[Rule 4 — Slower target, bigger relative win](#slower-target-bigger-relative-win).** Read the table above down its base&rarr;spec columns: the slower dense **31B** out-gains the faster MoE **26B-A4B** on both drafters — **+94% vs +80%** with MTP, **+59% vs +41%** with EAGLE3.

### gpt-oss — EAGLE3 only, the draft is everything

No native MTP head, so EAGLE3 is the only option — and gpt-oss is where the *draft-is-everything* rule is sharpest:

| engine · draft | base → EAGLE3 | Δ | note |
|---|---|---|---|
| SGLang · LMSYS draft | 140.3 → 171.9 | **+22%** | mixed engine images |
| vLLM · LMSYS draft | 252.8 → 246.7 | **−2.4%** | neutral |
| vLLM · NVIDIA draft | 252.8 → 138.5 | **−45%** | wrong draft, saturated model |

**Table 6 — gpt-oss-120b EAGLE3, engine × draft.** The draft dominates: on vLLM the same model swings from −45% (NVIDIA draft) to −2.4% (LMSYS), and no spec config beats vLLM's no-spec baseline (252.8).
{: .figcaption}

**[Rule 3 — Drafters are brittle](#drafters-are-brittle).** The draft alone moves 43 points. Same model, workload, and engine (vLLM): swapping NVIDIA's throughput-tuned draft (~9% accept) for LMSYS/SpecForge (~29% accept) rescues −45% to roughly neutral.

**[Rule 5 — Speculation can't rescue a bad config](#speculation-cant-rescue-a-bad-config).** SGLang *with* the best draft (171.9) is still ~32% below vLLM with **no speculation at all** (252.8). The fastest gpt-oss-120b we measured is vLLM, no spec.

### Qwen3-Coder-30B-A3B + DFlash — the exception

DFlash is redeemed by the *workload*, not the configuration. Qwen3-Coder-30B-A3B only accepts 2.25 of 16 tokens on chat datasets, which is disappointingly low for a high-cost drafter like DFlash.

**[Rule 2 — Agreement](#agreement-is-critical-to-performance).** On templated, low-entropy code (HumanEval), the same Qwen3-Coder-30B-A3B drafter nails long spans: **[7.96 of 16](https://gauravmm.github.io/autobench/configs/qwen3-coder-30b-a3b-ddtree-humaneval/)**, a 2.7× decode win. A drafter that's dead weight on hard, high-entropy text can be the fastest thing on the box when the continuation is easy to guess.

## The future

Its early days yet, so we're waiting

### DDTree (draft *trees*, not draft *lines*)

DFlash bets everything on *one* long draft line, which quickly decays to nothing, wasting most of the draft compute. [Block Diffusion Draft Trees](https://liranringel.github.io/ddtree/) (arXiv [2604.12989](https://arxiv.org/abs/2604.12989), Ringel & Romano) builds a tree of likely continuations and verifies it in a single pass.

<p class="token-stream fork tree">
<span class="ctx"><span class="tok c0">I</span><span class="tok c1">saw</span><span class="tok c2">her</span><span class="tok c3">duck</span></span>
<span class="guess"><span class="ell">…</span></span>
<span class="guess"><span class="ell">…</span> <span class="tok c4">under</span><span class="tok c5">the</span></span>
<span class="sub"><span class="ell">…</span></span>
<span class="sub"><span class="ell">…</span> <span class="tok c0">branch</span><span class="tok c1">.</span></span>
<span class="sub"><span class="ell">…</span> <span class="tok c0">table</span><span class="tok c1">.</span></span>
<span class="guess"><span class="ell">…</span> <span class="tok c4">waddle</span></span>
<span class="sub"><span class="ell">…</span></span>
<span class="sub"><span class="ell">…</span> <span class="tok c5">away</span><span class="tok c0">.</span></span>
<span class="sub"><span class="ell">…</span> <span class="tok c5">across</span><span class="tok c0">the</span><span class="tok c1">pond</span><span class="tok c2">.</span></span>
</p>

This is new technology, hot off the presses, so it isn't in vLLM or SGLang. We ran Qwen3-Coder-30B-A3B (same-size 30B-A3B MoE, standard attention) using the research-grade code and found:

| workload | method | decode tok/s | accept-len | vs base |
|---|---|--:|--:|--:|
| chat (mt-bench) | autoregressive | 18.52 | 1.00 | — |
| | DFlash (single line) | 16.95 | 2.25 | **−8.4%** |
| | **DDTree (budget 64)** | **20.75** | 3.22 | **+12.1%** |
| | DDTree (budget 256) | 17.32 | 3.69 | −6.5% |
| code (HumanEval) | autoregressive | 17.66 | 1.00 | — |
| | DFlash (single line) | 47.87 | 7.96 | **2.7×** |
| | **DDTree (budget 64)** | **49.34** | 9.74 | **2.8×** |
| | DDTree (budget 256) | 41.30 | 10.50 | 2.3× |

**Table 7 — DDTree recovers DFlash's loss.** Qwen3-Coder-30B-A3B at batch-1 in the paper's PyTorch harness; rebuilt as a tree, the same block-diffusion draft turns chat's −8.4% DFlash loss into a +12.1% win, while code holds ~2.8×.
{: .figcaption}

Where the workload already suits DFlash (code), DDTree performs about the same. On the hard workfloads where DFlash fails, DDTree can rescue its performance.

**Bigger tree isn't better — there's a budget optimum.** Budget 256 has the *highest* acceptance (3.69 / 10.50) but is *slower* than budget 64 both times: past ~64 nodes the extra acceptance costs more to verify than it saves.

**The tree earns its keep on HARD workloads.** DDTree beats the single line by **+22% on chat** but only **+3.1% on code** — once DFlash already accepts ~8-of-16 (templated code) there's little headroom left; on high-entropy chat the extra branches matter.

**Two honest caveats.** The paper reports **8.22× lossless** on HumanEval; we see 2.7-2.8× — but that is our number on a **single DGX Spark (GB10)**, not the authors' datacenter GPUs, and it is smaller because this is batch-1, unquantized bf16, with a tree-attention mask that forces the slow torch-SDPA verify kernel (flash-attn can't express it), over only 12 problems. The *shape* transfers (tree > line, budget optimum, workload-driven); the absolute multiple doesn't. And **concurrency is still open** — the harness doesn't batch, so whether the tree still pays once the batch saturates the GB10 (the trade-off question above) is unmeasured. Serving-engine support is the blocker: DDTree is not merged into vLLM or SGLang (only [SGLang discussion #24605](https://github.com/sgl-project/sglang/discussions/24605)); a batched run needs an implementation like [CaDDTree](https://github.com/ZhangShuai1230/CaDDTree) to land in an engine first.

**Gemma-4 and our Qwen3.6 hybrids can't play — yet.** DDTree needs a *block-diffusion* drafter; Gemma-4 has none (its speculative decoders are EAGLE3 heads, e.g. `RedHatAI/gemma-4-31B-it-speculator.eagle3`), and Qwen3.6-27B/35B-A3B are hybrid-GDN, which the harness can't roll back. The proxy above is standard-attention Qwen3-Coder.

> TODO(graphic): draft-*line* (DFlash, one decaying path) vs draft-*tree* (DDTree, branching, verified in one pass) side-by-side schematic — the single clearest "what changed" visual.

### Drafter-assisted prefill

Everything above speeds up **decode** — emitting tokens one at a time. But the *first* token has its own cost: **prefill**, where the model reads your entire prompt before it says anything. For long prompts that read is the dominant latency (the "time to first token"), and ordinary speculation does nothing for it.

**Drafter-assisted prefill points the small model at the prompt instead of the output.** The drafter skims the whole prompt cheaply and flags which tokens actually matter; the big model then prefills only that important subset rather than every token. Fewer tokens through the expensive model means a faster first token — [SpecPrefill](https://arxiv.org/abs/2502.02789) (ICML 2025) reports up to **~7.7× faster time-to-first-token** on a 405B model, and it needs no training. It can even reuse the *same* drafter you already run for decode-time speculation, so the extra cost is small; follow-ups extend the idea to [cross-family draft models for long-context compression](https://arxiv.org/abs/2603.02631).

We didn't benchmark this on the Spark — it's a separate lever from the decode speedups above — but for long-context, low-concurrency work (exactly the Spark's niche) it's the natural next thing to try.

## So... what should I do?

For a general chat gateway on one Spark, native MTP is the default winner for the Qwen3.6 / Gemma-4 families — it won at *every* concurrency we measured, +26-56% even with the batch full. EAGLE3 is competitive only with a validated, workload-matched draft; DFlash is a single-stream / latency-critical special case.

Two rules fall out of the data. First, **the drafter's cost decides the curve**: a near-free drafter (native MTP, ~67% acceptance) wins everywhere, while a heavy one (DFlash, ~3.7-of-11) buys low-concurrency latency with spare parallelism and gives it back under load. Second, **speculative decoding can't fix a bad config**: 35B-A3B NVFP4 with no speculative decoder out-decodes FP8 with MTP (430.8 vs 407.9), and vLLM with no speculative decoder out-decodes SGLang with the best draft (252.8 vs 171.9). Pick the right engine and quant first — the speculative decoder is a multiplier, not a rescue.
