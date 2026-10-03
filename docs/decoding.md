# Decoding acceleration research

Method landscape for fast decode beyond the drafters TensorFold already ships (MTP heads,
DFlash2 block-diffusion drafters, DSpark), with memory overhead and reported decode speedups.
All numbers are lossless (output distribution preserved) unless marked lossy, come from each
paper's own harness, and are single-stream temperature 0 unless noted. Numbers from different
harnesses are not directly comparable; treat them as orders of magnitude, not a leaderboard.

Surveyed October 2026 via arXiv (abstracts plus DFlash/DSpark/EAGLE-3/DeepSeek-V3 full texts)
and live license checks against the GitHub and Hugging Face APIs (2026-10-03).

The per-family tables keep raw AR-relative numbers. The ranking table below re-anchors
everything to **MTP = 1.0x**, which is how these methods actually compare against the
MTP heads TensorFold already ships.

## The cost model that explains every ranking

Per generated token, speculative decoding costs
`L = (T_draft + T_verify) / tau`, where `tau` is the accepted length per verify round
(DSpark 2607.05147, Eq. 1). There are only three levers:

1. **Draft faster** — lower `T_draft`. Autoregressive drafters (MTP chains, EAGLE) pay
   `T_draft ~ gamma * t_step` (grows with block length); parallel drafters (DFlash) fill
   the whole block in one forward pass, so `T_draft` is nearly independent of `gamma`.
2. **Draft better** — raise `tau` (acceptance length). Driven by target conditioning,
   intra-block causality, on-policy training.
3. **Verify smarter** — lower effective `T_verify`: trees spread the budget over
   alternatives; block verification accepts more per round; adaptive scheduling stops
   verifying low-survival-probability suffixes.

Two systematic caveats:

- **Batch size.** At high concurrency decoding becomes compute-bound and speculation loses
  its memory-bandwidth advantage; naive block verification wastes batch capacity. MagicDec,
  DSpark and DScale exist specifically to fix this regime.
- **Entropy and generation length.** Acceptance collapses when the target's entropy rises
  (RL rollouts, Bebop 2606.12370) or far past the drafter's training-length distribution
  (Test-Time Speculation 2605.09329).

## Ranking with MTP as the 1.0x baseline

Anchor: MTP delivers 1.2-1.8x over AR in practice (DeepSeek-V3 reports 1.8x TPS; 1.2-1.7x
for native MTP on Qwen3.5 in SGLang). Single-stream multiples below divide each method's
AR-relative speedup by ~1.5x; the batched column uses direct same-stack measurements where
they exist (DFlash2 vs native MTP on identical models; DSpark vs MTP-1 in DeepSeek-V4
serving). Cross-harness ratios are approximate — single-stream research harnesses flatter
drafting-heavy methods relative to production serving numbers.

### Ranked, permissively licensed (commercial use OK)

| # | Method | vs MTP, single-stream | vs MTP, batched serving | Concurrency | Code / checkpoints |
| --- | --- | --- | --- | --- | --- |
| 1 | DFlash2 | ~4x (6.5x AR) | ~1.4-2.1x, typically ~2x (Qwen3.5 SGLang, same models) | SGLang/vLLM native; add verify-length scheduling past concurrency ~16 | MIT (`z-lab/dflash`), Apache-2.0 checkpoints (`z-lab/*-DFlash2`) |
| 2 | DSpark | ~4x, est. (semi-AR, same class as DFlash; +18-22% over DFlash at concurrency 8-32 per DScale cross-comparisons) | **1.6-1.85x, direct measurement in DeepSeek-V4 production** | Best in class: confidence-scheduled verify length; production-proven | checkpoints MIT (`deepseek-ai/DeepSeek-V4-Pro-DSpark`) and Apache-2.0 (`RedHatAI`, `openbmb`); community code MIT |
| 3 | EAGLE-3 | 2.3-4.3x (own harness 6.5x AR; 3.5x in DFlash's harness) | +40% throughput at batch 64 (SGLang) | Engine-native, but decays to ~1.0-1.2x by concurrency 32 (B200 table in DFlash paper) | Apache-2.0 (`SafeAILab/EAGLE`); Apache-2.0 checkpoints (`yuhuili/EAGLE3-*`) |
| 4 | PARD-2 | ~4.6x (6.94x AR) | — | vLLM integration; COD training 3x cheaper | MIT (`AMD-AGI/PARD`); no official checkpoints — train your own |
| 5 | DominoTree | ~4.9x (7.3x AR) | +12% over Domino in SGLang single-request | SGLang out-of-tree plugin; tree budget discipline needed at concurrency | MIT (`slin-zhq/Domino-Tree`); needs a compatible drafter checkpoint |
| 6 | MagicDec | — | ~1.7x at batch 32-256 on long context (2.51x AR) | Built for batch; sparse-KV draft | Apache-2.0 (`Infini-AI-Lab/MagicDec`) |
| 7 | LongSpec | ~2.2x (3.26x AR) | 2.25x wall-clock on AIME24 (QwQ 32B) | Constant draft KV; SGLang tree attention | MIT (`sail-sg/LongSpec`); MIT checkpoints (`sail/longspec-*`) |
| 8 | Medusa / Hydra++ | 1.5-2.4x | — | vLLM/HF native; typical-acceptance mode is lossy — use tree + rejection | Apache-2.0 code (`FasterDecoding/Medusa`, `zankner/Hydra`); some original Medusa checkpoints untagged |
| 9 | PLD / REST / Lookahead | 1.1-1.6x | — | Trivial — the draft is ~free (copy, retrieve, or Jacobi) | Apache-2.0 (`FasterDecoding/REST`, `hao-ai-lab/LookaheadDecoding`); PLD built into vLLM |
| 10 | Block verification + adaptive draft stopping | x1.05-1.6 multiplier on any stack above | same multiplier | Engine-side; this is the fix that keeps speculation alive at high concurrency | In vLLM/SGLang (Apache-2.0); AdaEDL is a few lines to reimplement |

### License-blocked or paper-only (not commercial-ready as-is)

| Method | vs MTP, single-stream | Blocker |
| --- | --- | --- |
| DARTree | **~6.5x (9.73x AR) — best number found** | `VILA-Lab/DARTree` has no LICENSE file |
| Domino | ~3.7x (5.49x AR) | `jianuo-huang/Domino` has no LICENSE file |
| SEED | ~1.8x (2.7x AR), zero drafter memory | `lhk2004/SEED` has no LICENSE file |
| AngelSpec | +10-12% over DFlash | `Tencent/AngelSpec` custom license — needs review |
| Draft-OPD / TTS / BV loss / DRelay / DScale / DDTree / PCTree / DeLS-Spec / DBloom / ASD / AdaEDL / EntMTP / post-training MTP | +10-72% stack multipliers | No official code (reimplement from paper) |
| LayerSkip | 0.9-1.4x | **CC BY-NC 4.0 — explicitly non-commercial, code and checkpoints** |
| Kangaroo / Sequoia / SpecExec / DistillSpec / ASPIRE | 1.0-2.7x | No LICENSE file in repo, or no official code |

Licensing nuance: a missing LICENSE file blocks use of that repo's code as-is (default
copyright), but does not block reimplementing the method from the arXiv paper — algorithms
are not copyright-restricted, and arXiv's distribution license places no constraint on
implementations. So DARTree is "blocked" only in the copy-the-code sense; writing your own
tree scorer from the paper is unrestricted (patents are a separate question outside this
survey's scope — engineering triage, not legal advice).

### License fine print for DFlash2 and DSpark (checked 2026-10-03)

DFlash2 itself is **not** CC-NC: the method's code and the official checkpoints TensorFold
ships are permissive. One third-party upload re-licenses a GLM adapter as CC BY-NC-ND 4.0 —
that artifact is non-commercial; the z-lab originals are not.

| Artifact | Kind | License | Commercial |
| --- | --- | --- | --- |
| `z-lab/dflash` | DFlash2 code | MIT | yes |
| `z-lab/Qwen3.8-27B-DFlash2`, `z-lab/gemma-4-26B-A4B-it-DFlash` | official drafter checkpoints | Apache-2.0 | yes |
| `incoai/Qwen3.8-27B-DFlash2` | third-party drafter checkpoint | Apache-2.0 | yes |
| `incoai/GLM-5.3-Flash-DFlash2` | third-party drafter checkpoint | **CC BY-NC-ND 4.0** | **no — avoid** |
| `deepseek-ai/DeepSeek-V4-Pro-DSpark` | DSpark drafter checkpoint | MIT | yes |
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | DSpark drafter checkpoint | Apache-2.0 | yes |
| `TensorFold/DeepSeek-V4-Flash-DSpark-MLX` | DSpark drafter (TensorFold conversion) | MIT | yes |
| `RadixArk/Qwen3.8-27B-DSpark` | third-party DSpark checkpoint | "other" | review before use |
| `TensorFold/Qwen3.8-27B-MLX-4bit` | base target | Apache-2.0 | yes |
| `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` | base target (MTP included) | MIT | yes |
| `TensorFold/DeepSeek-V4-Flash-MTP-MLX` | base target (MTP included) | MIT | yes |

**GLM-5.3-Flash specifics.** There is no official z-lab GLM-5.3-Flash DFlash2 drafter (verified against the
full z-lab org listing); every GLM DFlash2 on HF is a third-party conversion. TensorFold's CUDA family pinned
`incoai/GLM-5.3-Flash-DFlash2` (`src/tensorfold/families/glm5_next/__init__.py`) until the 2026-10-03
integration, which switched the pin to the MIT-licensed RedHatAI DSpark preview (speculators format); the
incoai checkpoint's CC BY-NC-ND 4.0 terms are why the swap matters commercially. Consequences:

| Option for GLM-5.3-Flash | License | Commercial |
| --- | --- | --- |
| Included MTP head (Mac path; `-MTP` checkpoints) | MIT | yes — the safe default |
| `RedHatAI/GLM-5.3-Flash-speculator.dspark-preview` | MIT | **yes — the CUDA engine's default drafter pin as of the 2026-10-03 integration** (vLLM speculator format) |
| `incoai/GLM-5.3-Flash-DFlash2` (still loads via `--drafter`) | CC BY-NC-ND 4.0 | **no** |
| `canada-quant/GLM-5.3-Flash-DFlash2-E/-F/-G` | Apache-2.0 | likely, but verify weight provenance + calibration before shipping |
| `modal-labs/GLM-5.3-Flash-DFlash` | MIT | likely, same caveat (named DFlash, not DFlash2) |
| Solstice-AI `...UNCENSORED...-DFlash2` | MIT | abliterated weights — not a drop-in |
| Train your own DFlash2 head for GLM (z-lab MIT code + paper recipe) | yours | yes — cleanest commercial path to GLM + DFlash2 |

Drafter-family coverage for GLM-5.3-Flash, checked 2026-10-03: MTP yes (in-checkpoint, MIT);
DSpark yes (RedHatAI preview, MIT; `AlayaNeW/GLM-5.3-DSpark` covers non-Flash GLM-5.3, MIT);
DFlash2 third-party only (popular uploads NC); **no EAGLE-3 drafter exists for
GLM-5.3-Flash** (only GLM-4.7-Flash, Apache-2.0, and GLM-5.1, MIT, via third parties).

So commercially, GLM-5.3-Flash serves with MTP or the MIT DSpark preview; the incoai DFlash2
path stays non-commercial. Swapping the `DRAFTER` pin to a permissive artifact is a one-line
change, but requires re-checking block size, mask token id and draft calibration against the
base checkpoint first. Do not confuse z-lab's `-PARO` checkpoints with drafters — PARO is
ParoQuant 4-bit quantization (arXiv 2511.10645), a memory-reduction method orthogonal to
drafting.

Two general rules this table illustrates: (1) the drafter adapter and the base model carry
separate licenses — both must be commercial-clear for the served pair to be; (2) any random
third-party re-upload of a drafter can re-license it (as `incoai` did with the GLM adapter),
so pin official-source checkpoints.

### Is it easy to add concurrency to these?

Yes for anything already inside a continuous-batching engine — MTP, EAGLE-3, DFlash2,
DSpark, PARD, Medusa all run batched draft+verify today. Two real caveats:

1. **Speedups compress as batch grows**, because decoding turns compute-bound and the
   free memory-bandwidth headroom that speculation exploits disappears. AR-tree drafting
   collapses first: EAGLE-3 drops from 1.6x at concurrency 1 to ~1.0-1.2x at concurrency 32
   on a B200 (DFlash's serving table). Block-parallel drafters hold their advantage longer
   because one draft pass covers many positions.
2. **Static verify lengths waste batch capacity** at concurrency ~16+ — every request pays
   to verify its full block regardless of survival probability. The fix is per-request,
   confidence-scheduled verification: DSpark's dynamic verify-length allocation (deployed
   in DeepSeek-V4), DScale's path-aware tiles (+43-49% throughput over DFlash at
   concurrency 8-32), AdaEDL-style entropy stopping (+10-57%), or ASPIRE's fully
   asynchronous per-request schedule.

Practical readout: DSpark is the only method concurrency-proven in production (+60-85%
per-user speed vs MTP-1 at matched throughput), and it is exactly "semi-AR block drafting +
confidence-scheduled verification". Trees also need budget discipline at concurrency —
DScale exists for that. Self-speculation is the awkward one at high batch: drafting
competes with serving for compute, which is why ASPIRE schedules it asynchronously.

## Composite scorecard

Five axes, each 0-10, weighted for a production serving stack. Scores are a synthesis of the
verified numbers above, not new measurements — the anchors are listed so they can be audited
or re-weighted.

Weights and anchors:

- **Speed (30%)** — MTP-normalized speedup, 50/50 single-stream vs batched where both are
  known; domain-limited methods are discounted for generality. 10 = >=5x MTP everywhere;
  7 = ~4x MTP; 5 = ~2x MTP; 3 = ~1.3x MTP in a limited regime.
- **Memory (20%)** — extra weights + KV + buffers vs vanilla AR of the same target.
  10 = zero; 8 = buffers only or <3% KV; 6.5 = ~6-13% weights (block drafter class);
  5 = ~1-2 GB standalone draft.
- **Concurrency (20%)** — how much of the gain survives as batch grows. 10 = designed for
  and/or production-proven at high batch; 4 = collapses by batch ~32.
- **License (15%)** — 10 = MIT/Apache-2.0 code **and** checkpoints; 8-9 = permissive code,
  train-your-own or untagged artifacts; 6 = permissive code but a license-murky dependency;
  2 = repo has no LICENSE file; 0 = CC BY-NC.
- **Integration (15%)** — 9-10 = drop-in or already shipped/training-free; 5-7 = engine port
  with existing checkpoints; 4 = training required; 2 = reimplement from paper.

### Ranking

| # | Method | Speed | Mem | Conc | Lic | Integ | **Score** | Regime / caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DFlash2 | 7 | 6.5 | 7 | 10 | 10 | **7.8** | General default; already shipped in TensorFold |
| 2 | DSpark | 7 | 6.5 | 10 | 9 | 6 | **7.7** | High-concurrency serving; wired for DeepSeek-V4-Flash only |
| 3 | MagicDec | 4.5 | 10 | 10 | 10 | 5 | **7.6** | Long context, batch 32+; ~no gain outside that regime |
| 4 | PLD (prompt lookup) | 3 | 10 | 9 | 9 | 9 | **7.4** | Free, but only helps on context-overlap tasks |
| 5 | LongSpec | 5.5 | 9 | 7 | 10 | 4 | **7.0** | Long context; constant draft KV |
| 6 | REST | 3.5 | 9.5 | 8.5 | 9 | 6 | **6.9** | Retrieval-friendly corpora; CPU datastore |
| 7 | MTP (baseline) | 2 | 9 | 7 | 8 | 10 | **6.5** | The reference point — cheap, modest gain |
| 8 | Lookahead | 4 | 8 | 6 | 10 | 5 | **6.3** | No draft model; trades FLOPs for steps |
| 9 | EAGLE-3 | 5.5 | 6 | 4 | 10 | 7 | **6.2** | Excellent single-stream, poor batch survival |
| 10 | DominoTree | 7 | 6.5 | 5.5 | 6 | 4 | **6.0** | Best clean speed, but needs a self-trained Domino-style drafter |
| 11 | Medusa / Hydra++ | 5 | 6 | 6 | 8 | 5 | **5.9** | Head training; memory scales with vocab |
| 12 | PARD-2 | 6.5 | 5 | 5 | 8 | 4 | **5.8** | Strong number, train-your-own draft |
| 13 | DARTree | 8 | 7 | 3 | 2 | 2 | **5.0** | Fastest anywhere (9.73x AR); code unlicensed, single-stream |
| 14 | SEED | 5 | 10 | 4 | 2 | 2 | **4.9** | Zero drafter memory, but repo unlicensed |
| 15 | Kangaroo | 3 | 9 | 4 | 2 | 4 | **4.4** | Repo unlicensed |
| 16 | LayerSkip | 3 | 10 | 4 | 0 | 4 | **4.3** | CC BY-NC 4.0 — non-commercial |

Read the top of this table as: **DFlash2 and DSpark are co-leaders** — 7.8 vs 7.7 is inside
the noise of the underlying harness variance. DFlash2 wins on drop-in availability (it is
already the shipped drafter across Qwen/GLM/Gemma) and on having an official repo; DSpark
wins the moment concurrency matters, and TensorFold already lists a DSpark drafter for
DeepSeek-V4-Flash. MagicDec, PLD and LongSpec rank high because they cost almost nothing —
they are regime specialists worth stacking when their regime applies, not general-purpose
replacements.

### Stack add-ons (multipliers, scored as additions rather than standalone methods)

| Add-on | Effect | Mem | Conc | Lic | Integ | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Block verification | x1.05-1.08 speed, never worse | 10 | 9 | 10 | 9 | Always on |
| AdaEDL entropy draft-stop | x1.1-1.6 accepted tokens | 10 | 9 | 10* | 9 | Always on (*few lines, no code released) |
| Training-free tree on DFlash2 | up to +98% accepted tokens | 8 | 5 | 7 | 6 | Next upgrade; needs cc budgeting |
| Semi-AR head (Domino/DSpark recipe) | +60-85% vs MTP-1 in production | 7 | 10 | 9 | 5 | The big win; only training spend |
| Verify-length scheduling (DSpark DVL / DScale) | +44-49% throughput at cc 8-32 | 10 | 10 | 10* | 6 | Required above cc ~16 |
| Drafter post-training (Draft-OPD / BV loss / TTS) | +13-72% acceptance | 10 | 9 | 10* | 3 | Cheap quality lift, no serving change |

### Sensitivity — who wins if you re-weight

- **Speed only:** DARTree (8) first, then DominoTree / DFlash2 / DSpark / PARD-2 cluster
  (~7).
- **Concurrency-heavy (weight 30%):** DSpark (~7.9) > MagicDec (~7.9) > DFlash2 (~7.5).
- **Memory-first:** MagicDec and PLD (10), then LongSpec (9); SEED would lead its class but
  is license-blocked.
- **Strict license filter:** drops DARTree, SEED, Kangaroo, LayerSkip, and demotes
  DominoTree (Domino dependency) — the commercial-clean podium becomes DFlash2, DSpark,
  MagicDec.

### Verdict

The composite **"DFlash2 + block verification + training-free tree (+ semi-AR head and
verify-length scheduling later)"** stack projects to Speed 8.5 / Mem 6.5 / Conc 8.5 /
Lic 9 / Integ 7 = **8.0 — higher than any single method in the table**. That is the same
recommendation as the adoption list below, now with the arithmetic attached: the ceiling
(~6-7x MTP single-stream, ~2.5-3.5x MTP serving) comes from stacking orthogonal wins, not
from picking the single fastest paper.

## Why MTP is the weakest modern drafter

MTP (DeepSeek-V3 2412.19437) chains D sequential prediction modules, so drafting cost grows
with depth. Verified consequences:

- DeepSeek-V3 reports **1.8x TPS** from MTP at high acceptance (tech report).
- DFlash's SGLang table, same models: native MTP acceptance is **5.2-7.1** yet end-to-end
  speedup is only **1.2-1.7x** (Qwen3.5-4B/9B/35B-A3B); DFlash turns similar acceptance into
  **2.3-3.9x** because its one-pass drafting is nearly free.
- Structural flaws identified in 2026 work: first-token head competes with the backbone LM
  head (CLP 2606.10935); fixed tree topology ignores local entropy (EntMTP 2606.27550);
  acceptance is bounded by entropy fluctuation under RL (Bebop: negative-linear in entropy,
  ~95% acceptance achievable with end-to-end TV loss); heads are set at pretraining time
  (post-training heads match joint pretraining with 10^3-10^4x fewer tokens,
  2610.00888).

## The information floor of parallel drafting

"Beyond Parallel Blindness" (2608.27339) separates the two losses in block drafting:

- The **all-parallel floor**: even a perfect marginal drafter cannot beat ~0.286 rejection
  at the final slot on Qwen3-4B (max 71% per-slot acceptance) because slot k has no
  information about slots < k.
- **One realized predecessor token removes 86-100% of the floor** — locality, confirmed by
  mutual information.
- Current drafters sit far above their floors: the final-slot **model gap** is 43-64% of
  DFlash's rejection, 85-92% of DSpark's oracle-conditioned rejection.

Theory verdict: purely parallel marginal drafting is capped; **semi-AR / path-conditioned
drafting** (Domino, DSpark, DARTree trees, DRelay repair) is the principled fix, and there
is still large headroom in drafter quality.

## Method list

Memory overheads are for an 8B-class dense target (bf16, ~16.5 GB weights, ~4.6 GB KV at
32K context with GQA); scale linearly-ish with target size. "est." marks estimates from
architecture, not paper tables.

### Shipped baselines

| Method | arXiv | Extra memory | Reported decode speedup | Theoretical note | Code / checkpoints |
| --- | --- | --- | --- | --- | --- |
| Vanilla AR decoding | — | none | 1.0x | Sequential bound; memory-bandwidth limited | — |
| Speculative sampling (separate small draft) | 2302.01318 | +1-2 GB draft weights + draft KV (est.) | 2-2.5x (Chinchilla 70B) | Original rejection-sampling verifier | Any permissively licensed small model |
| MTP head (DeepSeek-V3 style, D=1) | 2412.19437 | +~1-2% weights (one block, shared embed/head), +~3% KV | 1.8x TPS (V3); 1.2-1.7x on Qwen3.5 in SGLang | `T_draft ~ gamma * t_step` caps speedup despite tau 5-7 | Ships with the model release; license follows the base model |
| DFlash / DFlash2 (block-diffusion drafter) | 2602.06036 | +~6-13% weights (5 draft layers + KV-injection proj; shares frozen target LM head), +~3% KV for projected context features | >6x lossless, up to 2.5x over EAGLE-3; per-task 4.2-7.9x on Qwen3-4B/8B | One forward pass per block: `T_draft` independent of block size; 7B-parameter diffusion drafters (DiffuSpec line) were rejected as too heavy | MIT (`z-lab/dflash`); Apache-2.0 checkpoints (`z-lab/*-DFlash2`) |

### Structurally better drafters (the "better than MTP+DFlash2" tier)

| Method | arXiv | Extra memory | Reported decode speedup | Theoretical note | Code / checkpoints |
| --- | --- | --- | --- | --- | --- |
| Domino (parallel backbone + GRU causal head) | 2605.29707 | ~DFlash + tiny head | 5.49x e2e; 5.8x throughput in SGLang | Adds intra-block causality that pure marginals cannot represent (attacks the information floor) | `jianuo-huang/Domino` — no LICENSE file (blocked) |
| DSpark (semi-AR: parallel backbone + Markov + RNN head, confidence-scheduled verify) | 2607.05147 | ~DFlash + small heads | **+60-85% per-user speed vs MTP-1 at matched throughput in DeepSeek-V4 production serving** | The production-proven answer for high concurrency; lossless scheduling + hardware-aware prefix scheduler | No official repo; checkpoints MIT (`deepseek-ai/DeepSeek-V4-Pro-DSpark`) and Apache-2.0 (`RedHatAI`, `openbmb`); community code MIT |
| PARD (target-independent parallel draft) | 2504.18583 | +~1-2 GB draft (est.) | 3.67x Llama3.1-8B in vLLM, 1.15x over EAGLE-3 | One drafter serves a whole model family; COD converts AR drafts to parallel | MIT (`AMD-AGI/PARD`); no official checkpoints |
| PARD-2 (acceptance-aligned objective, dual-mode) | 2605.08632 | same | **6.94x** Llama3.1-8B; 1.9x over EAGLE-3 | Training objective aligned with consecutive-acceptance, not token accuracy | MIT (`AMD-AGI/PARD`); no official checkpoints |
| SEED (self-spec, no drafter checkpoint) | 2609.36590 | **zero** (reinterprets target as encoder-decoder; drafts from cached deep reps) | 2.7x avg on 4B-scale, 28% faster than EAGLE-3 (NeurIPS 2026) | Draft quality of MTP at near-zero draft cost; best memory profile | `lhk2004/SEED` — no LICENSE file (blocked) |
| EAGLE-3 (feature-level AR, reference point) | 2503.01840 | +~0.7-1.5 GB (1-2 layers + fusion + embedding), +3-6% KV | up to 6.5x (own harness); ~3.0-3.5x in DFlash's Qwen3 harness; +40% SGLang throughput at batch 64 | Strongest AR drafter; harness gap vs DFlash shows parallel drafting's structural edge | Apache-2.0 (`SafeAILab/EAGLE`); Apache-2.0 checkpoints (`yuhuili/EAGLE3-*`) |

### Training-free tree extensions (stack on any block drafter)

| Method | arXiv | Extra memory | Reported decode speedup | Theoretical note | Code / checkpoints |
| --- | --- | --- | --- | --- | --- |
| DDTree (best-first tree from block marginals) | 2604.12989 | 0 weights; transient tree buffers | best-in-class acceptance on DFlash | Ancestor-only mask verifies the tree in one target pass | No official code |
| DARTree (AR-correction head expanded to trees) | 2608.13524 | 0 (reuses pretrained AR head) | **9.73x lossless**, up to 12.97 accepted tokens/round (+98.6% vs DFlash) | Best single-stream number found; path-conditional scoring, batch-then-prune | `VILA-Lab/DARTree` — no LICENSE file (blocked; reimplementable from paper) |
| DominoTree (conditional tree on Domino) | 2607.08642 | 0 | up to 7.3x on Qwen3-8B; +12% vs Domino in SGLang; +29-36% accepted length at long context | Recomputing the causal correction per root-to-node path is the gain (+4.7%) | MIT (`slin-zhq/Domino-Tree`); needs a compatible drafter checkpoint |
| PCTree (parent-conditioned children on DSpark) | 2608.02123 | 0 | 6.60x vs 6.14x (DSpark, Qwen3-4B GSM8K, B=16); tau 9.41 -> 11.16 | Converts a linear semi-AR draft into a tree with no retraining | No official code |

### Training-side upgrades (no inference memory change)

| Method | arXiv | Extra memory | Reported gain | Theoretical note | Code |
| --- | --- | --- | --- | --- | --- |
| Draft-OPD (on-policy distillation) | 2605.29343 | 0 | >5x; +23% over EAGLE-3, +13% over DFlash | Fixes the offline-to-inference mismatch of SFT-trained drafters | `Simplified-Reasoning/Draft-OPD` — no LICENSE file |
| Test-Time Speculation (online draft adaptation) | 2605.09329 | 0 | +41-72% acceptance, scaling with output length | Verification already invokes the target: free teaching signal | No official code |
| BV loss (block-verification-aware objective) | 2609.34832 | 0 | +13-21% accepted tokens | Derives the loss from the sequence-level acceptance rule | No official code |
| AdaEDL (entropy-based draft stopping) | 2410.18351 | 0 | +10-57% over static draft length | Lower bound on acceptance from draft-logit entropy | No official code (few-line algorithm) |
| EntMTP (entropy-scheduled tree topology) | 2606.27550 | 0 | 1.15x over Hydra, 1.36x over Medusa | Speculation depth should track local predictability | No official code |
| DistillSpec (draft-target distillation) | 2310.08461 | 0 | +10-45% over vanilla SD | Divergence choice per task/temperature | No official code |
| Post-training MTP heads + relaxed verification | 2610.00888 | 0 | matches joint-pretrained MTP with 10^3-10^4x fewer tokens; +12-16% from bounded-drift verification | MTP quality is a post-training problem, not a pretraining constraint | No official code |
| DBloom (block-size expansion 16 -> 24) | 2608.30427 | ~0 | +0.8-1.37 committed tokens on ceiling-bound workloads | Recovering "stranded speedup" when every block is fully accepted | No official code |

### Verification-side upgrades (drop-in, lossless, zero memory)

| Method | arXiv | Extra memory | Reported gain | Theoretical note | Code |
| --- | --- | --- | --- | --- | --- |
| Block verification (BV) | 2403.10444 | 0 | +5-8% wall-clock over token verification | **Provably optimal** among on-path verifiers; never worse — should be the default | No dedicated repo; in vLLM/SGLang (Apache-2.0) |
| Greedy multi-path BV (GBV) | 2602.16961 | 0 | +30% block efficiency, -15% walltime vs BV; >15% over SOTA multi-path on Llama-3 70B | LP shows BV is optimal even with off-path probabilities; extends to multi-path | No dedicated repo |
| SpecTr-GBV (multi-draft + BV as optimal transport) | 2604.25925 | 0 | best block efficiency of i.i.d. multi-draft methods | Proves the optimal expected acceptance bound, improving with number of drafts | No dedicated repo |
| Approximate Speculative Decoding (ASD, lossy) | 2608.03447 | 0 | +3-15% throughput | Budgeted acceptance of mismatches; reduces to strict at zero budget | Repo linked in paper; license unverified |

### No-draft and low-memory methods

| Method | arXiv | Extra memory | Reported decode speedup | Theoretical note | Code / checkpoints |
| --- | --- | --- | --- | --- | --- |
| Lookahead (Jacobi + n-gram pool) | 2402.02057 | 0 weights; ~0.5-1 GB buffers (est.) | 1.8x MT-bench, 4x code on multiple GPUs | Exact, no auxiliary model; trades FLOPs for steps | Apache-2.0 (`hao-ai-lab/LookaheadDecoding`) |
| LLMA / prompt-lookup drafting | 2304.04487 | 0 | >2x on overlap-heavy tasks (summarization, editing, agents) | Copies spans from context; the cheapest possible drafter | Built into vLLM (Apache-2.0) |
| REST (retrieval drafts) | 2311.08252 | 0 GPU weights; CPU datastore | 1.62-2.36x | Plug-and-play, training-free | Apache-2.0 (`FasterDecoding/REST`) |
| LayerSkip (early-exit self-spec) | 2404.16710 | 0 (training recipe) | 1.34-2.16x | Shares compute and KV between draft and verify — lowest-memory speculation class | **CC BY-NC 4.0 — non-commercial** (`facebookresearch/LayerSkip`, code and checkpoints) |
| Kangaroo (double early exit) | 2404.18911 | +67M params (0.13 GB) vs 591M for Medusa-1 | up to 1.68x | Confidence-threshold draft stopping | `Equationliu/Kangaroo` — no LICENSE file |
| GliDe with a CaPE (draft reuses target KV) | 2402.02082 | shares target KV | 2.17x -> 2.61x | Draft reads target's cached keys/values: no duplicate KV | No official code found |
| Looped-model self-spec (LoopSpec / WaveFront / DAS-Wave) | 2609.17184 / 2609.23033 / 2609.34538 | 0 | 6.83x / 2.4-4.8x / 4.0-7.0x | Intermediate recurrence depths are free drafters for weight-tied models | Per-paper repos; licenses unverified |

### Long context and high concurrency

| Method | arXiv | Extra memory | Reported gain | Theoretical note | Code / checkpoints |
| --- | --- | --- | --- | --- | --- |
| MagicDec (sparse-KV draft, target itself) | 2408.11049 | 0 | 2.51x at batch 32-256 on long contexts (Llama3.1-8B) | Bottleneck-shift analysis: KV (not weights) dominates long-context batches; drafting strategy provably improves with batch | Apache-2.0 (`Infini-AI-Lab/MagicDec`) |
| LongSpec (constant-size draft KV) | 2502.17421 | draft KV reduced from O(ctx) to O(1) | 3.26x long-context; 2.25x wall-clock on AIME24 with QwQ | Fixes the three long-context failure modes of SD (draft KV, train-test mismatch, tree attention) | MIT (`sail-sg/LongSpec`); MIT checkpoints (`sail/longspec-*`) |
| ASPIRE (asynchronous per-request self-spec) | 2609.17943 | 0 | 1.70-4.58x batched long-ctx; +27% over prior self-spec | Per-request schedules beat synchronized draft-verify phases | No official code found |
| DScale (concurrency scaling for block-SD) | 2609.37532 | +112K-param predictor | +43.9-48.8% throughput over DFlash at concurrency 8-32 | Path-aware tiling + dynamic verify-length reclaim padding waste | No official code |
| SpecExec (massive parallel speculation, offload) | 2406.02532 | transient tree KV | up to 20 tokens per target pass; 50B+ models on consumer GPUs | Offloading makes batch-of-tokens nearly free | `williamsentosa95/SpecExec` — no LICENSE file |
| Sequoia (DP-optimal tree, hardware-aware) | 2402.12374 | draft + tree buffers | 4.04x (7B), 3.73x (13B), 2.27x (33B); ~10x offloading 70B | Closed-form optimal tree for a given verify budget | `Infini-AI-Lab/Sequoia` — no LICENSE file |
| SpecInfer (tree verification) | 2305.09781 | tree KV | 1.5-2.8x distributed; 2.6-3.5x offloading | The original token-tree verifier | Apache-2.0 (`flexflow/FlexFlow`) |

### Newest refinements (late September 2026 scan)

A fresh arXiv sweep found no fundamentally new axis, but eight more refinements of the same
winning family (block-parallel drafting + causal correction + trees + scheduling):

| Method | arXiv | What it adds | Reported result |
| --- | --- | --- | --- |
| UBTree | 2609.39972 | Unigram proposer + bigram transition selector for tree construction; entropy-robust draft diversity | 5.84-6.94x AR; beats DARTree in all 28 comparisons; beats DSpark at production scale |
| DPara | 2609.27396 | Overlaps drafting with verification ("parallel speculative decoding"); precomputes draft representations for every acceptance boundary | 3.21-3.52x; removes drafting from the critical path — a genuinely new serving axis |
| DEdit | 2609.38510 | Diffusion drafter that iteratively edits its own draft (token-to-token repair, bidirectional context) | 5.72-5.97x macro-avg, best of evaluated drafters on Qwen3-4B/8B |
| H-Spec | 2609.24197 | Hybrid Mamba-attention drafter with **no drafter-side KV cache** (reuses target KV in place + last-token hidden state) | +5.0-13.3% accepted length; lowest KV utilization under concurrency |
| RheoSampling | 2609.21827 | Fixes the one-hot collapse of dynamic trees (EAGLE-3-style) at temperature > 0; lossless stochastic trees | Improves acceptance over dynamic trees at sampling temperature |
| Acceptance-aware training (EAL/WTV losses) | 2609.24150 | Trains drafters on expected accepted length directly, not CE/KL; +GRPO stage | Consistent acceptance gains over KL-based training |
| SpecStream | 2609.33184 | Streams KV from CPU during verification; drafts inside transfer bubbles (long-context, memory-bound) | 1.32-1.41x over KV-offload baselines; +55.4% tokens/GPU |
| Spexis | 2609.34370 | Speculation as a multi-GPU parallelism axis (overlaps with pipeline/tensor parallelism) | up to +34% over optimal PP/TP |

None of these has shipped production evidence yet, and code availability is unverified as
of this scan. UBTree and DPara are the two worth tracking: UBTree's "beats DARTree 28/28"
claim makes it the current single-stream candidate to watch, and DPara's draft-verify
overlap is the only new serving-level idea in the batch.

**UBTree vs DSpark integration verdict:** UBTree posts the better paper number (and even
beats DSpark in its own evaluation), but it is days old, single-source, has no released
code or checkpoints found, and its tree costs batch capacity at concurrency unless you add
scheduling anyway. DSpark has the concurrency machinery built in (confidence-scheduled
verify length, hardware-aware prefix scheduler), live-traffic production evidence at
DeepSeek, and MIT/Apache-2.0 checkpoints for all three model families TensorFold serves.
Integrate DSpark; revisit UBTree once it ships code and replication. The two compose —
UBTree's entropy-robust trees can layer on a DSpark-style backbone later.

## Observable Operator Models: elegant theory, open gap

Observable Operator Models (Jaeger, *Observable Operator Models for Discrete Stochastic Time
Series*, Neural Computation 12(6), 2000) represent a stochastic process as a linear dynamical
system over a compact "predictive state" `w`: with one linear operator `tau_a` per symbol,
a readout functional `sigma`, and state updates `w' = tau_a w / P(a|w)`.

**Why the theory is attractive for fast decoding:**

- The probability of any length-k word `u` is a single linear product
  `P(u) = sigma * tau_{u_k} ... tau_{u_1} * w_0`, with **no per-step renormalization**. The
  full lookahead tree over k steps — every branch, exactly — is k batched matrix products
  (einsums), so an OOM drafter has O(1) sequential rollout depth: it can emit an entire
  path-conditioned block in parallel.
- That is precisely the structure the information-floor result wants: word operators compose
  **along paths**, so slot k is conditioned on slots 1..k-1 exactly, in closed form — what
  Domino's GRU head and DSpark's Markov/RNN heads only approximate, and what marginal block
  diffusion (vanilla DFlash) provably lacks.
- The state is a sufficient statistic (a "predictive state"): it could be distilled directly
  from the target model's hidden state once per round, then roll out k steps with tiny linear
  ops — no draft KV cache at all.
- Rich theory exists: OOMs strictly generalize HMMs/PFAs at equal dimension; norm-observable
  OOMs are equivalent to predictive state representations (Singh et al., UAI 2004,
  arXiv:1207.4167); PSRs/OOMs are equivalent to stochastic weighted automata and to matrix-
  product-state Born machines (arXiv:2010.10653), and connect to causal-state complexity
  (arXiv:1108.3984). Spectral learning is consistent and O(n) for continuous data
  (arXiv:1609.00932), and an approximation theory for infinite-dimensional processes is
  under construction (arXiv:2404.12070).

**Why nothing shipped at LLM scale (status as of Oct 2026):**

- Naive parameterization needs one d-by-d operator per vocabulary symbol: d^2 * |A| floats.
  For d=1024 and a 128k vocab that is ~1.3e11 floats (~260 TB) — infeasible. Feasible
  variants require factorized, token-embedding-conditioned low-rank operators, which is
  structurally the territory of linear attention / SSM state updates — i.e., it stops being
  a pure OOM and becomes a small linear-recurrence draft model that must still be proven to
  track a deep target.
- Learnability is the classic weakness: unconstrained OOM learning hits the
  *negative-probability problem* and state drift; stable training needs constrained
  parameterizations (e.g., retraction-based learning on the Stiefel manifold,
  arXiv:1912.02098) whose scaling behavior at LLM vocabulary/dimension is untested.
- An arXiv full-text search for "observable operator models" returns only seven papers, none
  applied to LLM decoding. The empirical descendants of the same idea — fill a block of
  future tokens in one parallel step — are the block-diffusion and semi-AR drafters above,
  which solve the problem with learned deep nets instead of exact linear operators.

**Theoretical upside if it worked:** exact k-step lookahead distributions at k parallel
matmuls, zero draft KV, drafter memory ~O(d^2 + d*|A|*r) for rank-r factorized operators
(tens to hundreds of MB), and provable acceptance bounds (acceptance probability is
1 - TV distance to the target, so a well-distilled OOM gives a distribution-level guarantee
no neural drafter offers). It is the most promising *unexplored* direction found in this
survey, and the right target shape — per the information floor — is a path-conditioned
OOM/PSR drafter distilled from the target, not a marginal one.

## What to adopt, given MTP + DFlash2 already ship

Ordered by (theoretical gain) / (integration cost). License status from the 2026-10-03
check is folded in — everything below is shippable commercially without reimplementation,
except where noted.

1. **Block verification as the default verifier** (2403.10444). Provably never worse,
   +5-8% wall-clock, zero memory, no training. Engine-side in vLLM/SGLang (Apache-2.0).
   Then GBV (2602.16961) for multi-path setups.
2. **Trees on top of DFlash2** (DDTree/DARTree/DominoTree style). Training-free given a
   block drafter; converts per-position marginals into path coverage; papers report up to
   +98.6% accepted tokens and the best lossless number found (9.73x). Cost: transient tree
   buffers. DominoTree's code is MIT; DARTree's repo is unlicensed, so reimplement its
   scoring from the paper.
3. **Semi-AR correction heads** (Domino 2605.29707, DSpark 2607.05147). The information-floor
   result is the theoretical argument; the DeepSeek-V4 production deployment (+60-85% per-user
   speed vs MTP-1) is the empirical one. Same memory class as DFlash2. DSpark checkpoints are
   commercially usable (MIT/Apache-2.0); Domino's repo is unlicensed.
4. **Adaptive draft/verify scheduling** (AdaEDL, EntMTP, DSpark's confidence-scheduled
   verification, DScale). Essential above concurrency ~8, where static speculation backfires.
   AdaEDL is a few lines to reimplement; DScale's recipe is in the paper.
5. **On-policy drafter post-training** (Draft-OPD, BV loss, TTS). No serving-stack change;
   +13-23% acceptance on the same checkpoints; also the cheapest fix for long-generation
   acceptance decay. No official code — reimplement from the papers.
6. **Long-context mode**: MagicDec-style sparse-KV drafting (Apache-2.0) + LongSpec
   constant-KV drafts (MIT); ASPIRE for asynchronous batches (reimplement). Matters once
   context passes ~16K.
7. **Memory-constrained alternative**: SEED (2609.36590) gets EAGLE-3-class speed with zero
   drafter weights, but its repo has no license — for shipping today use PLD/REST
   (Apache-2.0, zero drafter memory, 1.1-1.6x MTP) until SEED's licensing is clarified.

## References

Drafting, block-parallel family: DFlash 2602.06036; Domino 2605.29707; DominoTree 2607.08642;
DSpark 2607.05147; PCTree 2608.02123; DRelay 2610.01439; DeLS-Spec 2607.07409; HyperDFlash
2606.26744; AngelSpec/DFly 2607.25852; DScale 2609.37532; DBloom 2608.30427; ReTrace 2608.29748;
DARTree 2608.13524; DDTree 2604.12989; information floor 2608.27339; multimodal diffusion-drafting
survey 2608.20743.

MTP family: DeepSeek-V3 2412.19437; post-training heads 2610.00888; AdaMTP 2608.00434;
EntMTP 2606.27550; CLP 2606.10935; Bebop 2606.12370; PARD 2504.18583; PARD-2 2605.08632.

Self-speculative: LayerSkip 2404.16710; Kangaroo 2404.18911; SEED 2609.36590; ASPIRE 2609.17943;
SparseSpec-L 2607.27735; S2-MoE 2608.15018; LoopSpec 2609.17184; WaveFront 2609.23033;
DAS 2609.34538; FlexEE 2609.17008.

Verification theory: block verification 2403.10444; GBV 2602.16961; SpecTr-GBV 2604.25925;
BV for diffusions 2606.13426; ASD 2608.03447; BV loss 2609.34832; AdaEDL 2410.18351.

No-draft / retrieval / trees: Lookahead 2402.02057; LLMA 2304.04487; REST 2311.08252;
DistillSpec 2310.08461; SpecInfer 2305.09781; Sequoia 2402.12374; SpecExec 2406.02532;
MagicDec 2408.11049; LongSpec 2502.17421; speculative sampling 2302.01318; EAGLE 2401.15077;
EAGLE-2 2406.16858; EAGLE-3 2503.01840; Medusa 2401.10774; Hydra 2402.05109; GliDe+CaPE 2402.02082.

Draft training: Draft-OPD 2605.29343; Test-Time Speculation 2605.09329.

Late September 2026 additions: UBTree 2609.39972; DPara 2609.27396; DEdit 2609.38510;
H-Spec 2609.24197; RheoSampling 2609.21827; acceptance-aware training 2609.24150;
SpecStream 2609.33184; Spexis 2609.34370; HAWK (multimodal drafting) 2610.00623;
Redline (risk-guaranteed serving configs) 2609.33887. ParoQuant (quantization, not
drafting): 2511.10645.

Surveys: Xia et al. 2401.07851; Hu et al. 2502.19732; Ryu and Kim 2411.13157.

Observable operator models: Jaeger, Neural Computation 12(6), 2000; PSRs arXiv:1207.4167;
OOM/WFA/tensor-network equivalences arXiv:2010.10653; process dimension arXiv:1108.3984;
spectral learning arXiv:1609.00932; stable learning on the Stiefel manifold arXiv:1912.02098;
approximation theory arXiv:2404.12070.
