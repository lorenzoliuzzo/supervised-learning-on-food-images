# Training roadmap

**Status:** Phases A-D done, including the addendum and the single 90-epoch full run — **63.83% val-test top-1 / 81.79% top-3 / 87.39% top-5**, the report's headline number (`checkpoints/full-90ep-lr0.8-best.pth.tar`, epoch 86); baseline trunk, GAP head, plain recipe, lr=0.8, batch 256, crop-scale-min 0.08, all unmodified from what this phase settled on. Everything under baseline (narrower trunks, alternate pooling heads, batch size 160/256/512, crop-scale 0.25/0.40, six regularization axes total) was proxied and none beat it. **No run has ever been seeded (#33)**, including this one — every accuracy figure in this file is a point estimate, not an exactly reproducible one. Phase E (self-supervised track) and the report remain; Phase E's SimSiam scaffolding (`src/simsiam.py`, `main.py --init-encoder`, `notebooks/colab_gpu_probe.ipynb`) landed 2026-08-03, no pretraining run has happened yet · **Baseline `main`:** `820f347` · **Last measured:** 2026-08-03

Every number here was measured on the project box (RTX 5050 Laptop, 8 GB VRAM,
16 threads) at 176 px / bf16 / `channels_last`, in `performance` power profile,
and is reproducible with `benchmarks/`. No accuracy figure appears anywhere in
this file, because no training run has happened yet. Nothing enters here that we
have not run.

**On reading the throughput figures.** The GPU runs against a ~50 W
`SW_POWER_CAP` with its clock oscillating 2175–2340 MHz against a 3090 MHz
maximum, so short timing windows are unreliable — an early version of the
benchmark swung 45% on one variant between sweeps. `benchmarks/bench.py` now
times three 40-step rounds and keeps the best; back-to-back sweeps agree within
2–5% and the ranking is stable. These are GPU-only rates on synthetic batches,
so treat them as an upper bound on real epoch time, and as *ratios* between
variants rather than promises about the wall clock.

---

## 1. Where we are

Phase 0 (unblock) and most of Phase 1 (architecture) are done — #1, #2, #3 and #8
are closed, 12 tests pass, `ruff check .` is clean.

| Was | Now |
| --- | --- |
| `FoodCNN` could not be instantiated (#1) | fixed |
| validation read `train_set` (#2) | `dataset_paths()` + tests pinning labels, not just directories |
| `_identity_block` discarded the residual (#3) | real `ResidualBlock`, activation after the add |
| 6.42M params in one FC layer (#8) | true GAP head; `Linear(512, 251)` = 129k |
| `test_parameter_budget` did not exist | exists, plus gradient-reachability and pooling tests |

Current model: **6,578,939 params** (65.8% of budget), 98.0% of them in
`features`; 1688 img/s, peak 1.26 GiB, 1.2 min/epoch, ~1.8 h for 90 epochs.

## 2. Measurements that shape everything below

**Neither the parameter cap nor VRAM binds any more.** 3.42M params and 6.7 of
8 GiB are unspent. Wall-clock is the only real constraint left.

**The train split is near-balanced, so long-tail machinery is out of scope.**
Median 471 images/class, p10 366, p90 580; exactly one class has fewer than 200
(34 images). The 19.3x "imbalance ratio" is that single outlier. Class-balanced
loss, logit adjustment and decoupled classifier training are dropped. Label
*noise* is still real — train is web-crawled, val is clean — and stays in scope.

**The loader stays ahead, so augmentation is free — but use 8 workers.** At
176 px against ~1.7k img/s of GPU demand, `crop+flip` delivers 4.4-4.7k at any
worker count, and TrivialAugment 4.3k at 8 workers (2.5x headroom) falling to
2.4k at 16 (1.4x). Oversubscribing workers actively costs throughput on 15 GB of
RAM. Augmentation strength is an accuracy decision, not a throughput trade, but
the margin is 2.5x rather than the 5x an earlier under-clocked measurement
suggested — worth re-checking if the trunk gets faster still.

## 3. Spending the 3.42M headroom — measured, not guessed

All variants under the 10M cap, at 176 px / batch 160:

| Variant | Params | img/s | 90 ep | Peak | Final map |
| --- | --- | --- | --- | --- | --- |
| baseline `[2,2,2,1]` 64-512 | 6.58M | 1688 | 1.8 h | 1.26 GiB | 6x6 |
| `[2,2,3,1]` 64-512 | 7.76M | 1585 | 1.9 h | 1.31 GiB | 6x6 |
| `[2,2,4,1]` 64-512 | 8.94M | 1475 | 2.0 h | 1.36 GiB | 6x6 |
| `[2,3,3,1]` 64-512 | 8.06M | 1460 | 2.0 h | 1.39 GiB | 6x6 |
| `[2,2,2,2]` 64-448 | 9.46M | 1608 | 1.8 h | 1.30 GiB | 6x6 |
| wider@11 64-320-512 | 7.98M | 1494 | 2.0 h | 1.30 GiB | 6x6 |
| **5 stages 48-512 `[2,2,2,2,1]`** | **9.08M** | **1960** | **1.5 h** | **1.04 GiB** | **3x3** |
| uniformly wider 72-576 | 8.31M | 1062 | 2.8 h | 1.41 GiB | 6x6 |
| 5 stages, no stem MaxPool | 9.08M | 684 | 4.3 h | 2.16 GiB | 6x6 |

- **Widths must be multiples of 32.** The 72-144-288-576 trunk has *fewer* params
  than the 5-stage variant and runs 46% slower — it falls off the tensor-core
  fast path. This is the single largest own-goal available here.
- **The 5-stage variant is +38% params, +16% throughput and -17% VRAM at once.**
  It pays for this with a fourth downsample: the final map goes 6x6 to 3x3, which
  is precisely the resolution fine-grained classes need. Candidate, not decision —
  Phase C settles it.
- **Resolution dominates cost.** Dropping the stem MaxPool to keep the map at 6x6,
  at identical parameter count, costs 2.9x throughput (1960 to 684 img/s) and 2x
  VRAM. Every other knob here is worth less than this one.

---

## Phase A — pre-run gates (no GPU-hours)

- [x] **BatchNorm after the three shortcut projections** (`src/model.py:24`). They are
      bare `Conv2d`s, so an unnormalized branch is added to a BN'd one; torchvision's
      ResNet uses `conv1x1 + norm` in `downsample` for exactly this reason.
- [x] **`bias=False` on every conv followed by BatchNorm** — 3,776 no-op parameters.
- [x] **Zero-init the second BN's gamma in each block**, so each block starts as
      identity. This is what makes warmup plus a high LR safe.
- [x] **`main.py` training loop**, keeping the argparse / `main_worker` shape:
  - [x] `torch.autocast("cuda", dtype=torch.bfloat16)` — **no `GradScaler`**; bf16
        does not need one.
  - [x] `channels_last` on model and inputs.
  - [x] cosine schedule with 5-epoch linear warmup, replacing `StepLR(30, 0.1)`.
  - [x] `CrossEntropyLoss(label_smoothing=0.1)`; SGD with `nesterov=True`.
  - [x] `--workers` default 4 to 8; train crop 224 to 176.
  - [x] drop the `nn.DataParallel` wrapper on a single GPU — it costs a
        scatter/gather per step and prefixes every checkpoint key with `module.`.
- [x] **Tests** (never reading `food251/`):
  - [x] single-batch overfit gate — 200 steps on fixed `randn` drives loss below
        0.1. The cheapest possible proof the trunk can learn, and the one thing
        that would otherwise waste a 2-hour run.
  - [x] `channels_last` survives the forward pass.
  - [x] warmup+cosine LR hits expected values at epochs 0, 5 and last.

Params after the BN/bias fixes: **6,576,955** (bias removal costs slightly more
than the new shortcut BNs add back). 17 tests pass, `ruff check .` clean.

## Phase B — freeze the protocol before any run

- [x] **Split the 11,994-image val set 50/50, stratified, into `val-dev` and
      `val-test`** (decided). Ablations and checkpoint selection read `val-dev`
      only; `val-test` is touched once, for the report's headline number. Without
      this the headline is selected-on. The split must not drift between runs —
      every one of the 251 classes lands within 1 image of an exact 50/50 split
      (6,063 dev / 5,931 test).
      Implemented in `src/make_val_split.py` (per-class shuffle-and-halve, fixed
      seed 251) as a pure function of `food251/meta/val_labels.csv`, so it's
      byte-identical on every regeneration — that determinism is what replaces
      committing the output: `splits/val_split.csv` is gitignored (12k rows,
      no information not already in the dataset + one seed), regenerate with
      `python src/make_val_split.py` before the first training run on a fresh
      checkout.
- [ ] **Fixed seed** and [x] **a fixed 15-epoch proxy protocol** for every
      comparison. At ~1.2 min/epoch a proxy run is ~18 minutes, so comparisons
      that would be unaffordable at 90 epochs are routine. Rank on the proxy,
      confirm once at full length.
      `main.py --val-subset dev` (default) reads `splits/val_split.csv` and
      restricts validation to val-dev automatically; `--val-subset test` is the
      explicit, one-time opt-in for the report's headline number.

      **The seed half of this was never implemented** (found 2026-08-02, see
      issue #33). `main.py`'s `--seed` defaults to `None`, nothing passes it,
      and `proxy_sweep.py` doesn't set one either, so *every* Phase C and
      Phase D number came from an unseeded, unrepeatable run. Unticked here
      rather than quietly left green. Wherever the phases below say "single
      seed each", read "unseeded". The proxy-protocol half is real and is
      what the comparisons below actually rest on.
- [x] **Per-run logging**: config hash, per-epoch train/val top-1 and top-5,
      wall-clock, peak VRAM. Capture what the report needs the first time.
      `src/runlog.py` (`RunLog`), wired into `main.py`'s epoch loop and into
      `benchmarks/proxy_sweep.py`; writes gitignored JSON under `runs/`.

## Phase C — pick the architecture (~1.2 GPU-h)

- [x] Four 15-epoch proxy runs on `val-dev` top-1, real data via
      `benchmarks/proxy_sweep.py` (logs in gitignored `runs/phaseC/`), single
      seed each, ~83 min total wall clock:

      | Variant | Params | val-dev top1 | val-dev top5 | Peak VRAM |
      | --- | --- | --- | --- | --- |
      | `[2,2,2,2]` 64-448 | 9.46M | **49.99%** | 78.29% | 2.04 GiB |
      | `[2,2,4,1]` 64-512 | 8.94M | 49.73% | 77.93% | 2.14 GiB |
      | baseline `[2,2,2,1]` 64-512 | 6.58M | 48.82% | 77.12% | 1.99 GiB |
      | 5-stage 48-512 `[2,2,2,2,1]` | 9.08M | 47.42% | 75.94% | 1.61 GiB |

      **The 5-stage variant's 3x3 final map does cost accuracy** — it's
      1.4-2.6 points behind the three 6x6-final-map trunks despite having the
      second-most parameters, so its throughput/VRAM win from §3 does not
      carry over to a real training signal. Ruled out.

      The other three are within 1.2 points of each other on a *single*
      15-epoch run with no repeated seeds — not enough to call a winner with
      confidence. `[2,2,2,2]` 64-448 nominally leads but only by 0.26 points
      over `[2,2,4,1]`, well inside plausible run-to-run noise. Baseline
      trails the leader by 1.17 points at 30% fewer parameters. **This is an
      architecture decision, not a measurement one — recorded here, not
      acted on.** See "Open decision" below.

      (`benchmarks/trunk_variants.py`'s stem conv was missing `bias=False`, a
      Phase A change that hadn't propagated there; fixed after this sweep ran,
      so the four runs above trained with a 64-parameter stem bias none of
      the graded architecture has. Negligible next to the multi-point gaps
      above, not worth rerunning for; future sweeps use the corrected stem.)

## Phase C addendum — below baseline, and the head (2026-08-02, ~1.6 GPU-h)

Phase C only ever swept 6.58M-9.46M: baseline and up. Diminishing returns
going up says nothing about going down, and the head had never been an
experimental axis at all. Five 15-epoch legs (issues #28, #29), all through
`benchmarks/proxy_sweep.py` in one pass under the **Phase D recipe**
(plain, lr=0.8, batch 256) rather than Phase C's pre-LR-search 0.1 — so
these are not comparable to the Phase C table above, which is why the
baseline was re-run here as its own leg rather than quoted from it.

| Variant | Params | val-dev top1 | Δ vs baseline | McNemar p | Peak VRAM |
| --- | --- | --- | --- | --- | --- |
| **narrow `[2,2,2,1]` 64-384** | **5.18M** | **55.22%** | **+0.00** | **1.0000** | 1.96 GiB |
| baseline `[2,2,2,1]` 64-512 | 6.58M | 55.22% | — | — | 1.99 GiB |
| head: spatial attention | 6.59M | 55.01% | -0.21 | 0.6826 | 1.99 GiB |
| narrow `[2,2,2,1]` 32-256 | 1.68M | 48.64% | -6.58 | <0.0001 | 1.08 GiB |
| head: GAP+GMP concat | 6.71M | 37.51% | -17.71 | <0.0001 | 1.99 GiB |

(Accuracies are each leg's saved per-image correctness vector, which
`proxy_sweep.py` scores in bf16; the per-epoch logs score in fp32 and read
~0.03 lower. All five vectors are produced identically, which is what makes
the pairing valid.)

- **There is an accuracy floor, and baseline sits well above it.** 64-384 at
  21% fewer parameters is not merely close to baseline, it is a dead heat —
  `b`=441 discordant one way, `c`=441 the other, p=1.0000. 32-256 then loses
  6.58 points at p<0.0001. So the flat region extends below baseline but not
  far, and it ends in a cliff rather than a slope, somewhere between 1.68M
  and 5.18M. **Capacity is not the binding constraint at 6.58M and wasn't at
  5.18M either** — the §2 claim that the parameter cap doesn't bind now has
  evidence on both sides of baseline instead of one.
- **Neither pooling head helps.** Spatial attention is a clean null (-0.21,
  p=0.68): learned pooling weights bought nothing over uniform averaging.
- **GAP+GMP underfits rather than overfits, and its LR is the suspect.** Its
  train loss starts *above* baseline's at epoch 0 (5.595 vs 4.996) and never
  closes (3.851 vs 2.994 at epoch 14), with val top-1 still climbing steeply
  at the end (31.4% to 37.5% over the last two epochs). No divergence, no
  NaN — a slower trajectory, not a broken one. Max-pooled activations are
  unbounded and much larger than mean-pooled ones, so concatenating hands the
  classifier two input blocks on very different scales, and lr=0.8 was tuned
  for a head without that problem. **Same caveat as GCE and EMA below: this
  says "needs its own LR or a norm on the concatenated features," not "wrong
  for this dataset."** Not spending more budget on it — the attention head's
  clean null is weak evidence that pooling isn't where the accuracy is.

**Adopting 64-384 is an architecture decision, not a measurement one, and is
not taken here.** For: 21% fewer parameters at a literal dead heat, on a
project graded against a parameter cap. Against: it buys no accuracy, no
meaningful wall-clock (20 vs 21 min) and no VRAM that matters, and baseline
is the trunk every Phase D number was measured on. `model.py` unchanged.

## Smoke test — pipeline validated, not a Phase D result

A full 90-epoch run on the unmodified baseline `FoodCNN` (Phase A recipe,
`--val-subset dev`, no Phase D recipe tuning yet), run to prove the pipeline
survives full length before spending Phase D's GPU-hours on it. It is not the
"one 90-epoch run" Phase D calls for — that one happens after the trunk and
recipe are both decided.

- **Converged at 61.57% val-dev top-1 / 85.67% top-5** (best epoch 82:
  62.08% / 85.52% — `model_best.pth.tar`, not the final checkpoint, is the
  one to keep). Up from ~7% at epoch 0, matching the tail-heavy climb the
  15-epoch proxy already hinted at in §Phase C, and well past both the
  Food-101-from-scratch loose anchor (mid-50s%) and the proxy's own 47-50%.
- **No crashes, NaNs, or OOM across the full run.** Checkpointing (regular +
  best) worked correctly. Peak VRAM 1.99 GiB — still far under the 8 GiB
  ceiling even at the full 90 epochs.
- **Wall clock: 2.31 h, not the ~1.8 h estimated in §3.** Per-batch time rose
  from ~0.16 s early on to ~0.26 s past epoch ~60 (`clocks_event_reasons.active`
  showed `SW_POWER_CAP` again) — sustained 90-epoch load heat-soaks this
  laptop GPU more than the short benchmark sweeps in `benchmarks/` ever
  triggered. The existing power-cap caveat at the top of this file already
  says to treat throughput figures as an upper bound; this is the concrete
  case where a full run ran ~28% slower than a synthetic estimate. Phase D's
  budget should assume real runs, not benchmark rates.

## Phase D — recipe, then the full run (~4.2 GPU-h)

- [x] 15-epoch proxies on the winning trunk: LR in {0.05, 0.1, 0.2, 0.4} at
      batch 256; TrivialAugment vs RandAugment; Mixup/CutMix on/off; EMA on/off.

      LR sweep (baseline trunk, batch 256, ~23-27 min/run — see
      `runs/plots/comparison-val_acc1.png`):

      | LR | val-dev top-1 | top-3 | top-5 |
      | --- | --- | --- | --- |
      | 0.05 | 43.96% | — | — |
      | 0.10 (prior default) | 49.41% | — | — |
      | 0.20 | 52.56% | — | — |
      | **0.40** | **54.02%** | 74.16% | 81.02% |

      (Top-3/top-5 only logged from the 0.4 leg onward — top-3 tracking
      landed mid-sweep; see "Decisions taken".) Follow-up at the range's
      edge, same protocol:

      | LR | val-dev top-1 | Δ vs previous |
      | --- | --- | --- |
      | 0.40 | 54.02% | — |
      | 0.60 | 54.96% | +0.94 |
      | **0.80** | **55.58%** | +0.62 |

      **Stopping the LR search at 0.8.** Gains shrink cleanly each doubling
      (+5.45, +3.15, +1.46, +0.94, +0.62) — a textbook diminishing-returns
      curve, so the next doubling (1.6) would likely buy ~0.3-0.4 points.
      0.8's train-loss curve is smooth (lowest final loss, no divergence),
      but its val-accuracy curve is visibly noisier mid-run than 0.4/0.6's
      (a dip around epochs 8-9) — not instability, since loss never wobbles,
      but a sign of pushing into a noisier optimization regime. Diminishing
      reward plus growing noise flips the risk/reward; **0.8 is the LR
      Phase D carries forward.**

      **Augmentation, 15-epoch proxy (baseline trunk, lr=0.8, batch 256):**

      | Recipe | val-dev top-1 | top-3 | top-5 |
      | --- | --- | --- | --- |
      | none (control) | **55.58%** | 74.75% | 81.40% |
      | TrivialAugment | 52.09% | 72.24% | 79.71% |
      | RandAugment | 52.60% | 72.84% | 79.93% |

      Both augmentation policies land ~3 points *below* the no-augmentation
      control at 15 epochs, TrivialAugment and RandAugment effectively tied
      with each other. Checked whether this is a top-1-only illusion —
      `benchmarks/analyze_errors.py` against all three checkpoints, not just
      the headline number — and it isn't at this proxy length: control leads
      on macro-F1 (53.38% vs 49.63%/50.22%), weighted-F1, and the
      worst-30-classes average (16.77% vs 14.05%/14.78%), so augmentation
      isn't quietly buying rare-class recall at the cost of overall accuracy
      here. Calibration (15-bin ECE) is a wash across all three. One thing
      the deep-dive did confirm as real rather than a fluke: the same
      near-duplicate class pairs (beef tartare/steak tartare, chicken
      wing/buffalo wing, oyster/huitre, ...) turn up under all three
      training recipes — cross-validates that those are dataset labeling
      collisions, not an artifact of one run.

      But at epoch 15 both augmented runs' train loss was still meaningfully
      above the control's (3.32-3.43 vs 3.02) — still mid-convergence — so
      the 15-epoch protocol was suspected of being too short for
      augmentation to pay off. **Resolved with a matched-epoch rematch:**
      no-augmentation and RandAugment both re-run fresh for 30 epochs
      (cosine schedule needs the total epoch count set upfront, so this
      could not be a resume of the 15-epoch checkpoints):

      | Recipe (30ep) | val-dev top-1 | macro-F1 | weighted-F1 | worst-30 avg | ECE |
      | --- | --- | --- | --- | --- | --- |
      | none (control) | **61.27%** | **59.38%** | **60.90%** | **23.62%** | 0.144 |
      | RandAugment | 60.43% | 58.31% | 59.84% | 23.21% | 0.157 |

      The 15-epoch proxy exaggerated the gap in magnitude (~3 points vs
      0.84 at matched epochs), and control leads on every point-estimate
      axis in the deep-dive. **But checked for statistical significance
      with `benchmarks/significance_test.py`** — paired McNemar's test on
      the same 6,063 val-dev images (`b`=454 images only control got right,
      `c`=401 only RandAugment got right, n=855 discordant) — and the
      0.84-point gap is *not* significant
      (p=0.075, two-sided exact binomial on the discordant pairs).** The
      honest read: this proxy has no power to distinguish "control is
      slightly better" from "these are statistically tied." Conclusion,
      revised from an earlier draft of this section that called it a clean
      win: **no proxy evidence that RandAugment beats the plain recipe at
      this schedule length, but also none that it's meaningfully worse** —
      plain still carries forward on simplicity, not on a proven margin.
      Single seed per config throughout (no repeated-seed runs anywhere in
      Phase C or D — see that Phase's own caveat), so this is silent on
      training-noise variance on top of the sampling variance McNemar
      covers; a second seed on both legs would be the way to actually
      settle it.

      *Incident, 2026-07-31:* the worktree running the first 30-epoch
      RandAugment attempt was recycled mid-training (session-level infra,
      not a code bug), which deleted its gitignored `runs/`/`checkpoints/`
      along with the dataset symlink the run was reading from — it crashed
      on its final epoch with `FileNotFoundError`. The per-epoch curve
      survived in an out-of-worktree scratchpad and was reconstructed into
      `runs/phaseD/reconstructed-*.json` (flagged `"reconstructed": true`;
      val metrics exact, train metrics are the epoch's last running average
      rather than the true epoch mean — kept for the record, not used
      above). No checkpoint survived that attempt, so it was re-run clean
      for the table above. All Phase D work now runs from the main checkout
      (`/home/lollinux/life/supervised-learning-on-food-images`), not a
      worktree, so a recycle can't repeat this.

      **Mixup / CutMix / EMA / GCE loss, 15-epoch proxy** (same protocol,
      against the 15-epoch control above — these were not extended to 30
      epochs; see reasoning per-row):

      | Recipe | val-dev top-1 | macro-F1 | worst-30 avg | zero-acc classes | ECE |
      | --- | --- | --- | --- | --- | --- |
      | none (control, 15ep) | 55.58% | 53.38% | 16.77% | 4 | 0.182 |
      | Mixup | 42.95% | 40.05% | 5.13% | 11 | 0.231 |
      | CutMix | 48.84% | 45.61% | 7.19% | 5 | 0.172 |
      | EMA (decay 0.999) | 39.88% | 38.32% | 4.52% | 9 | 0.107 |
      | GCE loss (q=0.7) | 22.23% | 14.86% | 0.00% | **119** | 0.328 |

      Unlike the RandAugment rematch, these four don't need a significance
      caveat — paired McNemar's test (same protocol as above, each vs the
      15-epoch control) puts all four at p < 0.0001. These are large,
      unambiguous effects, not proxy noise.

      - **Mixup/CutMix**: both well below control, CutMix less damaging than
        Mixup. Consistent with the augmentation finding — soft-label mixing
        is an even stronger regularizer than TrivialAugment/RandAugment, and
        15 epochs isn't enough runway for it. Given RandAugment's own 30-epoch
        rematch still lost, extending Mixup/CutMix to 30 epochs was not judged
        worth the further GPU-hours here — flagged as future work rather than
        measured, since it needs its own justification, not an assumption
        that they'd close the gap the way RandAugment partially did.
      - **EMA**: worst-30-classes and macro-F1 both collapse alongside top-1.
        Plausible cause, not confirmed: at decay 0.999 the effective
        averaging window is ~1,000 steps, and this proxy is only ~6,945
        steps total (463 batches x 15 epochs) — the EMA weights may still be
        lagging well behind the raw, rapidly-changing (lr=0.8) weights when
        the proxy ends, especially early in training before warmup settles.
        A longer schedule or a faster (lower) decay would be needed to tell
        EMA's ceiling from an artifact of this specific proxy length.
      - **GCE loss**: by far the worst result, and the shape of the failure
        is the tell — 119 of 251 classes at zero accuracy, and this is *not*
        underconfidence: mean confidence on wrong predictions is 0.49 and
        ECE is the worst of every recipe tested (0.328), i.e. GCE is
        confidently wrong on the classes it collapsed on. lr=0.8 was tuned
        for CE's loss landscape; GCE's bounded loss (`L_q = (1 - p_y^q)/q`)
        produces much smaller gradients as confidence rises, so a fixed
        high LR tuned for CE is not a fair test of GCE's ceiling. **This
        result says "GCE needs its own LR search," not "GCE fails on this
        dataset."** Revisit only if noise-robustness becomes a priority —
        not spending more budget on it now.

      **What Phase D's recipe search carries forward: the plain recipe**
      (label-smoothed CE, no extra augmentation, no Mixup/CutMix, no EMA) at
      lr=0.8. Confidence differs by candidate, not uniform across the row:
      Mixup, CutMix, EMA and GCE loss are decisively ruled out
      (p < 0.0001 each, large point-estimate gaps, and the deep-dive rules
      out a rare-class or calibration trade-off hiding behind the numbers).
      RandAugment is not decisively ruled out — its matched-epoch gap
      (0.84 points) is not statistically significant (p=0.075) on a single
      seed each — so "plain wins" there is really "plain is simpler and
      nothing forces a switch," not "plain is proven better." GCE and EMA's
      poor showings are additionally attributable to untuned
      hyperparameters for this specific proxy length rather than the
      techniques being wrong for this problem; noted as open follow-ups,
      not settled negatives.
- [x] **Batch size in {160, 256, 512}, LR scaled with it (linear scaling rule).**
      Measured on the baseline trunk at 176 px: throughput is flat across this
      range (1738-1785 img/s, batch 160-768), VRAM scaling linearly from 1.28 to
      5.64 GiB — the GPU is compute-saturated at batch 160, not idle, so this
      buys nothing in wall-clock. The reason to sweep it anyway is gradient
      noise and BatchNorm statistics, which do change with batch size; that is
      an accuracy question, not a throughput one.

      15-epoch proxy, plain recipe, LR scaled linearly from the batch-256
      control (lr=0.8): batch 160 at lr=0.5, batch 512 at lr=1.6. Against the
      `phaseD-lr0.8` checkpoint via `benchmarks/significance_test.py`:

      | Batch | LR | val-dev top1 | Δ vs control | McNemar p | Peak VRAM |
      | --- | --- | --- | --- | --- | --- |
      | 160 | 0.5 | 55.85% | +0.18 | 0.7407 | 1.30 GiB |
      | 256 (control) | 0.8 | 55.67% | — | — | 1.99 GiB |
      | 512 | 1.6 | 55.30% | -0.36 | 0.4868 | 3.83 GiB |

      All three tied. **Batch size is a non-lever for this recipe** — neither
      throughput (already known) nor accuracy (now measured) moves across
      160-512 with the LR scaled to match. Batch 256 stays, on no evidence
      against it and because every other Phase D number was measured there.

      One infra finding on the way: batch 512 at the default 8 workers
      crashed with `RuntimeError: unable to allocate shared memory` one epoch
      in — not VRAM, `/dev/shm` (7.7 GiB on this box). Each of the 8 workers
      prefetches collated batches, and at batch 512 that overruns shm before
      it overruns the GPU. Re-ran clean at `--workers 4`. Worth knowing if a
      future run pushes batch size further: **`/dev/shm`, not VRAM, is what
      binds first on this box at large batch.**

- [x] **RandomResizedCrop scale floor** (not in the original Phase D list;
      added 2026-08-02 after the matched-epoch 30-epoch runs showed val top-1
      running 7-14 points *above* train top-1 with the gap widening, not
      closing — the model underfits, it doesn't overfit, on every recipe axis
      tried so far. `RandomResizedCrop(176)` had been running at the
      ImageNet-inherited `scale=(0.08, 1.0)` since Phase A; a train view can
      be cropped from 8% of the image, tuned for 1.28M images against this
      dataset's ~90k. Made tunable via `--crop-scale-min`, default unchanged.

      15-epoch proxy, plain recipe, lr=0.8, batch 256, against `phaseD-lr0.8`:

      | Crop scale min | val-dev top1 | Δ | McNemar p | train_acc1 | gap (val-train) |
      | --- | --- | --- | --- | --- | --- |
      | 0.08 (control) | 55.58%\* | — | — | 44.53% | +11.05 |
      | 0.25 | 55.71% | +0.05 | 0.9465 | 51.36% | +4.31 |
      | 0.40 | 54.49% | -1.17 | **0.0232** | 55.19% | -0.56 |

      (\*control's val-dev top1 shown here is the checkpoint re-score used
      for every McNemar comparison, 55.67%, not the epoch-end log value used
      elsewhere in this table — same run, two ways of reading it.)

      **The hypothesis half-confirms and the accuracy half doesn't.** Raising
      the crop floor closes the train/val gap exactly as predicted — it goes
      from +11.05 to +4.31 to slightly negative, i.e. ordinary train-above-val
      behaviour, by 0.40. But val accuracy does not follow: 0.25 ties control,
      and 0.40 is significantly *worse*, not better. `RandomResizedCrop`'s
      low-scale cropping is doing double duty as both regularizer and
      view-diversity source — it teaches scale/translation invariance and
      effectively multiplies the training set, which ordinary regularizers
      like Mixup/RandAugment don't. Relaxing the floor makes the task easier
      to fit but removes that diversity without buying back any epochs, so at
      a fixed 15-epoch budget it's a net loss. No matched-epoch rematch run
      for this one, unlike RandAugment's: 0.40 already trains *faster* per
      epoch than control (55.19% vs 44.53% train_acc1 at epoch 15) and still
      loses on val, so there is no "not converged yet" argument for more
      epochs to close the gap. **`--crop-scale-min` stays in `main.py`,
      default unchanged at 0.08** — sixth Phase D-style axis tested,
      sixth to lose.
- [x] One proxy for noise handling: CE + label smoothing vs GCE. Adopt only if it
      wins. Co-teaching stays out — two networks is a 2x cost with no evidence
      behind it here.

      Run above (GCE loss row, 22.23% vs control's 55.58%) — GCE did not win,
      so label-smoothed CE is kept. Caveat carried over from that row: this
      used lr=0.8, tuned for CE, and GCE's collapse pattern (119/251
      zero-accuracy classes, confidently wrong per its ECE) is consistent
      with an LR mismatch rather than GCE being unsuitable for this
      dataset's label noise. Not reopening without a dedicated LR search for
      GCE, which hasn't been budgeted.
- [x] **One** 90-epoch run at 176 px, evaluated with FixRes-style test-resolution
      correction (train 176, test 224 centre crop).

      Baseline trunk, GAP head, plain recipe, lr=0.8, batch 256, crop-scale-min
      0.08 — every setting this phase and its addendum settled on, unmodified.
      `python src/main.py food251 --epochs 90 --lr 0.8 -b 256
      --run-label full-90ep-lr0.8 --log-dir runs/final`. The FixRes correction
      needed no extra flag: `main.py` already validates at `Resize(256)` +
      `CenterCrop(224)` while training at 176, by default.

      | | top-1 | top-3 | top-5 |
      | --- | --- | --- | --- |
      | val-dev, best (epoch 86) | **64.36%** | 82.22% | 87.51% |
      | val-dev, final (epoch 90) | 64.26% | 82.14% | 87.61% |
      | **val-test (headline, touched once)** | **63.83%** | **81.79%** | **87.39%** |

      2h05m wall clock, 1.99 GiB peak VRAM, `model_best.pth.tar` at epoch 86
      kept per the smoke test's own convention — though here best and final are
      within 0.1 points of each other, not the ~0.5-point gap the smoke test
      saw, which is itself evidence the model is no longer overfitting late in
      training (see below). Val-test lands 0.5 points under val-dev's best,
      the expected direction since val-dev drove checkpoint selection, and a
      small enough gap that the split isn't doing anything unusual.

      **Up 2.3 points on the smoke test's 61.57%** (pre-Phase-D default lr=0.1,
      no crop/batch/head tuning), at the same trunk and epoch count — the
      full recipe search bought a real, if modest, gain. **Continues the
      15/30/90-epoch trend cleanly**: 55.6% to 61.3% to 63.8% (val-test) /
      64.4% (val-dev, best) — diminishing but still real returns from length
      alone, as the cosine schedule design implies.

      **Confirms the underfitting diagnosis the crop-scale investigation was
      built on.** Train top-1 at the final epoch was 64.88%, val top-1 64.26%
      — the +11.05-point val-over-train gap measured at epoch 15, and the
      +7.3-point gap at epoch 30, has closed to near zero by epoch 90. The
      model needed the full 90-epoch schedule to converge; nothing tested in
      this phase sped that up, and per the crop-scale result, nothing needed
      to — the "regularization" axes were mostly not costing convergence, the
      schedule length was.

      **No thermal slowdown this time.** The smoke test saw per-batch time
      rise from 0.16s to 0.26s past epoch 60 under a lower GPU power cap;
      this run held steady at 0.147-0.169s/step at epochs 5, 30, 60 and 89
      alike, after the box was switched to its high-performance power
      profile partway through a related proxy run earlier this phase. Not
      isolated as a controlled comparison, but consistent with the smoke
      test's own read that the slowdown was a power-cap effect, not a hard
      thermal limit.

      **Caveat carried from #33: this run was not seeded**, consistent with
      every other run in this project, kept that way rather than risk
      `cudnn.deterministic`'s unmeasured slowdown on the single most
      expensive run so far. The headline number above is a point estimate,
      not an exactly reproducible one.
- [ ] Optional, cheap to measure: `torch.compile`.

## Phase E — self-supervised track (~18 GPU-h)

**Decided: the full setting — 200 pretrain epochs at 176 px.** SimSiam/BYOL needs
two augmented views per step, so an epoch costs ~2x: ~8 h pretrain, ~1.8 h
finetune, ~8 h for the control. This is more GPU time than the entire rest of the
plan combined, and it is deliberate — it keeps the result comparable to the
published SimSiam/BYOL settings instead of inviting "you under-trained it".

**SimSiam over BYOL, decided 2026-08-03**: no momentum/target encoder to add
(no second ~6.5M-param shadow copy of `FoodCNN` sitting in VRAM), no EMA-decay
schedule to tune, and the original recipe matches this project's batch sizes
without a large-batch trick like LARS. Scaffolding is in: `src/simsiam.py`
(`ProjectionMLP`/`PredictionMLP`/`SimSiamModel`, negative cosine similarity
loss with stop-gradient, `TwoCropsTransform` + SimSiam's augmentation recipe,
checkpointed every epoch and resumable via `--resume`, same pattern as
`main.py`); `FoodCNN.forward_features` (`src/model.py`) exposes the pooled
512-d trunk output the projector attaches to, no parameter-budget impact;
`main.py --init-encoder PATH` loads only the pretrained trunk into a fresh
`FoodCNN` for the finetune half (mutually exclusive with `--resume`). Covered
by `tests/test_simsiam.py`, all on random tensors per this project's
no-dataset-in-tests rule.

- [ ] Pretrain 200 epochs at 176 px, then finetune at 176.
- [ ] **The control is not optional**: SSL-pretrain + finetune must be compared
      against spending those same GPU-hours on longer supervised training.
      Without it the result is uninterpretable.
- [x] Checkpoint pretraining often enough that an interrupted 8 h run is
      resumable — at this length that is a practical requirement, not a nicety.

SimCLR is excluded — it degrades below ~1k batch, unreachable in 8 GB.

**Where to run it**: `notebooks/colab_gpu_probe.ipynb` benchmarks
`FoodCNN` on whatever GPU Colab assigns that session, using
`benchmarks/bench.py::measure` unchanged (synthetic batches, no dataset
needed) at the same settings behind the 1688 img/s local baseline above —
its printed speedup multiplier is what decides whether the 200-epoch
pretrain runs locally or there. Not run yet; needs a manual pass on
colab.research.google.com before its numbers can be trusted.

## Phase F — report

- [ ] Typst sources already exist under `report/`. Every number traceable to a
      logged run.

---

## Budget

| Phase | GPU-h |
| --- | --- |
| C — architecture proxies (4 x 18 min) | 1.2 |
| C addendum — width floor + heads (5 x 19 min, measured) | 1.6 |
| D — recipe proxies (8 x 18 min) | 2.4 |
| D — full supervised run (90 ep) | 1.8 |
| E — SSL track including control | ~18 |
| slack / reruns | 3 |
| **total** | **~26** |

The supervised half of this is ~5 GPU-h; Phase E is the other 70%. Note the
earlier estimate of ~29 h was measured in `balanced` power profile and under
contention, understating throughput by ~1.7x — the totals happen to land close
together, but for unrelated reasons.

## Decisions taken

- **Val split: 50/50 stratified `val-dev` / `val-test`.** Model selection never
  touches `val-test`; the report quotes it once.
- **SSL scope: the full setting, 200 epochs at 176 px**, with the
  equal-GPU-hours supervised control.
- **Trunk for Phase D: the baseline `[2,2,2,1]` 64-512, provisionally.**
  Phase C ruled out the 5-stage variant but left baseline / `[2,2,4,1]` /
  `[2,2,2,2]` 64-448 within 1.17 points on a single seed each — not enough
  to call with confidence. Going with the baseline because it's statistically
  indistinguishable from the other two here, uses 30% fewer parameters, and
  that headroom hasn't measured as worth anything yet; better spent on Phase
  D's recipe tuning. `model.py` needs no change — the baseline is already
  what's there. **Explicitly revisitable**: if the Phase D-tuned baseline
  underperforms expectations, `[2,2,4,1]` and `[2,2,2,2]` 64-448 are still
  defined in `benchmarks/trunk_variants.VARIANTS` and can be re-proxied (or
  re-run with a second seed, per the option not taken here) without redoing
  any of the measurement work above.

  **Strengthened 2026-08-02 by the Phase C addendum**, which tested the
  direction Phase C never did: 64-384 at 5.18M ties baseline exactly
  (p=1.0000), 32-256 at 1.68M loses 6.58 points. The argument above was
  "that headroom hasn't measured as worth anything yet" — one-sided, since
  nothing below baseline had been tried. It is now two-sided: baseline sits
  on a plateau, and the cliff is at least 1.4M parameters below it. The
  choice stands, and 64-384 is a live alternative that costs nothing rather
  than a risk. Neither pooling head (#29) displaces the GAP head either.
- **Phase D recipe: plain (label-smoothed CE, no augmentation/Mixup/CutMix/
  EMA), lr=0.8.** All five recipe axes were proxied at 15 epochs; nothing
  beat plain. The one close call, RandAugment, got a matched 30-epoch
  rematch specifically because the 15-epoch gap (~3 points) coincided with
  its train loss still being well above the control's — a real "is 15
  epochs long enough to judge this" signal that the other three additions
  didn't show as clearly. At 30 epochs the gap narrowed to 0.84 points and
  control led on every point-estimate axis of the deep-dive, but a paired
  McNemar's test on the shared val-dev images found that gap **not**
  statistically significant (p=0.075, single seed each) — so plain carries
  forward on simplicity and an unbeaten track record, not a proven margin
  over RandAugment specifically. Mixup/CutMix/EMA/GCE's losses, by
  contrast, are all p < 0.0001 — decisively real. GCE's collapse (119/251
  zero-accuracy classes, confidently wrong per its calibration numbers) and
  EMA's is each attributed to an untuned setting for this proxy length
  (lr=0.8 was picked for CE; GCE's bounded loss wants its own LR search;
  EMA's decay=0.999 implies a ~1,000-step averaging window against a
  ~6,945-step proxy) rather than a settled verdict against the technique —
  noted as open follow-ups. **Explicitly revisitable** the same way the
  trunk choice above is: if the eventual full run underperforms, RandAugment
  is the candidate worth another look first, given how much its gap closed
  once epoch count was matched.
- **Phase D recipe axes are implemented in `main.py`, opt-in, all defaulting
  to off**: `--augment {trivial,rand}`, `--mix {mixup,cutmix}`, `--ema`,
  `--loss gce`. All five have now been proxied (see Phase D); none beat the
  plain recipe within the budgets measured, and it carries forward for the
  eventual full 90-epoch run. Checkpoints are namespaced under
  `checkpoints/<run-label>.pth.tar`, fixing
  a real bug where sequential runs silently overwrote each other's weights
  (the smoke test's converged checkpoint was lost this way; its logged
  metrics in `runs/` were not affected).
- **Analysis tooling**: `benchmarks/plot_runs.py` (learning-curve and
  multi-run comparison plots), `benchmarks/analyze_errors.py`
  (classification report, top-confused class pairs, per-class accuracy,
  confidence histogram, t-SNE of penultimate features — CPU-default, reads
  a checkpoint, doesn't require the GPU), and `benchmarks/significance_test.py`
  (paired McNemar's test between two checkpoints on the same val split —
  point-estimate accuracy gaps alone don't say whether a comparison had the
  power to detect a real difference; added once the Phase D recipe search
  needed it, see that Phase for why). Per-epoch `lr` and top-3 accuracy
  are now logged in `RunLog`; older logs in `runs/` predate both fields and
  are read by `plot_runs.py` without crashing rather than backfilled.

Both were settled on 2026-07-30. Reopen them here rather than silently
diverging in code.
