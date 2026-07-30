# Training roadmap

**Status:** Phase A not started · **Baseline `main`:** `820f347` · **Last measured:** 2026-07-30

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

- [ ] **BatchNorm after the three shortcut projections** (`src/model.py:24`). They are
      bare `Conv2d`s, so an unnormalized branch is added to a BN'd one; torchvision's
      ResNet uses `conv1x1 + norm` in `downsample` for exactly this reason.
- [ ] **`bias=False` on every conv followed by BatchNorm** — 3,776 no-op parameters.
- [ ] **Zero-init the second BN's gamma in each block**, so each block starts as
      identity. This is what makes warmup plus a high LR safe.
- [ ] **`main.py` training loop**, keeping the argparse / `main_worker` shape:
  - [ ] `torch.autocast("cuda", dtype=torch.bfloat16)` — **no `GradScaler`**; bf16
        does not need one.
  - [ ] `channels_last` on model and inputs.
  - [ ] cosine schedule with 5-epoch linear warmup, replacing `StepLR(30, 0.1)`.
  - [ ] `CrossEntropyLoss(label_smoothing=0.1)`; SGD with `nesterov=True`.
  - [ ] `--workers` default 4 to 8; train crop 224 to 176.
  - [ ] drop the `nn.DataParallel` wrapper on a single GPU — it costs a
        scatter/gather per step and prefixes every checkpoint key with `module.`.
- [ ] **Tests** (never reading `food251/`):
  - [ ] single-batch overfit gate — 200 steps on fixed `randn` drives loss below
        0.1. The cheapest possible proof the trunk can learn, and the one thing
        that would otherwise waste a 2-hour run.
  - [ ] `channels_last` survives the forward pass.
  - [ ] warmup+cosine LR hits expected values at epochs 0, 5 and last.

## Phase B — freeze the protocol before any run

- [ ] **Split the 11,994-image val set 50/50, stratified, into `val-dev` and
      `val-test`.** Ablations and checkpoint selection read `val-dev` only;
      `val-test` is touched once, for the report. Without this the headline number
      is selected-on. (4 classes have <20 val images and 1 has 2; those per-class
      numbers stay noisy either way.)
- [ ] **Fixed seed and a fixed 15-epoch proxy protocol** for every comparison.
      At ~1.2 min/epoch a proxy run is ~18 minutes, so comparisons that would be
      unaffordable at 90 epochs are routine. Rank on the proxy, confirm once at
      full length.
- [ ] **Per-run logging**: config hash, per-epoch train/val top-1 and top-5,
      wall-clock, peak VRAM. Capture what the report needs the first time.

## Phase C — pick the architecture (~1.2 GPU-h)

- [ ] Four 15-epoch proxy runs on `val-dev` top-1: baseline 6.58M ·
      `[2,2,4,1]` 8.94M · `[2,2,2,2]` 64-448 9.46M · 5-stage 48-512 9.08M.
      This answers whether the 5-stage variant's 3x3 final map costs accuracy.
      Either answer belongs in the report.

## Phase D — recipe, then the full run (~4.2 GPU-h)

- [ ] 15-epoch proxies on the winning trunk: LR in {0.05, 0.1, 0.2, 0.4} at
      batch 256; TrivialAugment vs RandAugment; Mixup/CutMix on/off; EMA on/off.
- [ ] One proxy for noise handling: CE + label smoothing vs GCE. Adopt only if it
      wins. Co-teaching stays out — two networks is a 2x cost with no evidence
      behind it here.
- [ ] **One** 90-epoch run at 176 px, evaluated with FixRes-style test-resolution
      correction (train 176, test 224 centre crop).
- [ ] Optional, cheap to measure: `torch.compile`.

## Phase E — self-supervised track (~8 GPU-h as scoped)

SimSiam/BYOL needs two augmented views per step, so an epoch costs ~2x. At 200
pretrain epochs and 176 px that is ~8 h, plus ~1.8 h finetune, plus the
equal-GPU-hours supervised control the comparison requires — ~18 h unscoped.
Scoped to 100 epochs at 128 px it is ~2 h pretrain + 1.8 h finetune + ~4 h
control. The 128 px figure is scaled from the measured 176 px rate, not measured
directly — confirm it before committing to the schedule.

- [ ] Pretrain at 128 px, finetune at 176 — standard, and roughly halves the cost.
- [ ] Cap pretraining at 100 epochs, and report the cap as a limitation rather
      than concluding SSL "does not work" from an under-trained run.
- [ ] **The control is not optional**: SSL-pretrain + finetune must be compared
      against spending those same GPU-hours on longer supervised training.
      Without it the result is uninterpretable.

SimCLR is excluded — it degrades below ~1k batch, unreachable in 8 GB.

## Phase F — report

- [ ] Typst sources already exist under `report/`. Every number traceable to a
      logged run.

---

## Budget

| Phase | GPU-h |
| --- | --- |
| C — architecture proxies (4 x 18 min) | 1.2 |
| D — recipe proxies (8 x 18 min) | 2.4 |
| D — full supervised run (90 ep) | 1.8 |
| E — SSL track including control | ~8 |
| slack / reruns | 3 |
| **total** | **~17** |

Roughly half the previous estimate: the earlier figures were measured in
`balanced` power profile and under contention, which understated throughput by
~1.7x.

## Open decisions

1. **`val-dev` / `val-test` split — yes or no?** Recommended yes; it is the
   difference between a defensible headline number and a selected-on one.
2. **SSL scope**: 100 epochs at 128 px (recommended, ~16 h with control) versus
   the full 200 at 176 px (~32 h).
