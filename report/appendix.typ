#import "@preview/bloated-neurips:0.8.0": botrule, midrule, toprule

= Learning-rate sweep <appendix-lr>

@sec-recipe-ablation reports the learning-rate sweep as final accuracies. The
per-epoch curves behind those numbers, all 15-epoch `val-dev` proxies on the
baseline trunk at batch 256:

#figure(
  caption: [
    `val-dev` top-1 per epoch across the learning-rate sweep. The visible
    mid-run dip on lr 0.8 around epochs 8-9, absent from 0.4 and 0.6, is the
    growing-noise half of the argument for stopping the search there; its
    training loss (not shown) is the smoothest of the sweep and reaches the
    lowest final value.
  ],
  image("figures/lr-sweep-val-acc1.png", width: 88%),
) <figure-lr-sweep>

#figure(
  caption: [
    Learning-rate sweep, 15-epoch `val-dev` proxies, baseline trunk, batch
    256. Top-3 and top-5 were only logged from the lr 0.4 leg onward, when
    top-3 tracking was added to `RunLog` mid-sweep.
  ],
  table(
    columns: 5,
    align: (right, right, right, right, right),
    stroke: none,
    toprule,
    table.header([LR], [top-1], [top-3], [top-5], [$Delta$ vs. previous]),
    midrule,
    [0.05], [43.96%], [---], [---], [---],
    [0.10], [49.41%], [---], [---], [+5.45],
    [0.20], [52.56%], [---], [---], [+3.15],
    [0.40], [54.02%], [74.16%], [81.02%], [+1.46],
    [0.60], [54.96%], [---], [---], [+0.94],
    [*0.80*], [*55.58%*], [74.75%], [81.40%], [+0.62],
    botrule,
  ),
) <table-lr-sweep>

lr 0.10 is the reference script's default, and the setting the pipeline
validation run in @sec-full-run used; lr 0.80 is what @sec-recipe-ablation
carries forward.

= Batch size and crop scale <appendix-nonlevers>

Both axes are summarized in @sec-recipe-ablation; the full tables follow.
Each row is a 15-epoch `val-dev` proxy on the baseline trunk with the plain
recipe, compared against the lr 0.8 / batch 256 control by a paired McNemar
exact test.

#figure(
  caption: [
    Batch size with the learning rate scaled linearly from the batch-256
    control. All three are statistically tied.
  ],
  table(
    columns: 6,
    align: (right, right, right, right, right, right),
    stroke: none,
    toprule,
    table.header([Batch], [LR], [top-1], [$Delta$], [McNemar $p$], [Peak VRAM]),
    midrule,
    [160], [0.5], [55.85%], [+0.18], [0.7407], [1.30 GiB],
    [256], [0.8], [55.67%], [---], [---], [1.99 GiB],
    [512], [1.6], [55.30%], [-0.36], [0.4868], [3.83 GiB],
    botrule,
  ),
) <table-batch-size>

One infrastructure finding came out of this leg: batch 512 at the default
eight dataloader workers crashed one epoch in with `RuntimeError: unable to
allocate shared memory`, and re-ran clean at four workers. Each worker
prefetches collated batches, and at batch 512 that overruns this machine's
7.7 GiB `/dev/shm` well before it overruns the 8 GiB of VRAM. *At large batch
sizes on this box, shared memory binds before GPU memory does* -- worth
knowing before reading a batch-size ceiling off VRAM figures alone.

#figure(
  caption: [
    `RandomResizedCrop` scale floor. The train/validation gap behaves exactly
    as the underfitting hypothesis predicts; accuracy does not follow it.
  ],
  table(
    columns: 6,
    align: (right, right, right, right, right, right),
    stroke: none,
    toprule,
    table.header([Scale min], [top-1], [$Delta$], [McNemar $p$], [train top-1], [gap]),
    midrule,
    [0.08], [55.67%], [---], [---], [44.53%], [+11.05],
    [0.25], [55.71%], [+0.05], [0.9465], [51.36%], [+4.31],
    [0.40], [54.49%], [-1.17], [*0.0232*], [55.19%], [-0.56],
    botrule,
  ),
) <table-crop-scale>

No matched-epoch rematch was run for this axis, unlike RandAugment's. The
reason is visible in the table: the 0.40 leg already trains *faster* per epoch
than the control (55.19% vs. 44.53% train top-1 at epoch 15) and still loses
on validation, so there is no "not converged yet" argument for more epochs to
close the gap.

The control row's top-1 here is the checkpoint re-score used for every McNemar
comparison in this report, rather than the epoch-end log value quoted in
@sec-recipe-ablation -- the same run read two ways. Checkpoint re-scores run in
bf16 and read about 0.03 points above the fp32 per-epoch logs; every
comparison pairs like with like.

= Full architecture-screening sweep <appendix-sweep>

@table-arch-search in the main text reports 15-epoch accuracy proxies for four
trunk variants. Those four were selected from a larger synthetic screening
pass over throughput, parameter count, and memory alone (`benchmarks/bench.py`,
`benchmarks/trunk_variants.py`), all at 176px training resolution and batch
160, before any accuracy was measured. The full sweep:

#figure(
  caption: [
    Synthetic throughput/memory sweep over trunk variants under the
    10M-parameter cap, 176px training resolution, batch 160. "90 ep" is the
    projected wall-clock for a full 90-epoch run at the measured throughput;
    "Final map" is the trunk's spatial resolution immediately before global
    average pooling.
  ],
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: none,
    toprule,
    table.header([Variant], [Params], [img/s], [90 ep], [Peak VRAM], [Final map]),
    midrule,
    [baseline `[2,2,2,1]` 64-512], [6.58M], [1688], [1.8 h], [1.26 GiB], [6x6],
    [`[2,2,3,1]` 64-512], [7.76M], [1585], [1.9 h], [1.31 GiB], [6x6],
    [`[2,2,4,1]` 64-512], [8.94M], [1475], [2.0 h], [1.36 GiB], [6x6],
    [`[2,3,3,1]` 64-512], [8.06M], [1460], [2.0 h], [1.39 GiB], [6x6],
    [`[2,2,2,2]` 64-448], [9.46M], [1608], [1.8 h], [1.30 GiB], [6x6],
    [wider\@11 64-320-512], [7.98M], [1494], [2.0 h], [1.30 GiB], [6x6],
    [5 stages 48-512 `[2,2,2,2,1]`], [9.08M], [*1960*], [*1.5 h*], [1.04 GiB], [3x3],
    [uniformly wider 72-576], [8.31M], [1062], [2.8 h], [1.41 GiB], [6x6],
    [5 stages, no stem max-pool], [9.08M], [684], [4.3 h], [2.16 GiB], [6x6],
    botrule,
  ),
) <table-full-sweep>

The last row isolates the resolution effect cited in @discussion: identical
parameter count to the row above it, but with the stem's max-pool removed to
keep the final feature map at 6x6 instead of 3x3, at a cost of 2.9x throughput
and roughly 2x peak memory. The "uniformly wider 72-576" row isolates the
tensor-core alignment effect: fewer parameters than the five-stage variant,
at 46% lower throughput, purely from using channel widths that are not
multiples of 32.

= The rejected depthwise-separable trunk <appendix-depthwise>

Before the residual trunk in @method, a MobileNetV2-style depthwise-separable
version of the same network was implemented and measured head-to-head on the
same hardware. It is kept in `src/model.py`, unused by `FoodCNN.features`, as
a documented negative result rather than deleted:

#figure(
  caption: [
    Depthwise-separable vs. plain-convolution trunk, measured head-to-head on
    the project GPU (176px, batch 160, bf16, `channels_last`).
  ],
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    toprule,
    table.header([Trunk], [Params], [img/s], [Peak VRAM]),
    midrule,
    [Depthwise-separable (MobileNetV2-style)], [2.55M], [621], [3.83 GiB],
    [Plain-convolution residual (adopted)], [6.58M], [*918*], [*1.28 GiB*],
    botrule,
  ),
) <table-depthwise>

Fewer parameters did not mean faster or lighter here: depthwise convolutions
trade parameter count for activation memory and wall-clock, and under this
project's binding constraints -- throughput and memory, not raw parameter
count -- that trade loses.#footnote[
  This comparison's 918 img/s baseline figure is lower than the 1688 img/s
  recorded for the same trunk in @table-full-sweep. The two were measured in
  separate benchmarking passes -- this comparison predates
  `benchmarks/bench.py`'s current three-round, keep-the-best methodology --
  and, per the thermal-throttling behavior discussed in @discussion, absolute
  throughput on this hardware is not stable across passes. The qualitative
  conclusion, that the depthwise-separable trunk trades parameters for memory
  and wall-clock, does not depend on which figure is used.
] The plain-convolution trunk was adopted on this basis before the
architecture search in @sec-arch-search ever began.
