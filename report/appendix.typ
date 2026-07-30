#import "@preview/bloated-neurips:0.8.0": botrule, midrule, toprule

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
  placement: top,
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
  placement: top,
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
