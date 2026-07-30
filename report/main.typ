#import "@preview/bloated-neurips:0.8.0": appendix, botrule, midrule, neurips2026, paragraph, toprule, url

#let todo(body) = highlight(fill: yellow.lighten(60%))[*TODO --- #body*]

#let authors = (
  (name: "Lorenzo Liuzzo", affl: ("airi", "skoltech"), email: "lorenzoliuzzo@outlook.com", equal: true),
)

#show: neurips2026.with(
  title: [Supervised and Self-Supervised Learning on Food-251],
  authors: authors,
  keywords: (
    "fine-grained image classification",
    "convolutional neural networks",
    "parameter-efficient architectures",
    "self-supervised learning",
  ),
  abstract: [
    We train a residual convolutional network from scratch to classify the
    251 fine-grained food categories of FoodX-251, under a hard budget of ten
    million trainable parameters and a single 8GB laptop GPU. We search among
    parameter-matched trunk variants using short, fixed-seed proxy runs on a
    held-out development split, then validate the selected trunk with a full
    training run reaching 61.57% top-1 / 85.67% top-5 accuracy before any
    recipe tuning. We describe a set of opt-in recipe extensions -- stronger
    augmentation, Mixup/CutMix, an exponential moving average, a noise-robust
    loss -- implemented and awaiting ablation, and outline a planned
    comparison against self-supervised pretraining under an equal-GPU-hour
    budget; both are reported where results exist and marked otherwise.
    Throughout, we find that measured wall-clock and memory behavior on
    commodity hardware -- channel-width alignment, spatial resolution, and
    thermal throttling -- shape the final design as much as raw parameter or
    FLOP counts do.
  ],
  bibliography: bibliography("main.bib"),
  accepted: false,
)

= Introduction <introduction>

FoodX-251 is a fine-grained food image classification benchmark spanning 251
categories, built from a large, web-crawled, noisily labeled training set with
a smaller, manually curated and clean validation set @kaur2019foodx.
Fine-grained food categories are hard to separate because inter-class visual
differences (plating, preparation, garnish) can be subtler than the
intra-class variation induced by lighting, angle, and portioning, and the
label noise inherent to a web-crawled training set compounds the problem.

This work targets FoodX-251 under two constraints imposed by the setting
rather than chosen for effect: the classifier is trained from scratch, with no
ImageNet or other pretraining, under a budget of fewer than 10,000,000
trainable parameters, and every run happens on a single consumer laptop GPU
with 8GB of VRAM. Neither constraint is negotiable, and together they shape
everything that follows: an architecture search that screens on measured
throughput and memory before ever measuring accuracy (@sec-arch-search), and a
training recipe built to spend the wall-clock the 8GB budget buys, not just
the parameter budget itself.

We make four contributions, in the order the underlying work was carried out:
a measured, budget-constrained architecture search over residual-trunk
variants (@sec-arch-search); a from-scratch training recipe modernized with
mixed-precision training, a cosine schedule, and label smoothing, plus a set
of opt-in extensions -- stronger augmentation, Mixup/CutMix, an exponential
moving average, a noise-robust loss -- now implemented and awaiting ablation
(@sec-recipe-ablation); a full-length training run validating the selected
trunk (@sec-full-run); and a planned comparison of supervised training against
self-supervised pretraining under an equal-GPU-hour budget (@sec-ssl).

One design choice is worth flagging before the rest. Despite FoodX-251's
long-tailed reputation, the training split we measured is close to
class-balanced (median 471 images per class; a single 34-image class drives
the oft-cited 19.3x imbalance ratio), so class-balanced losses, logit
adjustment, and long-tail-specific training are out of scope here -- the real
distributional problem is label noise, not long-tailedness
(@dataset-protocol). More generally: the architecture search and one full
training run are complete at the time of writing; the recipe ablation and the
self-supervised comparison are in progress, and are marked accordingly below
rather than reported with numbers we have not measured.

= Related work <related-work>

We organize related work into three lanes, matching the three axes of this
project: architecture, training recipe, and pretraining objective.

#paragraph[Parameter-efficient architecture design.] Recent efficient-CNN
work targets mobile and edge deployment under similar per-parameter and
per-FLOP pressure: MobileNetV4 @qin2024mobilenetv4 and MobileOne
@vasu2022mobileone optimize explicitly for on-device latency rather than
parameter count alone, and ConvNeXt V2 @woo2023convnextv2 revisits
pure-convolutional design at a scale where masked-autoencoder pretraining is
available -- a regime this project's from-scratch constraint excludes. None
of these target fine-grained food classification, and our own literature
search turned up no source addressing FoodX-251, Food-101, or Food-2K under a
hard parameter budget (`research/2026-07-30-architecture-brief.md`) -- the
architecture decisions in @method rest on general efficient-CNN principles
and our own measurements rather than a food-specific prior.

#paragraph[Augmentation and label-noise robustness.] RandAugment
@cubuk2020randaugment and TrivialAugment @muller2021trivialaugment establish
that automated or tuning-free augmentation policies match or beat searched
policies at near-zero search cost; mixup @zhang2018mixup and CutMix
@yun2019cutmix are complementary regularizers that mix inputs and labels
rather than choosing among fixed operators. Generalized Cross Entropy
@zhang2018generalized gives a noise-robust alternative to label-smoothed
cross-entropy, directly relevant to FoodX-251's web-crawled training labels.
As with the architecture literature, none of these ablate sub-10M-parameter
models trained for tens of epochs
(`research/2026-07-30-augmentation-noise-brief.md`) -- whether the accepted
wisdom transfers to that regime is precisely what @sec-recipe-ablation is
designed to check.

#paragraph[Self-supervised pretraining.] SimCLR @chen2020simclr established
contrastive pretraining as a strong alternative to supervised training but
relies on a large negative-sample batch to work well; at this project's 8GB
budget, reachable batch sizes fall below where SimCLR is known to degrade,
which is why it is excluded here. BYOL @grill2020byol, SimSiam
@chen2021simsiam, and Barlow Twins @zbontar2021barlowtwins instead avoid
negative pairs entirely and remain viable at small batch sizes, making them
the basis for the pretraining design in @sec-ssl.

= Dataset and protocol <dataset-protocol>

FoodX-251 @kaur2019foodx contains 251 fine-grained food categories: a large,
web-crawled training set with noisy labels, and a smaller, manually curated
validation set of 11,994 images.

#paragraph[Class balance.] Despite FoodX-251's long-tailed reputation,
per-class training counts are close to balanced: median 471 images per class,
10th/90th percentile 366/580, and exactly one class below 200 images (34) --
that single outlier drives the frequently cited 19.3x imbalance ratio. We
therefore do not use class-balanced losses, logit adjustment, or decoupled
classifier retraining; the real distributional challenge is label noise,
since training labels are web-crawled and validation labels are clean, not
long-tailedness.

#paragraph[Validation split.] Ablations and checkpoint selection must never
touch the number this report ultimately quotes, or that number is implicitly
selected-on. We split the 11,994-image validation set 50/50, stratified per
class (every one of the 251 classes lands within one image of an exact half),
into `val-dev` (6,063 images) and `val-test` (5,931 images), deterministically:
`src/make_val_split.py` is a pure function of the official validation labels
and a fixed seed (251), so the split is byte-identical on every regeneration
and is not committed (`splits/val_split.csv` is gitignored and regenerated
once per checkout). Every ablation and architecture comparison in this report
reads `val-dev`; `val-test` is read exactly once, for the headline number in
@sec-full-run.

#paragraph[Proxy protocol.] At the measured throughput of the selected trunk
(about 1.2 minutes/epoch, @sec-arch-search), a 15-epoch run costs about 18
minutes, making comparisons that would be unaffordable at full length
routine. Every architecture and recipe comparison below ranks candidates on a
fixed-seed, 15-epoch `val-dev` proxy; the winning configuration is then
confirmed with one full-length run.

#paragraph[Logging.] Every run cited in this report is logged with a config
hash, per-epoch train/validation top-1 and top-5, wall-clock time, and peak
VRAM (`src/runlog.py`), so every number below traces back to a specific run.

= Method <method>

This section describes `FoodCNN`'s architecture, then the training recipe
used to fit it.

== Architecture

`FoodCNN` (`src/model.py`) is a plain residual CNN, sized to spend its
parameter budget where it can affect accuracy rather than where convention
places it. A stem (3x3 stride-2 convolution, batch normalization, ReLU, 3x3
stride-2 max-pool) takes the input to a quarter of its input resolution
before any residual stage runs. Four residual stages follow, each halving
spatial resolution and doubling channel width relative to the last -- 64,
128, 256, then 512 channels, with block counts $[2, 2, 2, 1]$ -- so that the
final stage, carrying one block rather than two, is what keeps the trunk
under budget.

Each residual block places batch normalization after both convolutions in
the main branch and, whenever the shortcut is not the identity, after the 1x1
shortcut projection as well -- an unnormalized branch should not be added to
a normalized one. The second batch norm's scale ($gamma$) in every block is
zero-initialized, so every block begins training as an identity map; this is
what makes a high learning rate safe under warmup. The nonlinearity is
applied after the residual addition, not before it, so each block is an
actual nonlinearity rather than an affine detour.

#paragraph[A genuinely global head.] Pooling to $(1, 1)$ rather than the
$(7, 7)$ map convolutions naturally leave behind matters: at $(7, 7)$,
flattening produces a $512 times 7 times 7 = 25,088$-feature vector, and a
single subsequent linear layer to 251 classes costs 6.4M parameters -- 94% of
the entire model, spent on its least expressive layer. Pooling to $(1, 1)$
first reduces that same layer to `Linear(512, 251)`, 129k parameters.

#paragraph[Parameter budget.] With the fixes above, `FoodCNN` totals
6,576,955 trainable parameters -- 65.8% of the 10M budget -- with roughly 98%
of them inside the convolutional trunk rather than the classifier head.

#paragraph[A rejected alternative.] A depthwise-separable version of the same
trunk (MobileNetV2-style inverted residuals, retained in `src/model.py` as a
documented negative result rather than deleted) was measured head-to-head
with the plain-convolution trunk on this hardware: 2.55M parameters, 621
img/s, 3.83 GiB peak memory, against the plain-convolution trunk's 6.58M
parameters, 918 img/s, 1.28 GiB peak (full detail in @appendix-depthwise).
Depthwise convolutions here cost activation memory and wall-clock far more
than they save in parameter count, and under this project's constraints,
throughput and memory -- not parameter count -- are what actually bind
(@discussion).

== Training recipe

The reference training loop is modernized in ways that cost nothing at this
parameter budget: `torch.autocast("cuda", dtype=torch.bfloat16)` without a
`GradScaler` (bf16 does not need dynamic loss scaling), `channels_last`
memory format on both model and inputs, a cosine learning-rate schedule with
five-epoch linear warmup in place of step decay, `CrossEntropyLoss` with
label smoothing 0.1, SGD with Nesterov momentum, eight dataloader workers (up
from four), a 176px training crop (down from 224), and no `DataParallel`
wrapper -- there is only one GPU to shard across, and the wrapper's
scatter/gather adds a per-step cost and a `module.` prefix to every
checkpoint key for no benefit here.

== Recipe extensions <recipe-extensions>

Four further axes are implemented as opt-in flags to the training script, all
defaulting to off so that nothing about the recipe above changes unless
explicitly requested: an augmentation switch between no augmentation,
TrivialAugment @muller2021trivialaugment, and RandAugment
@cubuk2020randaugment; batch-wise Mixup @zhang2018mixup and CutMix
@yun2019cutmix, scored against the pre-mix hard label since a mixed batch has
no single correct class to evaluate top-1/top-5 against; an exponential
moving average of model weights (`torch.optim.swa_utils.AveragedModel`,
batch-norm running statistics left live rather than averaged); and
Generalized Cross Entropy @zhang2018generalized as an alternative to
label-smoothed cross-entropy for the training set's label noise.
@sec-recipe-ablation reports which, if any, improve on the recipe above.

= Experiments <experiments>

Accuracy figures below are `val-dev` top-1/top-5 unless stated otherwise;
`val-test` is read exactly once, in @sec-full-run, per the protocol in
@dataset-protocol. All measurements were taken on the project GPU (RTX 5050
Laptop, 8GB VRAM) in its `performance` power profile; @discussion explains
why that qualifier matters for every throughput number below.

== Architecture search <sec-arch-search>

A first pass screened trunk variants by synthetic throughput and memory
alone (`benchmarks/bench.py`, `benchmarks/trunk_variants.py`, full sweep in
@appendix-sweep), before spending any GPU-hours on accuracy. Two findings
from that pass decided which variants were worth an accuracy proxy at all.
First, stage widths must be multiples of 32: a 72-144-288-576 trunk has
*fewer* parameters than the five-stage candidate below yet runs 46% slower,
falling off the tensor-core fast path. Second, final spatial resolution
dominates cost more than any other single knob: dropping the stem's
max-pool to preserve a 6x6 final feature map, at identical parameter count,
costs 2.9x throughput (1960 to 684 img/s) and doubles peak memory.

Four candidates under the parameter cap were then measured for accuracy with
fixed-seed, 15-epoch `val-dev` proxies (`benchmarks/proxy_sweep.py`):

#figure(
  caption: [
    15-epoch `val-dev` proxy results for four candidate trunks, all under the
    10M-parameter cap.
  ],
  placement: top,
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    toprule,
    table.header([Variant], [Params], [val-dev top-1], [val-dev top-5], [Peak VRAM]),
    midrule,
    [`[2,2,2,2]` 64-448], [9.46M], [*49.99%*], [78.29%], [2.04 GiB],
    [`[2,2,4,1]` 64-512], [8.94M], [49.73%], [77.93%], [2.14 GiB],
    [baseline `[2,2,2,1]` 64-512], [6.58M], [48.82%], [77.12%], [1.99 GiB],
    [5-stage 48-512 `[2,2,2,2,1]`], [9.08M], [47.42%], [75.94%], [1.61 GiB],
    botrule,
  ),
) <table-arch-search>

The five-stage variant's extra downsample to a 3x3 final map costs 1.4-2.6
accuracy points despite being the second-largest candidate by parameter
count, so the throughput and memory advantage it showed in the synthetic
pass does not transfer to a real training signal; it is ruled out. The
remaining three sit within 1.2 points of each other on a single seed each --
not enough to call a winner with confidence. We adopt the baseline
`[2,2,2,1]` 64-512 trunk provisionally: it is statistically indistinguishable
from the other two here, uses 30% fewer parameters than the nominal leader,
and that headroom has not measured as worth anything yet -- better spent, for
now, on the recipe tuning in @sec-recipe-ablation. `[2,2,4,1]` and
`[2,2,2,2]` 64-448 remain defined in `benchmarks/trunk_variants.VARIANTS` and
can be re-proxied without repeating the measurement work above, if the tuned
baseline underperforms expectations.

== Recipe ablation <sec-recipe-ablation>

#todo[
  Not yet run. Planned: fixed-seed, 15-epoch `val-dev` proxies on the
  selected trunk, sweeping (i) learning rate in \{0.05, 0.1, 0.2, 0.4\} at
  batch 256; (ii) TrivialAugment vs. RandAugment; (iii) Mixup/CutMix on/off;
  (iv) EMA on/off; (v) batch size in \{160, 256, 512\} with learning rate
  scaled linearly -- throughput is flat across this range at fixed
  resolution (1738-1785 img/s, batch 160-768), so this axis is a
  gradient-noise and batch-norm-statistics question, not a throughput one;
  (vi) cross-entropy with label smoothing vs. Generalized Cross Entropy,
  adopted only if it wins. Results and the resulting recipe decision replace
  this paragraph once measured.
]

== Full supervised training run <sec-full-run>

A full 90-epoch run on the unmodified baseline trunk, with the training
recipe above and none of @sec-recipe-ablation's extensions, validated that
the pipeline survives full length before spending further GPU-hours tuning
it. It converged to *61.57% `val-dev` top-1 / 85.67% top-5* (best checkpoint
at epoch 82: 62.08% / 85.52%), rising from about 7% at epoch 0 and
comfortably past both a from-scratch Food-101 anchor (mid-50s%) and this
trunk's own 15-epoch proxy (@table-arch-search, 47-50%). No crashes, `NaN`s,
or out-of-memory errors occurred across the full run; peak memory was 1.99
GiB, still far under the 8 GiB ceiling at full length. Wall-clock was 2.31
hours, 28% above the approximately 1.8 hour synthetic estimate from the
architecture search's throughput sweep -- per-batch time rose from about
0.16s early in the run to about 0.26s past epoch 60 as sustained load
heat-soaked the laptop GPU (@discussion). This run validates the pipeline; it
is not the project's final supervised number, which is what
@sec-recipe-ablation's tuned recipe produces, read once against `val-test`
with FixRes-style test-time resolution correction @touvron2019fixres (train
at 176px, evaluate at 224px center crop).

#todo[`val-test` headline number: pending the recipe decision in @sec-recipe-ablation.]

== Self-supervised comparison <sec-ssl>

#todo[
  Not started. Planned: SimSiam/BYOL pretraining for 200 epochs at 176px
  (about 8 GPU-hours), finetuned at 176px (about 1.8 GPU-hours), compared
  against an equal-GPU-hour supervised control (about 8 GPU-hours) trained on
  the same trunk -- without that control, the comparison cannot separate
  "self-supervised pretraining helped" from "this trunk keeps improving with
  more epochs regardless of objective." SimCLR @chen2020simclr is excluded:
  it is known to degrade below roughly 1,000-example batches, unreachable
  within an 8GB budget. Results replace this paragraph once measured.
]

= Discussion <discussion>

#paragraph[Hardware measurement mattered as much as the architecture.]
Several findings above only exist because throughput and memory were
measured rather than assumed. Stage widths must be multiples of 32 to stay
on the tensor-core fast path (a 46% throughput gap at equal parameter count,
@sec-arch-search); final spatial resolution costs more than almost any other
knob (2.9x throughput, 2x memory, for an unchanged parameter count); the
dataloader is never the bottleneck at this resolution -- plain crop-and-flip
sustains 4.4-4.7k images/second at any worker count against roughly 1.7-1.9k
images/second of GPU demand, and even TrivialAugment keeps 2.5x headroom at
eight workers, falling to 1.4x at sixteen as oversubscription costs
throughput on this machine's 15GB of system RAM; and batch size is
throughput-flat from 160 to 768 (1738-1785 img/s) while memory scales
linearly from 1.28 to 5.64 GiB, meaning the GPU is compute-saturated at the
smallest batch size tried, not idle.

#paragraph[Thermal throttling makes a short benchmark an upper bound, not an
estimate.] The GPU runs against a roughly 50W power cap with its clock
oscillating 2175-2340 MHz against a 3090 MHz maximum, and an early version of
the throughput benchmark swung 45% on one variant between sweeps before being
changed to time three 40-step rounds and keep the best (sweeps now agree
within 2-5%, and rankings are stable). Even so, the full run in
@sec-full-run took 28% longer than its own synthetic estimate: sustained
90-epoch load heat-soaks this laptop GPU in a way no short benchmark window
reproduces. Every throughput figure in this report is an upper bound on real
epoch time and a ratio between variants, not a wall-clock promise.

#paragraph[Limitations.] The architecture comparison in @sec-arch-search
rests on a single 15-epoch seed per variant; the 1.2-point spread separating
three of the four candidates is well within plausible run-to-run noise,
which is why the baseline trunk is adopted provisionally rather than
declared a winner outright. All measurements are specific to one machine (RTX
5050 Laptop, 8GB VRAM) and need not transfer, even in ratio, to other
hardware. Training is from scratch throughout, by the project's own
constraint, so absolute accuracy figures here are not directly comparable to
pretrained food-classification baselines without accounting for that.

= Conclusion <conclusion>

Under a 10M-parameter budget and a single 8GB laptop GPU, a plain residual
CNN with a genuinely global pooling head reaches 61.57% `val-dev` top-1 /
85.67% top-5 on FoodX-251 in a full 90-epoch run, before any recipe tuning.
Reaching that number meant treating measured wall-clock and memory behavior
-- channel-width alignment, spatial resolution, thermal throttling -- as
design constraints alongside the parameter cap itself: a fifth residual
stage and a depthwise-separable trunk both looked attractive on paper and
measurably lost once actually run. The recipe ablation
(@sec-recipe-ablation) and the self-supervised comparison against an
equal-compute supervised control (@sec-ssl) remain open.

#todo[Revisit this conclusion once @sec-recipe-ablation and @sec-ssl have real results.]

#show: appendix

#v(-8pt)

= Technical appendices and supplementary material

Supplementary tables referenced from @method and @sec-arch-search follow:
the full synthetic throughput sweep behind the architecture search, and
additional detail on the rejected depthwise-separable trunk.

#include "appendix.typ"

#include "checklist.typ"
