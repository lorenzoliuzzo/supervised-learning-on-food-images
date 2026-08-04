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
    parameter-matched trunk variants and pooling heads using short, fixed-seed
    proxy runs on a held-out development split, then tune a training recipe
    -- learning rate, augmentation, Mixup/CutMix, an exponential moving
    average, a noise-robust loss, batch size, and crop scale -- with the same
    proxy protocol before confirming the result with one full-length run. The
    tuned recipe reaches *63.83% top-1 / 81.79% top-3 / 87.39% top-5* on a
    held-out test split touched exactly once, up from 61.57% top-1 for the
    same trunk under an untuned recipe. We outline a planned comparison
    against self-supervised pretraining under an equal-GPU-hour budget,
    reported once measured. Throughout, we find that measured wall-clock and
    memory behavior on commodity hardware -- channel-width alignment, spatial
    resolution, and thermal throttling -- shape the final design as much as
    raw parameter or FLOP counts do, and that most proposed recipe
    improvements do not survive a fixed-seed proxy and a paired significance
    test at this parameter budget.
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
variants and pooling heads, including a check of the parameter budget below
the adopted trunk as well as above it (@sec-arch-search); a from-scratch
training recipe modernized with mixed-precision training, a cosine schedule,
and label smoothing, then tuned across learning rate, augmentation,
Mixup/CutMix, an exponential moving average, a noise-robust loss, batch size,
and crop scale, each checked against the untuned recipe with a paired
significance test rather than a point estimate alone
(@sec-recipe-ablation); a full-length training run confirming the tuned
recipe on a test split read exactly once (@sec-full-run); and a planned
comparison of supervised training against self-supervised pretraining under
an equal-GPU-hour budget (@sec-ssl).

One design choice is worth flagging before the rest. Despite FoodX-251's
long-tailed reputation, the training split we measured is close to
class-balanced (median 471 images per class; a single 34-image class drives
the oft-cited 19.3x imbalance ratio), so class-balanced losses, logit
adjustment, and long-tail-specific training are out of scope here -- the real
distributional problem is label noise, not long-tailedness
(@dataset-protocol). More generally: the architecture search, the recipe
tuning, and one full training run confirming the tuned recipe are complete at
the time of writing; the self-supervised comparison is in progress, and is
marked accordingly below rather than reported with numbers we have not
measured.

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
confirmed with one full-length run. A point-estimate gap between two
15-epoch proxies does not by itself say whether the comparison had the power
to detect a real difference, so most comparisons below additionally report a
paired McNemar's test between the two checkpoints' per-image correctness on
the shared `val-dev` images (`benchmarks/significance_test.py`) --
discordant predictions only, two-sided exact binomial on the discordant
pairs.

#paragraph[Seeding.] `main.py` accepts a `--seed` argument, but no run
behind any number in this report passed one: every architecture and recipe
comparison is a single, unseeded run per configuration. This was found only
after most of the recipe search had already run (tracked as a known gap
rather than silently left unaddressed), and it means every accuracy figure
below is a point estimate, not an exactly reproducible one -- a limitation we
return to in @discussion rather than paper over by re-running everything
under a fixed seed after the fact.

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

#paragraph[The parameter budget is not what binds, on either side of the
adopted trunk.] @sec-arch-search only ever swept trunks from 6.58M
parameters up; a later pass (@table-below-baseline) checked below it and
tested the pooling head itself for the first time. A narrower trunk
(`[2,2,2,1]` 64-384, 5.18M parameters, 21% fewer than baseline) ties the
baseline exactly under a paired significance test ($p = 1.0000$), and only a
further 3.5M-parameter cut (32-256, 1.68M parameters) costs accuracy, and
does so sharply (-6.58 points, $p < 0.0001$). Neither a spatial-attention
pooling head nor a concatenated GAP+GMP head improves on plain global average
pooling; the concatenated head's apparent underperformance is attributable to
an untuned learning rate for its differently scaled inputs rather than a
capacity problem (@discussion). Together with @sec-arch-search's finding that
9.46M parameters buys at most 1.17 points over the baseline on a single seed,
this places baseline on a plateau extending from at least 5.18M to 9.46M
parameters -- the parameter cap is not the binding constraint at this trunk
depth in either direction, which is why @sec-recipe-ablation, not further
architecture search, is where the remaining project budget goes.

#figure(
  caption: [
    Sub-baseline trunk widths and alternative pooling heads, 15-epoch
    `val-dev` proxies under the tuned Phase D recipe (@sec-recipe-ablation),
    each compared against the baseline trunk with a paired McNemar's test on
    the same `val-dev` images.
  ],
  placement: top,
  table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    toprule,
    table.header([Variant], [Params], [val-dev top-1], [$Delta$ vs. baseline], [McNemar $p$]),
    midrule,
    [narrow `[2,2,2,1]` 64-384], [5.18M], [55.22%], [+0.00], [1.0000],
    [baseline `[2,2,2,1]` 64-512], [6.58M], [55.22%], [--], [--],
    [head: spatial attention], [6.59M], [55.01%], [-0.21], [0.6826],
    [narrow `[2,2,2,1]` 32-256], [1.68M], [48.64%], [-6.58], [$<0.0001$],
    [head: GAP+GMP concat], [6.71M], [37.51%], [-17.71], [$<0.0001$],
    botrule,
  ),
) <table-below-baseline>

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
checkpoint key for no benefit here. Two hyperparameters this loop leaves open
-- peak learning rate and the crop's minimum scale -- are swept rather than
guessed; @sec-recipe-ablation reports the search and the values it settled
on: batch 256, learning rate 0.8, and the default `RandomResizedCrop` scale
floor of 0.08 kept unchanged despite starting from an ImageNet-scale
assumption. Validation reads a 224px center crop from a 256px resize
regardless of training resolution, a FixRes-style correction
@touvron2019fixres for the systematic scale shift `RandomResizedCrop`
introduces between train and test time; because it is applied by default
rather than as an opt-in flag, it is already folded into every accuracy
figure in this report rather than a separate, uncosted improvement.

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

Six recipe axes were each swept independently against the same 15-epoch
`val-dev` control, all on the baseline trunk. Every gap below is reported
with the McNemar significance test from @dataset-protocol; several axes with
a real-looking point-estimate gap did not survive it.

#paragraph[Learning rate.] Doubling the learning rate five times over
$\{0.05, 0.1, 0.2, 0.4, 0.6, 0.8\}$ produced a clean diminishing-returns
curve (+5.45, +3.15, +1.46, +0.94, +0.62 top-1 points per doubling), so the
search stopped at 0.8 rather than continue into a regime of shrinking reward
and growing optimization noise (0.8's val-accuracy curve is visibly noisier
mid-run than 0.4's or 0.6's, despite a smooth, undiverged training loss):

#figure(
  caption: [Learning-rate sweep, 15-epoch `val-dev` proxy, batch 256.],
  placement: top,
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    toprule,
    table.header([LR], [val-dev top-1], [top-3], [top-5]),
    midrule,
    [0.05], [43.96%], [--], [--],
    [0.10 (prior default)], [49.41%], [--], [--],
    [0.20], [52.56%], [--], [--],
    [0.40], [54.02%], [74.16%], [81.02%],
    [0.60], [54.96%], [--], [--],
    [*0.80 (adopted)*], [*55.58%*], [--], [--],
    botrule,
  ),
) <table-lr-sweep>

#paragraph[Augmentation.] TrivialAugment and RandAugment each score roughly
3 points below a no-augmentation control at 15 epochs, and a deep-dive
(macro-F1, weighted-F1, worst-30-class average, calibration) confirms this
is not a top-1-only illusion -- the control leads on every axis, so
augmentation is not quietly buying rare-class recall at the cost of overall
accuracy. But both augmented runs' training loss was still well above the
control's at epoch 15, so a matched-epoch rematch at 30 epochs was run
before drawing a conclusion:

#figure(
  caption: [
    RandAugment vs. no augmentation, matched at 30 epochs, same protocol.
  ],
  placement: top,
  table(
    columns: 6,
    align: (left,) + (right,) * 5,
    stroke: none,
    toprule,
    table.header([Recipe (30 ep)], [val-dev top-1], [macro-F1], [weighted-F1], [worst-30 avg], [ECE]),
    midrule,
    [none (control)], [*61.27%*], [*59.38%*], [*60.90%*], [*23.62%*], [0.144],
    [RandAugment], [60.43%], [58.31%], [59.84%], [23.21%], [0.157],
    botrule,
  ),
) <table-augment-30ep>

The 15-epoch proxy exaggerated the gap (about 3 points vs. 0.84 at matched
epochs), and the control leads on every point estimate in the rematch -- but
a paired McNemar's test on the 855 discordant `val-dev` predictions puts the
0.84-point gap at $p = 0.075$, not significant. The honest read is that this
proxy cannot distinguish "control is slightly better" from "the two are
tied"; the plain recipe carries forward on simplicity and an unbeaten record,
not a proven margin over RandAugment specifically.

#paragraph[Mixup, CutMix, EMA, and Generalized Cross Entropy.] Unlike
augmentation, these four are decisively ruled out -- all four land at
$p < 0.0001$ against the same 15-epoch control:

#figure(
  caption: [
    Mixup/CutMix/EMA/GCE, 15-epoch `val-dev` proxy against the same control
    as @table-lr-sweep.
  ],
  placement: top,
  table(
    columns: 6,
    align: (left,) + (right,) * 5,
    stroke: none,
    toprule,
    table.header([Recipe], [val-dev top-1], [macro-F1], [worst-30 avg], [zero-acc classes], [ECE]),
    midrule,
    [none (control)], [55.58%], [53.38%], [16.77%], [4], [0.182],
    [Mixup @zhang2018mixup], [42.95%], [40.05%], [5.13%], [11], [0.231],
    [CutMix @yun2019cutmix], [48.84%], [45.61%], [7.19%], [5], [0.172],
    [EMA (decay 0.999)], [39.88%], [38.32%], [4.52%], [9], [0.107],
    [GCE @zhang2018generalized ($q=0.7$)], [22.23%], [14.86%], [0.00%], [*119*], [0.328],
    botrule,
  ),
) <table-mixup-ema-gce>

GCE's collapse is not underconfidence hiding behind a low top-1: mean
confidence on its wrong predictions is 0.49 and its ECE is the worst of
every recipe tested, i.e. it is confidently wrong on the 119 of 251 classes
it collapsed on. Both GCE and EMA's failures are attributable to
hyperparameters tuned for a different loss landscape -- learning rate 0.8
was found for label-smoothed cross-entropy, and GCE's bounded loss produces
much smaller gradients as confidence rises, while EMA's 0.999 decay implies
a roughly 1,000-step averaging window against a 6,945-step proxy -- rather
than a settled verdict against either technique; this is flagged as an open
question in @discussion, not reopened here.

#paragraph[Batch size.] Throughput is flat from batch 160 to 768
(1738-1785 img/s), so this axis is a gradient-noise and batch-norm-statistics
question rather than a throughput one. Swept at $\{160, 256, 512\}$ with
learning rate scaled linearly from the batch-256 control, all three are
statistically tied ($p = 0.74$ and $p = 0.49$ respectively against the
control); batch 256 is kept because every other axis in this section was
measured there, on no evidence against it.

#paragraph[Crop scale floor.] `RandomResizedCrop`'s minimum crop area had
run at the ImageNet-inherited default of 0.08 through every result above --
aggressive for this dataset's roughly 90k training images. Raising the floor
to 0.25 and 0.40 was motivated by a diagnosis that recurs throughout this
section: validation accuracy exceeds training accuracy in every recipe
tried, the opposite of the usual overfitting pattern, and the gap widens
rather than closes with more epochs. Raising the crop floor does close that
gap (val-minus-train goes from +11.05 points at 0.08 to +4.31 at 0.25 to
$-0.56$ at 0.40, ordinary train-above-val behavior by the third), but
accuracy does not follow: 0.25 ties the control and 0.40 is significantly
*worse* ($p = 0.0232$). `RandomResizedCrop`'s low-scale cropping is doing
double duty as both a regularizer and a view-diversity multiplier that nothing
else in the recipe replaces, so relaxing it trades away diversity without
buying back the epochs needed to use the easier task -- a net loss at a fixed
15-epoch budget. The default of 0.08 is kept.

#paragraph[A further candidate, not part of the six axes above.] A
similarity-smoothed cross-entropy loss -- label smoothing redistributed
toward classes flagged as likely near-duplicates by bidirectional confusion
in the trunk's own error analysis (@discussion), rather than spread
uniformly -- was implemented and logged at the same protocol (lr 0.8, batch
256, 15 epochs): 56.06% val-dev top-1, the best 15-epoch number measured
anywhere in this search. It has not been checked against the control with a
significance test or replicated, so it is reported here as a logged,
promising result rather than an adopted part of the recipe -- the standard
this section holds every other axis to, and a natural next step before the
self-supervised comparison in @sec-ssl.

#paragraph[Adopted recipe.] Label-smoothed cross-entropy, no extra
augmentation, no Mixup/CutMix, no EMA, learning rate 0.8, batch 256,
`RandomResizedCrop` scale floor 0.08 -- unmodified from the base recipe in
@method except for the tuned learning rate. This is the recipe
@sec-full-run's headline run trains with.

== Full supervised training run <sec-full-run>

#paragraph[Pipeline validation, pre-tuning.] Before spending GPU-hours
tuning the recipe, a full 90-epoch run on the unmodified baseline trunk with
the base recipe (learning rate 0.1, none of @sec-recipe-ablation's tuning)
confirmed the pipeline survives full length. It converged to *61.57%
`val-dev` top-1 / 85.67% top-5* (best checkpoint at epoch 82: 62.08% /
85.52%), rising from about 7% at epoch 0 and comfortably past both a
from-scratch Food-101 anchor (mid-50s%) and this trunk's own 15-epoch proxy
(@table-arch-search, 47-50%). No crashes, `NaN`s, or out-of-memory errors
occurred; peak memory was 1.99 GiB, still far under the 8 GiB ceiling at
full length. Wall-clock was 2.31 hours, 28% above the approximately 1.8 hour
synthetic estimate from the architecture search's throughput sweep --
per-batch time rose from about 0.16s early in the run to about 0.26s past
epoch 60 as sustained load heat-soaked the laptop GPU (@discussion).

#paragraph[Tuned headline run.] With @sec-recipe-ablation's adopted recipe
-- unchanged trunk and head, learning rate 0.8, batch 256, crop-scale floor
0.08 -- a second full 90-epoch run was trained and read against `val-test`
exactly once, per @dataset-protocol:

#figure(
  caption: [
    Tuned recipe, full 90-epoch run. `val-test` is read once, here; every
    other number in this report reads `val-dev`.
  ],
  placement: top,
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    toprule,
    table.header([], [top-1], [top-3], [top-5]),
    midrule,
    [val-dev, best (epoch 86)], [64.36%], [82.22%], [87.51%],
    [val-dev, final (epoch 90)], [64.26%], [82.14%], [87.61%],
    [*val-test (headline)*], [*63.83%*], [*81.79%*], [*87.39%*],
    botrule,
  ),
) <table-full-run>

This is up 2.3 points on the untuned pipeline-validation run above, at the
same trunk and epoch count, and continues that run's trend cleanly across
15/30/90-epoch schedules (55.6% to 61.3% to 63.8% `val-test` / 64.4%
`val-dev` best) -- diminishing but still real returns from schedule length
alone, consistent with the cosine schedule's own design. It also confirms
the underfitting diagnosis @sec-recipe-ablation's crop-scale investigation
was built on: the +11.05-point val-over-train gap measured at epoch 15 has
closed to near zero by epoch 90 (train top-1 64.88% vs. val-dev top-1
64.26% at the final epoch) -- the model needed the full schedule to
converge, and nothing in @sec-recipe-ablation sped that up because, per the
crop-scale result, nothing needed to. Best and final checkpoints are within
0.1 points of each other here, unlike the roughly 0.5-point gap in the
pipeline-validation run, itself evidence against late-training overfitting.
Wall-clock was 2h05m at 1.99 GiB peak memory, with no thermal slowdown this
time (0.147-0.169s/step steady from epoch 5 through epoch 89) after the
machine was switched to its high-performance power profile partway through
an earlier proxy run this phase -- consistent with, though not a controlled
replication of, the pipeline-validation run's own read that its slowdown was
a power-cap effect rather than a hard thermal limit.

As in every result in this report (@dataset-protocol), this run was not
seeded; the headline number above is a point estimate.

== Self-supervised comparison <sec-ssl>

The design is fixed and implemented; no pretraining run has completed at the
time of writing. SimSiam @chen2021simsiam is chosen over BYOL
@grill2020byol: BYOL's target network is a second, momentum-averaged copy of
the trunk, the same mechanism implicated in @table-mixup-ema-gce's EMA
collapse, and SimSiam avoids it entirely with a stop-gradient and predictor
asymmetry instead, at no second ~6.5M-parameter shadow copy in an
already-tight 8GB budget. SimCLR @chen2020simclr is excluded outright: it is
known to degrade below roughly 1,000-example batches, unreachable here.

Two design choices depart from a naive port of the supervised pipeline.
First, pretraining draws on the FoodX-251 challenge's 28,377-image unlabeled
test split in addition to the 118,475 labeled training images (146,852
total) -- pretraining on only the images already carrying labels would not
exercise the premise a self-supervised comparison exists to test. Second,
because SimSiam's training loss can and does keep falling even after the
representation has collapsed to a constant output, pretraining is monitored
with a weighted $k$-nearest-neighbor probe on frozen features
@wu2018unsupervised against `val-dev`, alongside two collapse diagnostics --
mean per-dimension feature standard deviation and the effective rank of the
feature covariance -- computed every few epochs rather than trusting loss
alone.

The plan calls for 200 pretrain epochs at 176px (about 8 GPU-hours),
finetuning at 176px (about 1.8 GPU-hours), against an equal-GPU-hour
supervised control (about 8 GPU-hours) on the same trunk -- without that
control, the comparison cannot separate "self-supervised pretraining helped"
from "this trunk keeps improving with more epochs regardless of objective."
Results, and the finetuning protocol they are read under, replace this
paragraph once measured.

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

#paragraph[Near-duplicate classes are a real dataset property, not a
training artifact.] Bidirectional confusion analysis on the recipe-search
checkpoints repeatedly flags the same class pairs as mutually confused --
beef tartare/steak tartare, chicken wing/buffalo wing, oyster/huitre, among
others -- and these pairs recur under all three augmentation conditions in
@table-augment-30ep, not just one run. That consistency across independently
trained checkpoints is what makes them a property of the label space rather
than a fluke of one training run, and is the motivation behind
@sec-recipe-ablation's similarity-smoothed loss candidate.

#paragraph[Ruled-out recipe axes may only be ruled out at this learning
rate.] GCE and EMA's collapses in @table-mixup-ema-gce are large and
statistically decisive, but both use a learning rate of 0.8, tuned for
label-smoothed cross-entropy's loss landscape rather than either technique's
own. A bounded loss like GCE produces systematically smaller gradients as
confidence rises, and EMA's effective averaging window is a function of its
decay rate relative to schedule length; either could plausibly change under
its own learning-rate search. This report treats them as ruled out under the
recipe actually adopted, not as a claim that no learning rate would make
them competitive.

#paragraph[Limitations.] The architecture comparison in @sec-arch-search
rests on a single 15-epoch seed per variant; the 1.2-point spread separating
three of the four candidates is well within plausible run-to-run noise,
which is why the baseline trunk is adopted provisionally rather than
declared a winner outright. More broadly, no run in this report -- across
architecture search, recipe ablation, or the full 90-epoch confirmation --
was seeded (@dataset-protocol), so every accuracy figure here is a point
estimate from a single training run rather than a distribution; where a
McNemar test is reported it addresses sampling variance on the shared
validation images, not this run-to-run variance. All measurements are
specific to one machine (RTX 5050 Laptop, 8GB VRAM) and need not transfer,
even in ratio, to other hardware. Training is from scratch throughout, by
the project's own constraint, so absolute accuracy figures here are not
directly comparable to pretrained food-classification baselines without
accounting for that.

= Conclusion <conclusion>

Under a 10M-parameter budget and a single 8GB laptop GPU, a plain residual
CNN with a genuinely global pooling head reaches *63.83% `val-test` top-1 /
81.79% top-3 / 87.39% top-5* on FoodX-251 -- up from 61.57% `val-dev` top-1
for the same trunk under an untuned recipe -- after a six-axis recipe search
in which most proposed improvements (stronger augmentation, Mixup, CutMix,
an exponential moving average, a noise-robust loss, a wider crop-scale
floor) failed a paired significance test against the plain baseline, and a
parameter-budget check in both directions found the trunk sitting on a
plateau rather than starved of capacity. Reaching the headline number meant
treating measured wall-clock and memory behavior -- channel-width alignment,
spatial resolution, thermal throttling -- as design constraints alongside
the parameter cap itself: a fifth residual stage and a depthwise-separable
trunk both looked attractive on paper and measurably lost once actually run.
The self-supervised comparison against an equal-compute supervised control
(@sec-ssl) remains open.

#todo[Revisit this conclusion once @sec-ssl has real results.]

#show: appendix

#v(-8pt)

= Technical appendices and supplementary material

Supplementary tables referenced from @method and @sec-arch-search follow:
the full synthetic throughput sweep behind the architecture search, and
additional detail on the rejected depthwise-separable trunk.

#include "appendix.typ"

#include "checklist.typ"
