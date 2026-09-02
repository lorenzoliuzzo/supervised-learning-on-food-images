#import "@preview/bloated-neurips:0.8.0": appendix, botrule, midrule, neurips2026, paragraph, toprule, url

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
    million trainable parameters and a single 8GB laptop GPU. Short 15-epoch
    proxy runs on a held-out development split screen trunk variants and
    recipe choices cheaply; one full-length run on the selected configuration
    then reaches *63.83% top-1 / 87.39% top-5* on a test split read exactly
    once, using 66% of the parameter budget. Two results shape that outcome.
    Capacity never binds: a trunk 21% smaller ties the adopted one exactly
    (McNemar $p = 1.0$), and the accuracy cliff lies somewhere below that,
    bracketed between 1.68M and 5.18M parameters. Nor does overfitting: six regularization axes were each proxied and
    none beat the plain recipe, because the model underfits at every schedule
    length tested -- the validation-over-training accuracy gap closes from
    $+11.1$ points at 15 epochs to zero only at 90. Throughout, measured
    wall-clock and memory behavior on commodity hardware shapes the design as
    much as parameter or FLOP counts do. A comparison against self-supervised
    pretraining under an equal-GPU-hour budget is designed and implemented but
    not yet run, and is reported as such.
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

Two constraints are imposed by the setting rather than chosen for effect: the
classifier trains from scratch, with no ImageNet or other pretraining, under
fewer than 10,000,000 trainable parameters, and every run happens on a single
consumer laptop GPU with 8GB of VRAM. Together they shape everything that
follows -- most of all by making wall-clock, not parameters, the resource
actually in short supply.

We make three contributions, in the order the underlying work was carried out.
First, a budget-constrained architecture search (@sec-arch-search) that
screens residual-trunk variants on measured throughput and memory before
accuracy, and that probes *below* the adopted trunk as well as above it --
finding that the parameter cap never binds, and bracketing the width at
which accuracy does start to fall. Second, a six-axis recipe search
(@sec-recipe-ablation) in which every regularizer tried lost to the plain
recipe, with a paired significance test separating the losses that are real
from the one that is merely unproven, and a single diagnosis -- underfitting
-- that accounts for all six. Third, a full-length run confirming the
resulting configuration on a test split touched exactly once
(@sec-full-run). A fourth strand, a comparison against self-supervised
pretraining under an equal-GPU-hour budget (@sec-ssl), is designed and
implemented but unrun; it is presented as a design, without numbers.

One design choice is worth flagging before the rest. Despite FoodX-251's
long-tailed reputation, the training split we measured is close to
class-balanced (median 471 images per class; a single 34-image class drives
the oft-cited 19.3x imbalance ratio), so class-balanced losses, logit
adjustment, and long-tail-specific training are out of scope here -- the real
distributional problem is label noise, not long-tailedness
(@dataset-protocol).

= Related work <related-work>

Three lanes matter here, and all three share one gap.

#paragraph[Parameter-efficient architecture.] MobileNetV4 @qin2024mobilenetv4
and MobileOne @vasu2022mobileone optimize for on-device latency rather than
parameter count alone -- the same distinction @sec-arch-search arrives at by
measurement -- while ConvNeXt V2 @woo2023convnextv2 revisits convolutional
design at a scale where masked-autoencoder pretraining is available, a regime
the from-scratch constraint excludes.

#paragraph[Augmentation and label-noise robustness.] RandAugment
@cubuk2020randaugment and TrivialAugment @muller2021trivialaugment establish
that automated or tuning-free policies match searched ones at near-zero search
cost; mixup @zhang2018mixup and CutMix @yun2019cutmix mix inputs and labels
instead of choosing among fixed operators; and Generalized Cross Entropy
@zhang2018generalized offers a noise-robust loss, directly relevant to
web-crawled training labels. @sec-recipe-ablation tests all five.

#paragraph[Self-supervised pretraining.] SimCLR @chen2020simclr needs a large
negative-sample batch, unreachable at 8GB. BYOL @grill2020byol and SimSiam
@chen2021simsiam avoid negative pairs entirely and stay viable at small batch
sizes, which is what makes them the basis of the design in @sec-ssl.

#paragraph[The gap.] We found no source evaluating any of this on FoodX-251,
Food-101 or Food-2K under a hard parameter budget, and none ablating
sub-10M-parameter models trained for tens of epochs. Whether the accepted
wisdom transfers to that regime is an open question here, which is why
@sec-recipe-ablation re-tests techniques well established elsewhere -- and
why several do not survive.

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

#paragraph[Proxy protocol and logging.] At about 1.2 minutes/epoch a
15-epoch run costs roughly 18 minutes, making comparisons that would be
unaffordable at full length routine. Every comparison below ranks candidates
on a 15-epoch `val-dev` proxy, with the winner confirmed by one full-length
run; every run is logged with a config hash, per-epoch train/validation
accuracies, wall-clock and peak VRAM (`src/runlog.py`), so each number here
traces back to a specific run.

#paragraph[Seeding.] No run in this project was seeded, so every accuracy
figure below is a point estimate from one unrepeatable run rather than a mean
over seeds. This bounds what the comparisons can claim: the paired McNemar
tests used throughout account for *sampling* variance over images and are
silent on the training variance reseeding would expose. Two comparisons here
are close enough for that to matter; both are flagged where they occur.

= Method <method>

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

Each residual block normalizes after both convolutions and, whenever the
shortcut is not the identity, after the 1x1 shortcut projection too -- an
unnormalized branch should not be added to a normalized one. The second batch
norm's scale ($gamma$) is zero-initialized, so every block begins as an
identity map, which is what makes a high learning rate safe under warmup; and
the nonlinearity is applied after the residual addition, so each block is an
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

#paragraph[A rejected alternative.] A MobileNetV2-style depthwise-separable
version of the same trunk measured 2.55M parameters, 621 img/s and 3.83 GiB
peak against the plain trunk's 6.58M, 918 img/s and 1.28 GiB
(@appendix-depthwise). Fewer parameters, slower and heavier: depthwise
convolutions trade parameter count for activation memory and wall-clock, which
is the wrong trade when throughput and memory are what bind (@discussion).

== Training recipe

The reference training loop is modernized in ways that cost nothing at this
parameter budget: mixed-precision `torch.autocast`, with the dtype chosen from
the GPU's compute capability rather than hardcoded -- bf16 without a
`GradScaler` on Ampere and later (bf16 needs no dynamic loss scaling), fp16
with one below it, since bf16 has no Tensor Core path on older cards and falls
back to an emulation roughly an order of magnitude slower -- `channels_last`
memory format on both model and inputs, a cosine learning-rate schedule with
five-epoch linear warmup in place of step decay, `CrossEntropyLoss` with
label smoothing 0.1, SGD with Nesterov momentum, eight dataloader workers (up
from four), a 176px training crop (down from 224), and no `DataParallel`
wrapper -- there is only one GPU to shard across, and the wrapper's
scatter/gather adds a per-step cost and a `module.` prefix to every
checkpoint key for no benefit here.

== Recipe extensions <recipe-extensions>

Four further axes are opt-in flags, all defaulting to off: an augmentation
switch between none, TrivialAugment @muller2021trivialaugment and RandAugment
@cubuk2020randaugment; batch-wise Mixup @zhang2018mixup and CutMix
@yun2019cutmix, scored against the pre-mix hard label since a mixed batch has
no single correct class; an exponential moving average of weights
(`AveragedModel`, batch-norm statistics left live rather than averaged); and
Generalized Cross Entropy @zhang2018generalized in place of label-smoothed
cross-entropy, for the training set's label noise.

= Experiments <experiments>

Accuracy figures below are `val-dev` unless stated otherwise; `val-test` is
read exactly once, in @sec-full-run. All measurements come from the project
GPU (RTX 5050 Laptop, 8GB VRAM) in its `performance` power profile, a
qualifier @discussion shows matters for every throughput number here.

== Architecture search <sec-arch-search>

A first pass screened trunk variants on synthetic throughput and memory alone
(@appendix-sweep), before spending GPU-hours on accuracy. Two findings from it
decided which variants earned an accuracy proxy: stage widths must be
multiples of 32, since a 72-144-288-576 trunk has *fewer* parameters than the
five-stage candidate yet runs 46% slower off the tensor-core fast path; and
final spatial resolution dominates every other knob, with the stem max-pool's
removal costing 2.9x throughput and 2x memory at identical parameter count.
The four survivors were then measured for accuracy over 15-epoch `val-dev`
proxies (`benchmarks/proxy_sweep.py`):

#figure(
  caption: [
    15-epoch `val-dev` proxy results for four candidate trunks, all under the
    10M-parameter cap.
  ],
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
remaining three sit within 1.2 points of each other on one unseeded run each
-- not enough to call a winner with confidence. We adopt the baseline
`[2,2,2,1]` 64-512 trunk provisionally: it is statistically indistinguishable
from the other two here, uses 30% fewer parameters than the nominal leader,
and that headroom has not measured as worth anything yet -- better spent, for
now, on the recipe tuning in @sec-recipe-ablation. `[2,2,4,1]` and
`[2,2,2,2]` 64-448 remain defined in `benchmarks/trunk_variants.VARIANTS` and
can be re-proxied without repeating the measurement work above, if the tuned
baseline underperforms expectations.

#paragraph[Below the baseline, and the head.] That sweep spans 6.58M to 9.46M
-- the baseline and upward -- but diminishing returns going up say nothing
about going down, and the pooling head had never been an experimental axis at
all. Five further 15-epoch legs close both gaps. They ran under the recipe
@sec-recipe-ablation settles on (plain, lr 0.8) rather than the lr 0.1 above,
so they are not comparable to @table-arch-search, which is why the baseline
was re-run as its own leg. Each is compared against it by a paired McNemar
exact test on the same 6,063 images (`benchmarks/significance_test.py`),
since a point-estimate gap alone cannot say whether a comparison had the
power to detect a difference:

#figure(
  caption: [
    Trunk widths below the baseline, and two alternative pooling heads.
    15-epoch `val-dev` proxies, plain recipe, lr 0.8, batch 256; $p$ from a
    paired McNemar exact test against the baseline leg.
  ],
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: none,
    toprule,
    table.header([Variant], [Params], [top-1], [$Delta$], [McNemar $p$], [Peak VRAM]),
    midrule,
    [narrow `[2,2,2,1]` 64-384], [*5.18M*], [55.22%], [*+0.00*], [*1.0000*], [1.96 GiB],
    [baseline `[2,2,2,1]` 64-512], [6.58M], [55.22%], [---], [---], [1.99 GiB],
    [head: spatial attention], [6.59M], [55.01%], [-0.21], [0.6826], [1.99 GiB],
    [narrow `[2,2,2,1]` 32-256], [1.68M], [48.64%], [-6.58], [$< 0.0001$], [1.08 GiB],
    [head: GAP+GMP concat], [6.71M], [37.51%], [-17.71], [$< 0.0001$], [1.99 GiB],
    botrule,
  ),
) <table-width-floor>

*The parameter cap does not bind, and now there is evidence on both sides of
the baseline rather than one.* At 21% fewer parameters, 64-384 is not merely
close to the baseline but a literal dead heat: 441 images the baseline alone
got right, 441 the narrow trunk alone got right, $p = 1.0000$. Halving again
to 32-256 then costs 6.58 points at $p < 0.0001$. So the plateau extends
below the adopted trunk, and it ends in a cliff rather than a slope,
somewhere between 1.68M and 5.18M parameters. We keep the 6.58M baseline
anyway, since 64-384 buys no accuracy, no meaningful wall-clock (20 vs. 21
minutes) and no VRAM that matters, while the baseline is the trunk every
recipe measurement below was taken on. On a task graded against a parameter
cap, it is worth stating plainly that the cap was never the difficulty.

Neither pooling head displaces global average pooling. Spatial attention is a
clean null ($-0.21$, $p = 0.68$): learned pooling weights bought nothing over
uniform averaging. The GAP+GMP concatenation's collapse is a different
failure -- its train loss starts *above* the baseline's at epoch 0 (5.595 vs.
4.996) and never closes, so it underfits rather than diverges. Max-pooled
activations are unbounded and much larger than mean-pooled ones, so
concatenating hands the classifier two feature blocks on very different
scales, against a learning rate tuned for a head without that problem. It
needs its own learning rate or a normalization layer; that is not the same as
being wrong for this dataset.

== Recipe ablation <sec-recipe-ablation>

Six axes were swept on the selected trunk with 15-epoch `val-dev` proxies:
learning rate, augmentation policy, Mixup/CutMix, weight averaging, training
loss, and batch size, plus the `RandomResizedCrop` scale floor. *Every one lost
to the plain recipe.* This section separates the losses a significance test
calls real from the one it does not, then gives the diagnosis behind all six.

#paragraph[Learning rate, the one axis that paid.] Sweeping lr at batch 256
gives 43.96%, 49.41%, 52.56%, 54.02%, 54.96%, 55.58% top-1 for lr
0.05, 0.1, 0.2, 0.4, 0.6, 0.8 (per-epoch curves in @appendix-lr). The gain
per doubling falls cleanly -- $+5.45$, $+3.15$, $+1.46$, $+0.94$, $+0.62$ --
so a further doubling to 1.6 would be worth perhaps 0.3-0.4 points. Against
that, lr 0.8's validation curve is visibly noisier mid-run than 0.4's or
0.6's, though its training loss never wobbles and reaches the lowest final
value of the sweep: a noisier optimization regime, not an unstable one.
Shrinking reward against growing noise is where we stop. *lr 0.8 is what the
rest of this section, and @sec-full-run, carry forward* -- an 8x increase
over the reference script's default, and worth 5.6 points over the lr 0.1 it
replaces.

#paragraph[Everything meant to regularize.] The remaining axes are compared
against a plain-recipe control at the same lr, each with a paired McNemar
exact test on the same 6,063 `val-dev` images. Since top-1 alone could hide a
rare-class or calibration trade-off, `benchmarks/analyze_errors.py` scores
every checkpoint on macro-F1, the mean accuracy of the 30 worst classes, and
15-bin expected calibration error:

#figure(
  caption: [
    Five regularization axes against the plain-recipe control. 15-epoch
    `val-dev` proxies, lr 0.8, batch 256. "worst-30" is the mean accuracy of
    the 30 lowest-accuracy classes; ECE is 15-bin expected calibration
    error.
  ],
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: none,
    toprule,
    table.header([Recipe], [top-1], [macro-F1], [worst-30], [ECE], [McNemar $p$]),
    midrule,
    [*plain (control)*], [*55.58%*], [*53.38%*], [*16.77%*], [0.182], [---],
    [TrivialAugment], [52.09%], [49.63%], [14.05%], [0.180], [$< 0.0001$],
    [RandAugment], [52.60%], [50.22%], [14.78%], [0.184], [$< 0.0001$],
    [CutMix], [48.84%], [45.61%], [7.19%], [0.172], [$< 0.0001$],
    [Mixup], [42.95%], [40.05%], [5.13%], [0.231], [$< 0.0001$],
    [EMA (decay 0.999)], [39.88%], [38.32%], [4.52%], [*0.107*], [$< 0.0001$],
    [GCE loss ($q = 0.7$)], [22.23%], [14.86%], [0.00%], [0.328], [$< 0.0001$],
    botrule,
  ),
) <table-recipe-ablation>

The control leads on top-1, macro-F1 and worst-30 simultaneously, so none of
these is quietly buying rare-class recall at the cost of overall accuracy.
Two of the losses deserve a caveat rather than a verdict. EMA's averaging
window at decay 0.999 is roughly 1,000 steps against a proxy only 6,945 steps
long, so its weights may still be lagging the raw, rapidly-moving ones when
the proxy ends -- note that it is the best-calibrated run in the table, which
is what a lagging average would look like. GCE's collapse is more extreme
than a merely bad hyperparameter usually produces -- 119 of 251 classes at
zero accuracy -- but the shape of the failure is diagnostic: mean confidence
on wrong predictions is 0.49 and its ECE is the worst measured, so it is
confidently wrong rather than underconfident. GCE's bounded loss
$L_q = (1 - p_y^q) \/ q$ produces smaller gradients as confidence rises, and
lr 0.8 was tuned for cross-entropy's landscape. *Both results say "needs its
own learning-rate search," not "wrong for this dataset."*

#paragraph[The one close call, and why a point estimate was not enough.] At
15 epochs both augmentation policies' training loss was still well above the
control's (3.32-3.43 vs. 3.02) -- a signal that the proxy might simply be too
short for augmentation to pay off, which none of the other axes showed as
clearly. RandAugment therefore got a matched-epoch rematch: both legs re-run
fresh for 30 epochs, since a cosine schedule needs its total length fixed
upfront and cannot be resumed into. The gap narrowed from about 3 points to
*0.84* (61.27% vs. 60.43%), with the control still ahead on macro-F1,
weighted-F1 and worst-30. On point estimates alone that reads as a clean win.
It is not: McNemar's test on the shared images finds 454 the control alone
got right against 401 for RandAugment, $n = 855$ discordant, *$p = 0.075$ --
not significant*. The honest reading is that this proxy lacks the power to
separate "the plain recipe is slightly better" from "these are tied." The
plain recipe carries forward on simplicity and an unbeaten record, not on a
demonstrated margin over RandAugment.

#paragraph[Two non-levers.] Batch size is flat: 160, 256 and 512 with the
learning rate scaled linearly land at 55.85%, 55.67% and 55.30%, all tied
against the control ($p = 0.74$, $p = 0.49$), and throughput is flat across
the same range (@discussion), so batch size moves neither thing it could
have. Raising the `RandomResizedCrop` scale floor above its
ImageNet-inherited 0.08 -- plausible, since that floor was tuned for 1.28M
images against this set's 90k -- half-works instructively: the
train/validation gap closes exactly as predicted ($+11.05$ to $+4.31$ to
$-0.56$), but accuracy does not follow, tying at 0.25 and losing
significantly at 0.40 ($p = 0.023$). Low-scale cropping is doing double duty
as regularizer *and* view-diversity source; relaxing it makes the task easier
to fit while removing diversity, a net loss at fixed epochs
(@appendix-nonlevers).

#paragraph[Why all six lost.] The crop-scale result names the cause the other
five share. At every schedule length tested, validation accuracy runs *above*
training accuracy -- $+11.05$ points at 15 epochs, $+7.3$ at 30. *This model
underfits; it does not overfit.* Regularizing a model that has not finished
fitting can only slow it down, which is exactly the ordering observed: the
stronger the regularizer, the worse the result (Mixup below CutMix below
RandAugment below plain). The one axis that did pay, learning rate, helps a
model fit faster rather than generalize better. What it needed was schedule
length, and @sec-full-run shows the gap closing to zero at 90 epochs.

*Carried forward: the plain recipe* -- label-smoothed cross-entropy, no
augmentation beyond crop-and-flip, no Mixup/CutMix, no EMA -- at lr 0.8, batch
256, crop-scale floor 0.08.

== Full supervised training run <sec-full-run>

One 90-epoch run was budgeted, and it was spent on exactly the configuration
the sections above settled: the baseline `[2,2,2,1]` 64-512 trunk, GAP head,
plain recipe, lr 0.8, batch 256, crop-scale floor 0.08, nothing tuned for this
run. Evaluation applies FixRes-style test-time resolution correction
@touvron2019fixres -- train at 176px, evaluate at `Resize(256)` +
`CenterCrop(224)` -- which `main.py` already does by default.

#figure(
  caption: [
    The full 90-epoch run. `val-test` is read once, here.
  ],
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    toprule,
    table.header([], [top-1], [top-3], [top-5]),
    midrule,
    [`val-dev`, best (epoch 86)], [64.36%], [82.22%], [87.51%],
    [`val-dev`, final (epoch 90)], [64.26%], [82.14%], [87.61%],
    [*`val-test` (headline)*], [*63.83%*], [*81.79%*], [*87.39%*],
    botrule,
  ),
) <table-full-run>

`val-test` lands 0.53 points below `val-dev`'s best -- the expected direction,
since `val-dev` selected the checkpoint, and small enough to say the split is
not doing anything surprising. Best and final differ by 0.1 points, against
roughly 0.5 in the earlier validation run: no late overfitting.

#paragraph[What the recipe search bought.] An earlier 90-epoch run on the same
trunk at the pre-search default lr 0.1 reached 61.57% `val-dev` top-1; it was
run to prove the pipeline survives full length, and it did, without crashes,
`NaN`s or out-of-memory errors. The tuned run's 64.36% is *+2.79 points at
identical trunk and epoch count*, almost all of it the learning rate, since
that is the only one of the six axes that moved.

#paragraph[Schedule length is what was buying accuracy.] Read across the three
schedule lengths at the same recipe -- 55.6% at 15 epochs, 61.3% at 30, 64.4%
at 90 (`val-dev`) -- and the returns diminish but never stop. The training
signal says why: train top-1 finished at 64.88% against 64.26% val, so the
*+11.05-point* validation-over-training gap measured at 15 epochs, and
$+7.3$ at 30, has closed to roughly zero only by epoch 90. The model spent
every one of those epochs still fitting. This is the same diagnosis
@sec-recipe-ablation arrives at from the other direction, and it is why
regularization had nothing to offer here.

Wall-clock was 2h05m against the earlier run's 2h31m for identical work, with
no slowdown over the run (@discussion), and peak memory 1.99 GiB -- under a
quarter of what was available.

== Self-supervised comparison <sec-ssl>

This comparison is implemented but *not run*; no numbers are claimed for it.
SimSiam @chen2021simsiam is chosen over BYOL @grill2020byol on memory and
tuning grounds rather than reported accuracy: no momentum encoder, so no
second 6.5M-parameter shadow trunk in VRAM alongside the two augmented views
an epoch already costs, and no EMA-decay schedule to tune -- and
@sec-recipe-ablation is a cautionary result about adding untuned
hyperparameters to this budget. SimCLR @chen2020simclr is excluded outright,
degrading below roughly 1,000-example batches that 8GB cannot reach here.
`FoodCNN.forward_features` exposes the pooled 512-d trunk output for a
projector, at no cost to the parameter budget, and `main.py --init-encoder`
loads a pretrained trunk into a fresh classifier to finetune.

*The equal-compute control is not optional.* Pretraining (200 epochs, about 8
GPU-hours at two views per step) plus finetuning (1.8) must be compared
against spending those same hours on longer supervised training of the same
trunk, or the result cannot separate "pretraining helped" from "this trunk
keeps improving with more epochs regardless of objective." @sec-full-run has
already shown it still gaining at epoch 90, which makes that alternative the
likely one rather than a pedantic caveat.

= Discussion <discussion>

#paragraph[Hardware measurement mattered as much as the architecture.]
Several findings above exist only because throughput and memory were measured
rather than assumed: the tensor-core alignment and spatial-resolution effects
of @sec-arch-search; the dataloader never being the bottleneck at this
resolution, where crop-and-flip sustains 4.4-4.7k img/s against 1.7-1.9k of
GPU demand and even TrivialAugment keeps 2.5x headroom at eight workers,
falling to 1.4x at sixteen as oversubscription costs throughput on 15GB of
system RAM; and batch size being throughput-flat from 160 to 768 while memory
scales linearly to 5.64 GiB, meaning the GPU is compute-saturated at the
smallest batch tried, not idle.

#paragraph[A power cap, not a thermal wall, and a short benchmark cannot tell
you which.] The GPU runs against a roughly 50W cap with its clock oscillating
2175-2340 MHz against a 3090 MHz maximum, and an early version of the
throughput benchmark swung 45% on one variant between sweeps before being
changed to time three 40-step rounds and keep the best (sweeps now agree
within 2-5%, and rankings are stable). The 90-epoch pipeline-validation run
then came in 28% over its own synthetic estimate, with per-batch time
degrading from 0.16s to 0.26s past epoch 60 -- which looks exactly like
heat-soak that no short benchmark window reproduces. It was not: the same
trunk, at the same length, held 0.147-0.169s/step throughout after the
machine was switched to its `performance` power profile (@sec-full-run). The
two runs are not a controlled comparison, but the reading changes the
conclusion one would draw from either alone. Throughput figures here remain
upper bounds and ratios between variants rather than wall-clock promises.

#paragraph[Limitations.] *No run was seeded* (@dataset-protocol), so every
figure is a single point estimate. This binds hardest on the two closest
comparisons -- the 1.2-point spread across three candidates in
@sec-arch-search, and RandAugment's 0.84-point matched-epoch deficit that
McNemar's test already declines to call significant -- and a second seed on
those legs is the cheapest available strengthening of this report. Separately,
every recipe axis was evaluated at the learning rate tuned for the plain
recipe, so GCE and EMA plausibly lost to that choice rather than on their
merits: unproven here, not ruled out. All measurements are specific to one
machine and need not transfer even in ratio, and training is from scratch
throughout, so these accuracies are not comparable to pretrained baselines.

= Conclusion <conclusion>

A plain residual CNN with a genuinely global pooling head reaches *63.83%
top-1 / 87.39% top-5* on FoodX-251 from scratch, on a test split read once,
using 66% of the parameters it was allowed. Both findings behind that number
cut against where effort naturally goes: capacity never constrained this
model, and neither did overfitting -- what it wanted was learning rate and
time, and the budget spent ruling out six regularizers is what established
that. Throughput and memory acted as design constraints alongside the
parameter cap throughout. Two things would most improve this report: reseeding
the two comparisons too close for one run to settle, and running @sec-ssl
against its equal-compute control.

#show: appendix

#v(-8pt)

= Technical appendices and supplementary material

Supplementary material referenced from @method, @sec-arch-search and
@sec-recipe-ablation follows: the learning-rate sweep in full, the batch-size
and crop-scale tables, the synthetic throughput sweep behind the architecture
search, and additional detail on the rejected depthwise-separable trunk.

#include "appendix.typ"

#include "checklist.typ"
