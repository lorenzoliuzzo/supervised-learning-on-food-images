#import "@preview/touying:0.6.1": *

// Bamboo theme -- carried over unchanged from the author's personal
// template (typst-templates/templates/presentation.typ), reused here for
// consistency across projects.

#let title-slide(..args) = touying-slide-wrapper(
  self => {
    let info = self.info + args.named()

    let body = {
      set align(center + horizon)
      if info.title != none {
        block(
          fill: self.colors.primary,
          width: 80%,
          inset: (y: 1em),
          radius: 1em,
          text(
            size: 1.5em,
            fill: self.colors.neutral-lightest,
            weight: "bold",
            info.title
          ),
        )
      }

      set text(fill: self.colors.neutral-darkest)
      if info.subtitle != none {
        block(text(size: 1.1em, info.subtitle))
      }
      if info.author != none {
        block(info.author)
      }
      if info.date != none {
        block(utils.display-info-date(self))
      }
    }

    touying-slide(self: self, body)
  }
)

#let slide(title: auto, ..args) = touying-slide-wrapper(
  self => {

    let header-content = {
      set align(top)
      show: components.cell.with(fill: self.colors.primary, inset: 1em)

      set align(horizon)
      set text(fill: self.colors.neutral-lightest, size: 1.1em)
      utils.call-or-display(self, self.info.title)
      linebreak()

      set text(size: 1.5em)
      if title != auto {
        utils.call-or-display(self, title)
      } else {
        utils.display-current-heading(level: 2)
      }
    }

    let footer-content = {
      set align(bottom)
      show: components.cell.with(fill: self.colors.primary, inset: 1em)
      set align(horizon)
      set text(
        fill: self.colors.neutral-lightest,
        size: .8em
      )
      utils.call-or-display(self, self.info.author)
      h(1fr)
      context utils.slide-counter.display() + " / " + utils.last-slide-number
    }

    let conf = config-page(header: header-content, footer: footer-content)
    touying-slide(self: utils.merge-dicts(self, conf), ..args)
  }
)

#let new-section-slide(self: none, body) = touying-slide-wrapper(
  self => {
    let main-body = {
      set align(center + horizon)
      set text(
        size: 2.3em,
        fill: self.colors.primary,
        weight: "bold",
        style: "italic"
      )
      utils.display-current-heading(level: 1)
    }
    touying-slide(self: self, main-body)
  }
)

#let focus-slide(body) = touying-slide-wrapper(
  self => {
    set text(fill: self.colors.neutral-lightest, size: 2em)
    let config = config-page(fill: self.colors.primary, margin: 2.3em)
    touying-slide(
      self: utils.merge-dicts(self, config),
      align(horizon + center, body)
    )
  }
)

#let bamboo-theme(aspect-ratio: "16-9", ..args, body) = {
  show: touying-slides.with(
    config-page(
      paper: "presentation-" + aspect-ratio,
      margin: (top: 5.2em, bottom: 3em, x: 2em),
    ),
    config-colors(
      primary: rgb("#5E8B65"),
      neutral-lightest: rgb("#ffffff"),
      neutral-darkest: rgb("#000000"),
    ),
    config-methods(alert: utils.alert-with-primary-color),
    config-common(
      slide-fn: slide,
      new-section-slide-fn: new-section-slide
    ),
    config-info(..args)
  )

  set text(size: 20pt)
  body
}

#show: bamboo-theme.with(
  title: "Supervised and Self-Supervised Learning on Food-251",
  subtitle: "FoodCNN under a 10M-parameter budget on a single 8GB GPU",
  author: "Lorenzo Liuzzo",
  date: datetime.today(),
)

// Convenience wrapper matching the personal template's slide-data hand-off.
#let slide(title, body, notes: none) = [
  == #title
  #body
  #if notes != none and notes != "" [#speaker-note[#notes]]
]

// === body ===

#title-slide()

= The problem

#slide("FoodX-251, under two hard constraints")[
  - 251 fine-grained food categories, web-crawled *noisy* training labels,
    small clean validation set (Kaur et al., 2019)
  - Fine-grained: inter-class differences (plating, garnish) can be
    *subtler* than intra-class variation (lighting, angle, portioning)
  - Two constraints, not chosen for effect:
    - *From scratch* -- no ImageNet or other pretraining
    - *Under 10,000,000 trainable parameters*, on a single *8GB* laptop GPU
  - These two constraints shape every decision that follows
]

#slide("What's actually long-tailed here? Not the classes.")[
  - Median 471 images/class, p10/p90 366/580 -- close to *balanced*
  - The famous 19.3x imbalance ratio is *one* 34-image outlier class
  - #alert[Class-balanced losses, logit adjustment, long-tail training: out of
    scope] -- the real distributional problem is *label noise*
    (web-crawled train, clean val), not long tails
]

= Protocol

#slide("Splitting the validation set, once, deterministically")[
  - 11,994 val images -> 50/50 stratified split, fixed seed 251,
    every class within 1 image of exact half
  - `val-dev` (6,063): every ablation, every architecture comparison
  - `val-test` (5,931): touched *exactly once*, for the headline number
  - Without this split, the headline number is selected-on
  - 15-epoch fixed-seed proxy on `val-dev` (\~18 min) stands in for a full
    90-epoch run (\~2h) during search; winner gets one full confirmation run
]

= Architecture

#slide("FoodCNN: spend the budget where it moves accuracy")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      - Stem (3x3 s2 conv + BN + ReLU + 3x3 s2 maxpool) -> 1/4 resolution
      - 4 residual stages, doubling width each stage: *64-128-256-512*,
        block counts $[2,2,2,1]$
      - BN after *every* conv incl. shortcut projections; zero-init the
        2nd BN's $gamma$ -- each block starts as identity, safe for a high LR
      - ReLU *after* the residual add
    ],
    [
      *The one fix that mattered most:*
      - Pool to $(1,1)$, not $(7,7)$
      - $(7,7)$ map -> flatten -> `Linear` costs *6.4M params* (94% of the
        whole model) on the *least* expressive layer
      - Global pool first -> `Linear(512, 251)` = *129k params*
      - Result: *6.58M params*, 65.8% of budget, 98% inside the trunk
    ],
  )
]

#slide("Search by throughput and memory first, accuracy second")[
  Screened trunk variants *synthetically* (no GPU-hours on accuracy yet):
  - Stage widths must be *multiples of 32* -- an off-alignment trunk has
    *fewer* params yet runs *46% slower* (falls off the tensor-core fast path)
  - Final spatial resolution dominates cost more than any single knob --
    keeping a 6x6 map instead of 3x3 at *equal params* costs *2.9x*
    throughput and *2x* memory
  - Lesson: hardware-aware screening before spending a single accuracy run
]

#slide("Four candidates, one seed each")[
  #table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: none,
    table.hline(),
    table.header([Variant], [Params], [top-1], [top-5], [Peak VRAM]),
    table.hline(),
    [`[2,2,2,2]` 64-448], [9.46M], [*49.99%*], [78.29%], [2.04 GiB],
    [`[2,2,4,1]` 64-512], [8.94M], [49.73%], [77.93%], [2.14 GiB],
    [baseline `[2,2,2,1]` 64-512], [6.58M], [48.82%], [77.12%], [1.99 GiB],
    [5-stage 48-512], [9.08M], [47.42%], [75.94%], [1.61 GiB],
    table.hline(),
  )
  - Top three within *1.2 points* -- not enough to call on one seed
  - Adopt *baseline*: statistically indistinguishable, *30% fewer params*,
    that headroom hadn't earned its keep yet
  - A follow-up sweep *below* baseline found a dead heat down to 5.18M
    (`p=1.0000`) and a cliff before 1.68M -- capacity was never the
    binding constraint near baseline
]

= Training recipe

#slide("Recipe: six axes swept, plain wins")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      *Recipe, free at this budget:*
      - bf16 autocast, no `GradScaler`
      - `channels_last`, cosine LR + 5-epoch warmup
      - label smoothing 0.1, SGD+Nesterov
      - 176px crop, 8 workers, no `DataParallel`
      - LR search: 0.05 -> *0.8* (diminishing returns each doubling)
    ],
    [
      *Ablated, one seed each, plain recipe wins:*
      - TrivialAugment / RandAugment: *lose* at matched epochs
        (RandAugment's 0.84-pt gap: *not* significant, p=0.075)
      - Mixup, CutMix, EMA, GCE loss: *lose decisively* ($p < 0.0001$ each)
      - Batch size 160-512: no lever, throughput *and* accuracy flat
      - `RandomResizedCrop` scale floor: closes train/val gap but
        *loses* accuracy -- it's doing double duty as regularizer
    ]
  )
  #v(-0.6em)
  #align(center)[*Carries forward: plain label-smoothed CE, lr=0.8, batch 256.*]
]

= Results

#slide("The full run")[
  90 epochs, baseline trunk, tuned recipe, FixRes-style eval (train 176px,
  test 224px center crop):

  #align(center)[
    #table(
      columns: 4,
      align: (left, right, right, right),
      stroke: none,
      table.hline(),
      table.header([Split], [top-1], [top-3], [top-5]),
      table.hline(),
      [val-dev, best], [64.36%], [82.22%], [87.51%],
      [*val-test (headline, touched once)*], [*63.83%*], [*81.79%*], [*87.39%*],
      table.hline(),
    )
  ]
  - Train/val gap closes from *+11 points* at epoch 15 to *near zero* at
    epoch 90 -- the model was underfitting, not overfitting; it needed the
    full schedule, not more regularization
  - 2h05m wall clock, 1.99 GiB peak VRAM -- comfortably under the 8GB ceiling
]

#slide("Self-supervised track -- in progress")[
  - *Plan (decided):* SimSiam pretraining, 200 epochs @ 176px (\~8 GPU-h),
    finetune (\~1.8 GPU-h), against an *equal-GPU-hour supervised control*
    (\~8 GPU-h) -- without the control, "SSL helped" is indistinguishable
    from "more epochs helped, regardless of objective"
  - SimSiam over BYOL: no momentum encoder, no second \~6.5M-param shadow
    network to fit in 8GB
  - SimCLR excluded: degrades below batch sizes this budget can reach
  - Scaffolding lands: `src/simsiam.py`, `--init-encoder`, tested on
    synthetic tensors -- #alert[the run itself has not happened yet]
]

= Discussion

#slide("Hardware measurement mattered as much as architecture")[
  - Channel widths, spatial resolution, and *thermal throttling* moved
    results as much as any architectural choice
  - GPU runs against a \~50W power cap; a short benchmark swung 45% between
    sweeps before fixing the timing methodology
  - The full 90-epoch run itself ran *28% slower* than its own synthetic
    estimate -- sustained load heat-soaks a laptop GPU in a way no short
    benchmark window reproduces
  - Every throughput number here is an *upper bound*, and a *ratio*
    between variants -- not a wall-clock promise
]

#slide("Conclusion")[
  - A plain residual CNN with a genuinely global pooling head reaches
    *63.83% top-1 / 87.39% top-5* on FoodX-251, from scratch, under 10M
    params, on a single 8GB GPU
  - Two attractive-on-paper ideas -- a fifth residual stage, a
    depthwise-separable trunk -- both *measurably lost* once actually run
  - Treating wall-clock and memory as design constraints, not just the
    parameter cap, is what got here
  - *Open:* the self-supervised comparison against an equal-compute
    supervised control
]

#focus-slide[
  Questions?
]
