#import "@preview/touying:0.6.1": *

// Bamboo theme. Originally carried over from the author's personal
// presentation template; the palette, chrome, and the title/section/focus
// slides were reworked for this deck and now diverge from that template.

#let palette = (
  primary: rgb("#35603F"),
  mid: rgb("#5E8B65"),
  pale: rgb("#EDF2EB"),
  accent: rgb("#C2703A"),
  ink: rgb("#1B1F1B"),
  muted: rgb("#71806F"),
  rule: rgb("#CFD9CD"),
  paper: rgb("#FFFFFF"),
)

// Deck chrome is sized off these, so the page margins stay in sync with the
// header band when either one changes.
#let header-height = 4.5em
#let bar-height = 3.5pt

#let progress-strip = utils.touying-progress(ratio => grid(
  columns: (ratio * 100%, 1fr),
  rows: bar-height,
  components.cell(fill: palette.accent),
  components.cell(fill: palette.pale),
))

#let section-label = context {
  let heads = query(selector(heading.where(level: 1)).before(here()))
  if heads.len() > 0 {
    text(
      size: 0.62em,
      fill: palette.pale,
      weight: "medium",
      tracking: 0.12em,
      upper(heads.last().body),
    )
  }
}

#let title-slide(..args) = touying-slide-wrapper(self => {
  let info = self.info + args.named()

  let body = {
    set align(left + horizon)
    block(width: 100%, inset: (x: 0.6em), {
      block(width: 3.4em, height: bar-height, fill: palette.accent)
      v(0.9em)

      if info.title != none {
        block(
          width: 92%,
          text(size: 1.8em, fill: palette.primary, weight: "bold", info.title),
        )
      }
      if info.subtitle != none {
        v(0.1em)
        block(
          width: 80%,
          text(size: 1.0em, fill: palette.muted, info.subtitle),
        )
      }

      v(1.1em)
      line(length: 100%, stroke: 0.6pt + palette.rule)
      v(0.7em)

      set text(size: 0.78em, fill: palette.ink)
      grid(
        columns: (1fr, auto),
        align: (left + horizon, right + horizon),
        if info.author != none { info.author },
        if info.date != none {
          text(fill: palette.muted, utils.display-info-date(self))
        },
      )
    })
  }

  touying-slide(self: self, body)
})

#let bamboo-slide(title: auto, ..args) = touying-slide-wrapper(self => {
  let header-content = {
    set align(top)
    block(
      width: 100%,
      fill: palette.primary,
      inset: (x: 1.4em, top: 0.75em, bottom: 0.7em),
      {
        set align(left)
        section-label
        linebreak()
        set text(fill: palette.paper, size: 1.28em, weight: "medium")
        if title != auto {
          utils.call-or-display(self, title)
        } else {
          utils.display-current-heading(level: 2)
        }
      },
    )
    progress-strip
  }

  let footer-content = {
    set align(bottom)
    block(width: 100%, inset: (x: 1.4em, bottom: 0.9em), {
      set text(size: 0.62em, fill: palette.muted)
      line(length: 100%, stroke: 0.6pt + palette.rule)
      v(0.5em)
      utils.call-or-display(self, self.info.author)
      h(1fr)
      context text(
        fill: palette.primary,
        weight: "medium",
        utils.slide-counter.display(),
      ) + text(fill: palette.rule, " / ") + utils.last-slide-number
    })
  }

  let conf = config-page(header: header-content, footer: footer-content)
  touying-slide(self: utils.merge-dicts(self, conf), ..args)
})

#let new-section-slide(self: none, body) = touying-slide-wrapper(self => {
  let main-body = context {
    // The heading counter has not advanced yet on its own section slide, so
    // count the level-1 headings up to this point instead.
    let n = query(selector(heading.where(level: 1)).before(here())).len()
    set align(left + horizon)
    block(inset: (x: 0.6em), {
      text(
        size: 3.4em,
        fill: palette.pale,
        weight: "bold",
        if n < 10 { "0" + str(n) } else { str(n) },
      )
      v(-0.5em)
      block(width: 2.6em, height: bar-height, fill: palette.accent)
      v(0.6em)
      text(
        size: 1.9em,
        fill: palette.primary,
        weight: "bold",
        utils.display-current-heading(level: 1),
      )
    })
  }
  touying-slide(
    self: utils.merge-dicts(self, config-page(margin: (top: 2em, bottom: 2em, x: 2em))),
    main-body,
  )
})

#let focus-slide(body) = touying-slide-wrapper(self => {
  let config = config-page(fill: palette.primary, margin: 2.3em)
  touying-slide(
    self: utils.merge-dicts(self, config),
    align(horizon + center, {
      block(width: 2.6em, height: bar-height, fill: palette.accent)
      v(0.8em)
      text(fill: palette.paper, size: 2.2em, weight: "bold", body)
    }),
  )
})

// Section-opening line for a column inside a two-column slide.
#let col-head(body) = block(
  below: 0.7em,
  text(fill: palette.primary, weight: "bold", size: 0.95em, body),
)

// Two columns separated by a hairline, so the split reads as deliberate.
// Set a notch smaller than single-column body text: two columns carry roughly
// twice the words in the same height, and the densest slide overflows at 1em.
#let two-col(left-body, right-body, ratio: (1fr, 1fr)) = {
  set text(size: 0.90em)
  // The rule is a grid stroke, not a rotated `line`: `line(length: 100%)`
  // resolves against the body *width*, which made the row taller than the
  // slide and pushed everything after it onto a second page.
  grid(
    columns: ratio,
    inset: (x: 0.9em),
    stroke: (x, y) => if x == 1 { (left: 0.6pt + palette.rule) },
    left-body,
    right-body,
  )
}

// Numeric results table: tinted header, rules only where they carry meaning.
#let data-table(columns: (), align-spec: auto, header-row: (), ..rows) = block(
  width: 100%,
  below: 0.9em,
  {
    set text(size: 0.9em)
    table(
      columns: columns,
      align: align-spec,
      stroke: none,
      inset: (x: 0.7em, y: 0.36em),
      fill: (_, y) => if y == 0 { palette.primary } else if calc.odd(y) { palette.pale },
      table.header(
        ..header-row.map(c => text(fill: palette.paper, weight: "bold", c)),
      ),
      ..rows,
    )
  },
)

#let bamboo-theme(aspect-ratio: "16-9", ..args, body) = {
  show: touying-slides.with(
    config-page(
      paper: "presentation-" + aspect-ratio,
      margin: (top: header-height + 1.1em, bottom: 2.6em, x: 2em),
    ),
    config-colors(
      primary: palette.primary,
      secondary: palette.mid,
      tertiary: palette.accent,
      neutral-lightest: palette.paper,
      neutral-darkest: palette.ink,
    ),
    config-methods(
      alert: (self: none, body) => text(fill: palette.accent, weight: "bold", body),
    ),
    config-common(
      slide-fn: bamboo-slide,
      new-section-slide-fn: new-section-slide,
    ),
    config-info(..args),
  )

  set text(size: 20pt, fill: palette.ink)
  set par(leading: 0.7em, justify: false)
  set list(indent: 0.5em, spacing: 0.75em, marker: (
    text(fill: palette.mid)[•],
    text(fill: palette.rule)[‣],
  ))
  set table(stroke: none)

  // `*bold*` carries the argument through these slides, so it gets the house
  // green; `#alert[]` stays amber for the handful of genuine interruptions.
  show strong: it => text(fill: palette.primary, weight: "bold", it.body)
  show raw: it => text(fill: palette.ink.lighten(15%), it)

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
]

#slide([What's actually long-tailed here? Not the classes.])[
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
  #two-col[
    #col-head[The trunk]
    - Stem -> 1/4 resolution, then 4 residual stages doubling width:
      *64-128-256-512*, blocks $[2,2,2,1]$
    - BN after *every* conv; zero-init each block's 2nd BN $gamma$, so
      blocks start as identity -- safe for a high LR
    - ReLU *after* the residual add
  ][
    #col-head[The one fix that mattered most]
    - $(7,7)$ map -> flatten -> `Linear` costs *6.4M params*: 94% of the
      model on its *least* expressive layer
    - Global pool to $(1,1)$ first -> `Linear(512, 251)` = *129k*
    - *6.58M params*, 66% of budget, 98% of it in the trunk
  ]
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
  #data-table(
    columns: (1fr, auto, auto, auto, auto),
    align-spec: (left, right, right, right, right),
    header-row: ([Variant], [Params], [top-1], [top-5], [Peak VRAM]),
    [`[2,2,2,2]` 64-448], [9.46M], [49.99%], [78.29%], [2.04 GiB],
    [`[2,2,4,1]` 64-512], [8.94M], [49.73%], [77.93%], [2.14 GiB],
    [baseline `[2,2,2,1]` 64-512], [6.58M], [48.82%], [77.12%], [1.99 GiB],
    [5-stage 48-512], [9.08M], [47.42%], [75.94%], [1.61 GiB],
  )
  - Top three within *1.2 points* -- not enough to call on one seed
  - Adopt *baseline*: statistically indistinguishable, *30% fewer params*,
    that headroom hadn't earned its keep yet
  - A follow-up sweep *below* baseline found a dead heat down to 5.18M
    (`p=1.0000`) and a cliff before 1.68M -- capacity was never the
    binding constraint near baseline
]

= Training recipe

#slide("Six axes swept; only the learning rate paid")[
  #two-col(ratio: (1fr, 1.5fr))[
    #col-head[Adopted recipe]
    - plain label-smoothed CE (0.1), SGD+Nesterov
    - *lr 0.8*, batch 256, cosine + 5-epoch warmup
    - bf16 autocast, `channels_last`, 176px crop
  ][
    #col-head[Swept on 15-epoch `val-dev` proxies]
    - *Learning rate* 0.05 -> 0.8: *+11.6 pts*, the one axis that paid
    - *Augmentation:* TrivialAugment, RandAugment lose at 15 ep
      ($p < 0.0001$); RandAugment *rematched at 30 ep* closes to
      0.84 pts -- *not* significant ($p = 0.075$)
    - *Mixup, CutMix, EMA, GCE:* lose decisively ($p < 0.0001$ each)
    - *Batch size* 160-512: flat ($p gt.eq 0.49$)
    - *Crop-scale floor:* 0.25 flat; 0.40 costs 1.17 pts ($p = 0.023$)
  ]
]

= Results

#slide("The full run")[
  90 epochs, baseline trunk, tuned recipe, FixRes-style eval (train 176px,
  test 224px center crop):

  #data-table(
    columns: (1fr, auto, auto, auto),
    align-spec: (left, right, right, right),
    header-row: ([Split], [top-1], [top-3], [top-5]),
    [val-dev, best], [64.36%], [82.22%], [87.51%],
    [*val-test (headline, touched once)*], [*63.83%*], [*81.79%*], [*87.39%*],
  )
  - Train/val gap closes from *+11 points* at epoch 15 to *near zero* at
    epoch 90 -- the model was underfitting, not overfitting; it needed the
    full schedule, not more regularization
  - 2h05m wall clock, 1.99 GiB peak VRAM -- comfortably under the 8GB ceiling
]

#slide([Self-supervised track -- in progress])[
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
    depthwise-separable trunk -- both *lost once actually run*: 1.4-2.6
    points and slower-and-heavier respectively
  - Treating wall-clock and memory as design constraints, not just the
    parameter cap, is what got here
  - *Open:* the self-supervised comparison against an equal-compute
    supervised control
]

#focus-slide[
  Questions?
]
