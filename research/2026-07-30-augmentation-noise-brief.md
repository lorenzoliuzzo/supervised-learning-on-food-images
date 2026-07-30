# Research brief: Training a CNN from scratch on noisy web-crawled food images: augmentation, label noise, long tail

## Summary
The supplied source set only covers one of the six questions in depth — data augmentation — via the RandAugment/TrivialAugment/mixup/CutMix lineage plus a 2024 survey consolidating mixup-family variants; none of the label-noise-robust-objective, long-tail-reweighting, optimizer/schedule, single-GPU-throughput, or test-time-resolution literature requested is present in this source list (the remainder are unrelated arXiv papers on Byzantine-robust distributed SGD, remote-sensing generative augmentation, risk theory, and other off-topic subfields). Within augmentation, the established result is that automated/random policies (RandAugment) and their tuning-free simplification (TrivialAugment) match or beat searched policies at near-zero search cost, while mixup and CutMix are complementary regularizers acting on the input/label space rather than the operator space, with the 2024 survey mapping how mixing-based methods have since diversified. No source here supports claims about the augmentation-strength crossover for under-parameterized models on short schedules, label-smoothing-vs-noise-robust-loss trade-offs, long-tail logit adjustment, AdamW-vs-SGD/EMA/cosine choices, mixed-precision/channels_last/progressive-resizing throughput effects, or FixRes/TTA — these must be sourced elsewhere before the brief's questions 2–6 can be answered with citations.

## Prerequisites
- Standard CNN training loop familiarity (data loading, loss, backprop) — not covered by these sources
- Basic understanding of empirical risk minimization and its failure modes under label noise/memorization, as motivated informally in the mixup source
- Familiarity with automated/search-based augmentation policies predating RandAugment (e.g. AutoAugment) is assumed but not supplied here

## Learning path
1. 1. Read mixup (arxiv:1710.09412v2) to understand convex label/input mixing as a regularizer against memorization — relevant given FoodX-251's label noise even though the paper does not target noisy labels directly.
2. 2. Read CutMix (arxiv:1905.04899v2) to contrast patch-level mixing/localization benefits against mixup's global blend; both are candidate augmentations for the from-scratch CNN.
3. 3. Read RandAugment (arxiv:1909.13719v2) for the automated-policy augmentation baseline and its claimed reduced search cost.
4. 4. Read TrivialAugment (arxiv:2103.10158v2) as the tuning-free comparator; note its claim of matching RandAugment for near-zero cost, and treat the under-parameterized/short-schedule crossover question as open since none of these ablate sub-10M-parameter models trained in tens of epochs.
5. 5. Read the 2024 mixup survey (arxiv:2409.05202v2) to survey how far the mixing-augmentation family has diversified since CutMix/mixup, and to scope what remains unresolved.
6. 6. Because this source set has no coverage of label-noise-robust losses (GCE/SCE/bootstrapping/co-teaching/ELR), long-tail reweighting/logit adjustment, optimizer/schedule/EMA choices, mixed-precision/channels_last/progressive-resizing throughput effects, or FixRes/TTA evaluation practice, treat questions 2–6 of the brief as requiring a separate literature search before any citation-grounded recommendation can be made.

## Reading list
1. **mixup: Beyond Empirical Risk Minimization** — <http://arxiv.org/abs/1710.09412v2>
   Foundational input-mixing regularizer (mixup); establishes the convex-combination training principle that CutMix and later mixing methods build on and that Mixup/CutMix ablations in food-noise pipelines typically cite as baseline.
2. **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** — <http://arxiv.org/abs/1905.04899v2>
   CutMix extends regional dropout/mixing to patch-level label mixing with localization benefits; the natural second augmentation primitive after mixup for the requested Mixup+CutMix comparison.
3. **RandAugment: Practical automated data augmentation with a reduced search space** — <http://arxiv.org/abs/1909.13719v2>
   RandAugment establishes the reduced-search-space automated augmentation policy that TrivialAugment later simplifies; needed before evaluating whether such policies help or hurt small from-scratch models.
4. **TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation** — <http://arxiv.org/abs/2103.10158v2>
   TrivialAugment is the most recent, tuning-free augmentation baseline in this set and directly informs the augmentation-crossover question (single random op vs. RandAugment's stacked policy) for constrained compute budgets.
5. **A Survey on Mixup Augmentations and Beyond** — <http://arxiv.org/abs/2409.05202v2>
   Most recent source overall; surveys the post-mixup/CutMix landscape and is the right vantage point to assess which mixing variants have actually replicated versus stalled, though it does not itself resolve the noisy-label or long-tail questions.

## Sources
- `arxiv:1909.13719v2` [RandAugment: Practical automated data augmentation with a reduced search space](http://arxiv.org/abs/1909.13719v2) (2019) · Ekin D. Cubuk, Barret Zoph, Jonathon Shlens
- `arxiv:2510.21391v1` [TerraGen: A Unified Multi-Task Layout Generation Framework for Remote Sensing Data Augmentation](http://arxiv.org/abs/2510.21391v1) (2025) · Datao Tang, Hao Wang, Yudeng Xin
- `arxiv:1907.02664v2` [Data Encoding for Byzantine-Resilient Distributed Optimization](http://arxiv.org/abs/1907.02664v2) (2019) · Deepesh Data, Linqi Song, Suhas Diggavi
- `arxiv:2005.07866v1` [Byzantine-Resilient SGD in High Dimensions on Heterogeneous Data](http://arxiv.org/abs/2005.07866v1) (2020) · Deepesh Data, Suhas Diggavi
- `arxiv:1110.5626v1` [Constraints on dark energy from H II starburst galaxy apparent magnitude versus redshift data](http://arxiv.org/abs/1110.5626v1) (2011) · Data Mania, Bharat Ratra
- `arxiv:2411.15497v3` [AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation](http://arxiv.org/abs/2411.15497v3) (2024) · Datao Tang, Xiangyong Cao, Xuan Wu
- `arxiv:2410.01088v2` [Exploring Empty Spaces: Human-in-the-Loop Data Augmentation](http://arxiv.org/abs/2410.01088v2) (2024) · Catherine Yeh, Donghao Ren, Yannick Assogba
- `arxiv:2108.06949v1` [Data Augmentation for Scene Text Recognition](http://arxiv.org/abs/2108.06949v1) (2021) · Rowel Atienza
- `arxiv:2103.10158v2` [TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation](http://arxiv.org/abs/2103.10158v2) (2021) · Samuel G. Müller, Frank Hutter
- `arxiv:1707.09430v1` [Human in the Loop: Interactive Passive Automata Learning via Evidence-Driven State-Merging Algorithms](http://arxiv.org/abs/1707.09430v1) (2017) · Christian A. Hammerschmidt, Radu State, Sicco Verwer
- `arxiv:1712.08887v3` [Efficient data augmentation techniques for some classes of state space models](http://arxiv.org/abs/1712.08887v3) (2017) · Linda S. L. Tan
- `arxiv:1710.05204v2` [Sequential Design and Spatial Modeling for Portfolio Tail Risk Measurement](http://arxiv.org/abs/1710.05204v2) (2017) · Michael Ludkovski, James Risk
- `arxiv:1710.09412v2` [mixup: Beyond Empirical Risk Minimization](http://arxiv.org/abs/1710.09412v2) (2017) · Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin
- `arxiv:2401.03328v6` [Optimal risk sharing, equilibria, and welfare with empirically realistic risk attitudes](http://arxiv.org/abs/2401.03328v6) (2024) · Jean-Gabriel Lauzier, Liyuan Lin, Peter Wakker
- `arxiv:2409.05202v2` [A Survey on Mixup Augmentations and Beyond](http://arxiv.org/abs/2409.05202v2) (2024) · Xin Jin, Hongyu Zhu, Siyuan Li
- `arxiv:2107.03979v1` [On the Selection of Loss Severity Distributions to Model Operational Risk](http://arxiv.org/abs/2107.03979v1) (2021) · Daniel Hadley, Harry Joe, Natalia Nolde
- `arxiv:1508.00310v2` [Statistical Emulators for Pricing and Hedging Longevity Risk Products](http://arxiv.org/abs/1508.00310v2) (2015) · James Risk, Michael Ludkovski
- `arxiv:1902.04489v3` [Evaluating Range Value at Risk Forecasts](http://arxiv.org/abs/1902.04489v3) (2019) · Tobias Fissler, Johanna F. Ziegel
- `arxiv:0909.4948v3` [Optimal Stopping for Dynamic Convex Risk Measures](http://arxiv.org/abs/0909.4948v3) (2009) · Erhan Bayraktar, Ioannis Karatzas, Song Yao
- `arxiv:2412.06830v1` [A New Strategy for the Exploration of Venus](http://arxiv.org/abs/2412.06830v1) (2024) · The VEXAG Exploration Strategy Study Analysis Workgroup
- `arxiv:1905.04899v2` [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](http://arxiv.org/abs/1905.04899v2) (2019) · Sangdoo Yun, Dongyoon Han, Seong Joon Oh
- `arxiv:2311.17717v3` [Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers](http://arxiv.org/abs/2311.17717v3) (2023) · Chi-Pin Huang, Kai-Po Chang, Chung-Ting Tsai
- `arxiv:1809.05247v2` [Revisiting Random Binning Features: Fast Convergence and Strong Parallelizability](http://arxiv.org/abs/1809.05247v2) (2018) · Lingfei Wu, Ian E. H. Yen, Jie Chen
- `arxiv:2301.10131v2` [Random perfect matchings in regular graphs](http://arxiv.org/abs/2301.10131v2) (2023) · Bertille Granet, Felix Joos
- `arxiv:2405.01606v2` [Enhancing the Trainability of Variational Quantum Circuits with Regularization Strategies](http://arxiv.org/abs/2405.01606v2) (2024) · Jun Zhuang, Jack Cunningham, Chaowen Guan
- `arxiv:1910.11775v2` [Physics Briefing Book](http://arxiv.org/abs/1910.11775v2) (2019) · European Strategy for Particle Physics Preparatory Group
