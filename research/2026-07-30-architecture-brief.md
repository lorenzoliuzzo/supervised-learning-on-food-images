# Research brief: Parameter-efficient CNN architectures under 10M parameters for fine-grained food classification

## Summary
Of the supplied sources, only a narrow slice bears on parameter-efficient CNN design (MobileNetV2, ShuffleNetV2, EfficientNet, EfficientNet-eLite, TinyNets/Model Rubik's Cube, plus two tangential 2023-2026 scaling-theory and vector-valued papers); none address food classification, FoodX-251/Food-101/Food-2K benchmarks, SE blocks, grouped convolutions, anti-aliased downsampling, or stochastic depth specifically, so questions 3, 4, and 5 cannot be answered from this source set and must be flagged as gaps. The available sources do let you answer questions 1 and 2: MobileNetV2 and ShuffleNetV2 establish that FLOPs/param-count and measured latency diverge for depthwise/grouped designs (ShuffleNetV2's explicit point), while EfficientNet and its sub-10M-oriented descendants (TinyNets, EfficientNet-eLite) are the closest matches to a compound width/depth/resolution budgeting argument for a fixed parameter ceiling. Treat the NAS-hardware, quantum-computing, GNN-accelerator, and pedagogy sources in the candidate list as irrelevant noise from a mis-targeted retrieval, not evidence for this brief.

## Prerequisites
- Standard CNN building blocks: convolution, batch norm, pooling, backpropagation through conv layers
- FLOPs vs. parameter-count vs. wall-clock-latency as distinct cost axes for a given architecture
- Depthwise-separable convolution and grouped convolution mechanics
- Basic familiarity with ImageNet-scale classification training pipelines (augmentation, LR schedules) since none of the sources here are food-specific

## Learning path
1. Read the Network-in-Network paper (1603.06759v1) to fix the 1x1-conv / global-average-pooling head vocabulary used everywhere downstream.
2. Read MobileNetV2 (1801.04381v4) to internalize the inverted-residual bottleneck block and its per-stage parameter accounting.
3. Read ShuffleNetV2 (1807.11164v1) to learn why parameter count and FLOPs diverge from measured latency/activation memory on real hardware, and note its explicit design guidelines.
4. Read EfficientNet (1905.11946v5) for the compound width/depth/resolution scaling framework under a fixed budget.
5. Read Model Rubik's Cube / TinyNets (2010.14819v2) to see that framework re-derived specifically for the small-model regime you are targeting.
6. Read EfficientNet-eLite (2009.07409v1) for an edge-hardware-constrained instantiation closer to your 8GB/laptop-GPU ceiling.
7. Skim Commutative Width and Depth Scaling (2310.01683v1) as a theoretical sanity check on whether width/depth choices are as separable as the compound-scaling heuristic assumes.
8. Skim V-EfficientNets (2505.05659v1) only to note it as a speculative, unvalidated-at-this-scale extension, not a design to adopt yet.
9. Read 'Are we done with ImageNet?' (2006.07159v1) as a methodological caution before trusting any accuracy comparison table you assemble, including for FoodX-251 — note this source itself says nothing about food data, so all Food-101/FoodX-251/Food-2K-specific claims (resolution dependence, attention/part-based methods, published baselines) must be sourced elsewhere; this source list has no evidence for questions 3–5 and that gap should be stated explicitly in the report.

## Reading list
1. **Convolution in Convolution for Network in Network** — <http://arxiv.org/abs/1603.06759v1>
   Establishes 1x1 convolutions and the MLP-in-conv idea that underlies both bottleneck blocks and global-average-pooling classifier heads used by every later efficient architecture; useful baseline before inverted residuals.
2. **MobileNetV2: Inverted Residuals and Linear Bottlenecks** — <http://arxiv.org/abs/1801.04381v4>
   Foundational depthwise-separable + inverted-residual-bottleneck design; the block you will almost certainly reimplement from scratch in model.py, and the reference point for parameter accounting per stage.
3. **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design** — <http://arxiv.org/abs/1807.11164v1>
   Directly answers the latency-vs-parameter-count decoupling question: derives practical guidelines (memory access cost, channel splits/shuffle, grouped-conv pitfalls) showing why depthwise-heavy nets can be parameter-cheap but slow/memory-hungry on real hardware, not just FLOP-cheap.
4. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** — <http://arxiv.org/abs/1905.11946v5>
   The canonical compound-scaling treatment of width/depth/resolution tradeoffs under a fixed budget; the theoretical backbone for deciding where to spend a 10M-parameter budget across stages vs. input resolution.
5. **Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets** — <http://arxiv.org/abs/2010.14819v2>
   Re-derives EfficientNet's scaling coefficients specifically for very small ('Tiny') models, i.e. the regime this project sits in; most directly answers 'where should the sub-10M budget go' rather than extrapolating from EfficientNet's larger-model fits.
6. **EfficientNet-eLite: Extremely Lightweight and Efficient CNN Models for Edge Devices by Network Candidate Search** — <http://arxiv.org/abs/2009.07409v1>
   Edge-device-targeted lightweight CNN search explicitly trading accuracy for resource usage; closest available source to your hardware ceiling (8GB VRAM, days not weeks) and to activation-memory-aware design choices.
7. **Commutative Width and Depth Scaling in Deep Neural Networks** — <http://arxiv.org/abs/2310.01683v1>
   Recent (2023) theoretical result on whether width- and depth-scaling limits commute; relevant as an open-controversy check on whether the width/depth split recommended by EfficientNet-style compound scaling is principled or just empirically convenient at small scale.
8. **V-EfficientNets: Vector-Valued Efficiently Scaled Convolutional Neural Network Models** — <http://arxiv.org/abs/2505.05659v1>
   2025 speculative extension of EfficientNet-style scaling to vector-valued (multi-channel-coherent) convolutions; read last as a frontier/uncertain direction, not an established design choice for a from-scratch sub-10M model.
9. **Are we done with ImageNet?** — <http://arxiv.org/abs/2006.07159v1>
   Not architecture-specific, but relevant caution when positioning reported accuracy numbers: shows published ImageNet-era gains can partly reflect label-noise artifacts rather than true model improvement — apply the same skepticism when comparing against any fine-grained baseline numbers you find elsewhere.

## Sources
- `arxiv:1801.04381v4` [MobileNetV2: Inverted Residuals and Linear Bottlenecks](http://arxiv.org/abs/1801.04381v4) (2018) · Mark Sandler, Andrew Howard, Menglong Zhu
- `arxiv:2207.13219v4` [Dalorex: A Data-Local Program Execution and Architecture for Memory-bound Applications](http://arxiv.org/abs/2207.13219v4) (2022) · Marcelo Orenes-Vera, Esin Tureci, David Wentzlaff
- `arxiv:2412.20486v2` [LSQCA: Resource-Efficient Load/Store Architecture for Limited-Scale Fault-Tolerant Quantum Computing](http://arxiv.org/abs/2412.20486v2) (2024) · Takumi Kobori, Yasunari Suzuki, Yosuke Ueno
- `arxiv:2009.00804v2` [Architectural Implications of Graph Neural Networks](http://arxiv.org/abs/2009.00804v2) (2020) · Zhihui Zhang, Jingwen Leng, Lingxiao Ma
- `arxiv:1306.0089v1` [A Novel Reconfigurable Architecture of a DSP Processor for Efficient Mapping of DSP Functions using Field Programmable DSP Arrays](http://arxiv.org/abs/1306.0089v1) (2013) · Amitabha Sinha, Mitrava Sarkar, Soumojit Acharyya
- `arxiv:1209.4451v1` [Inverted Classroom an der Hochschule Karlsruhe - ein nicht quantisierter Flip](http://arxiv.org/abs/1209.4451v1) (2012) · Isabel Braun, Gottfried Metzger, Stefan Ritter
- `arxiv:2505.09343v2` [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](http://arxiv.org/abs/2505.09343v2) (2025) · Chenggang Zhao, Chengqi Deng, Chong Ruan
- `arxiv:2006.07159v1` [Are we done with ImageNet?](http://arxiv.org/abs/2006.07159v1) (2020) · Lucas Beyer, Olivier J. Hénaff, Alexander Kolesnikov
- `arxiv:1901.01074v3` [Multi-Objective Reinforced Evolution in Mobile Neural Architecture Search](http://arxiv.org/abs/1901.01074v3) (2019) · Xiangxiang Chu, Bo Zhang, Ruijun Xu
- `arxiv:2007.16149v1` [HMCNAS: Neural Architecture Search using Hidden Markov Chains and Bayesian Optimization](http://arxiv.org/abs/2007.16149v1) (2020) · Vasco Lopes, Luís A. Alexandre
- `arxiv:1906.02869v2` [One-Shot Neural Architecture Search via Compressive Sensing](http://arxiv.org/abs/1906.02869v2) (2019) · Minsu Cho, Mohammadreza Soltani, Chinmay Hegde
- `arxiv:2104.10450v1` [Making Differentiable Architecture Search less local](http://arxiv.org/abs/2104.10450v1) (2021) · Erik Bodin, Federico Tomasi, Zhenwen Dai
- `arxiv:2509.26037v2` [CoLLM-NAS: Collaborative Large Language Models for Efficient Knowledge-Guided Neural Architecture Search](http://arxiv.org/abs/2509.26037v2) (2025) · Zhe Li, Zhiwei Lin, Yongtao Wang
- `arxiv:2605.08238v1` [Resource-Aware Evolutionary Neural Architecture Search for Cardiac MRI Segmentation](http://arxiv.org/abs/2605.08238v1) (2026) · Farhana Yasmin, Mahade Hasan, Haipeng Liu
- `arxiv:2010.08219v2` [How Does Supernet Help in Neural Architecture Search?](http://arxiv.org/abs/2010.08219v2) (2020) · Yuge Zhang, Quanlu Zhang, Yaming Yang
- `arxiv:1907.06511v4` [Reinforcement Learning with Chromatic Networks for Compact Architecture Search](http://arxiv.org/abs/1907.06511v4) (2019) · Xingyou Song, Krzysztof Choromanski, Jack Parker-Holder
- `arxiv:1905.11946v5` [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](http://arxiv.org/abs/1905.11946v5) (2019) · Mingxing Tan, Quoc V. Le
- `arxiv:2505.05659v1` [V-EfficientNets: Vector-Valued Efficiently Scaled Convolutional Neural Network Models](http://arxiv.org/abs/2505.05659v1) (2025) · Guilherme Vieira Neto, Marcos Eduardo Valle
- `arxiv:2010.14819v2` [Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets](http://arxiv.org/abs/2010.14819v2) (2020) · Kai Han, Yunhe Wang, Qiulin Zhang
- `arxiv:2310.01683v1` [Commutative Width and Depth Scaling in Deep Neural Networks](http://arxiv.org/abs/2310.01683v1) (2023) · Soufiane Hayou
- `arxiv:1607.01977v1` [Deep Depth Super-Resolution : Learning Depth Super-Resolution using Deep Convolutional Neural Network](http://arxiv.org/abs/1607.01977v1) (2016) · Xibin Song, Yuchao Dai, Xueying Qin
- `arxiv:1603.06759v1` [Convolution in Convolution for Network in Network](http://arxiv.org/abs/1603.06759v1) (2016) · Yanwei Pang, Manli Sun, Xiaoheng Jiang
- `arxiv:2603.01172v1` [Midterm Status Report of the ILC Technology Network Activities](http://arxiv.org/abs/2603.01172v1) (2026) · ILC Technology Network
- `arxiv:2009.07409v1` [EfficientNet-eLite: Extremely Lightweight and Efficient CNN Models for Edge Devices by Network Candidate Search](http://arxiv.org/abs/2009.07409v1) (2020) · Ching-Chen Wang, Ching-Te Chiu, Jheng-Yi Chang
- `arxiv:1807.11164v1` [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](http://arxiv.org/abs/1807.11164v1) (2018) · Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng
- `arxiv:2402.19171v1` [Towards Assessing Spread in Sets of Software Architecture Designs](http://arxiv.org/abs/2402.19171v1) (2024) · Vittorio Cortellessa, J. Andres Diaz-Pace, Daniele Di Pompeo
