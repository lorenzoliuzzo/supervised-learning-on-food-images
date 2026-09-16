#let url(uri) = link(uri, raw(uri))

#let answerNA = text(fill: gray, "[NA]")
#let answerNo = text(fill: rgb("ff8c00"), "[No]")
#let answerYes = text(fill: blue, "[Yes]")

#pagebreak(weak: true)

#set heading(numbering: none)

= NeurIPS Paper Checklist



#let claim(
  name: [], question: [], answer: [], justification: [], guidelines: [],
) = {
  set list(indent: 1em, tight: false)
  show list: set block(spacing: 10pt)
  set block(spacing: 5.8pt)
  [
    *#name*

    Question: #question

    Answer: #answer

    Justification: #justification

    Guidelines:

    #guidelines
  ]
}

+ #claim(
  name: [Claims],
  question: [
    Do the main claims made in the abstract and introduction accurately reflect
    the paper's contributions and scope?],
  answer: answerYes,
  justification: [The abstract and @introduction state the headline 63.83%
    top-1 / 87.39% top-5 on `val-test`, the parameter budget actually used
    (6.58M of 10M), and the two findings the paper rests on -- that capacity
    does not bind (@table-width-floor) and that the model underfits, so no
    regularizer helped (@table-recipe-ablation). The self-supervised
    comparison named in the title is explicitly marked as designed but unrun,
    in the abstract, @introduction and @sec-ssl alike, and no numbers are
    claimed for it.],
  guidelines: [
  - The answer NA means that the abstract and introduction do not include the
    claims made in the paper.

  - The abstract and/or introduction should clearly state the claims made,
    including the contributions made in the paper and important assumptions and
    limitations. A No or NA answer to this question will not be perceived well
    by the reviewers.

  - The claims made should match theoretical and experimental results, and
    reflect how much the results can be expected to generalize to other settings.

  - It is fine to include aspirational goals as motivation as long as it is
    clear that these goals are not attained by the paper.
  ])

+ #claim(
  name: [Limitations],
  question: [
    Does the paper discuss the limitations of the work performed by the
    authors?
  ],
  answer: answerYes,
  justification: [@discussion has a dedicated Limitations paragraph. It
    states that no run was seeded, which comparisons that most affects, that
    all measurements come from one machine and need not transfer, that
    from-scratch accuracies are not comparable to pretrained baselines, and
    that GCE and EMA were each evaluated at a learning rate tuned for a
    different loss and so are unproven rather than ruled out.],
  guidelines: [
    - The answer NA means that the paper has no limitation while the answer No
      means that the paper has limitations, but those are not discussed in the
      paper.

    - The authors are encouraged to create a separate "Limitations" section in
      their paper.

    - The paper should point out any strong assumptions and how robust the
      results are to violations of these assumptions (e.g., independence
      assumptions, noiseless settings, model well-specification, asymptotic
      approximations only holding locally). The authors should reflect on how
      these assumptions might be violated in practice and what the implications
      would be.

    - The authors should reflect on the scope of the claims made, e.g., if the
      approach was only tested on a few datasets or with a few runs. In
      general, empirical results often depend on implicit assumptions, which
      should be articulated.

    - The authors should reflect on the factors that influence the performance
      of the approach. For example, a facial recognition algorithm may perform
      poorly when image resolution is low or images are taken in low lighting.
      Or a speech-to-text system might not be used reliably to provide closed
      captions for online lectures because it fails to handle technical jargon.

    - The authors should discuss the computational efficiency of the proposed
      algorithms and how they scale with dataset size.

    - If applicable, the authors should discuss possible limitations of their
      approach to address problems of privacy and fairness.

    - While the authors might fear that complete honesty about limitations
      might be used by reviewers as grounds for rejection, a worse outcome
      might be that reviewers discover limitations that aren't acknowledged in
      the paper. The authors should use their best judgment and recognize that
      individual actions in favor of transparency play an important role in
      developing norms that preserve the integrity of the community. Reviewers
      will be specifically instructed to not penalize honesty concerning
      limitations.
  ])

+ #claim(
  name: [Theory Assumptions and Proofs],
  question: [
    For each theoretical result, does the paper provide the full set of
    assumptions and a complete (and correct) proof?
  ],
  answer: answerNA,
  justification: [The paper contains no theoretical results.],
  guidelines: [
    - The answer NA means that the paper does not include theoretical results.

    - All the theorems, formulas, and proofs in the paper should be numbered
      and cross-referenced.

    - All assumptions should be clearly stated or referenced in the statement
      of any theorems.

    - The proofs can either appear in the main paper or the supplemental
      material, but if they appear in the supplemental material, the authors
      are encouraged to provide a short proof sketch to provide intuition.

    - Inversely, any informal proof provided in the core of the paper should be
      complemented by formal proofs provided in appendix or supplemental
      material.

    - Theorems and Lemmas that the proof relies upon should be properly
      referenced.
  ])

+ #claim(
  name: [Experimental Result Reproducibility],
  question: [
    Does the paper fully disclose all the information needed to reproduce the
    main experimental results of the paper to the extent that it affects the
    main claims and/or conclusions of the paper (regardless of whether the code
    and data are provided or not)?
  ],
  answer: answerYes,
  justification: [@method gives the architecture in full, @sec-recipe-ablation
    and @appendix-lr give every hyperparameter and how it was chosen, and
    @dataset-protocol specifies the split protocol, including that
    `src/make_val_split.py` regenerates a byte-identical `val-dev`/`val-test`
    split from the official labels and a fixed seed (251). The exact command
    behind the headline run is recorded with the run log. One caveat is stated
    rather than hidden: training runs were not seeded (@dataset-protocol), so
    a rerun reproduces the setup and the conclusions but not the exact
    digits.],
  guidelines: [
    - The answer NA means that the paper does not include experiments.

    - If the paper includes experiments, a No answer to this question will not
      be perceived well by the reviewers: Making the paper reproducible is
      important, regardless of whether the code and data are provided or not.

    - If the contribution is a dataset and/or model, the authors should
      describe the steps taken to make their results reproducible or verifiable.

    - Depending on the contribution, reproducibility can be accomplished in
      various ways. For example, if the contribution is a novel architecture,
      describing the architecture fully might suffice, or if the contribution
      is a specific model and empirical evaluation, it may be necessary to
      either make it possible for others to replicate the model with the same
      dataset, or provide access to the model. In general. releasing code and
      data is often one good way to accomplish this, but reproducibility can
      also be provided via detailed instructions for how to replicate the
      results, access to a hosted model (e.g., in the case of a large language
      model), releasing of a model checkpoint, or other means that are
      appropriate to the research performed.

    - While NeurIPS does not require releasing code, the conference does
      require all submissions to provide some reasonable avenue for
      reproducibility, which may depend on the nature of the contribution. For
      example

      #set enum(numbering: "(a)")

      + If the contribution is primarily a new algorithm, the paper should make
        it clear how to reproduce that algorithm.
      + If the contribution is primarily a new model architecture, the paper
        should describe the architecture clearly and fully.
      + If the contribution is a new model (e.g., a large language model), then
        there should either be a way to access this model for reproducing the
        results or a way to reproduce the model (e.g., with an open-source
        dataset or instructions for how to construct the dataset).
      + We recognize that reproducibility may be tricky in some cases, in which
        case authors are welcome to describe the particular way they provide
        for reproducibility. In the case of closed-source models, it may be
        that access to the model is limited in some way (e.g., to registered
        users), but it should be possible for other researchers to have some
        path to reproducing or verifying the results.
  ])

+ #claim(
  name: [Open Access to Data and Code],
  question: [
    Does the paper provide open access to the data and code, with sufficient
    instructions to faithfully reproduce the main experimental results, as
    described in supplemental material?
  ],
  answer: answerYes,
  justification: [FoodX-251 @kaur2019foodx is a public benchmark obtained from
    its official release; it is not redistributed here. All training,
    benchmarking and analysis code is in the accompanying repository, and the
    validation split is regenerated by a script rather than shipped as data,
    so it cannot silently drift from the one used here.],
  guidelines: [
    - The answer NA means that paper does not include experiments requiring
      code.

    - Please see the NeurIPS code and data submission guidelines
      (#url("https://neurips.cc/public/guides/CodeSubmissionPolicy")) for more
      details.

    - While we encourage the release of code and data, we understand that this
      might not be possible, so "No" is an acceptable answer. Papers cannot be
      rejected simply for not including code, unless this is central to the
      contribution (e.g., for a new open-source benchmark).

    - The instructions should contain the exact command and environment needed
      to run to reproduce the results. See the NeurIPS code and data submission
      guidelines (#url("https://neurips.cc/public/guides/CodeSubmissionPolicy"))
      for more details.

    - The authors should provide instructions on data access and preparation,
      including how to access the raw data, preprocessed data, intermediate
      data, and generated data, etc.

    - The authors should provide scripts to reproduce all experimental results
      for the new proposed method and baselines. If only a subset of
      experiments are reproducible, they should state which ones are omitted
      from the script and why.

    - At submission time, to preserve anonymity, the authors should release
      anonymized versions (if applicable).

    - Providing as much information as possible in supplemental material
      (appended to the paper) is recommended, but including URLs to data and
      code is permitted.
  ])

+ #claim(
  name: [Experimental Setting/Details],
  question: [
    Does the paper specify all the training and test details (e.g., data
    splits, hyperparameters, how they were chosen, type of optimizer)
    necessary to understand the results?
  ],
  answer: answerYes,
  justification: [@method gives the optimizer (SGD with Nesterov momentum),
    schedule (cosine with five-epoch linear warmup), loss, precision policy,
    crop sizes and worker count; @sec-recipe-ablation gives the learning rate
    and batch size with the sweeps that chose them; @dataset-protocol gives
    the splits and the 15-epoch proxy protocol. @appendix-lr and
    @appendix-nonlevers give the sweeps in full.],
  guidelines: [
    - The answer NA means that the paper does not include experiments.

    - The experimental setting should be presented in the core of the paper to
      a level of detail that is necessary to appreciate the results and make
      sense of them.

    - The full details can be provided either with the code, in appendix, or as
      supplemental material.
  ])

+ #claim(
  name: [Experiment Statistical Significance],
  question: [
    Does the paper report error bars suitably and correctly defined or other
    appropriate information about the statistical significance of the
    experiments?
  ],
  answer: answerYes,
  justification: [Every close comparison is accompanied by a paired McNemar
    exact test on the same `val-dev` images
    (`benchmarks/significance_test.py`), reported as an exact $p$ in
    @table-width-floor, @table-recipe-ablation, @table-batch-size and
    @table-crop-scale. That test is what keeps two results honest: the
    64-384 trunk's tie with the baseline ($p = 1.0$) and RandAugment's
    0.84-point deficit, which is *not* significant ($p = 0.075$) and is
    reported as unproven rather than a win. We do *not* report error bars over
    random seeds, because no run was seeded; @dataset-protocol and
    @discussion both state this and name it as the first thing that should be
    added, and are explicit that McNemar covers sampling variance over images
    only, not training variance.],
  guidelines: [
    - The answer NA means that the paper does not include experiments.

    - The authors should answer "Yes" if the results are accompanied by error
      bars, confidence intervals, or statistical significance tests, at least
      for the experiments that support the main claims of the paper.

    - The factors of variability that the error bars are capturing should be
      clearly stated (for example, train/test split, initialization, random
      drawing of some parameter, or overall run with given experimental
      conditions).

    - The method for calculating the error bars should be explained (closed
      form formula, call to a library function, bootstrap, etc.)

    - The assumptions made should be given (e.g., Normally distributed errors).

    - It should be clear whether the error bar is the standard deviation or the
      standard error of the mean.

    - It is OK to report 1-sigma error bars, but one should state it. The
      authors should preferably report a 2-sigma error bar than state that they
      have a 96% CI, if the hypothesis of Normality of errors is not verified.

    - For asymmetric distributions, the authors should be careful not to show
      in tables or figures symmetric error bars that would yield results that
      are out of range (e.g. negative error rates).

    - If error bars are reported in tables or plots, the authors should explain
      in the text how they were calculated and reference the corresponding
      figures or tables in the text.
  ])

+ #claim(
  name: [Experiments Compute Resources],
  question: [
    For each experiment, does the paper provide sufficient information on the
    computer resources (type of compute workers, memory, time of execution)
    needed to reproduce the experiments?
  ],
  answer: answerYes,
  justification: [All runs used one RTX 5050 Laptop GPU (8GB VRAM) in its
    `performance` power profile, stated in @experiments. Per-run wall-clock
    and peak VRAM are logged for every run (`src/runlog.py`) and quoted in
    the text: about 18-27 minutes per 15-epoch proxy, 2h05m and 1.99 GiB peak
    for the 90-epoch headline run. The runs behind the reported results total
    roughly 7 GPU-hours. The project spent appreciably more than that, and the
    paper says so rather than only counting what worked: the synthetic
    screening sweep of @appendix-sweep, a separate 90-epoch
    pipeline-validation run (@sec-full-run), the two 30-epoch legs of the
    matched rematch, and one proxy leg lost to a shared-memory crash and
    re-run (@appendix-nonlevers) are all additional.],
  guidelines: [
    - The answer NA means that the paper does not include experiments.

    - The paper should indicate the type of compute workers CPU or GPU,
      internal cluster, or cloud provider, including relevant memory and
      storage.

    - The paper should provide the amount of compute required for each of the
      individual experimental runs as well as estimate the total compute.

    - The paper should disclose whether the full research project required more
      compute than the experiments reported in the paper (e.g., preliminary or
      failed experiments that didn't make it into the paper).
  ])

+ #claim(
  name: [Code of Ethics],
  question: [
    Does the research conducted in the paper conform, in every respect, with
    the NeurIPS Code of Ethics
    #url("https://neurips.cc/public/EthicsGuidelines")?
  ],
  answer: answerYes,
  justification: [The work uses a public benchmark under its terms, releases
    its code, involves no human subjects or personally identifying data, and
    discloses its negative results and its methodological gaps rather than
    reporting only what worked.],
  guidelines: [
    - The answer NA means that the authors have not reviewed the NeurIPS Code
      of Ethics.

    - If the authors answer No, they should explain the special circumstances
      that require a deviation from the Code of Ethics.

    - The authors should make sure to preserve anonymity (e.g., if there is a
      special consideration due to laws or regulations in their jurisdiction).
  ])

+ #claim(
  name: [Broader Impacts],
  question: [
    Does the paper discuss both potential positive societal impacts and
    negative societal impacts of the work performed?
  ],
  answer: answerNA,
  justification: [The work trains a small image classifier on a public food
    benchmark, entirely from scratch, and releases no model or data that
    carries foreseeable societal risk beyond that of image classification
    generally.],
  guidelines: [
    - The answer NA means that there is no societal impact of the work
      performed.

    - If the authors answer NA or No, they should explain why their work has no
      societal impact or why the paper does not address societal impact.

    - Examples of negative societal impacts include potential malicious or
      unintended uses (e.g., disinformation, generating fake profiles,
      surveillance), fairness considerations (e.g., deployment of technologies
      that could make decisions that unfairly impact specific groups), privacy
      considerations, and security considerations.

    - The conference expects that many papers will be foundational research and
      not tied to particular applications, let alone deployments. However, if
      there is a direct path to any negative applications, the authors should
      point it out. For example, it is legitimate to point out that an
      improvement in the quality of generative models could be used to generate
      deepfakes for disinformation. On the other hand, it is not needed to
      point out that a generic algorithm for optimizing neural networks could
      enable people to train models that generate Deepfakes faster.

    - The authors should consider possible harms that could arise when the
      technology is being used as intended and functioning correctly, harms
      that could arise when the technology is being used as intended but gives
      incorrect results, and harms following from (intentional or
      unintentional) misuse of the technology.

    - If there are negative societal impacts, the authors could also discuss
      possible mitigation strategies (e.g., gated release of models, providing
      defenses in addition to attacks, mechanisms for monitoring misuse,
      mechanisms to monitor how a system learns from feedback over time,
      improving the efficiency and accessibility of ML).
  ])

+ #claim(
  name: [Safeguards],
  question: [
    Does the paper describe safeguards that have been put in place for
    responsible release of data or models that have a high risk for misuse
    (e.g., pre-trained language models, image generators, or scraped datasets)?
  ],
  answer: answerNA,
  justification: [No data or model with a high risk of misuse is released. The
    trained classifier predicts one of 251 food categories and has no
    generative capability.],
  guidelines: [
    - The answer NA means that the paper poses no such risks.

    - Released models that have a high risk for misuse or dual-use should be
      released with necessary safeguards to allow for controlled use of the
      model, for example by requiring that users adhere to usage guidelines or
      restrictions to access the model or implementing safety filters.

    - Datasets that have been scraped from the Internet could pose safety
      risks. The authors should describe how they avoided releasing unsafe
      images.

    - We recognize that providing effective safeguards is challenging, and many
      papers do not require this, but we encourage authors to take this into
      account and make a best faith effort.
  ])

+ #claim(
  name: [Licenses for Existing Assets],
  question: [
    Are the creators or original owners of assets (e.g., code, data, models),
    used in the paper, properly credited and are the license and terms of use
    explicitly mentioned and properly respected?
  ],
  answer: answerYes,
  justification: [FoodX-251 is credited to @kaur2019foodx and used under its
    published terms. `src/main.py` is adapted from the PyTorch ImageNet
    reference training script and says so in the repository, keeping that
    script's structure deliberately; `src/resnet.py` is kept unmodified as a
    baseline for comparison. PyTorch and torchvision are used as ordinary
    dependencies under their own licenses. Every technique tested in
    @sec-recipe-ablation is cited to its originating paper.],
  guidelines: [
    - The answer NA means that the paper does not use existing assets.

    - The authors should cite the original paper that produced the code package
      or dataset.

    - The authors should state which version of the asset is used and, if
      possible, include a URL.

    - The name of the license (e.g., CC-BY 4.0) should be included for each
      asset.

    - For scraped data from a particular source (e.g., website), the copyright
      and terms of service of that source should be provided.

    - If assets are released, the license, copyright information, and terms of
      use in the package should be provided. For popular datasets,
      #url("https://paperswithcode.com/datasets") has curated licenses for some
      datasets. Their licensing guide can help determine the license of a
      dataset.

    - For existing datasets that are re-packaged, both the original license and
      the license of the derived asset (if it has changed) should be provided.

    - If this information is not available online, the authors are encouraged
      to reach out to the asset's creators.
  ])

+ #claim(
  name: [New Assets],
  question: [
    Are new assets introduced in the paper well documented and is the
    documentation provided alongside the assets?
  ],
  answer: answerNA,
  justification: [No new dataset or pretrained model is released. The
    accompanying code is documented in the repository, but it is an
    implementation of the paper rather than an asset offered for reuse.],
  guidelines: [
    - The answer NA means that the paper does not release new assets.

    - Researchers should communicate the details of the dataset/code/model as
      part of their submissions via structured templates. This includes details
      about training, license, limitations, etc.

    - The paper should discuss whether and how consent was obtained from people
      whose asset is used.

    - At submission time, remember to anonymize your assets (if applicable).
      You can either create an anonymized URL or include an anonymized zip file.
  ])

+ #claim(
  name: [Crowdsourcing and Research with Human Subjects],
  question: [
    For crowdsourcing experiments and research with human subjects, does the
    paper include the full text of instructions given to participants and
    screenshots, if applicable, as well as details about compensation (if any)?
  ],
  answer: answerNA,
  justification: [The paper involves no crowdsourcing and no human subjects.
    FoodX-251's validation labels were human-verified by its original authors,
    not by us.],
  guidelines: [
    - The answer NA means that the paper does not involve crowdsourcing nor
      research with human subjects.

    - Including this information in the supplemental material is fine, but if
      the main contribution of the paper involves human subjects, then as much
      detail as possible should be included in the main paper.

    - According to the NeurIPS Code of Ethics, workers involved in data
      collection, curation, or other labor should be paid at least the minimum
      wage in the country of the data collector.
  ])

+ #claim(
  name: [
    Institutional Review Board (IRB) Approvals or Equivalent for Research with
    Human Subjects
  ],
  question: [
    Does the paper describe potential risks incurred by study participants,
    whether such risks were disclosed to the subjects, and whether
    Institutional Review Board (IRB) approvals (or an equivalent
    approval/review based on the requirements of your country or institution)
    were obtained?
  ],
  answer: answerNA,
  justification: [The paper involves no research with human subjects.],
  guidelines: [
    - The answer NA means that the paper does not involve crowdsourcing nor
      research with human subjects.

    - Depending on the country in which research is conducted, IRB approval (or
      equivalent) may be required for any human subjects research. If you
      obtained IRB approval, you should clearly state this in the paper.

    - We recognize that the procedures for this may vary significantly between
      institutions and locations, and we expect authors to adhere to the
      NeurIPS Code of Ethics and the guidelines for their institution.

    - For initial submissions, do not include any information that would break
      anonymity (if applicable), such as the institution conducting the review.
  ])

// New in NeurIPS 2026.
+ #claim(
  name: [Declaration of LLM Usage],
  question: [
    Does the paper describe the usage of LLMs if it is an important, original,
    or non-standard component of the core methods in this research? Note that
    if the LLM is used only for writing, editing, or formatting purposes and
    does _not_ impact the core methodology, scientific rigor, or originality of
    the research, declaration is not required.
  ],
  answer: answerNA,
  justification: [No LLM is a component of the method. The classifier, the
    training pipeline and every experiment reported here are ordinary
    supervised computer vision, and no result depends on an LLM. LLM
    assistance was used for writing and for routine software engineering,
    which the policy above does not require declaring.],
  guidelines: [
    - The answer NA means that the core method development in this research
      does not involve LLMs as any important, original, or non-standard
      components.

    - Please refer to our LLM policy in the NeurIPS handbook for what should
      or should not be described.
  ])
