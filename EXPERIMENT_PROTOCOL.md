# SI representation audit — initial experiment protocol

**Version:** 0.2, October 21, 2026. **Stage:** BUILD; SI implementation and synthetic verification completed; no real-data experiment or training. Model/data settings remain pre-execution defaults. Changes must be versioned with reasons before final test evaluation. Version 0.2 clarifies numerical/tie conventions and records metric implementation; the six-run design is unchanged.

## Question and interpretation

Within one small architecture and dataset, does training-set or validation-set SI agree with the ranking of unseen classification accuracy under a controlled change in regularization? The provisional hypothesis is that training SI can be optimistic under overfitting. Neither the hypothesis nor a disagreement is guaranteed. A null or opposite result is reportable.

This is an empirical audit/adaptation, not an exact reproduction, new index, or established generalization predictor. Six runs and one dataset cannot support transferability claims. Accuracy on unseen examples is the primary target; train–test accuracy gap is a separate descriptive outcome.

**Metric source:** [Kalhor et al., 2019, v1](https://arxiv.org/abs/1906.05156v1), Section 2.2, equation (1), PDF page 5. For fixed representations z_i, nearest neighbour n(i) minimizes Euclidean distance over j != i. SI is the fraction with y_i = y_n(i). It is supervised and equals leave-one-out 1-NN accuracy on those fixed representations with matching conventions. This identity is not independent evidence for the method.

## Data and leakage rules

- Use original processed **optdigits.tra** and **optdigits.tes**, not an unverified convenience-loader split. [UCI source](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits): 64 inputs in 0–16, ten classes, official training/test sets from different writers, CC BY 4.0. Record attribution, download URL, file hashes, parsed counts and schema.
- Keep the official test partition. Make a fixed stratified 80/20 train/validation split inside official training with split seed **2027**. The processed files do not supply per-example writer IDs; no writer-independent validation or writer-level confidence interval is claimed.
- Scale inputs by the fixed constant **16**. No fitted feature scaler or PCA in version 0.1.
- Audit exact duplicates and conflicting labels; preserve sample IDs. If duplicates bridge splits or metadata contradicts independence, pause experimental interpretation and document a protocol amendment before running comparisons. Do not silently change the official benchmark.
- SI analysis: use fixed **60 examples per class per split** (600 training, 600 validation), sampled once with seed **2028**, after checking availability. Equal class counts/sample sizes reduce a confound in comparing SI. Save selected IDs. Report full training/validation prediction metrics separately. These 600-point metrics describe subsets, not the entire distribution.
- All model/probe choices and reporting decisions use training/validation only. Open test outcomes once the experiment/configuration/reporting plan is fixed; evaluate all predeclared runs together. Further changes after viewing test results require new independent confirmation data or an explicitly exploratory label.

## Models and controlled comparison

| Setting | Shared model/training | Sole comparison variable |
|---|---|---|
| A | MLP 64 → 64 → 32 → 10; ReLU after each hidden layer; cross-entropy; AdamW; learning rate 0.001; batch size 64; 50 epochs | Weight decay 0 |
| B | Identical | Weight decay 0.01 |

Run both settings with training seeds **11, 22, 33**: **six trained models**. Pair initialization and minibatch order within each seed. No dropout, augmentation, scheduler, or early stopping in this pilot. Use the fixed epoch-50 checkpoint for primary comparison. Save descriptive trajectories at epochs 0, 5, 20, 50; they are correlated observations, not additional independent runs.

Evaluate SI on input, first hidden ReLU, and penultimate ReLU representations in evaluation mode. **Penultimate SI at epoch 50 is the predeclared primary diagnostic**; other layers/epochs are secondary. Do not select the most favorable layer retrospectively. The baseline metric uses raw Euclidean hidden-space geometry; per-feature scaling or L2 normalization would be an explicitly separate ablation.

## Baselines, controls and reporting

- Input baselines: majority-class predictor, logistic regression (C=1, max_iter=1000), and Euclidean 1-NN. Fit on training only. Report convergence issues rather than silently increasing search budgets.
- Representation probes: logistic regression with the same fixed settings and 1-NN, fitted on full training penultimate features and evaluated on validation. Compare their validation behavior with SI within validation. This cross-split 1-NN protocol is different from SI within one set. Leave-one-out 1-NN on that same set is not a separate comparator.
- Standard model metrics: full train/validation accuracy and cross-entropy, final test accuracy and macro-F1, per-class test recall and confusion matrix. Record exact subset sizes for all SI results.
- SI controls: epoch-0 untrained representations; label-permutation control on frozen features with preserved class counts and a recorded permutation seed. These controls do not train additional models or change training labels.
- Report six individual runs, paired A/B differences for each seed, and mean/standard deviation as descriptive seed variability. Rank agreement of train SI, validation SI, validation accuracy, and negative validation loss with final test accuracy may be shown descriptively with ties handled. Six runs, shared data and paired seeds are insufficient for a significance claim. If a score is constant, correlation is undefined and must be reported that way.
- Show SI-versus-layer curves, SI-versus-unseen-accuracy scatter, and prediction errors. Report disagreements if present; otherwise explicitly report that the pilot did not reveal one. Do not manufacture a failure example to satisfy a deliverable.

## Metric verification before real-data interpretation

Required checks: hand-computed separated and alternating-label point sets; explicit self-exclusion; all-one-class input; N < 2 rejection; NaN/infinite input rejection; consistent labels/shapes; duplicate points and conflicting labels; distance ties; agreement of batched exact search with a tiny brute-force implementation. These are substantive checks of the scientific measurement.

Search all neighbours using float64 squared Euclidean distances formed from direct differences, without approximate search. Squaring preserves the mathematical neighbour ordering; exact computed squared-distance equality defines a tie. This is floating-point computation, not exact real arithmetic. Smallest persistent integer sample ID wins a tie. Report tie incidence and the fraction of rows participating in duplicate-vector groups. Default IDs are row indices; explicit persistent IDs are required to preserve tied-neighbour choices across row shuffles. Without ties, row reordering should preserve SI. Reject nonfinite/overflowing distances. A different precision or tie policy is an implementation change to record.

## Compute, reproducibility and stop criteria

- CPU is sufficient for the intended pilot; actual runtime remains unmeasured. Use bounded distance batches; at 600 points, a full float64 distance matrix alone is approximately 2.9 MB, excluding intermediates.
- Before the six-run experiment, time a short training segment on actual hardware and estimate the remaining cost. Pilot target: no more than **two CPU-hours** for the six-model training set, excluding setup/analysis. If the projection exceeds this, reduce scope consistently and record the amendment before final test evaluation; do not drop only inconvenient runs.
- No GPU use, paid services, package installation, or data acquisition has occurred. The metric runs in the existing Python 3.11.4 / NumPy 1.26.4 environment; unittest is from the standard library. requirements-metric.txt records the tested NumPy version. PyTorch CPU is installed but has not been exercised in this milestone. Save versions, device/thread settings, seeds, split/data hashes, complete training config, checkpoint and metrics when training begins. Record source-state hashes until an approved Git commit exists; do not invent a commit SHA.
- Stop interpretation on metric-test failure, label/ID misalignment, unresolved cross-split duplicates, invalid partition assumptions, nonfinite training, or severe baseline failures. Fix and log the reason first.

## Evidence and next milestones

Minimum deliverable: tested SI implementation; data/split manifest; six-run ledger with per-run metrics/runtime; one layer plot and diagnostic scatter; baseline table; honest limitations; a reproduction command actually run. No result cells or résumé outcome claims should be filled before execution.

1. **Completed milestone:** SI implementation and synthetic checks. See [verification report](results/VERIFICATION.md) and [synthetic evidence](results/synthetic_demo.json); these are not benchmark results.
2. **Next milestone:** validate original data and CPU training environment; establish loader and baselines.
3. Execute and analyze the fixed six-run experiment.
4. Decide from evidence whether to extend to a small CNN/Fashion-MNIST study and one additional experimental axis.

Effort estimate remains 15–25 focused hours to an auditable pilot, including reading/setup/analysis. This is not a measured runtime. At 10–15 hours/week, the pre-November-1 milestone remains tight. Show files, changes and proposed message, then obtain explicit approval before any commit or push.
