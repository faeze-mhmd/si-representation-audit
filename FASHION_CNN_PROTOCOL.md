# Fashion-MNIST CNN extension — protocol v0.1

Date: October 21, 2026. **Stage: preflight.** The user selected continuation of the CNN extension. This document specifies a bounded study; real data have not been downloaded or trained on in this milestone. No accuracy fields are populated. Freeze the data manifest and record implementation/config/source hashes after data QA and real-data timing, before substantive training. Amendments must state reasons and preserve earlier versions.

## Question and scope

Does the observed relationship between SI and unseen accuracy persist in a small CNN on a second dataset, when training-data availability changes? Specifically, can fixed penultimate train/validation SI scores distinguish final classifiers across two data budgets and three paired seeds?

The Optdigits pilot is complete and its test has been inspected. It motivated this extension's design; this is not an exact replication. Dataset, architecture and comparison axis change together across studies, so their contributions cannot be causally separated by comparing the studies. Within this new study the declared axis is training budget. No learned generalization predictor or universal SI validity claim is proposed.

The prior weak weight-decay perturbation produced narrow outcomes and tied SI. A data-budget comparison may expose a wider range, but that outcome is not guaranteed. Larger-budget performance or SI need not improve. Budget itself is a required reference predictor: apparent SI agreement might merely reflect training-data availability. No augmentation, noise, pruning or hyperparameter sweep is added in this phase.

## Data provenance and isolation

Use the four original IDX gzip files from the [official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist), verified October 21, 2026. Credit Han Xiao, Kashif Rasul and Roland Vollgraf (2017), [dataset paper](https://arxiv.org/abs/1708.07747); retain the repository's MIT notice. It describes 60,000 training and 10,000 test examples, 28x28 grayscale inputs, ten labels.

| Filename | Published MD5 |
|---|---|
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 |

The official README lists about 30 MB compressed in total. Do not clone the repository or install torchvision just to load IDX. Use verified HTTPS acquisition, record actual source URL/bytes/SHA-256 as well as published MD5, and validate magic numbers, image/label counts, dimensions, dtype, labels and exact payload length. MD5 is an upstream integrity identifier, not a security claim.

Keep the official test partition isolated. Acquisition checks may inspect format, hashes and exact duplicate overlaps, but must not calculate test predictions, SI, class errors or rankings. Separate training-pool loaders from test evaluation. Do not claim subject/product-disjoint splits without metadata. If exact cross-partition duplicates or conflicting labels appear, document them and amend the interpretation before comparisons; do not silently remove/reassign benchmark examples. Near duplicates remain a separate unperformed check.

From the official 60,000 training rows, create a fixed stratified 54,000/6,000 pool/validation split with seed 2027; record ascending original row IDs. From the pool, use one per-class permutation with seed 2028 to select nested budgets: A = first 300/class (3,000), B = first 1,200/class (12,000), with A entirely inside B. Remaining pool rows are unused. These are budget-limited experiments, not full-training-set benchmark claims.

Transform all images consistently to float32 /255 and N×1×28×28. No fitted scaling/PCA, stochastic augmentation or test-based normalization. Keep loaders single-worker with explicit CPU data-order generators.

## Models and compute confound

Use the same CNN for all six runs:

```text
1x28x28
-> Conv2d(1,8,3,padding=1), ReLU, MaxPool2d(2)       [8x14x14]
-> Conv2d(8,16,3,padding=1), ReLU, MaxPool2d(2)      [16x7x7]
-> AdaptiveAvgPool2d(4x4), flatten                  [256]
-> Linear(256,32), ReLU                            [32: primary SI]
-> Linear(32,10)                                   [logits]
```

There are 9,802 trainable parameters, to be checked in preflight. No pretrained weights, dropout or batch normalization. AdamW lr=0.001, weight decay=0.01, batch=128, 30 fixed epochs, no scheduler/early stopping. The optimizer setting is fixed for all conditions; it is not chosen as a winner from the Optdigits test. Seeds 11/22/33 pair initialization across budgets. Data-order generators share the seed, but differently sized datasets cannot have identical minibatch sequences; do not claim matched batch orders.

At fixed epochs, A has 24 steps/epoch and B has 94, hence 720 versus 2,820 updates/run (10,620 across six runs). The intervention changes data availability **and** training exposure/compute; this protocol does not isolate sample diversity at matched optimizer steps. Report both sample counts and measured update/time totals.

Primary checkpoint is epoch 30. Capture SI at 0/5/15/30 and classifier train/validation curves. Do not use trajectories/layers as independent model replications or select a best epoch. Save optimizer/RNG state for interruption recovery, atomically, with source/config/split fingerprints. Final portable-weight support for this CNN remains to be implemented; the original MLP-specific loader must not be reused as if it supported the CNN.

## Fixed SI subsets and controls

Retain the published SI definition and the existing verified distance/tie implementation. Primary train SI uses 60/class from the shared A core, seed 2029: identical 600 images for both budgets and all seeds, all in-sample for learning. Primary validation SI uses 60/class, identical 600 images for every model. Construct a 200/class validation sensitivity pool using seed 2030, then a nested 60/class primary subset using seed 2031. Persist IDs; no resampling by model or favorable outcomes.

Primary geometry: penultimate ReLU at epoch 30. Secondary layer views: normalized flattened input (784), pooled convolutional features (256), and penultimate (32), on the 600-point subsets. Cache the fixed input SI. Distance query batch=8 bounds wide-layer intermediates; exact nearest-neighbor cost remains quadratic. The resolution of a 600-point score is one agreement/600, approximately 0.167 percentage points.

At final epoch only, also compute validation penultimate SI on the fixed 2,000 examples (one agreement/2,000 = 0.05 percentage points). This is a predeclared sensitivity analysis, reported alongside the 600-point result regardless of which correlates better; it is not a replacement primary score. It changes both sample size and neighbor candidates and does not create 2,000 independent model replications. Do not compute 2,000-point wide-layer SI in this version.

Epoch-0 representations and 100 fixed-graph label permutations of primary 600-point penultimate features at epochs 0/30 are sanity controls. Seeds 2032/2033 for train/validation. Label counts remain fixed. These are not p-values for label-trained networks. Model-training labels are not shuffled as an extra experiment.

## Baselines and analysis

Before CNN training, fit majority, input logistic regression and brute Euclidean 1-NN separately for each budget: six input readouts. Same training IDs/normalization and full validation queries. Logistic C=1, lbfgs, max_iter=1000, random_state=2027; report convergence warnings. Bound 1-NN prediction query batches. No PCA or SVM search. At epoch 30 fit a linear and a 1-NN probe on each CNN's training penultimate features: twelve secondary readouts. Probes are not independent network replications.

Primary outcome: official test accuracy of **all six frozen CNN heads**, not a validation winner. Secondary: macro-F1, per-class confusion/recall and all twelve probes/six input baselines (24 total readouts). No refit with validation added. Freeze an executable final-test plan after validating the workflow and before any Fashion-MNIST test performance.

Fixed primary predictor scores: train SI600, validation SI600. Reference predictors: full validation accuracy, negative full validation cross-entropy, training budget. SI2000 is secondary. Compare descriptive Spearman with test head accuracy across six models; also report within-budget SI agreement with three seeds, preserving undefined correlations on constants. Report all six rows, budget-group means/sample SD, and paired B-A differences. Accuracy ranks are primary; do not select a different target after results.

Six correlated models and one new dataset do not support a significance or universal transferability claim. Do not pool the six Optdigits runs and six CNN runs into a supposedly independent 12-model dataset or one cross-dataset rank coefficient. Present each study and its limitations separately. A wider across-budget correlation may reflect the budget effect rather than reliable within-condition ranking.

Error gallery: prechosen fashion_A_seed11, first 12 errors in ascending official test row order, after the final test evaluation. Preserve full predictions and per-class metrics for all 24 readouts. Any post-test modification is exploratory and needs fresh confirmation evidence.

## Preflight gates and budget

1. **Current milestone:** write this plan/config, verify the CNN on synthetic tensors, and time throwaway CPU optimizer steps. Save source/config hashes; no real-data inference or saved synthetic weights.
2. Acquire/validate data, save IDs and duplicate audit; perform tiny-batch overfit and real-training-only timing/SI checks. A synthetic timing projection does not establish real-data cost or useful accuracy.
3. Estimate six-run training cost with 2x margin; training ceiling 7,200 CPU seconds. Separately budget SI/probes/validation/checkpoint work up to 3,600 seconds. Stop before launching substantive runs if the projected budget exceeds these caps; document consistent scope revision or choose available free GPU before training. Paid compute is not authorized. GPU runtime/version/determinism must be recorded; no cross-device bitwise equivalence is assumed.
4. Validate convergence warnings/finite values and checkpoint continuation. Freeze source/config/split state and run only within the established budget, retaining all conditions including unfavorable results.
5. Validate saved models/metrics on training/validation, freeze final evaluation plan, evaluate official test once, inspect errors and write the extension report.

No package installation, dataset download, full CNN training, cloud session, new result claim or Git action is authorized merely by a favorable synthetic timing number. Dataset acquisition and bounded experiments are part of the user's selected extension; carry them out at the corresponding verified milestone. Estimated human effort for the extension is 10–18 focused hours, subject to data QA and compute timing; this is a planning estimate, not measured labor or a guaranteed completion time.

The Optdigits sources, protocol, config, weights and final results remain a historical completed study. Add new modules/artifacts; do not overwrite their pinned sources. Configuration: [fashion_cnn_v1.json](configs/fashion_cnn_v1.json). Current CNN implementation: [cnn.py](src/si_audit/cnn.py). Preflight script: [smoke_fashion_cnn.py](scripts/smoke_fashion_cnn.py).
