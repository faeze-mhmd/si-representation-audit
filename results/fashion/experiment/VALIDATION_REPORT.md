# Fashion-MNIST CNN study — validation preview

October 21, 2026. Six of six predeclared CNN conditions completed. **Official test performance remains uninspected.** All results below concern training/validation; they are not final unseen-test claims.

## Design and evidence

Fixed 3,000 (A) versus 12,000 (B) nested training images, three paired seeds, one 9,802-parameter CNN, AdamW lr=0.001/weight_decay=0.01, batch 128, 30 fixed epochs. Validation has 6,000 examples. Primary geometry is the epoch-30 penultimate ReLU; primary SI subsets have 600 shared images per split. A nested 2,000-point validation SI is the separately declared sensitivity view.

A/B initialization tensors match within each seed. Their batch sequences differ because dataset sizes differ. B has 2,820 optimizer updates versus A's 720 at fixed epochs; the intervention combines more data and more training exposure. No causal diversity-at-matched-compute or cross-architecture claim is supported.

The [protocol](../../../FASHION_CNN_PROTOCOL.md), [pre-training freeze](../TRAINING_FREEZE.json), [data/split audit](../DATA_AND_PREFLIGHT.md), and [cost gate](../preflight/measurement_gate.json) identify the planned methods and executed source. No checkpoint, seed, budget or metric was selected after seeing outcomes.

## All six classifier heads

| Budget | Seed | Train accuracy | Validation accuracy | Macro-F1 | Errors | Train SI600 | Validation SI600 | Validation SI2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 11 | 76.60% | 74.97% | 0.7379 | 1502 | 70.83% | 71.67% | 73.65% |
| B | 11 | 84.52% | 83.62% | 0.8324 | 983 | 77.83% | 79.00% | 78.15% |
| A | 22 | 76.53% | 75.48% | 0.7428 | 1471 | 71.83% | 71.00% | 72.65% |
| B | 22 | 83.53% | 82.87% | 0.8244 | 1028 | 74.83% | 74.00% | 76.80% |
| A | 33 | 76.17% | 75.88% | 0.7486 | 1447 | 71.00% | 70.50% | 72.20% |
| B | 33 | 83.09% | 82.97% | 0.8244 | 1022 | 73.67% | 74.17% | 77.20% |

| Budget | Mean validation accuracy | Sample SD (percentage points) | Mean macro-F1 |
|---|---:|---:|---:|
| A | 75.44% | 0.460 | 0.7431 |
| B | 83.15% | 0.407 | 0.8271 |

Sample SD describes three training seeds on a shared validation set; it is not a confidence interval. Paired B minus A accuracy differences:

| Seed | Difference (percentage points) |
|---|---:|
| 11 | +8.650 |
| 22 | +7.383 |
| 33 | +7.083 |

Observed comparisons within each budget:

- A: 0/3 heads exceeded the strongest fixed input baseline (logistic_regression, 81.87%); 0/3 tied it. Validation SI had 3 distinct 600-point scores and 3 distinct 2,000-point scores across the three seeds. These counts describe this validation comparison, not significance or general validity.
- B: 2/3 heads exceeded the strongest fixed input baseline (logistic_regression, 82.88%); 0/3 tied it. Validation SI had 3 distinct 600-point scores and 3 distinct 2,000-point scores across the three seeds. These counts describe this validation comparison, not significance or general validity.

## Input baselines — same training rows

| Budget | Fixed model | Validation accuracy | Macro-F1 | Errors | Warnings |
|---|---|---:|---:|---:|---:|
| A | majority | 10.00% | 0.0182 | 5400 | 0 |
| A | logistic_regression | 81.87% | 0.8190 | 1088 | 0 |
| A | euclidean_1nn | 78.42% | 0.7858 | 1295 | 0 |
| B | majority | 10.00% | 0.0182 | 5400 | 0 |
| B | logistic_regression | 82.88% | 0.8288 | 1027 | 0 |
| B | euclidean_1nn | 81.70% | 0.8177 | 1098 | 0 |

Input baselines were fitted once per budget, not once per neural seed. Logistic regression uses C=1, lbfgs, max_iter=1000, seed 2027. 1-NN uses Euclidean brute search on ascending training IDs, with query batches bounded at 256. Inputs use fixed float32 /255; no fitted scaler/PCA. [Full baseline results](../baselines/baseline_results.json) and [all baseline predictions](../baselines/validation_predictions.csv) are retained.

## All twelve frozen-representation probes

| CNN | Linear accuracy | Linear macro-F1 | Linear warning count | 1-NN accuracy | 1-NN macro-F1 |
|---|---:|---:|---:|---:|---:|
| fashion_A_seed11 | 80.80% | 0.8052 | 1 | 74.25% | 0.7435 |
| fashion_B_seed11 | 85.05% | 0.8494 | 1 | 81.28% | 0.8132 |
| fashion_A_seed22 | 80.58% | 0.8027 | 1 | 73.63% | 0.7372 |
| fashion_B_seed22 | 84.15% | 0.8398 | 1 | 80.12% | 0.8022 |
| fashion_A_seed33 | 79.05% | 0.7864 | 1 | 72.97% | 0.7306 |
| fashion_B_seed33 | 84.10% | 0.8393 | 1 | 79.55% | 0.7965 |

A concrete disagreement already appears within budget A: seed 11 has higher validation SI600 (71.67%) than seed 33 (70.50%), but lower validation head accuracy (74.97% versus 75.88%). All three seed rows remain reported. This is a validation-stage counterexample to identical ranking in this setting, not a universal SI failure or an independent test result.

Probes with warnings: **6 of 12**. Full warning messages and iteration counts remain in each run record. A linear probe reaching max_iter=1000 is a fixed-budget, potentially non-converged readout, not a fully optimized linear-separability estimate. Settings were retained rather than silently increasing iterations or scaling hidden features. Probe comparisons inherit this limitation.

Probes fit only training representations; validation labels are used for reporting. A probe is a different readout of the same learned representation, not an additional independent trained network. Within-set SI uses 599 candidate neighbors on 600 points, whereas cross-split 1-NN probes use 3,000 or 12,000 training candidates. Their numerical scores have different evaluation protocols.

## What this preview can establish

The tables directly show how each fixed SI view and classifier behaves at the two budgets. They do not establish a general predictor of unseen performance. In particular, validation SI and validation accuracy use overlapping labeled examples. The primary independent comparison remains the future official test accuracy of all six frozen heads. Training budget itself is a required reference predictor in that analysis; an across-budget SI trend could reflect only the budget intervention.

The 600-point SI resolution is one agreement/600 (0.167 percentage points); the 2,000-point view has 0.05-point resolution and different neighbor candidates. Both must be reported regardless of which correlates more favorably. Different checkpoints/layers/subsets do not increase the number of independent models. Optdigits and Fashion-MNIST are reported as separate studies; their model rows are not pooled for a single correlation or significance test.

Figures: [learning curves](figures/learning_curves.svg), [layer SI](figures/layer_si.svg), [SI versus validation](figures/si_vs_validation.svg), validation errors (not included in this publication snapshot). Shading is the observed seed range. The gallery uses the prechosen A_seed11 model and its first twelve validation errors in ascending original IDs; it is not a random sample of all failures and contains no official test examples.

## Verification and runtime

All six portable, pickle-free CNN bundles were reloaded. Final train/validation head metrics, all final layer-SI values, twelve primary SI600 values, six SI2000 values, and fixed-graph shuffle controls reproduced. Twelve probes were refitted with identical settings on identical training features and reproduced their validation metrics/parameters/warnings/iterations. Six baseline metric summaries were independently recovered from saved predictions. All **24 validation readouts** are saved in [validation_predictions.csv](validation_predictions.csv); [validation_analysis.json](validation_analysis.json) contains the complete verified comparison.

- Training: 445.459 seconds.
- Measurements and probes: 664.947 seconds.
- Checkpoint writes: 1.813 seconds.
- Training-runner invocation wall: 1114.235 seconds.
- Six baseline workflow: 27.610 seconds.

Times are actual observations on the recorded one-thread CPU environment and exclude Python startup/imports where specified. The 7,200-second training cap and 3,600-second measurement cap were enforced. The earlier eight-minute training estimate was a projection, not the full workflow's measured duration.

Before main training, 68 unit tests passed in 13.055 seconds, including exact synthetic interruption/continuation, stale text-record recovery after checkpoint replacement, wrong-source rejection and corrupt tensor rejection. Initial atomic-rename tests failed in the Windows sandbox's private temporary directory; tests then passed in the named C: project cache (and the five targeted training tests passed outside that private sandbox). Explicit Python-owned checkpoint streams are closed before rename. No weakened checkpoint or numerical test was substituted. An initial isolated readout test emitted a joblib core-discovery warning; subsequent commands cap LOKY to one CPU. Baseline convergence warnings were absent; probe warnings are preserved above.

The experiment index's delivered bundle paths point to ignored checkpoints/fashion/ files; its original numeric types, model fingerprints and measurements are preserved. Native optimizer/RNG checkpoints remain in the C: temporary experiment cache for interruption recovery. No clean-environment, cross-device bitwise-reproduction, independent-authorship, novelty or publication claim is made.

## Reproduction and next milestone

From the project directory, with the raw cache and the recorded environment available:

```powershell
$fashionCache = Join-Path ([System.IO.Path]::GetTempPath()) 'codex-si-audit-fashion'
python -B scripts/run_fashion_experiment.py --data-dir (Join-Path $fashionCache 'raw') --output-root (Join-Path $fashionCache 'experiment_v1')
python -B scripts/analyze_fashion.py --experiment-root results/fashion/experiment --data-dir (Join-Path $fashionCache 'raw') --output-dir (Join-Path $fashionCache 'analysis_reproduction')
```

The runner reuses a completed matching condition and resumes a partial matching checkpoint; changed fingerprints are rejected. Verification refits probes without neural retraining and does not open official test files. The synthetic resume test has passed; interrupted real-run timing can vary.

Next: freeze a final evaluation plan naming all six preserved tensors, six input baselines, twelve probes and fixed predictor scores; then evaluate official test performance as a separate milestone. No final test metrics, Git initialization, staging, commit, push, package installation or publication occurred in this stage.
