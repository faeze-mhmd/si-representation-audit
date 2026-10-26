# Final test evaluation — SI representation audit

Executed October 21, 2026, under the [frozen plan](../pilot/FINAL_EVALUATION_PLAN.md). No neural retraining, checkpoint selection, hyperparameter changes or test SI. All six heads and 15 readouts reproduced their original validation metrics before test access.

## Primary results

3,058 training / 765 validation / 1,797 official test examples. All six epoch-50 models are reported. The official test writers differ from the original training pool; the inner validation split has no verified writer separation.

| Setting | Seed | Test accuracy | Macro-F1 | Errors | Train SI | Validation SI |
|---|---:|---:|---:|---:|---:|---:|
| A | 11 | 94.94% | 0.9493 | 91 | 96.17% | 97.67% |
| B | 11 | 94.99% | 0.9498 | 90 | 96.17% | 97.50% |
| A | 22 | 95.38% | 0.9537 | 83 | 97.17% | 97.67% |
| B | 22 | 95.49% | 0.9548 | 81 | 97.33% | 97.50% |
| A | 33 | 95.49% | 0.9548 | 81 | 96.67% | 97.67% |
| B | 33 | 95.44% | 0.9542 | 82 | 96.67% | 97.50% |

| Setting | Mean test accuracy | Sample SD (percentage points) | Mean macro-F1 |
|---|---:|---:|---:|
| A | 95.27% | 0.294 | 0.9526 |
| B | 95.31% | 0.275 | 0.9530 |

Sample SD describes three training seeds on one fixed test set; it is not a confidence interval. Paired B minus A accuracy differences:

| Seed | Difference (percentage points) |
|---|---:|
| 11 | +0.056 |
| 22 | +0.111 |
| 33 | -0.056 |

B changes the correct count by +1, +2 and -1 examples in the paired seeds. The average change is too narrow a basis for a general regularization advantage.

## Predeclared model-ranking analysis

| Fixed predictor | Six-model Spearman rho |
|---|---:|
| train si | 0.731 |
| validation si | -0.198 |
| validation accuracy | -0.364 |
| negative validation cross entropy | 0.551 |

Training SI agrees more closely with test accuracy than validation SI in this particular six-model comparison. That contrasts with its near-zero agreement with validation accuracy in the preview. Validation SI remains constant across seeds within each condition, so its within-condition correlation is undefined. These findings do not validate training SI as a general predictor or invalidate validation accuracy/SI. The model family, dataset and accuracy range are narrow; all models share data, architecture and paired seeds. No p-values, statistical superiority, universal metric ranking or novelty claim is made.

## Secondary readouts — all reported

Every probe was fitted only on the same 3,058 training representations. Input baselines used those same training rows, without refitting on validation. Logistic C=1, lbfgs, max_iter=1000, random_state=2027; 1-NN uses brute-force Euclidean distance. These are separate classifier readouts, not additional independent replications of the primary neural-head comparison.

| Readout | Test accuracy | Macro-F1 | Errors |
|---|---:|---:|---:|
| A_seed11__linear_probe | 94.99% | 0.9498 | 90 |
| A_seed11__cross_split_1nn_probe | 95.88% | 0.9588 | 74 |
| B_seed11__linear_probe | 95.05% | 0.9504 | 89 |
| B_seed11__cross_split_1nn_probe | 95.83% | 0.9582 | 75 |
| A_seed22__linear_probe | 95.83% | 0.9582 | 75 |
| A_seed22__cross_split_1nn_probe | 95.77% | 0.9577 | 76 |
| B_seed22__linear_probe | 95.77% | 0.9577 | 76 |
| B_seed22__cross_split_1nn_probe | 95.72% | 0.9571 | 77 |
| A_seed33__linear_probe | 95.72% | 0.9570 | 77 |
| A_seed33__cross_split_1nn_probe | 95.99% | 0.9599 | 72 |
| B_seed33__linear_probe | 95.66% | 0.9565 | 78 |
| B_seed33__cross_split_1nn_probe | 96.05% | 0.9604 | 71 |
| input__majority | 10.13% | 0.0184 | 1615 |
| input__logistic_regression | 94.88% | 0.9489 | 92 |
| input__euclidean_1nn | 97.27% | 0.9726 | 49 |

The input 1-NN baseline (97.27%) outperforms every trained head and every learned-feature probe here. Changing the readout to a probe does not remove that observed disadvantage. There was no post-test attempt to improve it.

## Errors and reproducibility

Full confusion matrices and per-class precision/recall/F1 for all 21 readouts are in [final_results.json](final_results.json). [predictions.csv](predictions.csv) stores every prediction: test_row is zero-based in optdigits.tes; sample_id = 3823 + test_row. Every error is recoverable from truth and readout columns. The gallery uses the prechosen A_seed11 model and its first 12 errors in ascending test row order; it is not a representative sample of all mistakes.

Figures: [test comparison](figures/test_performance.svg), [predictors versus test](figures/si_vs_test.svg), [fixed-model confusion](figures/test_confusion.svg), fixed-model errors (not included in this publication snapshot).

The saved prediction CSV is independently checked against all 21 confusion matrices, accuracies and macro-F1 scores. Source/config/split/plan/tensor hashes and software versions are recorded; no clean-environment or cross-platform reproduction claim is made. The historical pilot index and plan keep their pre-test status text; this report and final_results.json record completion.

Packaging note: the workspace editor converted result JSON from CRLF to LF and added a terminal newline to the start sentinel. The numeric text is unchanged. [packaging_manifest.json](packaging_manifest.json) retains both original and delivered hashes. The verifier reconstructs original line endings and requires the original byte hashes as well as delivered hashes. The prediction CSV is byte-identical to the executed output. Initial strict gallery verification correctly rejected the newline difference; the packaging check was added afterward, without rerunning models or modifying numerical results.

Measured runtime (excluding Python imports/startup):

- pretest_validation_gate: 8.631789 seconds.
- total_after_imports: 9.282890 seconds.

Per-class audit of the prechosen A_seed11 head:

| True digit | Test support | Recall | Errors |
|---|---:|---:|---:|
| 0 | 178 | 99.44% | 1 |
| 1 | 182 | 97.80% | 4 |
| 2 | 177 | 97.18% | 5 |
| 3 | 183 | 87.43% | 23 |
| 4 | 181 | 97.79% | 4 |
| 5 | 182 | 96.70% | 6 |
| 6 | 181 | 97.79% | 4 |
| 7 | 179 | 91.06% | 16 |
| 8 | 174 | 91.95% | 14 |
| 9 | 180 | 92.22% | 14 |

The largest directional confusions in this fixed model are 7 -> 5 (10 examples), 8 -> 1 (9), and 3 -> 2 (8). These are descriptive observations after evaluation, not evidence of a specific cause or a basis for tuning on this test.

No estimator warnings were emitted.

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/verify_saved_pilot.py --data-dir (Join-Path $digitCache 'raw')
# Final evaluator requires a fresh output folder; existing sentinel prevents overwrite/repeated runs.
python -B scripts/evaluate_final.py --data-dir (Join-Path $digitCache 'raw') --output-dir (Join-Path $digitCache 'final_v1')
# Saved-results verification/plotting does not refit or run model predictions:
python -B scripts/analyze_final.py --results-dir results/final --output-dir (Join-Path $digitCache 'final_figures')
```

Set $digitCache using the README acquisition workflow. The completed run used the C: temporary data cache; text/SVG deliverables were then preserved in this project. Drive C had about 94 GiB available before this stage. No unrelated file cleanup, package installation, Git initialization, staging, commit, push or publication was performed.

Next milestone: review the scientific interpretation and the user's own contribution before publication. Any later model changes are exploratory relative to this now-inspected test; a confirmatory extension needs a fresh evaluation protocol and dataset.
