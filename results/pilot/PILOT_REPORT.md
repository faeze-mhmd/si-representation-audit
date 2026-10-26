# Milestone 03 — six-run SI pilot, validation preview

October 21, 2026. **Six of six planned training runs completed. Final test performance has not been computed.**

## Measured results

Same original training/validation partition, fixed 600-example SI subsets, MLP 64→64→32→10, AdamW lr=0.001, batch=64, 50 epochs, one CPU thread. A/B pairs share initial tensors and all 50 minibatch-order hashes. Fixed epoch 50 and penultimate ReLU are the primary checkpoint/layer; no best-epoch selection.

| Run | Weight decay | Validation accuracy | Macro-F1 | Train SI | Validation SI |
|---|---:|---:|---:|---:|---:|
| A_seed11 | 0 | 97.39% | 0.9738 | 96.17% | 97.67% |
| B_seed11 | 0.01 | 97.25% | 0.9725 | 96.17% | 97.50% |
| A_seed22 | 0 | 97.25% | 0.9725 | 97.17% | 97.67% |
| B_seed22 | 0.01 | 97.39% | 0.9737 | 97.33% | 97.50% |
| A_seed33 | 0 | 96.99% | 0.9699 | 96.67% | 97.67% |
| B_seed33 | 0.01 | 96.99% | 0.9699 | 96.67% | 97.50% |

Prediction metrics use all 765 validation examples. Each SI uses its own fixed, balanced 600-example subset. SI percentages mean local label agreement, not classifier accuracy.

| Condition | Mean validation accuracy ± sample SD | Mean validation SI |
|---|---:|---:|
| A: decay 0 | 97.21% ± 0.20 percentage points | 97.67% |
| B: decay 0.01 | 97.21% ± 0.20 percentage points | 97.50% |

Three paired seeds per condition; SD is seed variability, not a confidence interval. B–A changes in correct validation predictions are −1, +1, 0 for seeds 11, 22, 33. The mean accuracy change is zero; this small study cannot establish that regularization has no effect generally.

The fixed input-space 1-NN baseline scored 98.04%. All six MLP heads scored below it. **Readout matters:** the separately fitted 1-NN probes on learned penultimate features averaged 98.21% in each condition, while linear probes averaged 97.12% (A) / 97.21% (B). These are predeclared secondary validation comparisons, not a selected model improvement or a significance claim. Complete probe metrics and warnings are retained in each run record.

## What the pilot shows

- Within each condition, validation SI is identical across all three seeds despite different prediction accuracies: A = 586/600; B = 585/600. It therefore cannot rank these final classifiers within that condition. Within-condition Spearman is undefined because SI is constant.
- Across all six related runs, descriptive Spearman with validation accuracy is 0.0615 for train SI, 0.0000 for validation SI, and −0.9562 for negative validation cross-entropy. All reference measures are reported; no p-values or population conclusions. Loss and accuracy can rank near-saturated classifiers differently.
- SI increased substantially from untrained penultimate representations to trained ones. That training trajectory does not establish accurate ranking between final models or prediction on an independent dataset.
- Final validation SI does not increase at every layer. Layer geometry and trained-head predictions measure different properties. The SI step size on 600 points is 1/600 (0.167 percentage points); A/B's validation-SI difference is exactly one agreement.
- The proposed training-SI optimism pattern was not observed here: selected training subsets had lower SI than validation subsets. Different examples/neighbourhoods prevent interpreting that difference as a generalization gap or rejecting the broader hypothesis.
- Fixed-representation label permutation controls used 100 shuffles, seeds 2030/2031, preserving class counts. Final control means span 0.0976–0.1006, around the analytical random-agreement expectation 59/599 ≈ 0.09850. This is a measurement sanity check, not a valid trained-network significance test.

This is a small, controlled diagnostic audit with useful limitations, not evidence that SI is universally useful/useless or that a new generalization predictor has been established. Validation SI uses labelled examples overlapping the validation accuracy target; independent test evaluation is still required.

## Evidence and figures

- [Analysis table, paired differences and rank comparisons](pilot_analysis.json).
- [Training-loop index and provenance](pilot_index.json).
- Per-run history, four observed SI checkpoints, controls, probes, confusion matrices, error IDs, initialization/order hashes: each run's metrics.json below this folder.
- Six small final weight bundles under the project's checkpoints directory; compressed NPZ inside JSON, loaded without pickle by model_from_bundle.
- [Learning curves](figures/learning_curves.svg), [layer SI](figures/layer_si.svg), [SI versus validation accuracy](figures/si_vs_validation.svg), error gallery (not included in this publication snapshot).
- Figures were rendered as PNG and visually inspected before publishing SVG. Shading denotes min–max seed range, not confidence intervals. Error gallery uses preselected A_seed11 and the first 12 error IDs, not a hand-picked best/worst run.

Measured training time: 47.87 seconds total. The six-run loop including measurement/checkpoint work took 65.29 seconds, excluding process imports/setup. Actual measurement time: 16.38 seconds. The earlier two-epoch timing projection was only an estimate.

## Verification and reproduction

**45 test methods passed** before the experiment, including meaningful checks for hidden activation extraction, exact portable-weight logits, preservation of RNG on model loading, paired initialization/order, synthetic optimizer/generator continuation, and corrupt-bundle rejection.

All six portable models were reloaded from the delivered folder. Their final train/validation prediction metrics and all 12 primary SI scores reproduced exactly. Source/split hashes and paired initialization/order checks passed. All six native optimizer/RNG checkpoints were also safe-loaded at epoch 50 and their tensor hashes matched the delivered models. See verification log (not included in this publication snapshot) and test output (not included in this publication snapshot).

The final read-only verifier also completed successfully after packaging: it recomputed the prediction metrics, twelve primary SI values and their permutation controls from the six delivered models, without accessing test data or writing files. Actual output (not included in this publication snapshot).

Actual training command, from the project directory:

```powershell
python -B scripts/run_pilot.py --data-dir 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-optdigits\raw' --data-artifact results/data_audit.json --config configs/pilot_v1.json --output-root 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-optdigits\pilot_v1'
python -B scripts/verify_saved_pilot.py --data-dir 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-optdigits\raw'
```

The analyzer was run on the cache and on the delivered folder. Its numerical verification passed on the delivered artifacts; a repeated figure export then failed because drive C was full. The original four complete figures had already been inspected and preserved. Do not mistake that later I/O failure for failed model/numerical verification.

During packaging, JSON reserialization initially changed integral-valued floats, causing a strict fingerprint mismatch. Original numeric types/text were restored; all 12 run/model JSON copies match the originals after CRLF/LF normalization. The final weights and scientific results did not change.

## Storage and current limits

Drive C reached zero free space. Only this project's redundant temporary files and, after all runs completed and verified final weights were retained, temporary optimizer checkpoints were removed (about 2.65 MB total). No unrelated/user files were removed. All raw training/test files, delivered final weight bundles, metrics and figures remain. Native resume checkpoints for these completed runs were cleaned; future interrupted runs still use the implemented resume mechanism. Re-executing from this cleaned cache requires recomputation or a new output directory.

Free space was still critically low after cleanup. A normal rerun/next stage needs storage with enough free space. No clean-environment or cross-device numerical-reproducibility claim is made.

## Next milestone

[Final evaluation plan](FINAL_EVALUATION_PLAN.md): keep all six epoch-50 models fixed, use original test only for final reporting, and compare predeclared diagnostics against those outcomes without changing hyperparameters/checkpoints. No final test scores, Git initialization, staging, commit, push or publication occurred.
