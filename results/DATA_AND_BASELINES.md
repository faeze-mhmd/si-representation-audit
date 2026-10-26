# Milestone 02 — official data, input baselines and CPU readiness

Date: October 21, 2026. **Status: completed.** All model-performance values in this report are **inner validation**, not final test results. The six-run neural study is not yet executed.

## Data and protocol checks

Official source: [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits); Alpaydin & Kaynak (1998), DOI 10.24432/C50P49; CC BY 4.0 as stated by UCI. Retained optdigits.tra, optdigits.tes and optdigits.names from the official archive.

- Archive size: 591,292 bytes. SHA-256: `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`.
- Original training: 3,823 rows. Each has 64 integer features in 0–16 and one target in 0–9; no missing/nonfinite fields.
- Original test: 1,797 rows. Used only for schema/class-domain/count and duplicate QA; no predictions, scores or tuning.
- Inner split: 3,058 training / 765 validation, stratified with seed 2027, ascending original row order.
- SI subsets: 600 training / 600 validation, 60 per class, sampled with NumPy seed 2028 after splitting.
- Zero exact duplicate vectors within either official file, across official train/test, or across inner train/validation. Zero conflicting-label duplicate groups.
- Original train/test writers are disjoint according to UCI. Inner validation writer independence cannot be checked from processed files; near-duplicates were not assessed.

Evidence: [data_audit.json](data_audit.json) includes provenance, all file hashes, class counts, audit gate and every split/subset row ID. No dataset change or sampling amendment was needed.

## Fixed baseline results

All features divided by the fixed constant 16. No learned preprocessing, hyperparameter search or test evaluation. BLAS/OpenMP budget: one thread; final script also limits joblib's process-local CPU count to one.

| Model | Validation accuracy | Macro-F1 | Errors / 765 | Fit seconds | Validation prediction seconds |
|---|---:|---:|---:|---:|---:|---:|
| Majority class | 10.20% | 0.0185 | 687 | 0.0009 | 0.0001 |
| Logistic regression, C=1 | 96.47% | 0.9647 | 27 | 0.1663 | 0.0006 |
| Euclidean 1-NN | 98.04% | 0.9803 | 15 | 0.0010 | 0.0432 |

[baseline_results.json](baseline_results.json) records full parameters, per-class precision/recall/F1, confusion matrices, warnings, runtime, source hashes, and the first 30 validation errors per model. Logistic regression converged in 84 iterations; all three final model runs recorded no warnings.

The 1-NN baseline is strong on this split. Ten of its fifteen validation errors involve true digits 8 or 9. This is a descriptive error pattern, not a general statement about digit difficulty. It does not establish that neural representations will improve prediction. Training 1-NN accuracy is in-sample and includes self; it must not be presented as SI/leave-one-out accuracy.

These are single-split results. No significance, independent-replication, writer-level uncertainty or final test-generalization claim is made.

## CPU timing check

Existing environment: Python 3.11.4, NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.9.0, threadpoolctl 3.6.0, PyTorch 2.12.1+cpu. No packages installed.

A throwaway MLP with the planned 64→64→32→10 architecture ran two training epochs on the 3,058 training examples. One CPU thread, seed 11, AdamW lr=0.001, weight_decay=0, batch_size=64; no checkpoint or accuracy evaluation.

- Observed epoch times: 0.0745, 0.0797 seconds.
- Linear projection for six 50-epoch runs: 23.13 seconds of training.
- A twofold margin gives 46.27 seconds, below the predeclared 7,200-second training budget.
- This is a two-epoch estimate, excluding imports, SI/probes, validation, logging, saving and interruptions. It is not a measured full-pilot runtime or a guarantee.

An initial recorded run emitted a Windows physical-core-discovery warning. Its JSON and warning are preserved in [baseline_first_run.json](baseline_first_run.json) and baseline_first_run_warning.txt (not included in this publication snapshot). The final script explicitly sets a one-core process budget, removing the hardware-probe warning. Accuracy/macro-F1 were identical across these two executions; they are not independent replications. The initial timing estimate was 51.46 seconds, illustrating timing/setup variability. Both timing runs are excluded from the six scientific pilot runs.

## Commands actually executed

From the project directory:

```powershell
python -B scripts/fetch_optdigits.py --archive 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-optdigits.zip' --data-dir 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-optdigits\raw'
python -B scripts/prepare_data.py --data-dir 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-optdigits\raw' --output -
python -B scripts/run_baselines.py --data-dir 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-optdigits\raw' --data-artifact results/data_audit.json --benchmark-cpu --output -
python -B -m unittest discover -s tests -v
python -B scripts/verify_artifacts.py
```

All final commands exited successfully. **39 test methods passed** (0.227 seconds reported by unittest), with additional parameterized cases. Tests include schema rejection, source hash changes, split leakage, balanced SI subsets, and a guard that training-only loading never opens optdigits.tes. Full output: data_baseline_tests.txt (not included in this publication snapshot).

The verifier independently recomputed accuracy and macro-F1 from all saved training/validation confusion matrices, checked error counts and source/data hashes, and confirmed zero full pilot runs. Verification output (not included in this publication snapshot).

## Acquisition and filesystem limitations

Python's HTTPS path reported an expired certificate. A Windows HTTPS request to the same official URL succeeded with certificate validation enabled; no TLS checks were disabled. The successful acquisition was:

```powershell
$digitTempArchive = Join-Path ([System.IO.Path]::GetTempPath()) 'codex-si-audit-optdigits.zip'
Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip' -OutFile $digitTempArchive -TimeoutSec 45
```

Windows denied shell creation/writes inside Documents, so raw archive/data are in the permitted temporary cache. Temporary files can be cleaned; re-fetch instructions and hashes are retained. The complete stdout JSON artifacts were saved using the workspace file-editing tool and their integrity checked afterward. This session has not established a clean-environment installation or general file-export behavior on another machine.

## Changes and next milestone

Added strict original-file loading, leakage-aware split/audit code, input baselines, a shared MLP definition with throwaway timing, acquisition and verification scripts, meaningful data tests, environment records, and the [experiment ledger](../experiments/experiment_log.csv). Protocol v0.2 remains frozen; status updates live in PROJECT_BRIEF.md/README.md.

Next: implement the six-run neural experiment with paired seeds, checkpoint/layer representations, fixed SI subsets, and validation-only development. Reserve final test outcomes until the reporting plan is fixed. No full pilot runs, staging, Git initialization, commits, pushes or publication occurred.
