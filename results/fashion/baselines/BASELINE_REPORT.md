# Fixed Fashion-MNIST input baselines — validation only

October 21, 2026. All six predeclared readouts completed before the main CNN experiment. A uses 3,000 training rows; B uses the nested 12,000 rows. All query the same 6,000 validation images. No official test file was opened by this workflow.

| Budget | Method | Validation accuracy | Macro-F1 | Errors | Logistic iterations |
|---|---|---:|---:|---:|---:|
| A | Majority | 10.00% | 0.0182 | 5400 | — |
| A | Logistic regression | 81.87% | 0.8190 | 1088 | 254 |
| A | Euclidean 1-NN | 78.42% | 0.7858 | 1295 | — |
| B | Majority | 10.00% | 0.0182 | 5400 | — |
| B | Logistic regression | 82.88% | 0.8288 | 1027 | 388 |
| B | Euclidean 1-NN | 81.70% | 0.8177 | 1098 | — |

No estimator warnings were emitted. The two input logistic regressions converged within the fixed max_iter=1000. C=1, lbfgs, random_state=2027; 1-NN uses brute Euclidean search on ascending original training IDs with at most 256 query images/batch. All inputs are flattened float32 /255 without fitted scaling/PCA. These are one fit per budget/method, not three neural-seed replications. No hyperparameter search or superiority significance test was performed.

Actual workflow wall after setup: 27.609933 seconds, including fitting, prediction and prediction-CSV construction, excluding imports/setup. The B logistic fit took 17.321748 seconds; the B 1-NN queries took 5.726470 seconds. Exact timings/parameters, per-class confusion/recall, warning lists and the first 30 error IDs are retained in [baseline_results.json](baseline_results.json).

Every validation prediction is in [validation_predictions.csv](validation_predictions.csv). Its exact executed byte hash matched after delivery, and all six metric summaries were recovered from it without refitting. The manifest/source digests in the result identify the executed versions.

Command actually executed from the project directory:

```powershell
python -B scripts/run_fashion_baselines.py --data-dir 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-fashion\raw' --output-dir 'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-fashion\baselines_v1'
```

The script refuses to overwrite an existing result artifact. No raw data, estimator binary, commit or push was published.
