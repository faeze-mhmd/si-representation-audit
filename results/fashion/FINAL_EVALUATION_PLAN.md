# Fashion-MNIST final evaluation plan — frozen before test performance

Prepared 2026-10-21T12:52:58.779766+00:00. Status: prepared; not executed. The validation study, including secondary probe convergence warnings, is known. No official Fashion-MNIST test predictions, SI or performance have been evaluated.

1. Use original official 10,000-example t10k IDX files. Verify published compressed identifiers and the exact SHA-256 below, as well as current source/config/split/freeze/baseline and six tensor digests before test inference. Test row IDs are zero-based; global sample_id = 60000 + test_row. No subject/product independence beyond the documented official partition is claimed.
2. Preserve all six epoch-30 SmallFashionCNN models. No checkpoint/model selection and no neural refit using validation. A uses the fixed 3,000 training IDs; B uses the fixed 12,000 IDs. These budgets combine more data with more updates under fixed epochs; no matched-compute causal claim.
3. Primary outcome: official test accuracy of all six trained heads. Secondary: macro-F1, confusion, per-class recall and complete prediction/error records. Report every seed, budget mean/sample SD, and paired B minus A. Seed SD is descriptive, not a confidence interval.
4. Primary predictor scores: saved epoch-30 penultimate train SI600 and validation SI600. References: full validation accuracy, negative validation cross-entropy, and training budget. Secondary sensitivity predictor: saved validation penultimate SI2000. No score is recomputed from test labels/features, and neither SI subset/layer/checkpoint is selected for favorable results.
5. Compare descriptive six-model Spearman rank agreement with head test accuracy, using average ranks for exact ties and null/undefined on constant vectors. Also report within-budget three-seed agreement. No p-values, universal predictor ranking or independent-model assumption. Do not pool Optdigits and Fashion-MNIST model rows into one correlation.
6. Secondary readouts: all twelve penultimate linear/1-NN probes and six input baselines, totaling 24 test readouts including the six heads. Fit each on its original budget's training samples/features only, with fixed C=1, lbfgs, max_iter=1000, random_state=2027 and brute Euclidean 1-NN. Use float32 /255 inputs and the same batching/tie conventions as validation. No fitted feature normalization or additional search. Reproduce saved validation readout summaries before test prediction.
7. Preserve and report all convergence warnings, including linear probes that reached the 1,000-iteration cap during validation. They are limited optimization-budget readouts, not fully optimized separability estimates. Do not silently increase iterations, rescale features, drop affected models or switch a readout after test inspection.
8. Compute no test SI. Save all 24 predictions per test image, full per-class metrics, configuration/source/tensor provenance and actual runtime. Prechosen error gallery: fashion_A_seed11, first twelve errors in ascending test-row order. It is diagnostic, not a random error sample. Verify saved predictions against every summary independently of the model execution.
9. Use an exclusive start marker to prevent accidental repeated test evaluation/overwrite. A technical repair after test access must be logged; model/method extensions after test inspection are exploratory and need fresh confirmation evidence. Source and validation-metric gates must pass before computing any test performance.
10. This plan prepares the next milestone only. No test execution, publication, Git initialization, staging, commit or push occurs through creating this document.

## Official test file hashes

- t10k-images-idx3-ubyte.gz: SHA-256 `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073`; MD5 `bef4ecab320f06d8554ea6380940ec79`.
- t10k-labels-idx1-ubyte.gz: SHA-256 `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5`; MD5 `bb300cfdad3c16e7a12a480ee83cd310`.

## Preserved epoch-30 tensor hashes

- fashion_A_seed11: `577362e3257ccbfc116813d7ecf4efcf70b679c2e3a58fdb5e4c8e4b81d7a731`.
- fashion_B_seed11: `7118cdb124ceddeb714efd89095dfb6705bd4b92c7dc003ddcad4942341da595`.
- fashion_A_seed22: `5a3f0cc5e74afe033b345e75bf62046754f8a0decaa910f0cf5e3ad29883a116`.
- fashion_B_seed22: `8f2e5567315aedf7b7056d8cbd91f55b8a85e27721ba8ca88fac058f884e01d8`.
- fashion_A_seed33: `b0e69579b619aebbcc27ed30b8818cc01f1071a8c17c097a222de52ca14d8ad4`.
- fashion_B_seed33: `f6b3f48e933811adfef32e90dae11b037f167217af4a176102c4df0137ce5524`.

Scientific methods: [protocol](../../FASHION_CNN_PROTOCOL.md). Training freeze: [TRAINING_FREEZE.json](TRAINING_FREEZE.json). Validation evidence: [report](experiment/VALIDATION_REPORT.md), [verified analysis](experiment/validation_analysis.json).
