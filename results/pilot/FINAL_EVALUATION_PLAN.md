# Final evaluation plan — frozen before opening test performance

Date: October 21, 2026. Status: prepared; not executed.

1. Keep all six existing A/B, seed 11/22/33, epoch-50 models. Verify the delivered tensor/payload hashes, data/split/source hashes, and original test SHA-256 6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8.
2. Use original optdigits.tes (1,797 examples). Do not refit neural models using validation or change model/checkpoint choices. Input baselines use the same 3,058 training examples for a fair comparison.
3. Primary outcome: test accuracy for each of the six classifier heads. Secondary: macro-F1, per-class recall/confusion and errors. Evaluate all six, not only the validation winner.
4. Fixed predictor scores: epoch-50 penultimate train SI and validation SI. Reference predictors: validation accuracy and negative validation cross-entropy. Compare descriptive six-model rank agreement with test accuracy; constant-score correlations remain undefined. No significance or acceptance/publication claims.
5. Report three-seed group mean/sample SD and paired B–A differences. Seeds/checkpoints/probes do not become independent datasets. Do not tune after inspecting test outcomes.
6. Secondary readouts, declared now: all twelve penultimate linear/1-NN probes plus three input baselines. Fit probe classifiers on training representations only using the same fixed settings as validation; no selection among the six. Keep their outcomes separate from the primary head-accuracy analysis.
7. Do not use test labels to construct predictor SI scores or choose normalizations. Any post-test methodological extension is exploratory and needs fresh confirmation data.
8. Save exact configurations, predictions/errors, source/tensor hashes and actual runtime. Storage space must be available before creating final artifacts.

Preserved final model tensor hashes:

- A_seed11: `e0eb15828d6dbfffdc0a2c3de9bc23bc4790b34ca2073d795c735f7fd44c98b3`
- B_seed11: `10f4b5bb6dd9f2cd815f2e1330e6b6eef30987ef4706b6707f656de3d493626a`
- A_seed22: `0842aaa1bea3fe9515f7a438fdc03706fc61b0741e274dfd9d6673d1fbfad424`
- B_seed22: `eb85aab5d48a4b22a0b4a2c3321472f3521d3054de200529fc43aa80b28d3d18`
- A_seed33: `3fabaa78198f7b338180410d35974699723772bd42f7b94eafe450e1cfc73737`
- B_seed33: `2b037a975539cade186bbe0ead2b08adedd7fb427e61a527bc58079743fb0633`
