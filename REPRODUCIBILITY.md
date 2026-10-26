# Reproduction levels and practical limits

No new environment has been installed or verified. The observed Python 3.11.4 environment is recorded in [requirements-fashion-observed.txt](requirements-fashion-observed.txt); those pins are observations, not a proven fresh-install lockfile. Scripts use the local src directory. No package installation is required in the already tested environment.

## 1. Synthetic understanding and code checks

From the project directory:

```powershell
python -B scripts/interview_practice.py
python -B scripts/run_synthetic_demo.py --output -
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -B -m unittest discover -s tests -v
```

The full 72-test suite passed in the existing environment before final evaluation. In this managed Windows session, sandbox-private temporary directories previously rejected checkpoint renames. The verified workaround was to use a named C: cache via Python's tempfile.tempdir. If that specific restriction applies, the following is the previously exercised pattern; it creates only scratch test files:

```powershell
python -B -c "import pathlib,tempfile,unittest; p=pathlib.Path(tempfile.gettempdir())/'codex-si-audit-fashion'; p.mkdir(exist_ok=True); tempfile.tempdir=str(p); s=unittest.defaultTestLoader.discover('tests'); r=unittest.TextTestRunner(verbosity=1).run(s); raise SystemExit(not r.wasSuccessful())"
```

The new interview demo checks three hand-computed cases and positive uniform scaling/translation on those cases. It does not prove general invariance numerically for every finite-precision input, and running it does not demonstrate independent user mastery.

## 2. Check saved final evidence without fitting or neural inference

Optdigits needs its delivered result/packaging files, without raw data or model weights:

```powershell
$digitCache = Join-Path $env:TEMP 'codex-si-audit-optdigits-review'
python -B scripts/analyze_final.py --results-dir results/final --output-dir (Join-Path $digitCache 'figures')
```

Fashion-MNIST additionally uses the original official test files to check true labels, provenance and gallery pixels. It does not fit a model or run the CNN:

```powershell
$fashionCache = Join-Path $env:TEMP 'codex-si-audit-fashion'
python -B scripts/analyze_fashion_final.py --results-dir results/fashion/final --data-dir (Join-Path $fashionCache 'raw') --output-dir (Join-Path $fashionCache 'saved_checks')
```

These saved checks reconstruct all 21 Optdigits and all 24 Fashion classification summaries, the declared rank/group summaries and the error selections. Fashion also checks the original partial-readout hashes from the factored representation. Class predictions do not reconstruct cross-entropy. Hashes and matching numbers are consistency evidence, not proof that all methods are scientifically valid or an independent replication.

Raw-data acquisition instructions and checksum records are in the README and data reports. Missing data are a prerequisite to acquire with the documented provenance; do not silently substitute torchvision/sklearn releases or different files. No download was needed during interview/publication preparation.

## 3. Replay frozen-model validation/test predictions

The final evaluators need the raw data and all six relevant portable weight bundles at the paths referenced by their historical indices. The local project retains twelve small JSON bundles under checkpoints; .gitignore excludes that directory from source tracking. The proposed supplementary manifest records their hashes and paths. A source-only publication will support level 2 checks, but cannot replay frozen networks until those reviewed assets are supplied.

The existing README contains the actually exercised final-evaluation command patterns. Use a fresh output directory; the evaluator deliberately refuses an existing start marker. Repeating the command checks reproducibility on an already inspected test. It does not restore independent confirmation evidence.

## 4. Train again from scratch

The original training scripts/configs and all saved provenance are available. Fashion's runner explicitly checks its historical baseline-results hash, cost gate, source/config/split freeze and observed environment before training. Those baseline artifacts include timings; regenerating them normally changes byte hashes. A new installation or changed artifacts therefore need a new, explicitly documented run/provenance setup rather than bypassing the original gates or overwriting the old evidence.

Exact full retraining on a fresh machine has not been verified. No one-command fresh-install-to-original-hashes claim is made. A future independent reproduction should record its own environment, artifact hashes and actual times, then compare scientific outcomes; it must retain the old record and label post-test changes as exploratory.

## What is unchanged in this milestone

Interview examples, review notes and publication manifests add explanation and packaging metadata. Frozen training sources/configs/plans, recorded results and model tensors are preserved. No real-data training or new test inference is needed for this milestone. See review (not included in this publication snapshot) and publication plan (not included in this publication snapshot).
