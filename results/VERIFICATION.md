# Milestone 01 — SI measurement verification

Date: October 21, 2026. Status: completed. Evidence is **synthetic metric correctness**, not learned-model performance.

## What changed

- Added a NumPy SI implementation with explicit self-exclusion, stable integer-ID ties, duplicate/tie diagnostics and per-example neighbour evidence.
- Added substantive unittest checks, including an independent Python-loop oracle and rejection of mixed labels that could otherwise silently become identical strings.
- Added an executable synthetic demonstration, source hashes and the tested NumPy requirement. No installation, dataset download, training, Git initialization, staging, commit, push or publication occurred.
- Protocol v0.2 records numerical conventions; the proposed six-model experiment remains unchanged.

## Actual commands and outcomes

Executed from the project directory in the existing Python environment:

1. `python -B -m unittest discover -s tests -v`
   - Exit code: 0.
   - **28 test methods passed**, including parameterized subcases.
   - Unittest-reported runtime: 0.303 seconds; tool wall time including Python startup: 2.467038 seconds.
   - Full output: test_output.txt (not included in this publication snapshot).
2. `python -B scripts/run_synthetic_demo.py --output results/synthetic_demo.json`
   - File export failed with FileNotFoundError in the managed shell, first on directory creation and then on file writing after the directory was created through the workspace editor.
   - The error was not a metric assertion failure. A file-export traceback is saved in file_export_error.txt (not included in this publication snapshot).
3. `python -B scripts/run_synthetic_demo.py --output -`
   - Exit code: 0; complete JSON artifact printed successfully.
   - Exact stdout JSON was parsed and saved using the authorized workspace file-editing tool.
   - Hand-computed and observed SI: separated = 1.0, alternating = 0.0, partial agreement = 0.75.
   - Evidence: [synthetic_demo.json](synthetic_demo.json), including inputs, labels, neighbours, environment and source SHA-256 hashes.

Test environment: Python 3.11.4, NumPy 1.26.4, Windows CPU. Standard-library unittest was used; pytest is not installed and was not needed. PyTorch CPU is present but has not been exercised in this milestone.

## Scope and limitations

Checks cover the measurement on deterministic, randomized small and deliberately invalid inputs. Search is exhaustive over supplied examples but arithmetic is float64. Extreme magnitudes can overflow (rejected) or underflow/lose precision. Ties are resolved by the smallest persistent ID; default row-number IDs are not persistent after shuffling. Batched search still has quadratic time complexity.

High SI on synthetic or training examples does not establish generalization. No accuracy, benchmark, training runtime or resume achievement is claimed. Shell-based file export remains unverified in this managed environment; stdout-based reproduction is verified.

## Next milestone

Validate original UCI digit files and split provenance, audit duplicates, establish the CPU training environment and basic baselines. Run the fixed pilot only after its runtime check. Commit/push approval remains user-controlled; no such action was performed or proposed here.
