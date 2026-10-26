# CNN runner verification before main training

October 21, 2026. All **68 unit tests passed in 13.055 seconds** inside unittest, before the training freeze and six main CNN conditions. This includes the prior 61 tests, two new readout checks and five new CNN training checks.

New readout checks cover training-only fitting, bounded query sizes, preserved fitting warnings and exact-tie prediction parity with the same scikit-learn implementation.

New CNN checks cover weighted cross-entropy across unequal evaluation batches, portable-weight logits/RNG preservation, format/payload rejection, source/config drift rejection and corrupt checkpoint tensors. The interrupted-continuation check compares final tensor digest, all epoch orders, histories, SI/control/sensitivity snapshots, final head metrics and probe metrics against uninterrupted execution. It also simulates an older text record accompanying a newer atomic tensor checkpoint and verifies recovery from the tensor state.

Initial runs reached the new tests but three checkpoint cases failed on atomic rename in the Windows sandbox-private temporary directory. Explicitly closed Python file streams were added; the sandbox-private directory still rejected renames. A separate atomic rename in the named C: project cache succeeded. The five focused CNN tests then passed outside the sandbox-private temp (5.223s), and the full 68-test suite passed with tempfile directed to the named C: cache. No model or optimization parameters were changed to address this filesystem restriction.

The first isolated readout test also emitted the known joblib Windows physical-core discovery warning. LOKY_MAX_CPU_COUNT=1 is now process-local in tests and execution scripts. All six actual input baselines completed without estimator warnings. The cost-calibration linear probe reached the fixed 1,000-iteration limit; its warning is retained in measurement_gate.json. Main-run probe warnings are to be reported rather than changing max_iter/scaling silently.

Verified command used for the full suite, from the project directory:

```powershell
python -B -c "import pathlib,tempfile,unittest; folder=pathlib.Path(r'C:\Users\Jafar\AppData\Local\Temp\codex-si-audit-fashion\unit-test-cache'); folder.mkdir(parents=True,exist_ok=True); tempfile.tempdir=str(folder); unittest.main(module=None,argv=['unittest','discover','-s','tests','-v'])"
```

On a normal writable temporary directory the standard `python -B -m unittest discover -s tests -v` command is appropriate; the named location above records the actual successful managed-session execution. Source/config/split hashes are pinned in ../TRAINING_FREEZE.json and measurement_gate.json. These are code/continuation checks, not evidence of generalization or personal interview mastery.
