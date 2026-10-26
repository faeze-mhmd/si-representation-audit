# Fashion-MNIST CNN extension: first verified milestone

Date: October 21, 2026. **No real-data results.**

The user selected continuation of the CNN extension. The [protocol](../../../FASHION_CNN_PROTOCOL.md) and [configuration](../../../configs/fashion_cnn_v1.json) now specify a six-run, nested training-budget comparison: 3,000 versus 12,000 training images, three paired seeds, fixed 30-epoch small CNN. A/B denote sample budgets in this extension; they denoted weight-decay settings in the completed Optdigits pilot. These studies must be reported separately.

## Work completed

- Implemented a 9,802-parameter CNN with a 32-dimensional primary representation.
- Added four substantive synthetic tests: representation/head consistency, paired initialization with caller-RNG preservation, invalid input rejection, and training gradient flow plus saved-weight logits preservation.
- Ran `python -B -m unittest discover -s tests -v`: **54 tests passed in 7.980 seconds** inside unittest. All previous tests also passed.
- Ran `python -B scripts/smoke_fashion_cnn.py --output <C:-temporary-cache>/synthetic_preflight.json`: completed, one CPU thread, batch 128, two warmup and twelve measured optimizer steps. No dataset opened, accuracy calculated or model weights saved.

## Measured timing and estimate

Mean measured synthetic optimizer step: **0.0305996 seconds**. The declared six-run schedule has 10,620 optimizer steps. Applying this measured step time gives an **estimate** of 324.968 seconds, or 649.935 seconds with a 2x margin, for optimizer work across all six models. This is approximately 5.4/10.8 minutes and lies below the declared 7,200-second training cap.

These are repeated random-tensor batches, not a timing measurement of real Fashion-MNIST training. Loading, validation, SI, probes, checkpoints, startup and interruptions are excluded. Actual data QA, tiny-batch overfit and real-training-only timing remain required before substantive runs. No GPU requirement or model accuracy is inferred from this timing.

## Important design limits

Larger data budget also causes more optimizer updates at fixed epochs (720 versus 2,820 per run). The study compares data availability under a fixed epoch policy, not data diversity at matched compute. A training-budget reference predictor is included in rank analysis. The 600-point validation SI is primary; a nested 2,000-point penultimate SI is a declared sensitivity check, not a metric selected afterward. Dataset, architecture and experimental axis differ from Optdigits; the comparison across studies is not a causal architecture test.

The original Fashion-MNIST source, published integrity checksums and license attribution are in the protocol. Data acquisition remains the next milestone. The C: drive had 101,304,954,880 bytes free when checked before preflight. No download, installation, cloud session, final-test performance, commit or push occurred.

## Reproduction

From the project directory in the existing environment:

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/smoke_fashion_cnn.py
```

The second command prints a new timing result and its source/config hashes; timing varies across invocations. The saved [actual preflight output](synthetic_preflight.json) identifies the executed versions. Its JSON was preserved with LF line endings by the workspace editor; contained source digests remain unchanged. No files belonging to the frozen Optdigits implementation/protocol/config/results were overwritten.
