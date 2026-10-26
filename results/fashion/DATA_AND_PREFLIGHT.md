# Fashion-MNIST data and real-training preflight

October 21, 2026. **Data QA and bounded preflight complete. Main six-run CNN experiment not yet executed.**

## Data acquisition and validation

Downloaded four original IDX gzip files from the [official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist), pinned to revision `b2617bb6d3ffa2e429640350f613e3291e10b141`. All four matched the published MD5 identifiers, and actual SHA-256/bytes/URLs were recorded. Compressed data total: **30,878,645 bytes**. HTTPS certificate validation remained enabled. The first sandboxed request could not open a socket; the authorized network-enabled execution succeeded.

Credit: Han Xiao, Kashif Rasul and Roland Vollgraf (2017), [Fashion-MNIST](https://arxiv.org/abs/1708.07747). The upstream [MIT notice](../../LICENSES/Fashion-MNIST.txt) is retained. Raw data are cached on C: in the Windows temporary folder, outside the deliverable source tree. No torchvision or other package was installed. Temporary files may be cleaned by the system; the pinned acquisition URLs and hashes allow recovery.

The strict loader validated IDX magic numbers, paired counts, 28x28 shape, byte dtype, labels 0–9 and exact decoded length. Published compressed checksums are checked before decompression, which is size-bounded. The training entry point opens only train-* files; the official test loader was called explicitly for schema and duplicate QA, without predictions, SI or performance metrics.

## Saved partitions and subsets

| Set | Images | Role |
|---|---:|---|
| Official training | 60,000 | Source of all training/validation selections |
| Internal training pool | 54,000 | Complement of fixed validation IDs |
| Validation | 6,000 | Reserved for development metrics |
| Budget A | 3,000 | 300 per class |
| Budget B | 12,000 | 1,200 per class; contains A |
| Shared training SI | 600 | 60/class within A; same IDs for all models |
| Primary validation SI | 600 | 60/class; within the sensitivity subset |
| Sensitivity validation SI | 2,000 | 200/class; final penultimate view only |
| Official test | 10,000 | Final performance remains uninspected |

split_manifest.json (not included in this publication snapshot) preserves all selected original row IDs and seeds through its configuration digest. The 54,000 pool IDs are exactly the sorted complement of validation IDs in 0–59,999, with their little-endian int64 content digest recorded, avoiding a redundant list. The manifest content digest is `7e8591baa4d311e61f8dbe32e69c3b5dbb1e24ba14b7e6384e3eea8d373cc6cd`. Each entry is regenerated and compared during verification; A/B nesting, validation isolation and SI balance are also explicitly checked.

SHA-256 candidates followed by exact uint8 array equality found **zero duplicate-image groups and zero conflicting-label groups across all 70,000 original images**. Every audited pool/validation/test and A/B evaluation boundary had zero exact overlap. This is an exact-pixel audit; near duplicates and subject/product independence are not established. The full data audit (not included in this publication snapshot) records provenance and checks.

## Real-training-only feasibility checks

All **61 tests passed in 4.791 seconds** inside unittest, including seven added checks of malformed IDX, compressed integrity/size bounds, training-loader isolation, deterministic nested selection, manifest tampering and duplicate/conflict accounting.

The tiny-batch diagnostic selected the first five A rows per class before outcomes (50 images), used the declared CNN/optimizer, and checked training accuracy every 20 updates up to a fixed 400-update cap. It reached **48/50 correct (96%) at update 100**, with cross-entropy 0.2532. Runtime: 1.131 seconds. This establishes a small memorization sanity check; it is not validation accuracy, generalization evidence, or a completed study run. These weights were discarded.

Two fresh, throwaway seed-11 models then trained for two epochs on A and B. Their initial tensor digests matched; differently sized datasets have different minibatch orders, as specified in the protocol.

| Budget | Steps/epoch | Measured epoch 1 | Measured epoch 2 |
|---|---:|---:|---:|
| A, 3,000 images | 24 | 0.548205 s | 0.535218 s |
| B, 12,000 images | 94 | 2.171572 s | 2.057346 s |

Extrapolating these two epoch times to the declared six 30-epoch runs gives **239.055 seconds** of training, or **478.111 seconds with a 2x margin** (about eight minutes). This passes the 7,200-second training cap. It is a projection, not the measured completion time of six models.

On the discarded two-epoch B model, shared training-only SI timing on 600 examples was 1.256371 s for input (784 dimensions), 0.376379 s for pooled conv features (256), and 0.049193 s for penultimate features (32); feature extraction took 0.061223 s. Batch size 8 bounded distance intermediates. Scores are retained as diagnostic outputs, not primary/validation study results. The full SI schedule, 2,000-example sensitivity calculation, probes, baselines, checkpoints and interruptions remain outside the training projection. **The separate full measurement-cost gate is not yet certified.**

The real preflight invocation took 12.879 seconds after imports. It did not open test files, evaluate validation/test performance or save weights. See [actual preflight](preflight/real_preflight.json).

## Verification, packaging and reproduction

The read-only [verifier](../../scripts/verify_fashion_preflight.py) checks delivered source/config digests, raw training checksums, every selected ID, nested/balanced subsets, the original data-audit digest, tiny-batch gate and projection arithmetic. No model fitting/inference or test-file access occurs on this path.

Workspace-editor delivery changes JSON CRLF to LF; the long manifest is wrapped only between JSON encoder tokens. The parsed manifest is unchanged. The verifier regenerates its typed content digest and reconstructs exact original CRLF for the data-audit hash embedded in preflight, rejecting any other difference. No training result was changed or rerun for packaging.

Actual execution used the C: temporary cache. From the project directory, the equivalent commands with a writable cache are:

```powershell
$fashionCache = Join-Path ([System.IO.Path]::GetTempPath()) 'codex-si-audit-fashion'
& scripts/fetch_fashion.ps1 -OutputDirectory (Join-Path $fashionCache 'raw')
python -B -m unittest discover -s tests -v
python -B scripts/prepare_fashion_data.py --data-dir (Join-Path $fashionCache 'raw') --output-dir (Join-Path $fashionCache 'data_v1')
python -B scripts/preflight_fashion_real.py --data-dir (Join-Path $fashionCache 'raw') --audit-dir (Join-Path $fashionCache 'data_v1') --output (Join-Path $fashionCache 'real_preflight.json')
python -B scripts/verify_fashion_preflight.py --data-dir (Join-Path $fashionCache 'raw')
```

The acquisition command originally used the identical script under the workspace work/ folder; the delivered copy is in scripts/. The historical protocol/config and synthetic preflight retain their issue-time “no download” text; this report records the later completed data stage. The previous Optdigits evidence remains a separate completed study.

## Next milestone

Fit and report the six fixed input baselines on validation, measure baseline/probe costs, implement the resumable CNN experiment runner, and verify optimizer/data-order continuation. Freeze the implementation/split/config state before the main six-run training. No main CNN accuracy, Fashion-MNIST test performance, new environment installation, paid compute, Git initialization, staging, commit or push is claimed.
