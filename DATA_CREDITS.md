# Data, method and licensing credits

The project is an implementation and empirical audit of a published measure. It does not claim a new index, copied author code, a paper's exact experimental reproduction, or pretrained third-party weights.

| Component | Source and retained evidence | Scope |
|---|---|---|
| Separation Index definition | [Kalhor et al., 2019, v1](https://arxiv.org/abs/1906.05156v1), Section 2.2, equation (1); [implementation](src/si_audit/metrics.py) | Published scientific definition, independently implemented in this project. Exact-distance ties and persistent IDs are explicit implementation conventions. |
| Optdigits | Alpaydin and Kaynak (1998), [UCI Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49), CC BY 4.0 as recorded in the [dataset audit](results/data_audit.json) and original access review | Original train/test files; raw files remain outside the project. Retained figures include diagnostic image examples. Give dataset credit when redistributing derived figures/results. |
| Fashion-MNIST | [Official Zalando Research repository](https://github.com/zalandoresearch/fashion-mnist), pinned revision b2617bb6d3ffa2e429640350f613e3291e10b141; audit (not included in this publication snapshot) | Official four compressed files, with published checksums verified. Raw images remain outside the project; diagnostic examples occur in saved error galleries. |
| Fashion-MNIST notice | [Retained MIT notice](LICENSES/Fashion-MNIST.txt) | This is the dataset/repository notice; it does not automatically choose a license for this project's newly generated code. |

No root project-code license has been selected. Choosing and approving that license, and confirming the intended copyright attribution, remain publication decisions. This document does not grant new permissions or replace any upstream notice.

Do not include raw datasets, temporary optimizer checkpoints, credentials or application attachments in the source repository. The twelve small project-generated model bundles can be reviewed as separate supplementary assets; their preparation does not authorize uploading them. See publication plan (not included in this publication snapshot).
