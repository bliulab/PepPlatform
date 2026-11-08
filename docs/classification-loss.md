# Classification training losses

In step 3 of the Web wizard, choose **Custom** and expand **Classification loss (optional)**. The section starts folded. Classification and semantic auxiliary losses are configured independently.

| Route | Choices | Historical default |
| --- | --- | --- |
| Generic sequence networks, including pretrained residue backbones | BCE, WBCE, ASL | BCE |
| Pooled ESM2, ProtBert, ProtT5, ESMC/ESM3 features with an MLP | BCE, WBCE, ASL | WBCE, weight cap 20 |
| PPI, HLA-I and HLA-II neural networks | BCE, WBCE | WBCE, weight cap 50 |

Pure classical classifiers and ensembles of trained models do not use this selector. In embedding-export pipelines, it controls classification training of the neural representation stage. Contrastive-only pretraining retains its own objective; the selected classification loss applies to classification fine-tuning.

## CLI parameters

Single-sequence tasks retain `--multilabel_loss bce`, `balanced_bce`, and `asymmetric_focal`, displayed as BCE, WBCE, and ASL. `auto` preserves the route's historical default shown above.

- WBCE: `--multilabel_pos_weight_cap` defaults to 20; it must be finite and at least 1. Positive weights use training labels only, with the historical formula and minimum weight 1.
- ASL: `--asl_gamma_pos` defaults to 1 and `--asl_gamma_neg` to 4; both must be finite and nonnegative. `--asl_clip` defaults to 0.05 and must be in `[0, 1)`.
- Pair tasks retain `--loss_type bce|wbce`. `--pos_weight_max` defaults to 50 and must be finite and at least 1. Pair tasks do not add ASL.

Web configurations retain explicitly supplied overrides; cloning an old configuration does not add loss fields. Artifacts record requested and resolved classification losses separately from semantic metadata.

## Saved runs

Historical defaults are preserved. Exact reproduction still requires the run's frozen source, configuration, data, caches, weights, seed, and environment. Older pooled MLP implementations ignored explicit loss overrides; selecting BCE/ASL or a different weight cap with the extended implementation creates a new experiment. Existing models and saved runs are not rewritten.
