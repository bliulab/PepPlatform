# ProtT5 single-file and paper-shard verification

[中文说明](prott5-equivalence.zh-CN.md)

On 2026-11-06, the pinned official ProtT5 checkpoint and the six original paper
shards passed a complete parameter comparison and the platform's feature-output
comparison. **Six shards are not a numerical requirement.** A separate loader RNG
check found different Torch random states, which matters when training a new
classifier. Model preparation now downloads the official file and automatically
restores the exact paper files locally, preserving their loading behavior.

The [parameter and feature report](prott5-equivalence-report.json) records every
tensor comparison, effective configuration, environment, source-file hashes and
output hashes. Its `PASS` covers checkpoint and feature equivalence; it does not
certify an identical training run.

## Choose a model for your purpose

For paper reproduction, use the prepared original-format model entry, a verified
original six-shard snapshot, or a verified offline package. Match the frozen
experiment configuration and environment as well as the model assets.

Independent experiments may use a compatible official ProtT5 snapshot unchanged.
Select its absolute directory with the training option `--prott5_model`; an
explicit path takes priority over installed local candidates. The preparation
option `--prott5-source` still performs restoration. Raw loading is a supported
choice for new experiments, while saved paper artifacts retain their model-identity
checks. See the [deployment guide](deployment.md#prepare-the-exact-paper-pretrained-assets)
for the two workflows and Docker paths.

## Checkpoint identities

| Item | Fixed identity |
| --- | --- |
| Official repository | `Rostlab/prot_t5_xl_uniref50` |
| Official revision | `973be27c52ee6474de9c945952a8008aeb2a1a73` |
| Official file | `pytorch_model.bin`, 11,275,562,724 bytes |
| Official file SHA-256 | `c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9` |
| Paper shard aggregate SHA-256 | `c1e7b7328276917e729bf11585ac6b4b7965aa33251edc882ee2fad7d47d307d` |
| Common canonical tensor fingerprint | `104950410ca044c20dd0dc2aa5515591cfdab660ffb2cff3e2de8975a5ee6315` |
| Accepted results | `function_policy_completion_20261022T0620Z_aupr_trapezoidal_v2` |

All 509 independent tensors, including the decoder, have identical names, shapes,
FP32 dtypes and bytes. The official state dictionary has 512 names because
`encoder.embed_tokens.weight`, `decoder.embed_tokens.weight` and `lm_head.weight`
alias `shared.weight`. Each alias was verified before deduplication.

| Parameter group | Independent tensors | Parameters | Bitwise equal |
| --- | ---: | ---: | --- |
| Shared embedding | 1 | 131,072 | Yes |
| Encoder excluding shared embedding | 194 | 1,208,010,752 | Yes |
| Decoder excluding shared embedding | 314 | 1,610,688,512 | Yes |

The raw configurations differ in serialization metadata and omitted defaults.
The platform explicitly loads `T5EncoderModel` for both formats. Their effective
configurations, loaded classes and loaded encoder states match. The tokenizer's
three source files are byte-identical.

## Actual platform outputs

The check used the original paper environment: Torch `2.5.1+cu121`, Transformers
`4.48.1`, Tokenizers `0.21.4`, SentencePiece `0.2.1`, Safetensors `0.7.0` and
Hugging Face Hub `0.36.0`. Execution used CPU, four threads, seed 42, deterministic
algorithms and Hugging Face offline flags.

The verifier audited the frozen registration, acceptance index, candidate metadata
and input hashes. It selected the first eight registered validation rows from each
of Function, PPI, MHC-I and HLA-II, yielding 56 sequence probes across seven streams.
The HLA-II peptide stream is an additional ProtT5 probe; that heterogeneous
candidate uses ESM2 for peptides. Seven additional probes cover normalization,
U/Z/O/B replacement, padding, truncation and the 511-residue boundary. Four invalid
inputs were also checked.

Across all **63 sequence probes in nine batches**, token IDs, attention and special
token masks, FP32 last hidden states, FP32 pooled embeddings, FP16 residue embeddings
and residue masks were bitwise equal. Maximum absolute output error was **0**.
This check did not retrain the 100 candidates or rerun their test metrics.

## Direct-file loading diagnostic and restoration

The [loader RNG diagnostic](../tools/verify_prott5_loader_rng.py) separately checks
Python, NumPy and Torch CPU RNG state after loading and a forward pass, then
initializes the registered Function CNN using its frozen metadata. It also compares
initialization after an explicit seed reset as a controlled diagnostic.

The original loader check found equal Python/NumPy states but different Torch CPU
states: the six-shard load advanced Torch's RNG, while the official single-file load
left its initial state unchanged. The training entry points set the seed before
loading the encoder. Consequently, matching pretrained weights and features alone
does not establish identical initialization for a subsequent neural classifier.
See the [classifier initialization report](prott5-loader-rng-report.json).

The registered CNN probe confirmed different initial weights with the same seed.
Explicitly resetting the seed before constructing each classifier made their states
identical in the controlled comparison:

| Check | Equal between formats |
| --- | --- |
| Loader Torch CPU RNG state | No |
| Registered CNN initialization with the original seed placement | No |
| CNN initialization after an explicit seed reset | Yes |

The local snapshot validator now recognizes the standard T5 default
`relative_attention_max_distance=128` when the official config omits it. Explicit
wrong values and missing XL dimensions remain rejected. Original model files and
configs were left untouched.

The preparation command accepts the pinned official single file as input and
restores every original file before training. `docker/models.lock.json`, the
artifact identity checks and the original seed placement remain unchanged.
Users download only the official source; the six-shard cache is generated locally.
The direct `.bin` loading diagnostic above remains valid and is retained as evidence.

## Automatic restoration audit

[The restoration report](prott5-restore-report.json) separately checks exact output
file identities, loaded encoder state, RNG after loading and inference, registered
CNN initialization, and three synthetic-data training steps. It also repeats the
63 sequence probes above. These checks target the prepared model entry; they do
not retrain the full candidate cohort. This full audit requires the registered
research assets. Normal model preparation needs neither the research directory
nor the original shards: the small restoration recipe ships with the source.

The restoration audit **passed** on 2026-11-06: all 13 files match the original
lock, every recorded RNG phase matches, all 26 CNN state tensors initialize
identically, and all three synthetic training losses and final weights match
bitwise. All 63 feature probes match with maximum absolute error 0. A container
with networking disabled also loaded and checked all six checkpoints successfully,
using the restored ProtT5 and read-only model mounts. Both repositories passed
96 related tests, and the previously exported model package remains compatible.

The earlier results above are the recorded CPU restoration checks. The subsequent
Docker acceptance downloaded all six checkpoints from an empty cache and passed
offline loading and ProtT5 comparisons in new GPU and CPU images: all 13 files
match the frozen hashes, with equal RNG states, 63-sequence features, CNN
initialization and three-step training within each device profile. Both repositories
passed 115 related tests; the [current runtime validation record](paper-runtime-validation.json)
records the measured environment and scope. Repeat an audit on CUDA by adding `--device cuda`
to the restoration verifier; CPU audits keep `--device cpu`.

## Repeat the checks

Use the frozen paper Python environment and retain the registered experimental
assets. Download only the fixed upstream revision to an isolated directory:

```bash
python - <<'PY'
import os
from huggingface_hub import hf_hub_download

for filename in [
    "config.json", "special_tokens_map.json", "spiece.model",
    "tokenizer_config.json", "pytorch_model.bin",
]:
    hf_hub_download(
        "Rostlab/prot_t5_xl_uniref50", filename=filename,
        revision="973be27c52ee6474de9c945952a8008aeb2a1a73",
        local_dir="backup/prott5-official-check",
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        token=False,
    )
PY

python tools/verify_prott5_equivalence.py \
  --official backup/prott5-official-check \
  --sharded /path/to/original/models/prott5 \
  --reference /path/to/PepPlatform/experiment_paper/current_results.json \
  --output backup/prott5-feature-verification --threads 4 --samples-per-task 8

python tools/verify_prott5_loader_rng.py \
  --official backup/prott5-official-check \
  --sharded /path/to/original/models/prott5 \
  --reference /path/to/PepPlatform/experiment_paper/current_results.json \
  --output backup/prott5-rng-verification --threads 4

python tools/verify_prott5_restore.py \
  --official backup/prott5-official-check \
  --sharded /path/to/original/models/prott5 \
  --restored /path/to/prepared/models/prott5 \
  --reference /path/to/PepPlatform/experiment_paper/current_results.json \
  --output backup/prott5-restoration-verification --threads 4
```

Set `HF_ENDPOINT=https://hf-mirror.com` for the download when needed. Both checks
verify the pinned official SHA-256 and original shard hashes before loading. They
read the source models without modifying them, require new output directories and
save JSON reports and source snapshots. The feature verifier also saves a Markdown
report and compared tensors. A diagnostic result of `DIFFERENT` returns exit code 1;
an execution failure is recorded as `ERROR`.
