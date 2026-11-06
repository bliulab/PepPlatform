# PepPlatform

English | [简体中文](README.zh-CN.md)

PepPlatform is a peptide modeling platform with a training CLI and a web
workspace for experiment management and model inference.

## Supported tasks

| Task | `--task_type` |
| --- | --- |
| Peptide function prediction | `pep_function` |
| Peptide–protein interaction | `pep_protein_interaction` |
| Peptide–HLA-I interaction | `pep_hlai` |
| Peptide–HLA-II interaction | `pep_hlaii` |

## Capabilities

- **Representations:** classical sequence features and pretrained protein language models.
- **Models:** traditional machine learning, sequence networks, and graph models.
- **Workflow:** training, validation-based model selection, evaluation, and model export.

## Python dependencies

Clone the source before following either installation path:

```bash
git clone https://github.com/bliulab/PepPlatform.git
cd PepPlatform
```

Use Python 3.10.19 and choose the requirements file for your workflow:

| File | Includes |
| --- | --- |
| `requirements.txt` | Shared CLI training libraries and classical features |
| `requirements-web.txt` | Shared training libraries plus the Web backend and database drivers |
| `requirements-pretrained.txt` | Shared training libraries plus Transformers, SentencePiece, Protobuf and ESM |
| `requirements-dev.txt` | Web and pretrained dependencies plus pytest, httpx and Ruff |

Scientific-library versions match the reference training environment, including
NumPy `2.2.6`, SciPy `1.15.3`, scikit-learn `1.7.2`, XGBoost `3.1.3` and
LightGBM `4.6.0`. The tested PyTorch version is `2.5.1`; pretrained routes also use the matching
`torchvision==0.20.1`. The default is the paper's CUDA 12.1 build; explicit CPU
installation commands are provided below. See the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/#v251).
Model weights, training data and BLAST reference databases are prepared separately.
The explicit versioned CLI also uses `requirements-web.txt` for its metadata and runner components.

## CLI quick start

Use **Python 3.10.19** and run commands from the repository root. See the
[installation guide](https://bliulab.github.io/pepplatform-docs/install/cli-only) for environment setup.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
python -m pip install -r requirements.txt
```

Prepare your data using the [dataset formats](https://bliulab.github.io/pepplatform-docs/datasets/format).
For single-sequence tasks, each split contains `seqs.fasta` and `labels.csv`.
Replace `/path/to/sequence-data` below with your dataset root directory.
Train a peptide-function model with k-mer / one-hot features:

```bash
python scripts/train_sequence_pipeline.py \
  --task_type pep_function --data_dir /path/to/sequence-data \
  --tokenizer kmer --encoder onehot --model cnn \
  --epochs 10 --artifact_dir output/pep-function
```

The command saves the model, configuration, and evaluation results under
`output/pep-function/`.

For peptide–protein, HLA-I, and HLA-II interaction tasks, use the pair entry point:

```bash
python scripts/train_pair_pipeline.py --help
```

See the [CLI reference](https://bliulab.github.io/pepplatform-docs/cli/overview) and
[training recipes](https://bliulab.github.io/pepplatform-docs/cli/recipes) for options and examples.

Training losses are displayed as **BCE**, **WBCE** (positive-class-weighted BCE), and **ASL**. Single-sequence CLI values remain `--multilabel_loss bce`, `balanced_bce`, and `asymmetric_focal`. `auto` preserves BCE for generic sequence networks and WBCE (weight cap 20) for pretrained pooled MLPs. Pair tasks retain `--loss_type bce|wbce`, defaulting to WBCE with a weight cap of 50. Historical candidate suffixes BalBCE and AFL mean WBCE and ASL.

The Web wizard exposes these choices in step 3, **Custom → Classification loss (optional)**, folded by default. See [classification training losses](docs/classification-loss.md) for parameters and historical compatibility.

The unified local CLI supports training, inference, local ensembles, and run history without a running Web service:

```bash
python -m pepplatform_cli
python -m pepplatform_cli --help
```


## Data and models

Datasets and pretrained weights are supplied separately. The example above uses
classical features and requires no pretrained weights. Pretrained routes need
the corresponding model snapshots; see the installation guide for setup.
Synthetic examples are intended only for workflow checks.
Data preparation tools are described in the [public dataset CLI](https://bliulab.github.io/pepplatform-docs/datasets/public-databases-cli).
PSSM/PSFM generation additionally needs BLAST+ (`ncbi-blast+`) and the route's reference data.

Dataset and model paths can be supplied through the CLI or Web configuration;
relative defaults resolve against the project directory. Web runtime storage can
be moved with `PEP_RUNTIME_ROOT`. Saved protocol/schema versions identify artifact
formats, and dated database migration IDs identify the upgrade chain. These
identifiers support compatibility; they do not select a private experiment run
or require a dated experiment directory.

For pretrained routes, use the paper's default CUDA 12.1 build and extensions:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1" "torchvision==0.20.1"
python -m pip install -r requirements-pretrained.txt
```

The expected versions are Torch `2.5.1+cu121`, torchvision `0.20.1+cu121`, CUDA
`12.1` and cuDNN `9.1`. For a CPU-only installation, choose the CPU wheels explicitly:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" "torchvision==0.20.1"
python -m pip install -r requirements-pretrained.txt
```

Then install the requirements layer you need. CPU and GPU builds can produce
numerical differences; reproducing a saved run also requires its recorded data,
configuration, seeds and model resources.

## Web interface

The web workspace provides dataset and experiment management, result comparison,
model inference, ensembling, and AI-assisted experiment planning.

New Web dataset uploads have the following default limits (1 MiB = 1,048,576 bytes):

| Environment variable | Default |
| --- | --- |
| `PEP_DATASET_UPLOAD_MAX_ARCHIVE_MIB` | 64 MiB ZIP file |
| `PEP_DATASET_UPLOAD_MAX_EXTRACTED_MIB` | 512 MiB total extracted data |
| `PEP_DATASET_UPLOAD_MAX_FILE_MIB` | 256 MiB per extracted file |
| `PEP_DATASET_UPLOAD_MAX_ENTRIES` | 1,000 files and directories |

Set positive integer values in `.env` and restart the backend to adjust these limits.
The request body allows 1 MiB beyond the ZIP limit for multipart form fields.
Uploads exceeding a limit return HTTP 413; damaged or unsafe ZIP archives return
HTTP 400. ZIP entries must use standard Deflate compression or be uncompressed
(Stored). Failed uploads are removed. Compression ratio alone does not reject an
archive. These limits apply to new Web uploads; CLI runs, built-in and existing
datasets, training settings and semantic-layer calculations keep their behavior.
When raising the ZIP limit, also check your reverse proxy's request limit; the
supplied Nginx configuration permits at most 512 MiB per request.

Install `requirements-web.txt` for the local backend, and additionally install
`requirements-pretrained.txt` when using pretrained routes. The supplied Docker
backend installs both. Frontend dependencies are installed separately with
`npm ci` inside `frontend/`.

For Docker, choose the complete guide for your host before running configuration and startup commands:

| Profile | Host requirements | Guide |
| --- | --- | --- |
| CPU | No NVIDIA GPU, or intentional CPU execution | [CPU deployment and operations](docs/deployment-cpu.md) |
| GPU | Compatible NVIDIA GPU, driver and NVIDIA Container Toolkit | [GPU deployment and operations](docs/deployment-gpu.md) |

Both guides cover `.env`, first login, model preparation, updates, backup/restore and troubleshooting.
Every CPU Compose operation retains both configuration files; the GPU guide uses the default CUDA 12.1
profile. For an existing service, use its update section and retain secrets, the project name and volumes.

The default address is <http://127.0.0.1:8080>. The stack uses PostgreSQL 16, Python 3.10, a Node.js 22
frontend build and Nginx 1.30. Database state, uploads, run artifacts and model caches persist in Docker
volumes. **Healthy containers do not establish pretrained-model readiness**; start with the basic
verification route that needs no pretrained weights.

For fixed paper assets, follow [CPU model preparation](docs/deployment-cpu.md#optional-prepare-all-six-pretrained-checkpoints)
or [GPU model preparation](docs/deployment-gpu.md#optional-prepare-all-six-pretrained-checkpoints).
Five model families require six checkpoints. ProtT5 restores the paper format from the pinned official
single file. Installed files total about 22.68 GB; current online preparation retaining source caches
uses about 45.37 GB, excluding overhead. The [shared reference](docs/deployment.md#prepare-the-exact-paper-pretrained-assets)
covers checkpoint identities, disk budgets, download sources and both ProtT5 workflows.

This preparation workflow targets paper reproduction. For your own experiments,
you may use a compatible official ProtT5 snapshot unchanged by setting the training
option `--prott5_model` to its absolute directory. The guide distinguishes this
option from the preparation script's `--prott5-source`, which still restores files.

For local development, use the Python environment above and install the Web layer.
In one terminal, configure the AI-key encryption secret and start the backend:

```bash
python -m pip install -r requirements-web.txt
export PEP_LLM_KEY_ENCRYPTION_SECRET='replace-with-a-stable-random-value-at-least-16-characters'
python -m uvicorn pepplatform.web.app:app --host 127.0.0.1 --port 8081
```

In another terminal, use Node.js 22 and run `cd frontend`, `npm ci`, then
`npm run dev`. Open <http://127.0.0.1:5173>. The backend loads a project `.env`
if present; configure the initial administrator or SMTP there for sign-in.

The [Docker deployment selector and shared reference](docs/deployment.md) links both hardware guides
and explains shared model and configuration details.
Follow the [online Docker deployment guide](https://bliulab.github.io/pepplatform-docs/install/docker) or
[local web development guide](https://bliulab.github.io/pepplatform-docs/install/local-web) to get started.

## Development and tests

For CPU development/tests, install the CPU Torch and torchvision pair explicitly:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" "torchvision==0.20.1"
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check --select E9,F63,F7,F82 .
```

Use `python -m ruff check .` for the full lint report, which also includes existing
style findings. Frontend checks run separately with `npm run test:unit` and
`npm run build` inside `frontend/`.

## Documentation

[English guides](https://bliulab.github.io/pepplatform-docs/quick-start/) · [中文手册](https://bliulab.github.io/pepplatform-docs/zh/quick-start/)

Documentation source is maintained in the separate [pepplatform-docs repository](https://github.com/bliulab/pepplatform-docs).

## License

[MIT](LICENSE). Dataset and pretrained-model licenses are determined by their respective providers.
