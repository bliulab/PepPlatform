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

Use Python 3.10 and choose the requirements file for your workflow:

| File | Includes |
| --- | --- |
| `requirements.txt` | Shared CLI training libraries and classical features |
| `requirements-web.txt` | Shared training libraries plus the Web backend and database drivers |
| `requirements-pretrained.txt` | Shared training libraries plus Transformers, SentencePiece, Protobuf and ESM |
| `requirements-dev.txt` | Web and pretrained dependencies plus pytest, httpx and Ruff |

Scientific-library versions match the reference training environment, including
NumPy `2.2.6`, SciPy `1.15.3`, scikit-learn `1.7.2`, XGBoost `3.1.3` and
LightGBM `4.6.0`. The tested PyTorch version is `2.5.1`; pretrained routes also use the matching
`torchvision==0.20.1`. Install CPU builds as shown below, or choose the matching
CUDA builds from the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/#v251).
Model weights, training data and BLAST reference databases are prepared separately.
The explicit versioned CLI also uses `requirements-web.txt` for its metadata and runner components.

## CLI quick start

Use **Python 3.10** and run commands from the repository root. See the
[installation guide](https://bliulab.github.io/pepplatform-docs/install/cli-only) for environment setup.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1"
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

For pretrained routes on CPU, add the matching torchvision build and extensions:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torchvision==0.20.1"
python -m pip install -r requirements-pretrained.txt
```

The reference GPU environment uses PyTorch `2.5.1+cu121` and torchvision
`0.20.1+cu121`. On a compatible NVIDIA host, replace the CPU install commands with:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1" "torchvision==0.20.1"
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

For Docker, copy the example configuration and edit `.env`:

```bash
cp .env.example .env
openssl rand -hex 32
```

Generate a separate value for each of `POSTGRES_PASSWORD`, `PEP_SESSION_SECRET`
and `PEP_LLM_KEY_ENCRYPTION_SECRET`; keep the latter two stable across restarts
and backups. Set both initial-admin fields to create your login account on first
startup. Configure SMTP for registration and password-reset emails. Then run:

```bash
docker compose up --build -d --wait --wait-timeout 300
```

Open <http://127.0.0.1:8080>. The stack uses PostgreSQL 16, Python 3.10, a Node.js 22
frontend build and Nginx 1.30. Database state, uploaded data, runs and downloaded
Hugging Face models persist in Docker volumes. The backend image includes CPU
PyTorch; GPU access needs separate image and device configuration. Local model
snapshots are supplied through an optional mount; see the Docker guide.

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

The repository includes a [Docker operations guide](docs/deployment.md) covering
first login, model mounts, updates, backups, recovery and troubleshooting.
Follow the [online Docker deployment guide](https://bliulab.github.io/pepplatform-docs/install/docker) or
[local web development guide](https://bliulab.github.io/pepplatform-docs/install/local-web) to get started.

## Development and tests

After installing the CPU PyTorch build above:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torchvision==0.20.1"
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
