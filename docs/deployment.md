# Docker deployment and operations

[简体中文](deployment.zh-CN.md) · [Online manual](https://bliulab.github.io/pepplatform-docs/install/docker)

## Run your own Web service

Our public service has limited storage and computing resources, so registration, training, inference and model allowances apply. If you have suitable resources, we encourage you to host your own Web service. You control users, quotas, model protection and retention, announcements and maintenance schedules. Choose the CPU or GPU guide for your hardware, then configure [resource policies](resource-governance.md) in the admin console.

## First deployment

Choose a complete guide for your host hardware. Each covers configuration, first login, basic
training verification, optional model preparation, updates, backup/restore and troubleshooting.
For an existing deployment, start with the selected guide's update section.

| Deployment | Host requirements | Complete guide |
| --- | --- | --- |
| CPU | No NVIDIA GPU, or intentional CPU execution | [CPU deployment and operations](deployment-cpu.md) |
| GPU | Compatible NVIDIA GPU, driver and NVIDIA Container Toolkit | [GPU deployment and operations](deployment-gpu.md) |

This page holds shared configuration and model explanations. Deployment and operations Compose
commands live in the hardware guides, so shared examples do not require manual hardware substitutions.
Both profiles use the same volumes and fixed model assets. The project supports Linux amd64,
Docker Engine and Docker Compose v2.35 or newer. This is the project's supported baseline.

Core dependency baseline:

| Component | Default paper GPU profile | Explicit CPU profile |
| --- | --- | --- |
| Python | 3.10.19 | 3.10.19 |
| Torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 | 2.5.1+cpu / 0.20.1+cpu |
| CUDA / cuDNN | 12.1 / 9.1 (90100) | Not used |
| Transformers / Tokenizers | 4.48.1 / 0.21.4 | Same |
| ESM / SentencePiece | 3.2.0 / 0.2.1 | Same |
| Hugging Face Hub / Safetensors | 0.36.0 / 0.7.0 | Same |

Service health, usable pretrained assets and numerical paper reproduction require separate checks.
Web routes without pretrained weights can run first; pretrained routes need their selected models.
Full reproduction also requires the source, splits, configurations, seeds and hardware to match.

### CPU-only deployment

Use the [complete CPU guide](deployment-cpu.md). Every operation retains both the base file and CPU
override, including model preparation, updates, backup/restore and optional proxies. The override
selects CPU wheels and clears GPU requests, without NVIDIA Container Toolkit. CPU execution is a
compatibility option; numerical equality with paper GPU training is not asserted.

### GPU deployment

Use the [complete GPU guide](deployment-gpu.md). The host needs a compatible NVIDIA GPU/driver and
configured [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
Builds check dependency versions; a separate runtime check verifies actual CUDA execution. Rebuild
the backend when switching CPU/GPU profiles.

### Download sources and mirrors in China

Configure each download layer separately:

| Download layer | Configuration |
| --- | --- |
| Docker base images | Docker daemon proxy or registry mirror on the host |
| Python packages and CUDA dependencies | Build setting `PEP_PIP_INDEX_URL` |
| Torch and torchvision wheels | Build setting `PEP_TORCH_INDEX_BASE_URL` |
| Pretrained checkpoint files | Runtime setting `HF_ENDPOINT` |

For a server in China, explicitly select these sources in the local `.env`:

```dotenv
PEP_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PEP_TORCH_INDEX_BASE_URL=https://mirrors.nju.edu.cn/pytorch/whl
PEP_PIP_TIMEOUT=120
HF_ENDPOINT=https://hf-mirror.com
```

Then use the normal build and model-preparation commands. The Torch base URL
must end at `whl`, without a channel suffix: the GPU profile appends `/cu121`
and the CPU profile appends `/cpu`. The Dockerfile downloads only the fixed
Torch/torchvision wheels from that channel, then installs their dependencies
from the selected PyPI index. It checks the installed Python, Torch, CUDA and
other core versions before the image build succeeds. See the
[TUNA PyPI guide](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) and the
[NJU PyTorch index](https://mirrors.nju.edu.cn/pytorch/whl/cu121/torch/).

The defaults remain the official package and Hugging Face endpoints. You can
select domestic package mirrors while keeping `HF_ENDPOINT=https://huggingface.co`.
A Hugging Face mirror can redirect large files to the original CDN, so test the
actual file transfer rather than assuming its hostname guarantees a faster path.
Your `HF_TOKEN` is sent to the chosen endpoint; gated models need authorization,
and a mirror may not contain every pinned revision. All downloaded model files
must pass the same frozen hashes, including the exact ProtT5 restoration checks.

Changing a Python/wheel source requires rebuilding the backend. Changing
`HF_ENDPOINT` applies to the next preparation container. Docker daemon registry
or proxy settings are separate host configuration; setting a pip/Hugging Face
mirror does not fix base-image pulls.

## Prepare the exact paper pretrained assets

### Choose your use case

| Goal | ProtT5 model to use |
| --- | --- |
| Reproduce the paper's ProtT5 training experiments | Use the preparation command below, an already verified original six-shard snapshot, or a verified offline model package. Train with the original-format model entry. |
| Run your own experiments or extract features | You may use a compatible official ProtT5 snapshot unchanged. Select your own model directory explicitly; restoration is optional for this use. |

The preparation command is the recommended way to obtain the paper's fixed assets.
Platform startup and independent experiments do not require this step. Reproducing
paper results also requires the matching source, data splits, configuration, seeds
and runtime environment described below.

`docker/models.lock.json` pins the resources used by the 100 accepted candidates across
all four tasks, including their ensemble members. Five model families require **six
checkpoints**, because the candidates used both ESM-C sizes:

| Resource | Fixed revision (file SHA-256 values and content fingerprints are in the lock) |
| --- | --- |
| ESM2 150M | `facebook/esm2_t30_150M_UR50D` at `a695f6045e2e32885fa60af20c13cb35398ce30c` |
| ProtBERT | `Rostlab/prot_bert` at `7a894481acdc12202f0a415dd567f6cfdb698908` |
| ProtT5-XL | Official single file at `973be27c52ee6474de9c945952a8008aeb2a1a73`; automatically restored to the original six shards |
| ESM-C 300M | `EvolutionaryScale/esmc-300m-2024-12` at `7f10b20ae75017b2dbc884070e03434515709a8d` |
| ESM-C 600M | `EvolutionaryScale/esmc-600m-2024-12` at `e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c` |
| ESM3 Open Small | `EvolutionaryScale/esm3-sm-open-v1` at `47f0545b2b6daf26a93439a3cd610f4f7f3d5478` |

ProtT5 preparation downloads the fixed official `.bin` file and uses the small
`docker/prott5.restore.json` recipe to reconstruct the exact paper files locally.
Every restored file must match the existing lock. This preserves the historical
loader and its RNG behavior; users do not need an original shard package.
The full checkpoint has about 3B parameters; the platform loads its approximately
1.2B encoder. See the [verification and restoration checks](prott5-equivalence.md).

Weights remain outside Git and the application image. Installed files total 22.68 GB (21.13 GiB).
Current online preparation retains source caches, bringing the file total to about 45.37 GB
(42.25 GiB), excluding additional overhead. See [disk budgets and space-saving paths](#disk-space-for-model-downloads).

### Download and prepare on the target server

Follow the complete hardware-specific steps: [CPU online preparation](deployment-cpu.md#online-preparation) ·
[GPU online preparation](deployment-gpu.md#online-preparation). Both guides also show how to reuse an
official ProtT5 snapshot. `--prott5-source` skips that download but still restores the paper format.
Preparation neither starts the database nor applies migrations, and reuses verified installations
and source caches on repeated runs.

For a bare Python installation, run `python docker/download_models.py --root backup/pretrained`
and set the training option `--prott5_model` to the absolute `backup/pretrained/models/prott5` path.
Use the prepared entry for paper training; directly loading the official `.bin` has different RNG behavior.

### Independent experiments: use an official ProtT5 snapshot unchanged

For your own training or feature extraction, you may keep a compatible official
ProtT5 snapshot unchanged, including its configuration, tokenizer and weights.
Add this option to your CLI training command to select the raw snapshot:

```text
--prott5_model /opt/prott5-official
```

An explicit absolute path takes priority over the installed paper-model cache.
Supplying only a repository name may still resolve to an existing local model.
In Docker, mount your download directory read-only at `/opt/prott5-official` and
use that container path in the training command. On a bare installation, use the
host directory's absolute path.

| Option | Where it is used | Behavior |
| --- | --- | --- |
| `--prott5-source DIR` | `docker/download_models.py` | Verifies a fixed official snapshot and restores the paper format before installation. |
| `--prott5_model DIR` | CLI training commands | Loads the selected compatible snapshot directly; does not run the restoration step. |

Direct `.bin` loading has different RNG behavior from the paper's original shards,
so this independent-experiment choice does not establish reproduction of the old
training results. Loading saved paper artifacts still requires a checkpoint that
matches their recorded model identity. The preparation command is not needed to
train a new model with your own compatible snapshot.

### Optional: export a portable package on the original server

Run from the updated source checkout, pointing `--reference` at the original research
registration. The tool verifies registration, acceptance index, candidate metadata and
actual model content before copying original files into a new output directory:

```bash
python3 docker/download_models.py --export backup/pretrained-models \
  --reference ../PepPlatform/experiment_paper/current_results.json
```

Copy the entire `backup/pretrained-models` directory to the same relative path on the
target server. It includes the lock, receipt and model files; cache symlinks become
ordinary files. The research assets stay untouched. The package need not be uploaded
to GitHub. Existing export directories are never overwritten.

### Optional: import a portable package and start

Complete import and device-check commands: [CPU offline import](deployment-cpu.md#offline-import) ·
[GPU offline import](deployment-gpu.md#offline-import). Import verifies every file and succeeds only
when all six pass. Models live under `/var/lib/pepplatform/pretrained` on `pep_runtime`, linked from
`/app/models` in the image. Runtime-volume backups include them; retain external packages separately too.

### Official endpoint, mirrors and offline mode

Downloads request only the fixed files listed by the lock and the pinned ProtT5
restoration recipe. ProtT5 conversion is deterministic and its output must match
the original lock; no alternate training checkpoint, version or endpoint is selected
automatically. The unchanged model lock also accepts previously exported packages.
The default `.env` setting is `HF_ENDPOINT=https://huggingface.co`. A deployment with
poor connectivity to the official endpoint can explicitly select a third-party mirror,
for example `HF_ENDPOINT=https://hf-mirror.com`. `HF_TOKEN` is sent to the selected
endpoint. Gated assets still require provider authorization; mirrors may lack some
fixed revisions. If a mirror does not support Xet, set `HF_HUB_DISABLE_XET=1`.
`HF_HUB_DOWNLOAD_TIMEOUT` and `HF_HUB_ETAG_TIMEOUT` configure timeouts.

After all six assets pass verification, set `HF_HUB_OFFLINE=1` and
`PEPPLATFORM_OFFLINE_MODE=1` in `.env` and recreate the backend for offline operation.
Package imports and `--check-only` always work without network access. Normal platform
startup does not prepare all models automatically. Existing optional mounts below
remain supported.

Identical model files are one reproducibility condition. The lock also records the
paper's reference dependencies. Full numerical reproduction additionally needs the
frozen source, splits, configurations, seeds and original CPU/CUDA environment. A short
CPU/GPU model check verifies usable assets on that device; it does not retrain the cohort.
The 2026-11-06 Docker acceptance completed all-six-checkpoint cold downloads,
offline GPU/CPU loading, and ProtT5 RNG, 63-sequence feature and three-step synthetic
training comparisons. See the [current runtime validation record](paper-runtime-validation.json)
for the measured image versions, network configuration and acceptance scope.

## Dataset upload limits

New Web dataset uploads have the following default limits (1 MiB = 1,048,576 bytes):

| Environment variable | Default |
| --- | --- |
| `PEP_DATASET_UPLOAD_MAX_ARCHIVE_MIB` | 64 MiB ZIP file |
| `PEP_DATASET_UPLOAD_MAX_EXTRACTED_MIB` | 512 MiB total extracted data |
| `PEP_DATASET_UPLOAD_MAX_FILE_MIB` | 256 MiB per extracted file |
| `PEP_DATASET_UPLOAD_MAX_ENTRIES` | 1,000 files and directories |

Set positive integer values in `.env` to adjust these limits. Restart a local backend after
editing it; for Compose, recreate services using the selected hardware guide's
environment-application command. A plain Compose restart does not reload `.env`.
The request body allows 1 MiB beyond the ZIP limit for multipart form fields.
Uploads exceeding a limit return HTTP 413; damaged or unsafe ZIP archives return
HTTP 400. ZIP entries must use standard Deflate compression or be uncompressed
(Stored). Failed uploads are removed. Compression ratio alone does not reject an
archive. These limits apply to new Web uploads; CLI runs, built-in and existing
datasets, training settings and semantic-layer calculations keep their behavior.
When raising the ZIP limit, also check your reverse proxy's request limit; the
supplied Nginx configuration permits at most 512 MiB per request.

## Pretrained models and SMTP

Download caches persist on the runtime volume under `HF_HOME=/var/lib/pepplatform/huggingface`.
They are independent files from the installed copies in `pretrained`, which can load locally after installation.

For existing local snapshots, create an optional `compose.models.yml`:

```yaml
services:
  backend:
    volumes:
      - ./models:/app/models:ro
    environment:
      PEPPLATFORM_OFFLINE_MODE: "1"
```

Place complete snapshots in the selected route's directory, such as `models/esm2`, `models/protbert`,
`models/prott5` or `models/EvolutionaryScale/esmc-600m-2024-12`. This mount replaces the image's default
`/app/models` entry, so ensure the mounted directory contains every model you need. Mounting alone
neither downloads models nor grants gated access. See the [CPU override examples](deployment-cpu.md#optional-configuration-and-overrides)
or [GPU override examples](deployment-gpu.md#optional-configuration-and-overrides) for startup commands;
retain persistent overrides on later operations. Remove offline settings for intended downloads,
and follow provider licenses and authorization requirements.

For email, set `PEP_SMTP_HOST`, `PEP_SMTP_PORT`, `PEP_SMTP_USER`, `PEP_SMTP_PASSWORD` and `PEP_SMTP_FROM`.
Port 587 normally uses STARTTLS; 465 selects implicit TLS. The sender should match the authenticated
mailbox; QQ Mail requires the mailbox authorization code. After editing `.env`, recreate services
using the selected hardware guide's environment-application command. A plain restart does not reload it.

## Local Web development

Use Python 3.10.19 and Node.js 22, and run backend commands from the repository root.
Install the paper's CUDA 12.1 PyTorch build by default:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
python -m pip install -r requirements-web.txt
```

For pretrained routes, also install the matching `torchvision==0.20.1` build and
`requirements-pretrained.txt`. For CPU-only development, select
`https://download.pytorch.org/whl/cpu` explicitly for both Torch and torchvision.

The backend loads `.env` from the project root. Configure the initial administrator or SMTP
for sign-in; the [CPU environment setup](deployment-cpu.md#new-installation-configure-the-environment)
and [GPU environment setup](deployment-gpu.md#new-installation-configure-the-environment)
describe the fields. Without `PEP_DATABASE_URL`, local metadata is stored in
`.pepplatform_web/metadata.sqlite3`. `PEP_RUNTIME_ROOT` changes the Web runtime storage root.
Saved protocol/schema versions identify artifact formats, and dated database migration IDs
identify the upgrade chain. These identifiers support compatibility; they do not select a
private experiment run or require a dated experiment directory.

Set `PEP_LLM_KEY_ENCRYPTION_SECRET` to a stable value of at least 16 characters before startup.
Generate it once with `openssl rand -hex 32` and save it in `.env`. AI provider keys saved by
users are encrypted with this secret. Keep it stable across restarts and backups; rotating it
without re-encrypting saved keys makes existing provider configurations unreadable.
If `.env` already contains the secret, the following export is unnecessary:

```bash
export PEP_LLM_KEY_ENCRYPTION_SECRET='replace-with-a-stable-random-value-at-least-16-characters'
python -m uvicorn pepplatform.web.app:app --reload --reload-dir pepplatform --reload-dir scripts --host 127.0.0.1 --port 8081
```

Scope `--reload` to `pepplatform` and `scripts`. A bare `--reload` watches staged source
snapshots under `.pepplatform_web/versions/staging/`, restarts the backend and cancels active
experiments. Omit `--reload` entirely when a training run must survive.

Start the frontend in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. If the backend uses another port, copy
`frontend/.env.example` to `frontend/.env.local`, set `VITE_API_TARGET` to the backend
address (for example, `http://127.0.0.1:8000`), then restart `npm run dev`.

Authentication is enabled by default. For local HTTP, leave `PEP_COOKIE_SECURE` unset or
set it to `false`; its default follows the `PEP_PUBLIC_BASE_URL` scheme. Use
`PEP_COOKIE_SECURE=true` behind HTTPS. For LAN testing, substitute your host's LAN address:

```bash
PEP_PUBLIC_BASE_URL=http://192.168.1.10:5173 \
PEP_COOKIE_SECURE=false \
python -m uvicorn pepplatform.web.app:app --reload --reload-dir pepplatform --reload-dir scripts --host 0.0.0.0 --port 8081
```

To temporarily run without login during development:

```bash
PEP_AUTH_ENABLED=0 python -m uvicorn pepplatform.web.app:app --reload --reload-dir pepplatform --reload-dir scripts --host 127.0.0.1 --port 8081
```

### Optional Agent web search

Agent web search uses a server-side Tavily Free/Researcher key. Set `PEP_TAVILY_API_KEY`
without enabling a paid or pay-as-you-go plan. The backend verifies the free plan and
remaining quota before every cached search window, uses one Basic Search credit per enabled
turn, and falls back to a normal offline answer when search is unavailable.

## Accounts and saved passwords

Regular users and administrators use the same password and session mechanisms.

| Feature | Behavior |
| --- | --- |
| Remember email | After a successful sign-in, stores the email and checkbox state in this browser's localStorage. Clearing it and signing in removes those saved values. |
| Browser password manager | Chrome, Edge or Firefox decides whether to offer to save, fill or update credentials, according to its own settings. |
| Login session | The `pep_session` Cookie keeps the account signed in for 14 days by default. Signing out revokes the current session. |
| Server password storage | The database stores an Argon2 password hash for verification. |

Remember email does not store a password or change the session lifetime, and its checkbox does not
control the browser's password manager. The authentication forms identify the account and existing
or new password to help browsers recognize them; a save prompt still depends on the browser.

If a credential is missing, use the same site address, including scheme, host and port, and check
the browser's password-saving setting and excluded sites. You can add the account in its password
manager. If the remembered email and a password suggestion name different accounts, choose the
intended saved account and check both fields before signing in. After a password reset, update
the browser's saved credential when prompted.

## Updates and persistence

Keep the checkout directory, Compose project name, three secrets and persistent overrides stable.
`postgres_data` stores the database; `pep_runtime` stores uploads, training artifacts, versioned inputs,
installed pretrained models and Hugging Face downloads. A backend restart cancels active experiments
in this single-backend deployment.

Complete upgrade steps: [CPU updates](deployment-cpu.md#updates-and-persistence) · [GPU updates](deployment-gpu.md#updates-and-persistence).

### Updating a clone after history consolidation

The public repository's history has been consolidated into one root commit titled `Initial commit`.
A clone based on the earlier history cannot use `git pull --ff-only` for its first update.
Complete a database/runtime backup using the instructions bundled with the current deployment,
preserve `.env` and persistent Compose overrides, and save local source edits before proceeding.

Run the following from the existing checkout on `main`. The first checks require clean tracked
files and index; the local backup branch retains the old commits. Untracked configuration remains
in place, and `--keep` stops if an update would overwrite conflicting local changes.

```bash
set -e
test "$(git branch --show-current)" = main
git diff --quiet
git diff --cached --quiet
migration_stamp=$(date -u +%Y%m%dT%H%M%SZ)
git branch "backup/pre-initial-$migration_stamp"
git fetch origin
git reset --keep origin/main
git branch --set-upstream-to=origin/main main
```

If a check stops, preserve the reported local changes and resolve the conflict before retrying.
Keep the backup branch local. These Git commands update source only; they do not restart services
or migrate the database. Then follow the normal CPU or GPU update sequence. Later updates use
`git pull --ff-only` as usual. Retain the old source backup with any deployment or experiment
records that reference its commit.

### Import legacy SQLite metadata

To import legacy local metadata into PostgreSQL, back up the source and target databases,
then run from the repository root with the target URL and administrator credentials set:

```bash
python scripts/import_legacy_sqlite.py \
  --source .pepplatform_web/metadata.sqlite3 \
  --target "$PEP_DATABASE_URL" \
  --admin-email "$PEP_INITIAL_ADMIN_EMAIL" \
  --admin-password "$PEP_INITIAL_ADMIN_PASSWORD"
```

This imports metadata and assigns imported records to the specified administrator. It does
not copy dataset or model files; retain the files referenced by the imported paths.

## Backup and restore

Finish training, stop both application services and leave the database running for a consistent
database/runtime backup. Retain environment files, overrides, source revision and external models.
Restore matching source and empty target volumes. Complete steps:
[CPU backup/restore](deployment-cpu.md#backup-and-restore) · [GPU backup/restore](deployment-gpu.md#backup-and-restore).

For a failed upgrade, restore the database and runtime backup with matching source/images;
reverting code alone does not revert database migrations.

## Troubleshooting

Start with the [CPU troubleshooting commands](deployment-cpu.md#troubleshooting) or
[GPU troubleshooting commands](deployment-gpu.md#troubleshooting). Docker base images, build
dependencies and runtime model downloads use separate network configuration.

If a Linux host proxy listens only on `127.0.0.1`, a build container cannot reach it through its own
loopback interface. With `http_proxy` / `https_proxy` set in the host shell, create a local
`compose.build-network.yml`:

```yaml
services:
  backend:
    build:
      network: host
  frontend:
    build:
      network: host
```

Build commands are in each hardware guide's optional-configuration section. This override changes
build networking only, retaining normal runtime Compose networking. Keep proxy addresses and
credentials out of Dockerfiles and Git.

### Disk space for model downloads

Distinguish the final installation, retained source files during preparation and business data.
The following sums file sizes from [`models.lock.json`](../docker/models.lock.json) and the
[ProtT5 restoration recipe](../docker/prott5.restore.json). GB is decimal; GiB is binary:

| Content or path | Model file size |
| --- | ---: |
| Six installed checkpoints | 22.68 GB / 21.13 GiB |
| Hugging Face source cache retained by online preparation | 22.68 GB / 21.13 GiB |
| Current online flow: source cache plus installed copies | 45.37 GB / 42.25 GiB |
| Offline package on another disk: installation on the target volume | 22.68 GB / 21.13 GiB |
| Offline package and installed files on the same disk | 45.37 GB / 42.25 GiB |

The script copies cached downloads into independent installed files. Verified staging directories
are activated by rename, without retaining another full copy on a normal first installation.
The official ProtT5 `.bin` and restored shards each occupy about 11.28 GB and are already included
in this table; do not add them again. These normal-flow file budgets exclude extra Xet caches,
failed downloads, corrupt installations preserved as `.previous-*` and other models. Leave additional
headroom. A 100 GB allocation is a generous example for extra caches and repair copies, **not a universal minimum**.

Budget separately for Docker images/build caches, the database, uploads, training artifacts,
additional model packages and backups. A full runtime backup includes installed files and caches,
and model weights may compress poorly.

Existing ways to save space:

- Put an offline package on another disk, so import writes only installed copies to the target volume.
  Image build dependencies still need separate preparation.
- After all six installations and device checks pass, use [CPU cache cleanup](deployment-cpu.md#optional-reclaim-model-download-caches)
  or [GPU cache cleanup](deployment-gpu.md#optional-reclaim-model-download-caches), then verify offline.
  This lowers long-term usage, not the initial online peak. Shared blobs and other caches affect reclaimed space.
- Moving the download cache to another disk distributes capacity requirements without lowering the
  total. The current script has no automatic cache-pruning mode between model installations.

By default the runtime volume and caches live in Docker-managed storage. Free space beside the
checkout does not establish volume capacity:

```bash
docker_root="$(docker info --format '{{.DockerRootDir}}')"
df -h "$docker_root" .
```

For a new deployment, bind the runtime volume to a larger disk. Create the actual directory first,
replace the path and save this local `compose.model-storage.yml`:

```yaml
volumes:
  pep_runtime:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/pepplatform-runtime
```

Check that directory's filesystem. This example is for a **new volume**; existing volumes do not
migrate when options change. Back up and migrate the complete runtime directory before choosing
a new volume name for an existing deployment. Retain this persistent override on every operation;
commands are in the selected hardware guide's optional-configuration section.

### Container proxy for model downloads

Configure Docker daemon, build and backend runtime proxies separately. A host-shell proxy is not
automatically forwarded into download containers. When needed, save local `compose.download-proxy.yml`
and set `PEP_DOWNLOAD_PROXY=http://host.docker.internal:7890` in `.env`, replacing the port:

```yaml
services:
  backend:
    extra_hosts:
      - host.docker.internal:host-gateway
    environment:
      HTTP_PROXY: ${PEP_DOWNLOAD_PROXY:?Set PEP_DOWNLOAD_PROXY}
      HTTPS_PROXY: ${PEP_DOWNLOAD_PROXY:?Set PEP_DOWNLOAD_PROXY}
      http_proxy: ${PEP_DOWNLOAD_PROXY:?Set PEP_DOWNLOAD_PROXY}
      https_proxy: ${PEP_DOWNLOAD_PROXY:?Set PEP_DOWNLOAD_PROXY}
      NO_PROXY: localhost,127.0.0.1,db,backend,frontend
      no_proxy: localhost,127.0.0.1,db,backend,frontend
```

The proxy must listen on a host address reachable from the container. A host-loopback-only listener
is unreachable through this bridge-network example. See the [CPU override examples](deployment-cpu.md#optional-configuration-and-overrides)
or [GPU override examples](deployment-gpu.md#optional-configuration-and-overrides) for commands.
The proxy changes only the connection route; fixed model files still come from the selected endpoint
and must pass hash checks.

If the proxy accepts only host-loopback connections, or bridge-network TLS keeps failing, establish
host-network connectivity using the same image first. For one-off preparation, add local
`compose.download-host.yml`:

```yaml
services:
  backend:
    network_mode: host
```

Set `PEP_DOWNLOAD_PROXY` to the actual host-loopback address, for example `http://127.0.0.1:7890`.
Use the combination commands in your hardware guide, retaining storage and other persistent
overrides. This host-network override is only for one-off preparation; the full Web stack retains
normal Compose networking.

### TLS compatibility for proxies

If Hugging Face requests from the same image keep failing with handshake timeouts
or `SSLEOFError`, and a TLS 1.2 comparison succeeds, set `PEP_HF_TLS12=1` in the
local `.env` and rerun preparation. This option affects the model-preparation script
only: it retains certificate and hostname verification, caps HTTPS at TLS 1.2,
and disables Xet's separate HTTP client in favor of standard HTTP downloads.
The default is `0`, retaining normal TLS/Xet behavior. Endpoints, revisions, file
hashes and ProtT5 restoration requirements stay fixed; offline import/checks are unaffected.
