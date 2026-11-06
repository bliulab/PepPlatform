# Docker deployment and operations

[简体中文](deployment.zh-CN.md) · [Online manual](https://bliulab.github.io/pepplatform-docs/install/docker)

## First deployment

The baseline is Linux amd64 with Docker Engine, Docker Compose v2.35 or newer,
Git, OpenSSL and curl. Compose must support `--wait`, `--wait-timeout`,
`depends_on.restart`, GPU device requests and the CPU override's `!reset` tag.
The default backend matches the paper's core GPU software baseline:

| Component | Default paper GPU profile | Explicit CPU profile |
| --- | --- | --- |
| Python | 3.10.19 | 3.10.19 |
| Torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 | 2.5.1+cpu / 0.20.1+cpu |
| CUDA / cuDNN | 12.1 / 9.1 (90100) | Not used |
| Transformers / Tokenizers | 4.48.1 / 0.21.4 | Same |
| ESM / SentencePiece | 3.2.0 / 0.2.1 | Same |
| Hugging Face Hub / Safetensors | 0.36.0 / 0.7.0 | Same |

The default host requires a compatible NVIDIA GPU/driver and configured
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
Check `nvidia-smi` on the host first. Image builds check compiled versions without
requiring GPU access; GPU execution is checked separately after container startup.
For a machine without GPU container support, select the [CPU profile](#cpu-only-deployment).
The host needs outbound access to Docker Hub, Debian, the selected PyTorch wheel
index and PyPI/npm. Reserve space for both images, datasets, runs and model weights.
Training memory needs depend on the selected route and dataset. Matching this core
software baseline does not replace the paper's data, configuration and hardware requirements.

Run from the repository root:

```bash
git clone https://github.com/bliulab/PepPlatform.git
cd PepPlatform
cp .env.example .env
openssl rand -hex 32
```

Edit `.env` before starting:

- Generate **three separate values** with the OpenSSL command for
  `POSTGRES_PASSWORD`, `PEP_SESSION_SECRET` and `PEP_LLM_KEY_ENCRYPTION_SECRET`.
  Use a hex database password: it is interpolated into a database URL.
- Set `PEP_INITIAL_ADMIN_EMAIL` and `PEP_INITIAL_ADMIN_PASSWORD` to create the
  first login account. Use a strong password of at least eight characters.
  This account can log in without SMTP. These fields do not reset an existing
  account's password on restart.
- Set `PEP_PUBLIC_BASE_URL` to the address users will visit. For another port,
  change both this URL and `PEP_HTTP_PORT`. For HTTPS behind a TLS reverse proxy,
  use the HTTPS URL; secure cookies are derived from that URL.
- Keep `PEP_ENV=production` and `PEP_AUTH_DEBUG_CODES=0`. Configure SMTP for
  registration and password-reset emails; SMTP is optional for the initial
  administrator's login. Tavily and Hugging Face tokens are optional integrations.

```bash
chmod 600 .env
docker compose config --quiet
docker compose build
docker compose run --rm --no-deps backend python docker/check_runtime.py --channel cu121 --device cuda
docker compose up -d --wait --wait-timeout 300
docker compose ps
curl --fail http://127.0.0.1:8080/api/health
```

Open <http://127.0.0.1:8080> and sign in with the initial administrator account.
The backend applies Alembic migrations before serving requests; Nginx starts
after the backend passes its health check. A fresh installation contains no
research datasets or pretrained weights. Upload your own data, or generate the
small synthetic workflow-check dataset:

```bash
python3 tools/build_web_route_smoke_dataset.py --output /tmp/pepplatform-demo.zip
```

Upload the ZIP as a **peptide-function** dataset. For a short workflow check without pretrained model
downloads, select **k-mer → One-hot → CNN**, one epoch and a small batch size.
Confirm that training finishes, a model appears, and inference returns scores
for `AAAAAAAAAAAA` and `CCCCCCCCCCCC`. Synthetic results have no scientific meaning.

For an automated check (Python 3, standard library only), set `PEP_SMOKE_EMAIL`
and `PEP_SMOKE_PASSWORD` to the login account, then run:

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json
```

This creates a synthetic dataset, a one-epoch run and a model, and checks
authenticated inference and version sealing. It retains these resources for
inspection. After a restart, update or restore, repeat with `--verify-existing`
to check the same resource IDs. Use `--base-url` for another address.

### CPU-only deployment

Use the same `.env` configuration, then add both Compose files to **every**
CPU command, including model preparation, updates, backup and restore:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  backend python docker/check_runtime.py --channel cpu --device cpu
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

The CPU override selects the CPU wheels and clears GPU requests, so it does not
require NVIDIA Container Toolkit. It uses the same frozen checkpoint files and
ProtT5 restoration. CPU execution is a compatibility alternative; numerical equality
with the paper's GPU training is not asserted. Rebuild the backend when switching profiles.
If you also need the paper assets, prepare and check them with the CPU prefix:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  backend python docker/download_models.py
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  backend python docker/download_models.py --check-only --smoke-test --device cpu
```


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
loader and its RNG behavior; users do not need your original shard package.
The full checkpoint has about 3B parameters; the platform loads its approximately
1.2B encoder. See the [verification and restoration checks](prott5-equivalence.md).

Weights remain outside Git and the application image. Installed assets total about
22.7 GB. Online preparation retains the upstream cache, including the approximately
11.3 GB ProtT5 `.bin`, as well as the approximately 11.3 GB restored ProtT5 files.
Allow at least 50 GB for download caches and installed assets; portable packages,
images, business data and backups need additional space.

### Download and prepare on the target server

For paper reproduction, configure `.env`, including provider access tokens where required, then run:

```bash
docker compose build
docker compose run --rm --no-deps backend python docker/download_models.py
docker compose run --rm --no-deps backend python docker/download_models.py --check-only
docker compose up -d --wait --wait-timeout 300
```

The preparation command covers all six checkpoints, reuses valid installations
and resumes cached downloads. ProtT5 is restored once into the persistent runtime
volume. Preparation does not start the database or apply database migrations.

To reuse an already downloaded official ProtT5 snapshot, mount its directory read-only:

```bash
docker compose run --rm --no-deps \
  -v "$PWD/backup/prott5-official:/official-prott5:ro" \
  backend python docker/download_models.py --prott5-source /official-prott5
```

The directory must contain the pinned `pytorch_model.bin`, `config.json`,
`spiece.model`, `special_tokens_map.json` and `tokenizer_config.json`. All five are
verified. This option makes no ProtT5 network requests and still restores the
paper format; other missing checkpoints follow normal preparation. It is separate
from export/import/check modes. For unchanged raw loading, use the training option
described in the independent-experiment section below.

For a bare Python installation, run `python docker/download_models.py --root
backup/pretrained` and set the training option `--prott5_model` to the absolute
`backup/pretrained/models/prott5` path. Use this prepared entry for paper training.
The raw official `.bin` directory has different loading RNG behavior.

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

Configure `.env` as above. Model preparation runs separately without starting the
database or applying database migrations:

```bash
docker compose build
docker compose run --rm --no-deps \
  -v "$PWD/backup/pretrained-models:/model-package:ro" \
  backend python docker/download_models.py --import-from /model-package
docker compose run --rm --no-deps backend python docker/download_models.py --check-only
docker compose up -d --wait --wait-timeout 300
```

Import uses no network. Each file must match the application lock before activation;
success requires all six checkpoints. Files live at `/var/lib/pepplatform/pretrained`
on the existing `pep_runtime` volume, linked from `/app/models` in the image. Public
snapshots retain `snapshots/<revision>` paths; local shards retain their original
content fingerprints. Existing runtime-volume backups include the installed assets;
also retain the portable package separately. Repeated preparation verifies and reuses
files. Corrupt installation directories are preserved as `.previous-*` before repair.
After a failure, fix the package and retry; a partial preparation is not success.

The default GPU runtime check loads each model on CUDA, embeds a short sequence,
checks its actual parameter devices, and releases it before the next model. CUDA
availability or version mismatches fail explicitly. Reserve sufficient host RAM and GPU memory:

```bash
docker compose run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cuda
```

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

## Pretrained models and SMTP

Downloaded Hugging Face files persist under `HF_HOME` on the runtime volume.
For existing local snapshots, create an optional `compose.models.yml`:

```yaml
services:
  backend:
    volumes:
      - ./models:/app/models:ro
    environment:
      PEPPLATFORM_OFFLINE_MODE: "1"
```

Place complete snapshots in the paths selected by the route, for example
`models/esm2`, `models/protbert`, `models/prott5` or
`models/EvolutionaryScale/esmc-600m-2024-12`. Start with both files:

```bash
docker compose -f docker-compose.yml -f compose.models.yml up --build -d --wait
```

Keep using the same file combination for all subsequent operations. Remove the
offline setting when downloads are intended. A mount alone does not download
weights or grant access to gated models; follow the provider's model license
and authorization requirements.

For email, set `PEP_SMTP_HOST`, `PEP_SMTP_PORT`, `PEP_SMTP_USER`,
`PEP_SMTP_PASSWORD` and `PEP_SMTP_FROM`. Port 587 normally uses STARTTLS; port
465 selects implicit TLS. The sender should match the authenticated mailbox.
For QQ Mail, use the mailbox authorization code. Apply environment changes with
`docker compose up -d --wait`; `docker compose restart` does not reload `.env`.

## Updates and persistence

Keep the same checkout directory and Compose project name: they determine the
named volumes. The stack stores PostgreSQL in `postgres_data` and uploads,
training artifacts, versioned inputs and Hugging Face downloads in `pep_runtime`.
The deployment is a single backend process; finish or cancel active training
before an update because a backend restart cancels in-progress experiments.

After a backup:

```bash
git pull --ff-only
docker compose up --build -d --wait --wait-timeout 300
docker compose ps
curl --fail http://127.0.0.1:8080/api/health
```

Verify login, an existing dataset and inference from an existing model. Keep the
three secrets unchanged. Changing the database password in `.env` does not
change an already initialized PostgreSQL user's password. Retain the session
secret for sessions and the AI encryption secret to read saved provider keys.
`docker compose down` preserves named volumes; **`down -v` deletes their data**.

## Backup and restore

Wait for training to finish, then stop both application services for a consistent
database/runtime backup. The database remains running. Run these commands from
the repository root, using a new backup directory each time:

```bash
mkdir -m 700 backup
docker compose stop frontend backend
docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > backup/database.dump
docker compose run --rm --no-deps -T backend tar -C /var/lib/pepplatform -czf - . > backup/runtime.tar.gz
cp .env backup/env
git rev-parse HEAD > backup/source-commit.txt
docker compose up -d --wait
```

Check each command succeeded and retain the backup securely outside the server;
the copied environment contains credentials. Also back up locally mounted
model snapshots or record where to retrieve their exact revisions. Keep a copy
of any Compose overrides and the matching application source/images.

Restore into a fresh checkout at the recorded source revision and empty volumes.
Restore the saved environment as `.env` first; copy Compose overrides and model
snapshots too. Use the same database user/name as the backup. Then:

```bash
docker compose build
docker compose up -d --wait db
docker compose run --rm --no-deps -T backend tar -C /var/lib/pepplatform -xzf - < backup/runtime.tar.gz
docker compose exec -T db sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --exit-on-error --no-owner --no-privileges' < backup/database.dump
docker compose up -d --wait --wait-timeout 300
```

Verify login, datasets and existing-model inference before accepting traffic.
For a failed upgrade, restore the matching database and runtime backup together
with its source/images; reverting code alone does not revert database migrations.

## Troubleshooting

```bash
docker compose ps -a
docker compose logs --tail=200 db backend frontend
```

- **Missing-secret error:** fill all three required secrets; run
  `docker compose config --quiet` again. Avoid sharing full rendered Compose
  configuration because it contains secrets.
- **Build download failure:** inspect the failed Dockerfile step and check the
  host/container network or proxy. For an npm mirror, pass
  `--build-arg NPM_REGISTRY=https://your-registry.example` to
  `docker compose build frontend`, then start normally.
- **Unhealthy backend / HTTP 502:** inspect backend migration/startup logs and
  database health; verify the health URL before trying training.
- **Cannot log in:** set both initial-admin fields for a new account, or configure
  working SMTP for registration/reset. An existing password is not replaced by
  editing the initial-admin fields. HTTP testing needs non-secure cookies; check
  `PEP_PUBLIC_BASE_URL` and any `PEP_COOKIE_SECURE` override.
- **Missing pretrained model:** supply the selected full snapshot or enable
  downloads and any required Hugging Face authorization.
- **Upload HTTP 413:** check the upload limits in `.env` and Nginx's 512 MiB
  request cap. Recreate the backend after environment changes; rebuild the
  frontend image after changing `docker/nginx.conf`.

If a Linux host's proxy listens only on `127.0.0.1`, the build container cannot
reach it through its own loopback interface. With `http_proxy` / `https_proxy`
already set on the host, a temporary build override can use the host network:

```yaml
# compose.build-network.yml — only needed for this build-network setup
services:
  backend:
    build:
      network: host
  frontend:
    build:
      network: host
```

```bash
docker compose -f docker-compose.yml -f compose.build-network.yml build --build-arg http_proxy --build-arg https_proxy
docker compose up -d --wait --wait-timeout 300
```

This changes the build network only. Keep proxy addresses and credentials out of
the Dockerfiles and Git. Container runtime networking remains on the Compose network.

### Disk space for model downloads

The six installed checkpoints occupy about 21.1 GiB, and Hugging Face retains
another copy of the source files in its download cache. Reserve at least 100 GB
for the cache, installation and retries. Check the filesystem containing Docker's
data directory; free space beside the checkout does not establish volume capacity.

For a fresh deployment, place the runtime volume on a large disk. Create its
directory first, replace `/mnt/data/pepplatform-runtime` below with the actual path,
and save this local override as `compose.model-storage.yml`:

```yaml
volumes:
  pep_runtime:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/pepplatform-runtime
```

Retain this override on every Compose command; CPU users also retain the CPU override:

```bash
docker compose -f docker-compose.yml -f compose.model-storage.yml run --rm --no-deps \
  backend python docker/download_models.py
```

This example creates a new runtime volume. For an existing deployment, back up and
migrate the complete runtime directory before selecting a new volume name.

### Container proxy for model downloads

The Docker daemon, build steps and backend runtime need separate proxy
configuration. A host-shell proxy is not automatically forwarded to the backend
container running the download command. When needed, save this local override as
`compose.download-proxy.yml` and set
`PEP_DOWNLOAD_PROXY=http://host.docker.internal:7890` in your local `.env`,
replacing the port with the actual value:

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

The proxy must listen on a host address reachable from the container. A proxy
bound only to the host's `127.0.0.1` is not reachable with this bridge-network
example. Add `-f compose.download-proxy.yml` to the download command, alongside any
storage or CPU overrides you need. The proxy changes the connection route; model
files still come from the selected endpoint and must match the frozen lock.

If the proxy accepts only host-loopback connections, or bridge-network TLS requests
keep failing, first verify connectivity with the same image on the host network.
For one-off preparation, add a local `compose.download-host.yml`:

```yaml
services:
  backend:
    network_mode: host
```

Set `PEP_DOWNLOAD_PROXY` to the actual host-loopback address, for example
`http://127.0.0.1:7890`, then run:

```bash
docker compose -f docker-compose.yml -f compose.download-proxy.yml -f compose.download-host.yml \
  run --rm --no-deps backend python docker/download_models.py
```

Retain any storage and CPU overrides you need. This host-network override is for
one-off model preparation; the full Web deployment keeps its normal Compose network.

### TLS compatibility for proxies

If Hugging Face requests from the same image keep failing with handshake timeouts
or `SSLEOFError`, and a TLS 1.2 comparison succeeds, set `PEP_HF_TLS12=1` in the
local `.env` and rerun preparation. This option affects the model-preparation script
only: it retains certificate and hostname verification, caps HTTPS at TLS 1.2,
and disables Xet's separate HTTP client in favor of standard HTTP downloads.
The default is `0`, retaining normal TLS/Xet behavior. Endpoints, revisions, file
hashes and ProtT5 restoration requirements stay fixed; offline import/checks are unaffected.
