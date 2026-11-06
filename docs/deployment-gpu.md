# GPU Docker deployment and operations

[简体中文](deployment-gpu.zh-CN.md) · [Deployment selector and shared reference](deployment.md) · [CPU deployment](deployment-cpu.md)

Use this guide for servers with a compatible NVIDIA GPU, driver and GPU container support. Run commands from the repository root.
For a **new installation**, configure the environment below. For an **existing deployment**, go to
[Updates and persistence](#updates-and-persistence), retaining its `.env`, project name and volumes.

The default profile selects the paper GPU baseline: Torch `2.5.1+cu121` / torchvision `0.20.1+cu121`, requesting NVIDIA GPU access. See the [shared reference](deployment.md#first-deployment) for core versions. Numerical reproduction also requires matching data, configurations, seeds and hardware.

## Environment and disk checks

Use Linux amd64, Docker Engine, Docker Compose v2.35 or newer, Git, OpenSSL and curl.
This is the project's supported Compose baseline; the CPU override uses the `!reset` tag.
Configure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) first and verify host `nvidia-smi` works. The container runtime check below also verifies actual CUDA execution.

Builds need Docker Hub, Debian repositories, the selected PyTorch wheel index, PyPI and npm.
Configure [download sources](deployment.md#download-sources-and-mirrors-in-china) and
[optional overrides](#optional-configuration-and-overrides) first if your network needs mirrors or proxies.

Check the disk containing Docker's data directory; the checkout may be on a different disk:

```bash
docker --version
docker compose -f docker-compose.yml version
docker_root="$(docker info --format '{{.DockerRootDir}}')"
df -h "$docker_root" .
```

| Preparation path | Model file budget, excluding additional overhead |
| --- | --- |
| Web startup with routes that need no pretrained weights | No six-checkpoint installation required |
| All six installed checkpoints | 22.68 GB (21.13 GiB) |
| Current online preparation, retaining source cache and installed copies | About 45.37 GB (42.25 GiB) |
| Offline package on another disk: target runtime volume only | About 22.68 GB |
| Offline package and installed copies on the same disk | About 45.37 GB |

These figures sum the current locked files. Extra download caches, retry files and corrupt installations
preserved as `.previous-*` need additional space. Budget separately for Docker images, build caches,
the database, uploads, training artifacts and backups. A 100 GB model-preparation budget is a generous
example, **not a universal minimum**. Host memory and GPU memory requirements depend on the model, route,
batch size and sequence length. If the runtime volume is bound to another disk, check that filesystem;
see [storage and space](deployment.md#disk-space-for-model-downloads).

## New installation: configure the environment

Copy the example configuration only for a new installation:

```bash
git clone https://github.com/bliulab/PepPlatform.git
cd PepPlatform
cp .env.example .env
openssl rand -hex 32
```

Edit `.env`:

- Run OpenSSL separately for three independent values: `POSTGRES_PASSWORD`, `PEP_SESSION_SECRET`
  and `PEP_LLM_KEY_ENCRYPTION_SECRET`. The generated hexadecimal database password is safe to embed in the connection URL.
- Set both `PEP_INITIAL_ADMIN_EMAIL` and `PEP_INITIAL_ADMIN_PASSWORD`. Use at least eight password
  characters. Initial-admin login needs no SMTP; these fields do not reset an existing account.
- Set `PEP_PUBLIC_BASE_URL` to the actual public address. If changing the port, set `PEP_HTTP_PORT`
  too. Use the HTTPS address behind a TLS reverse proxy so the backend enables secure cookies.
- Keep `PEP_ENV=production` and `PEP_AUTH_DEBUG_CODES=0`. Registration and password reset need SMTP.
  Tavily and Hugging Face tokens are optional integrations; gated downloads require an authorized `HF_TOKEN`.

```bash
chmod 600 .env
```

## Build, check and start

Every Compose command in this guide uses `docker compose -f docker-compose.yml`.
For persistent storage or model-mount overrides, append their `-f` arguments after this prefix and
before the subcommand. Retain the same file combination on subsequent operations.
Image builds check compiled versions without needing GPU access. The separate runtime-check container below verifies actual GPU execution.

```bash
set -e
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml run --rm --no-deps backend \
  python docker/check_runtime.py --channel cu121 --device cuda
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
docker compose -f docker-compose.yml ps
curl --fail http://127.0.0.1:8080/api/health
```

Adjust the health URL if you changed the port or address. The backend applies Alembic database
migrations before serving; Nginx waits for backend health. Open <http://127.0.0.1:8080> and sign in
with the initial administrator.

## Verify the service and basic training

Healthy containers establish service readiness; **they do not establish that the six pretrained
checkpoints are installed**. A new installation includes neither research data nor model weights.
Start with the **k-mer → One-hot → CNN** route, which needs no pretrained weights.

Set `PEP_SMOKE_EMAIL` and `PEP_SMOKE_PASSWORD` to your login credentials, then run on the host
(Python 3 standard library only):

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json
```

The script creates synthetic data, a one-epoch training run and a model, verifies authentication,
inference and versioned model inputs, and keeps the resources for inspection. Use `--base-url`
for another address. After a restart, update or restore, add `--verify-existing` to check the same resources.
Alternatively, generate a ZIP with
`python3 tools/build_web_route_smoke_dataset.py --output /tmp/pepplatform-demo.zip`, upload it as a
**peptide functional prediction** dataset, train for one epoch and infer on `AAAAAAAAAAAA` and
`CCCCCCCCCCCC`. Synthetic results have no scientific meaning.

## Optional: prepare all six pretrained checkpoints

Run this section only when you need these pretrained routes. Both hardware profiles use the same
fixed weights and ProtT5 restoration. [Model identity and reproduction requirements](deployment.md#prepare-the-exact-paper-pretrained-assets)
list all six checkpoints and fixed revisions. Preparation containers neither start the database nor apply migrations.

### Online preparation

Configure `.env` and build the image first. Supply an authorized `HF_TOKEN` for gated repositories.
Keep `HF_HUB_OFFLINE=0` and `PEPPLATFORM_OFFLINE_MODE=0` for online preparation.
Allow space for both the source cache and installed copies, then run:

```bash
set -e
docker compose -f docker-compose.yml build backend
docker compose -f docker-compose.yml run --rm --no-deps backend python docker/download_models.py
docker compose -f docker-compose.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cuda
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
```

The check loads each model, embeds a short sequence, checks its actual device, then releases it.
Missing CUDA access or incompatible dependencies fail explicitly; reserve enough host RAM and GPU memory.
All six file checks and device checks must pass before declaring these models ready.

### Reuse an official ProtT5 snapshot

For an existing fixed official snapshot, replace the host path with its actual location:

```bash
docker compose -f docker-compose.yml run --rm --no-deps \
  -v "$PWD/backup/prott5-official:/official-prott5:ro" \
  backend python docker/download_models.py --prott5-source /official-prott5
docker compose -f docker-compose.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cuda
```

The source directory must contain `pytorch_model.bin`, `config.json`, `spiece.model`,
`special_tokens_map.json` and `tokenizer_config.json`. All five must match the
[fixed restoration recipe](../docker/prott5.restore.json).
This skips ProtT5 downloading but still restores the paper format; other missing models follow normal
preparation. `--prott5-source` cannot be combined with export, import or check modes.
For your own experiments using an unchanged compatible snapshot, use the training option
`--prott5_model`; see [the option comparison](deployment.md#independent-experiments-use-an-official-prott5-snapshot-unchanged).

### Offline import

Obtain the complete package, including models, lock and receipt, using the
[original-server export instructions](deployment.md#optional-export-a-portable-package-on-the-original-server).
Replace the host path below with its actual location. The package can reside on another disk to
reduce space needed on the target runtime disk.

```bash
set -e
docker compose -f docker-compose.yml build backend
docker compose -f docker-compose.yml run --rm --no-deps \
  -v "$PWD/backup/pretrained-models:/model-package:ro" \
  backend python docker/download_models.py --import-from /model-package
docker compose -f docker-compose.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cuda
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
```

Import and checks need no network. Image builds still require software dependencies or an already
prepared matching image. Installed files live under `/var/lib/pepplatform/pretrained` on `pep_runtime`;
`/app/models` points to their model entries. Repeated preparation verifies and reuses installations.
Corrupt directories are preserved as `.previous-*` before repair, consuming additional space.
Partial success is not complete preparation.

### Enable offline operation

Once all six checkpoints pass verification, set in `.env`:

```dotenv
HF_HUB_OFFLINE=1
PEPPLATFORM_OFFLINE_MODE=1
```

Recreate services to apply the environment:

```bash
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
```

A plain restart does not reload `.env`. Package imports and `--check-only` never require network access.

## Optional configuration and overrides

The shared reference contains the override definitions. These examples retain this guide's hardware
selection. Append every additional file before the subcommand when combining options. Persistent
storage and local-model overrides must also be retained for updates, backups and restores.

| Purpose | Configuration reference |
| --- | --- |
| Local snapshots: `compose.models.yml` | [Model mounts](deployment.md#pretrained-models-and-smtp) |
| New runtime volume on a large disk: `compose.model-storage.yml` | [Disk and runtime volume](deployment.md#disk-space-for-model-downloads) |
| Host proxy during builds: `compose.build-network.yml` | [Build networking](deployment.md#troubleshooting) |
| Model download proxy: `compose.download-proxy.yml` | [Container proxy](deployment.md#container-proxy-for-model-downloads) |
| Host network for one-off preparation: `compose.download-host.yml` | [Host proxy](deployment.md#container-proxy-for-model-downloads) |

For a local-model mount:

```bash
docker compose -f docker-compose.yml -f compose.models.yml up --build -d --wait --wait-timeout 300
```

For model preparation on a new volume bound to another disk:

```bash
docker compose -f docker-compose.yml -f compose.model-storage.yml run --rm --no-deps \
  backend python docker/download_models.py
```

For a build proxy (set `http_proxy` / `https_proxy` in the host shell first):

```bash
docker compose -f docker-compose.yml -f compose.build-network.yml build --build-arg http_proxy --build-arg https_proxy
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
```

With both external storage and a download proxy:

```bash
docker compose -f docker-compose.yml -f compose.model-storage.yml -f compose.download-proxy.yml \
  run --rm --no-deps backend python docker/download_models.py
```

If the proxy accepts only host-loopback connections, first establish connectivity using the same
image on the host network, then use one-off preparation:

```bash
docker compose -f docker-compose.yml -f compose.download-proxy.yml -f compose.download-host.yml \
  run --rm --no-deps backend python docker/download_models.py
```

Retain any other persistent overrides in both proxy examples. Use `compose.download-host.yml` only
for one-off preparation; the full Web stack keeps its normal Compose network. For handshake failures,
see [TLS compatibility](deployment.md#tls-compatibility-for-proxies). After changing
[SMTP](deployment.md#pretrained-models-and-smtp) or other `.env` settings, apply them with this guide's `up` command.

## Optional: reclaim model download caches

This reduces **long-term usage after preparation**, not the roughly 45.37 GB initial online file
budget. Run only after all six installations and device checks pass. Back up first, finish training
and confirm no workload directly references the raw snapshots in these download caches.
The procedure stops application services, retaining the database. The script verifies the installed
files again, then uses the Hugging Face cache API to remove only the six pinned source revisions.
Other revisions, other models, the `pretrained` installation and runtime data remain.
Other caches and `.previous-*` copies are outside this cleanup.

```bash
set -e
docker compose -f docker-compose.yml stop frontend backend
docker compose -f docker-compose.yml run --rm --no-deps -T backend python - <<'PY_CACHE'
import json
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import replace
from huggingface_hub import scan_cache_dir

subprocess.run([sys.executable, "docker/download_models.py", "--check-only"], check=True)
lock = json.loads(Path("docker/models.lock.json").read_text())
recipe = json.loads(Path("docker/prott5.restore.json").read_text())
upstreams = [m for m in lock["models"] if m["origin"] == "huggingface"]
upstreams.append(recipe["source"])
wanted = {(m["repo_id"], m["revision"]) for m in upstreams}
cache = Path(os.environ["HF_HOME"]) / "hub"
if not cache.is_dir():
    print("No Hub download cache to clean.")
    raise SystemExit(0)
info = scan_cache_dir(cache)
for repo in info.repos:
    if repo.repo_type != "model":
        continue
    revisions = [r.commit_hash for r in repo.revisions if (repo.repo_id, r.commit_hash) in wanted]
    if not revisions:
        continue
    scoped = replace(info, repos=frozenset({repo}), size_on_disk=repo.size_on_disk)
    strategy = scoped.delete_revisions(*revisions)
    print(f"{repo.repo_id}: reclaiming {strategy.expected_freed_size_str}.", flush=True)
    strategy.execute()
PY_CACHE
docker compose -f docker-compose.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cuda
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
```

The final check loads installed copies offline to verify independence from the removed source cache.
Shared blobs and other caches may reduce the amount reclaimed below 22.68 GB; use the script's output
and filesystem measurements. A later installation repair may need to download source files again.

## Updates and persistence

Upgrade from the **original checkout directory and Compose project name**, retaining `.env` and all
persistent overrides. `postgres_data` stores the database; `pep_runtime` stores uploads, training
artifacts, versioned inputs, installed pretrained models and Hugging Face downloads.
Do not overwrite existing configuration with `.env.example`. Keep all three secrets stable;
editing the database password in `.env` does not change an initialized PostgreSQL user's password.

Finish or cancel training, then complete the backup below. A backend restart cancels active
experiments. After a successful backup:

```bash
set -e
git pull --ff-only
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml run --rm --no-deps backend \
  python docker/check_runtime.py --channel cu121 --device cuda
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
docker compose -f docker-compose.yml ps
curl --fail http://127.0.0.1:8080/api/health
```

Verify login, existing datasets and existing-model inference. If you saved the synthetic smoke state,
run on the host:

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json --verify-existing
```

Deployments using all six checkpoints should also repeat the model check above. Rebuild the backend
when switching CPU/GPU profiles. A plain `down` preserves named volumes; **`down -v` deletes their data**.

## Backup and restore

### Backup

Wait for training to finish and check each operation succeeds. The example creates a fresh private
directory. Model weights may compress poorly: check backup destination space first and put `backup_dir`
on another disk if needed. Keep the database running.

```bash
set -e
mkdir -p backup
backup_dir="$(mktemp -d "$PWD/backup/deployment-XXXXXXXX")"
docker compose -f docker-compose.yml stop frontend backend
docker compose -f docker-compose.yml exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$backup_dir/database.dump"
docker compose -f docker-compose.yml run --rm --no-deps -T backend \
  tar -C /var/lib/pepplatform -czf - . > "$backup_dir/runtime.tar.gz"
cp .env "$backup_dir/env"
git rev-parse HEAD > "$backup_dir/source-commit.txt"
for compose_file in docker-compose*.yml compose.*.yml; do
  if [ -f "$compose_file" ]; then
    cp "$compose_file" "$backup_dir/"
  fi
done
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
printf 'Backup: %s\n' "$backup_dir"
```

Record the backup path, inspect the files and exit statuses, and store a secure copy off the server.
Environment and proxy override files may contain credentials. Back up local model mounts separately
or record their exact revisions. Record any custom `COMPOSE_PROJECT_NAME` and retain matching source/images.

### Restore

Use the source revision recorded in the backup, an independent working directory and **empty target
volumes**. Restore `.env`, Compose overrides and external model mounts first. Verify that the target
project and storage paths do not point at the original deployment's volumes. Keep the database user
and name consistent. Replace the backup path:

```bash
set -e
backup_dir="/path/to/deployment-backup"
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300 db
docker compose -f docker-compose.yml run --rm --no-deps -T backend \
  tar -C /var/lib/pepplatform -xzf - < "$backup_dir/runtime.tar.gz"
docker compose -f docker-compose.yml exec -T db sh -c \
  'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --exit-on-error --no-owner --no-privileges' < "$backup_dir/database.dump"
docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
```

Verify health, login, datasets and existing-model inference before accepting traffic. Repeat the
model check when using pretrained assets. For a failed upgrade, restore the database and runtime
backup together with matching source/images; reverting code alone does not revert database migrations.

## Troubleshooting

```bash
docker compose -f docker-compose.yml ps -a
docker compose -f docker-compose.yml logs --tail=200 db backend frontend
docker compose -f docker-compose.yml config --quiet
```

- **GPU request failure or unavailable CUDA:** check the host driver, `nvidia-smi` and Toolkit, then repeat this guide's CUDA runtime check. Use the [CPU guide](deployment-cpu.md) for machines without GPU support.
- **Build downloads fail:** check the failing layer and [download/network settings](deployment.md#troubleshooting).
  For an npm mirror:

  ```bash
  docker compose -f docker-compose.yml build --build-arg NPM_REGISTRY=https://your-registry.example frontend
  docker compose -f docker-compose.yml up -d --wait --wait-timeout 300
  ```

- **Unhealthy backend / HTTP 502:** check database health and backend migration/startup logs.
- **Cannot log in:** check initial-admin or SMTP settings. Editing initial-admin fields does not reset
  existing passwords. Check whether HTTP testing incorrectly enables Secure cookies.
- **Missing pretrained model:** startup does not prepare all weights. Run preparation and checks above,
  or supply the correct full mounted snapshot and any required authorization.
- **Upload HTTP 413:** check `.env` upload limits and Nginx's 512 MiB request cap. Recreate the backend
  after environment changes; rebuild the frontend after editing `docker/nginx.conf`.

For missing secrets, fill all three required values and check configuration again. Do not share full
rendered configuration, `.env` or proxy settings containing credentials.
