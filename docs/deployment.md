# Docker deployment and operations

[简体中文](deployment.zh-CN.md) · [Online manual](https://bliulab.github.io/pepplatform-docs/install/docker)

## First deployment

The baseline is Linux amd64 with Docker Engine, Docker Compose v2.35 or newer,
Git, OpenSSL and curl. Compose must support `--wait`, `--wait-timeout` and
`depends_on.restart`. The host needs outbound access to Docker Hub, Debian,
the PyTorch CPU wheel index and PyPI/npm during the build. Allow several GB of
free disk space for images, plus space for your datasets, runs and model weights.
Training memory needs depend on the selected route and dataset. The supplied
backend uses CPU PyTorch; GPU deployment requires a CUDA image and NVIDIA
Container Toolkit/device configuration.

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
docker compose up --build -d --wait --wait-timeout 300
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

Upload the ZIP as a **peptide-function** dataset. For a CPU check without model
downloads, select **k-mer → One-hot → CNN**, one epoch and a small batch size.
Confirm that training finishes, a model appears, and inference returns scores
for `AAAAAAAAAAAA` and `CCCCCCCCCCCC`. Synthetic results have no scientific meaning.

For an automated check (Python 3, standard library only), set `PEP_SMOKE_EMAIL`
and `PEP_SMOKE_PASSWORD` to the login account, then run:

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json
```

This creates a synthetic dataset, a one-epoch CPU run and a model, and checks
authenticated inference and version sealing. It retains these resources for
inspection. After a restart, update or restore, repeat with `--verify-existing`
to check the same resource IDs. Use `--base-url` for another address.

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
