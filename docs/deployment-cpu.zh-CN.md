# CPU Docker 部署与运维

[English](deployment-cpu.md) · [部署选择与共享参考](deployment.zh-CN.md) · [GPU 部署](deployment-gpu.zh-CN.md)

本指南适用于没有 NVIDIA GPU 或不使用 GPU 的服务器。从仓库根目录执行命令。
**首次安装**按下文配置环境；**已有服务升级**直接阅读[更新与持久化](#更新与持久化)，保留原 `.env`、项目名和数据卷。

CPU 覆盖文件选择 Torch `2.5.1+cpu` / torchvision `0.20.1+cpu`，并清除默认 GPU 请求，无需 NVIDIA Container Toolkit。CPU 是兼容部署选项，不承诺与论文 GPU 训练数值逐位一致。

## 欢迎自行部署 Web 服务

公共服务的服务器存储与计算资源有限，因此会设置注册、训练、推理和模型数量等额度。具备计算和存储资源的个人或团队，欢迎自行部署 Web 服务，自主配置用户、额度、模型保护与保留期限、公告和维护时间。可先按硬件条件选择 CPU 或 GPU 指南，再在管理员后台调整[资源策略](resource-governance.zh-CN.md)。

## 环境与磁盘检查

使用 Linux amd64、Docker Engine、Docker Compose v2.35 或更新版本、Git、OpenSSL 和 curl。
这是本项目的 Compose 支持下限；CPU 覆盖文件使用 `!reset` 标签。
本流程的运行环境和模型检查均使用 CPU。

构建需要访问 Docker Hub、Debian 软件源、所选 PyTorch wheel 源、PyPI 和 npm。
需要国内镜像或代理时，先配置[下载源](deployment.zh-CN.md#国内镜像与下载源)及[可选覆盖文件](#可选配置与覆盖文件)。

先检查 Docker 数据目录所在磁盘；仓库所在磁盘可能是另一块盘：

```bash
docker --version
docker compose -f docker-compose.yml -f docker-compose.cpu.yml version
docker_root="$(docker info --format '{{.DockerRootDir}}')"
df -h "$docker_root" .
```

| 模型准备路径 | 模型文件预算，额外开销另计 |
| --- | --- |
| 只启动 Web，使用无需预训练权重的路线 | 无需预先安装六套模型 |
| 六套模型安装完成 | 22.68 GB（21.13 GiB） |
| 当前在线准备：源文件缓存与安装副本均保留 | 约 45.37 GB（42.25 GiB） |
| 离线包在另一块盘，仅计算目标运行卷 | 约 22.68 GB |
| 离线包与安装副本在同一块盘 | 约 45.37 GB |

这些数字按当前锁定文件求和，不包含额外下载缓存、重试文件和损坏安装的 `.previous-*` 副本。
还需为 Docker 镜像、构建缓存、数据库、上传数据、训练产物和备份分别留空间；100 GB 是宽裕的模型准备预算示例，**不是统一最低要求**。
内存需求取决于模型、路线、batch size 和序列长度。
如将运行卷绑定到另一块盘，应检查那个目录所在的磁盘；详见[空间与存储说明](deployment.zh-CN.md#模型下载的磁盘空间)。

## 首次安装：配置环境

以下复制配置的步骤仅用于全新安装：

```bash
git clone https://github.com/bliulab/PepPlatform.git
cd PepPlatform
cp .env.example .env
openssl rand -hex 32
```

编辑 `.env`：

- 分别运行 OpenSSL 命令生成三个独立值，填入 `POSTGRES_PASSWORD`、`PEP_SESSION_SECRET`、`PEP_LLM_KEY_ENCRYPTION_SECRET`。数据库密码建议使用生成的十六进制字符串，以便安全拼入连接 URL。
- 同时填写 `PEP_INITIAL_ADMIN_EMAIL` 与 `PEP_INITIAL_ADMIN_PASSWORD`。密码至少八个字符；初始管理员无需 SMTP 即可登录，已有账号密码不会被这些字段重置。
- 设置实际访问地址 `PEP_PUBLIC_BASE_URL`。改端口时同时设置 `PEP_HTTP_PORT`；使用 TLS 反向代理时填写 HTTPS 地址，以启用安全 Cookie。
- 保持 `PEP_ENV=production`、`PEP_AUTH_DEBUG_CODES=0`。注册和重置密码需要 SMTP；Tavily 和 Hugging Face token 为可选集成，受限模型下载需填写已获授权的 `HF_TOKEN`。

```bash
chmod 600 .env
```

## 构建、检查与启动

本指南每条 Compose 命令固定使用 `docker compose -f docker-compose.yml -f docker-compose.cpu.yml`。
如使用存储或模型挂载等持久覆盖文件，在这个前缀后、子命令前追加对应 `-f`，后续操作始终保留同一组文件。
CPU 的两个文件参数也必须用于一次性容器、模型准备、更新、备份和恢复。

```bash
set -e
docker compose -f docker-compose.yml -f docker-compose.cpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps backend \
  python docker/check_runtime.py --channel cpu --device cpu
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
docker compose -f docker-compose.yml -f docker-compose.cpu.yml ps
curl --fail http://127.0.0.1:8080/api/health
```

若更换端口或访问地址，相应调整健康检查 URL。后端自动执行 Alembic 数据库迁移，Nginx 等待后端健康后启动。
打开 <http://127.0.0.1:8080>，用初始管理员登录。

## 验收服务与基础训练

容器 healthy 只表示服务就绪，**不表示六套预训练模型已经准备好**。全新安装不附带研究数据或权重。
可以先用无需预训练权重的 **k-mer → One-hot → CNN** 路线验收。

设置 `PEP_SMOKE_EMAIL`、`PEP_SMOKE_PASSWORD` 为登录账号，在宿主机执行以下脚本（仅需 Python 3 标准库）：

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json
```

它会创建微型合成数据、一轮训练和模型，检查认证、推理与模型版本封存，并保留资源供后续检查。
其他地址加 `--base-url`；重启、更新或恢复后加 `--verify-existing` 检查同一批资源。
也可用 `python3 tools/build_web_route_smoke_dataset.py --output /tmp/pepplatform-demo.zip` 生成 ZIP，
上传为**肽段功能预测**数据集，训练一轮后对 `AAAAAAAAAAAA`、`CCCCCCCCCCCC` 做推理。
合成数据结果不具有科学意义。

## 可选：准备六套预训练模型

只有需要这些预训练路线时才执行本节。两种硬件方案使用相同的固定权重及 ProtT5 还原流程；
[模型身份与复现边界](deployment.zh-CN.md#准备与论文候选实验一致的模型)列出六套资源和固定 revision。
准备容器不启动数据库，也不执行数据库迁移。

### 在线准备

先完成 `.env` 配置和镜像构建，为受限资源提供已获授权的 `HF_TOKEN`。联网准备时保持 `HF_HUB_OFFLINE=0`、`PEPPLATFORM_OFFLINE_MODE=0`。确认空间足以同时容纳缓存和安装副本，然后执行：

```bash
set -e
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build backend
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps backend python docker/download_models.py
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cpu
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

检查会逐个加载模型、运行短序列并核对实际设备，之后释放模型。CPU 检查可能较慢，需预留足够主机内存。
只有全部六套校验并通过所选设备检查，才能认为这些模型就绪。

### 复用已有官方 ProtT5 快照

若已有固定版本的官方快照，将下面宿主机路径替换为实际目录：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  -v "$PWD/backup/prott5-official:/official-prott5:ro" \
  backend python docker/download_models.py --prott5-source /official-prott5
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cpu
```

源目录须包含 `pytorch_model.bin`、`config.json`、`spiece.model`、`special_tokens_map.json` 和 `tokenizer_config.json`，五个文件均按[固定还原配方](../docker/prott5.restore.json)校验。此方式只省去 ProtT5 下载，仍会还原原实验格式，其他缺失模型照常准备。
`--prott5-source` 不能与导出、导入、检查模式混用。
开展自己的实验、直接加载兼容官方快照时，使用训练参数 `--prott5_model`；见[两种参数的区别](deployment.zh-CN.md#独立实验原样使用官方-prott5)。

### 离线导入

先按[原服务器导出说明](deployment.zh-CN.md#可选原服务器导出离线模型包)取得包含模型、锁定清单和回执的完整模型包。
将下例宿主机路径改为实际位置；模型包可放另一块盘，以减少目标运行卷所在磁盘的占用。

```bash
set -e
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build backend
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  -v "$PWD/backup/pretrained-models:/model-package:ro" \
  backend python docker/download_models.py --import-from /model-package
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cpu
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

模型导入与检查无需联网；镜像构建仍需软件依赖，或预先准备好的对应镜像。
安装文件保存在 `pep_runtime` 卷的 `/var/lib/pepplatform/pretrained`，`/app/models` 指向其模型入口。
重复执行会校验并复用已有安装；损坏目录会保留为 `.previous-*` 后修复，额外占用空间。部分模型成功不代表全部完成。

### 启用离线运行

六套模型通过检查后，可在 `.env` 设置：

```dotenv
HF_HUB_OFFLINE=1
PEPPLATFORM_OFFLINE_MODE=1
```

重新创建服务以应用环境配置：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

普通重启不会重新加载 `.env`。模型包导入和 `--check-only` 本身始终不需要网络。

## 可选配置与覆盖文件

共享参考提供各覆盖文件内容；以下命令保留本指南的硬件选择。若组合多项配置，在子命令前依次追加所需文件。
存储和本地模型挂载等持久覆盖文件也要用于更新、备份和恢复。

| 用途 | 配置说明 |
| --- | --- |
| 本地模型快照 `compose.models.yml` | [模型挂载](deployment.zh-CN.md#预训练模型与邮件) |
| 新运行卷放到大容量磁盘 `compose.model-storage.yml` | [磁盘与运行卷](deployment.zh-CN.md#模型下载的磁盘空间) |
| 构建使用宿主机代理 `compose.build-network.yml` | [构建网络](deployment.zh-CN.md#故障排查) |
| 模型下载代理 `compose.download-proxy.yml` | [容器下载代理](deployment.zh-CN.md#模型下载的容器代理) |
| 一次性下载使用宿主机网络 `compose.download-host.yml` | [宿主机代理](deployment.zh-CN.md#模型下载的容器代理) |

本地模型挂载示例：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml -f compose.models.yml up --build -d --wait --wait-timeout 300
```

新运行卷使用独立磁盘后准备模型：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml -f compose.model-storage.yml run --rm --no-deps \
  backend python docker/download_models.py
```

构建代理示例（先在宿主机设置 `http_proxy` / `https_proxy`）：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml -f compose.build-network.yml build --build-arg http_proxy --build-arg https_proxy
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

同时使用独立存储和下载代理：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml -f compose.model-storage.yml -f compose.download-proxy.yml \
  run --rm --no-deps backend python docker/download_models.py
```

如果代理只接受宿主机回环地址，在确认同一镜像能用宿主机网络连通后，可进行一次性准备：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml -f compose.download-proxy.yml -f compose.download-host.yml \
  run --rm --no-deps backend python docker/download_models.py
```

这两个下载代理示例如还使用其他持久覆盖文件，需一并叠加。`compose.download-host.yml` 仅用于一次性准备；完整 Web 服务使用正常 Compose 网络。
TLS 握手故障可参考[代理兼容模式](deployment.zh-CN.md#代理的-tls-兼容模式)。
[SMTP 配置](deployment.zh-CN.md#预训练模型与邮件)或其他 `.env` 修改后，使用本指南的 `up` 命令应用。

## 可选：清理模型下载缓存

这只降低**准备完成后的长期占用**，不降低首次在线准备约 45.37 GB 的文件预算。
仅在六套安装校验和设备检查均通过后执行；先完成备份、结束训练，并确认业务没有直接引用这些下载缓存中的原始快照。
以下操作停止应用服务，保留数据库；脚本再次核对安装，再用 Hugging Face 缓存 API 删除六套固定源 revision。
它保留其他 revision、其他模型、`pretrained` 安装目录和运行数据。其他缓存及 `.previous-*` 不在清理范围。

```bash
set -e
docker compose -f docker-compose.yml -f docker-compose.cpu.yml stop frontend backend
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps -T backend python - <<'PY_CACHE'
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
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cpu
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

末尾检查在离线状态下加载安装副本，确认它们不依赖被删除的源缓存。共享 blob 或其他缓存可能使实际回收量小于 22.68 GB；
以脚本输出和磁盘检查为准。之后如需修复安装，可能需要重新下载源文件。

## 更新与持久化

在**原仓库目录和原 Compose 项目名**下升级，保留 `.env` 与所有持久覆盖文件。
`postgres_data` 保存数据库，`pep_runtime` 保存上传、训练产物、版本化输入、预训练安装和 Hugging Face 缓存。
不要重新复制 `.env.example` 覆盖已有配置；三个密钥保持稳定，修改环境文件不会修改已初始化数据库的用户密码。

若克隆来自公开仓库历史整理之前，请先完成[一次性源码更新](deployment.zh-CN.md#历史整理后的首次更新)，
再执行下列常规更新命令。登录与浏览器密码管理的区别见[账号登录与密码保存](deployment.zh-CN.md#账号登录与密码保存)。

若旧版本尚无 `docker-compose.cpu.yml`，先在原版本下使用原有部署配置与随附文档完成备份，
再执行下面的更新命令。拉取新版后才使用本指南的双文件参数；下节备份命令适用于已具备 CPU 覆盖文件的版本。

先结束或取消训练，再完成备份。后端重启会取消进行中的实验。备份成功后执行：

```bash
set -e
git pull --ff-only
docker compose -f docker-compose.yml -f docker-compose.cpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps backend \
  python docker/check_runtime.py --channel cpu --device cpu
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
docker compose -f docker-compose.yml -f docker-compose.cpu.yml ps
curl --fail http://127.0.0.1:8080/api/health
```

确认登录、已有数据集与已有模型推理；若保存了前述合成验收状态，在宿主机执行：

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json --verify-existing
```

需要六套模型的部署还应重复本指南的模型检查。切换 CPU/GPU 配置时重新构建后端。
普通 `down` 保留命名卷，**`down -v` 会删除卷内数据**。

## 备份与恢复

### 备份

等待训练结束，逐项确认成功后继续。下例使用新建的私有目录；权重压缩收益可能有限，
请先确认备份目标空间，必要时把 `backup_dir` 改到另一块盘。数据库保持运行。

```bash
set -e
mkdir -p backup
backup_dir="$(mktemp -d "$PWD/backup/deployment-XXXXXXXX")"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml stop frontend backend
docker compose -f docker-compose.yml -f docker-compose.cpu.yml exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$backup_dir/database.dump"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps -T backend \
  tar -C /var/lib/pepplatform -czf - . > "$backup_dir/runtime.tar.gz"
cp .env "$backup_dir/env"
git rev-parse HEAD > "$backup_dir/source-commit.txt"
for compose_file in docker-compose*.yml compose.*.yml; do
  if [ -f "$compose_file" ]; then
    cp "$compose_file" "$backup_dir/"
  fi
done
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
printf 'Backup: %s\n' "$backup_dir"
```

保存备份目录位置，逐项检查文件与退出状态，并安全复制到服务器之外。环境文件和代理覆盖文件可能含凭据。
本地挂载模型须另行备份或记录精确版本；记录自定义 `COMPOSE_PROJECT_NAME`，保留匹配的源码或镜像。

### 恢复

使用备份记录的源码版本、独立工作目录和**空目标卷**。先从备份恢复 `.env`、Compose 覆盖文件和外部模型挂载；
确认目标项目及存储路径没有指向原服务的数据卷。保持数据库用户名与库名一致。
将下例路径改为实际备份目录：

```bash
set -e
backup_dir="/path/to/deployment-backup"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300 db
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps -T backend \
  tar -C /var/lib/pepplatform -xzf - < "$backup_dir/runtime.tar.gz"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml exec -T db sh -c \
  'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --exit-on-error --no-owner --no-privileges' < "$backup_dir/database.dump"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

恢复后检查健康接口、登录、数据集和已有模型推理，再接入用户流量；如使用预训练模型，再运行对应模型检查。
升级失败时，数据库、运行文件与匹配源码或镜像一起恢复；只回退代码不会撤销数据库迁移。

## 故障排查

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml ps -a
docker compose -f docker-compose.yml -f docker-compose.cpu.yml logs --tail=200 db backend frontend
docker compose -f docker-compose.yml -f docker-compose.cpu.yml config --quiet
```

- **仍请求 GPU 或出现 CUDA wheel：** 确认每个命令都带本指南的两个文件参数，并重新构建后端；检查构建日志中的通道为 `cpu`。
- **构建下载失败：** 检查失败的下载层及[下载源与网络](deployment.zh-CN.md#故障排查)。npm 镜像示例：

  ```bash
  docker compose -f docker-compose.yml -f docker-compose.cpu.yml build --build-arg NPM_REGISTRY=https://your-registry.example frontend
  docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
  ```

- **后端不健康或 HTTP 502：** 检查数据库健康、后端迁移与启动日志。
- **无法登录：** 检查初始管理员或 SMTP；修改初始管理员字段不会重置已有密码。HTTP 测试需检查 Cookie 是否误设为 Secure。
- **找不到预训练模型：** 服务启动不会自动准备全部权重；执行本指南模型准备与检查，或提供正确的完整挂载快照及必要授权。
- **上传 HTTP 413：** 检查 `.env` 上传限制和 Nginx 的 512 MiB 请求上限；修改环境后重新创建后端，修改 `docker/nginx.conf` 后重新构建前端。

缺少密钥时补齐三个必填项后重新检查配置。避免公开完整渲染配置、`.env` 或带凭据的代理配置。
