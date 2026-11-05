# Docker 部署与运维

[English](deployment.md) · [在线手册](https://bliulab.github.io/pepplatform-docs/zh/install/docker)

## 首次部署

基础验收环境为 Linux amd64，需要 Docker Engine、Docker Compose v2.35 或更新版本、
Git、OpenSSL 和 curl；Compose 需支持 `--wait`、`--wait-timeout`、`depends_on.restart`。
构建期间需要访问 Docker Hub、Debian 软件源、PyTorch CPU wheel 源、
PyPI 和 npm。镜像需预留数 GB 磁盘空间，数据集、训练产物和模型权重需要额外空间；
训练内存需求取决于路线和数据规模。随附后端使用 CPU PyTorch；GPU 部署需另配
CUDA 镜像、NVIDIA Container Toolkit 和设备映射。

在仓库根目录操作：

```bash
git clone https://github.com/bliulab/PepPlatform.git
cd PepPlatform
cp .env.example .env
openssl rand -hex 32
```

启动前编辑 `.env`：

- 用 OpenSSL 命令**分别生成三个独立值**，填入 `POSTGRES_PASSWORD`、
  `PEP_SESSION_SECRET` 和 `PEP_LLM_KEY_ENCRYPTION_SECRET`。
  数据库密码会拼入连接 URL，建议直接使用生成的十六进制字符串。
- 填写 `PEP_INITIAL_ADMIN_EMAIL` 和 `PEP_INITIAL_ADMIN_PASSWORD`，创建首次登录账号。
  密码至少八个字符，建议使用高强度密码。此账号无需 SMTP 即可登录；
  重启时这些字段不会重置已有账号的密码。
- 将 `PEP_PUBLIC_BASE_URL` 改为用户实际访问地址。更换端口时同时修改此 URL 与
  `PEP_HTTP_PORT`。使用 TLS 反向代理时填写 HTTPS 地址，后端据此启用安全 Cookie。
- 保持 `PEP_ENV=production`、`PEP_AUTH_DEBUG_CODES=0`。注册和密码重置需要 SMTP；
  初始管理员登录不依赖 SMTP。Tavily 和 Hugging Face token 为可选集成配置。

```bash
chmod 600 .env
docker compose config --quiet
docker compose up --build -d --wait --wait-timeout 300
docker compose ps
curl --fail http://127.0.0.1:8080/api/health
```

打开 <http://127.0.0.1:8080>，用初始管理员账号登录。后端在提供服务前自动运行
Alembic 数据库迁移；Nginx 等待后端健康检查通过后启动。新安装不附带研究数据或
预训练权重。上传自己的数据，或生成用于验证流程的微型合成数据：

```bash
python3 tools/build_web_route_smoke_dataset.py --output /tmp/pepplatform-demo.zip
```

将 ZIP 上传为**肽段功能预测**数据集。无需下载权重的 CPU 检查可选择
**k-mer → One-hot → CNN**，训练一轮，使用较小 batch size。确认训练完成并出现模型，
再对 `AAAAAAAAAAAA`、`CCCCCCCCCCCC` 做推理，确认返回分数。合成数据结果不具有科学意义。

也可自动检查（仅需 Python 3 标准库）：设置环境变量 `PEP_SMOKE_EMAIL`、
`PEP_SMOKE_PASSWORD` 为登录账号，再执行：

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json
```

脚本会创建合成数据集、一轮 CPU 训练和模型，验证认证、推理与模型版本封存，
并保留资源供检查。重启、更新或恢复后，增加 `--verify-existing` 参数检查同一批资源。
其他地址通过 `--base-url` 指定。

## 预训练模型与邮件

下载的 Hugging Face 文件位于运行卷中的 `HF_HOME`，重建容器后仍保留。
如已有本地模型快照，可创建可选的 `compose.models.yml`：

```yaml
services:
  backend:
    volumes:
      - ./models:/app/models:ro
    environment:
      PEPPLATFORM_OFFLINE_MODE: "1"
```

将完整快照放到所选路线对应目录，例如 `models/esm2`、`models/protbert`、
`models/prott5` 或 `models/EvolutionaryScale/esmc-600m-2024-12`，然后组合启动：

```bash
docker compose -f docker-compose.yml -f compose.models.yml up --build -d --wait
```

后续操作始终使用同一组 Compose 文件。需要在线下载时移除离线设置。
挂载本身不会下载权重或授予受限模型访问权限；需遵循提供方的模型许可证和授权要求。

邮件配置使用 `PEP_SMTP_HOST`、`PEP_SMTP_PORT`、`PEP_SMTP_USER`、
`PEP_SMTP_PASSWORD`、`PEP_SMTP_FROM`。587 端口通常使用 STARTTLS，465 端口自动使用
隐式 TLS，发件人应与认证邮箱一致。QQ 邮箱密码填邮箱授权码。
修改环境后执行 `docker compose up -d --wait`；`docker compose restart` 不会重新加载 `.env`。

## 更新与持久化

保持仓库目录和 Compose 项目名稳定，它们决定使用哪些命名卷。
`postgres_data` 保存数据库；`pep_runtime` 保存上传数据、训练产物、版本化输入和
Hugging Face 缓存。部署使用单后端进程；更新前需等待训练完成或主动取消，
后端重启会取消进行中的实验。

完成备份后更新：

```bash
git pull --ff-only
docker compose up --build -d --wait --wait-timeout 300
docker compose ps
curl --fail http://127.0.0.1:8080/api/health
```

检查登录、已有数据集和已有模型推理。三个密钥保持不变：修改 `.env` 中的数据库密码
不会修改已初始化数据库的用户密码；会话密钥用于保留会话，加密密钥用于读取已保存的
AI 提供方 API key。`docker compose down` 保留命名卷，**`down -v` 会删除卷内数据**。

## 备份与恢复

等待训练结束，停止两个应用服务，取得一致的数据库与运行文件备份；数据库保持运行。
在仓库根目录执行，每次使用新的备份目录：

```bash
mkdir -m 700 backup
docker compose stop frontend backend
docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > backup/database.dump
docker compose run --rm --no-deps -T backend tar -C /var/lib/pepplatform -czf - . > backup/runtime.tar.gz
cp .env backup/env
git rev-parse HEAD > backup/source-commit.txt
docker compose up -d --wait
```

逐项确认命令成功，并将备份安全存放到服务器之外；备份环境文件包含凭据。
本地挂载的模型快照需额外备份，或记录其精确版本的获取方式；同时保存 Compose
覆盖文件以及匹配的应用源码或镜像。

恢复时，使用记录的源码版本、新工作目录与空卷。先将备份的环境文件恢复为 `.env`，
补回 Compose 覆盖文件及模型快照，保持数据库用户名和库名与备份一致。然后执行：

```bash
docker compose build
docker compose up -d --wait db
docker compose run --rm --no-deps -T backend tar -C /var/lib/pepplatform -xzf - < backup/runtime.tar.gz
docker compose exec -T db sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --exit-on-error --no-owner --no-privileges' < backup/database.dump
docker compose up -d --wait --wait-timeout 300
```

恢复后先检查登录、数据集和已有模型推理，再接入用户流量。
升级失败需将数据库、运行文件与对应源码或镜像一起恢复，单独回退代码不会撤销数据库迁移。

## 故障排查

```bash
docker compose ps -a
docker compose logs --tail=200 db backend frontend
```

- **缺少密钥：** 补齐三个必填密钥，再执行 `docker compose config --quiet`。
  避免公开完整渲染后的 Compose 配置，其中包含凭据。
- **构建下载失败：** 根据失败步骤检查主机和容器网络、代理。npm 镜像可通过
  `docker compose build frontend --build-arg NPM_REGISTRY=https://your-registry.example`
  指定，然后正常启动。
- **后端不健康或 HTTP 502：** 查看后端迁移、启动日志及数据库健康状态；
  健康接口正常后再尝试训练。
- **无法登录：** 新账号需同时填写两个初始管理员字段，或使用可用 SMTP 完成注册与重置。
  修改初始管理员字段不会替换已有密码。HTTP 本地测试需使用非 Secure Cookie，
  检查 `PEP_PUBLIC_BASE_URL` 和 `PEP_COOKIE_SECURE` 覆盖值。
- **找不到预训练模型：** 提供所选路线的完整快照，或允许在线下载并配置必要的 Hugging Face 授权。
- **上传 HTTP 413：** 检查 `.env` 上传限制及 Nginx 的 512 MiB 请求上限。
  改环境配置后重建后端容器，改 `docker/nginx.conf` 后重新构建前端镜像。

如果 Linux 主机的代理只监听 `127.0.0.1`，构建容器无法通过自己的回环地址访问它。
主机已设置 `http_proxy` / `https_proxy` 时，可临时创建构建网络覆盖文件：

```yaml
# compose.build-network.yml，仅供这种构建网络环境使用
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

此设置只改变构建网络，不改变服务运行时的 Compose 网络。代理地址和凭据不应写入
Dockerfile 或 Git。
