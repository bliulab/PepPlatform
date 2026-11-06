# Docker 部署与运维

[English](deployment.md) · [在线手册](https://bliulab.github.io/pepplatform-docs/zh/install/docker)

## 首次部署

基础环境为 Linux amd64，需要 Docker Engine、Docker Compose v2.35 或更新版本、
Git、OpenSSL 和 curl；Compose 需支持 `--wait`、`--wait-timeout`、`depends_on.restart`、
GPU 设备请求及 CPU 覆盖文件的 `!reset` 标签。默认后端对齐论文的核心 GPU 软件基线：

| 组件 | 默认论文 GPU 版 | 显式 CPU 版 |
| --- | --- | --- |
| Python | 3.10.19 | 3.10.19 |
| Torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 | 2.5.1+cpu / 0.20.1+cpu |
| CUDA / cuDNN | 12.1 / 9.1（90100） | 不使用 |
| Transformers / Tokenizers | 4.48.1 / 0.21.4 | 相同 |
| ESM / SentencePiece | 3.2.0 / 0.2.1 | 相同 |
| Hugging Face Hub / Safetensors | 0.36.0 / 0.7.0 | 相同 |

默认方案要求宿主机具有兼容的 NVIDIA 显卡、驱动及已配置好的
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)，
先在宿主机确认 `nvidia-smi` 可用。镜像构建只检查编译版本，不要求构建时能访问 GPU；
GPU 实际执行能力在容器启动后单独检查。没有 GPU 容器支持时，选择[CPU 方案](#cpu-版部署)。
构建期间需要访问 Docker Hub、Debian 软件源、所选 PyTorch wheel 源、PyPI 和 npm。
为镜像、数据集、训练产物及模型权重分别预留空间；训练内存需求取决于路线和数据规模。
核心软件基线一致仍需配合同一实验的数据、配置及硬件条件，才能评估数值复现。

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
docker compose build
docker compose run --rm --no-deps backend python docker/check_runtime.py --channel cu121 --device cuda
docker compose up -d --wait --wait-timeout 300
docker compose ps
curl --fail http://127.0.0.1:8080/api/health
```

打开 <http://127.0.0.1:8080>，用初始管理员账号登录。后端在提供服务前自动运行
Alembic 数据库迁移；Nginx 等待后端健康检查通过后启动。新安装不附带研究数据或
预训练权重。上传自己的数据，或生成用于验证流程的微型合成数据：

```bash
python3 tools/build_web_route_smoke_dataset.py --output /tmp/pepplatform-demo.zip
```

将 ZIP 上传为**肽段功能预测**数据集。无需下载预训练权重的短流程检查可选择
**k-mer → One-hot → CNN**，训练一轮，使用较小 batch size。确认训练完成并出现模型，
再对 `AAAAAAAAAAAA`、`CCCCCCCCCCCC` 做推理，确认返回分数。合成数据结果不具有科学意义。

也可自动检查（仅需 Python 3 标准库）：设置环境变量 `PEP_SMOKE_EMAIL`、
`PEP_SMOKE_PASSWORD` 为登录账号，再执行：

```bash
python3 tools/check_web_deployment.py --state /tmp/pepplatform-smoke.json
```

脚本会创建合成数据集、一轮训练和模型，验证认证、推理与模型版本封存，
并保留资源供检查。重启、更新或恢复后，增加 `--verify-existing` 参数检查同一批资源。
其他地址通过 `--base-url` 指定。

### CPU 版部署

沿用相同的 `.env`，在**每一次** CPU 命令中叠加两个 Compose 文件，包括模型准备、
后续更新、备份和恢复：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  backend python docker/check_runtime.py --channel cpu --device cpu
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --wait --wait-timeout 300
```

CPU 覆盖文件选择 CPU wheel，并清除 GPU 请求，无需 NVIDIA Container Toolkit。
两种方案使用相同的冻结权重和 ProtT5 还原流程。CPU 是兼容部署选项，不承诺与论文 GPU
训练数值逐位一致；切换方案时重新构建后端。如还需要论文模型资产，使用同一 CPU 前缀准备和检查：

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  backend python docker/download_models.py
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm --no-deps \
  backend python docker/download_models.py --check-only --smoke-test --device cpu
```


### 国内镜像与下载源

各层下载需要分别配置：

| 下载内容 | 配置位置 |
| --- | --- |
| Docker 基础镜像 | 宿主机 Docker 服务的代理或 registry mirror |
| Python 包及 CUDA 依赖 | 构建配置 `PEP_PIP_INDEX_URL` |
| Torch 与 torchvision wheel | 构建配置 `PEP_TORCH_INDEX_BASE_URL` |
| 预训练模型文件 | 运行时配置 `HF_ENDPOINT` |

国内服务器可在本地 `.env` 中显式选择：

```dotenv
PEP_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PEP_TORCH_INDEX_BASE_URL=https://mirrors.nju.edu.cn/pytorch/whl
PEP_PIP_TIMEOUT=120
HF_ENDPOINT=https://hf-mirror.com
```

随后使用正常的构建和模型准备命令。Torch 源地址应止于 `whl`，不要附加通道：
GPU 方案自动追加 `/cu121`，CPU 方案自动追加 `/cpu`。Dockerfile 先从该通道
下载固定版本的 Torch、torchvision wheel，再从所选 PyPI 源安装它们的依赖。
镜像构建成功前会检查 Python、Torch、CUDA 等核心版本。可参考
[TUNA PyPI 帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)和
[南大 PyTorch 索引](https://mirrors.nju.edu.cn/pytorch/whl/cu121/torch/)。

默认仍使用官方包源和 Hugging Face 源。可以只切换国内依赖源，保留
`HF_ENDPOINT=https://huggingface.co`。HF 镜像的大文件可能继续跳转到原 CDN，
实际是否加速需要测量文件传输。令牌会发送给所选 `HF_ENDPOINT`；
受限模型仍需授权，镜像也不保证包含全部固定 revision。
所有模型仍接受原锁定哈希校验，ProtT5 仍需通过原格式的精确还原检查。

修改 Python/wheel 源后重新构建后端；修改 `HF_ENDPOINT` 后的新准备容器使用新端点。
Docker 服务的基础镜像代理或 registry mirror 属于单独的宿主机配置，
设置 pip 或 Hugging Face 镜像不会改变基础镜像的拉取路径。


## 准备与论文候选实验一致的模型

### 先选择用途

| 使用目的 | ProtT5 模型选择 |
| --- | --- |
| 复现论文中使用 ProtT5 的训练实验 | 运行下方准备命令，或使用已验证的原六分片、离线模型包；训练使用与原实验一致的模型入口。 |
| 开展自己的实验或提取特征 | 可以直接使用平台兼容的官方 ProtT5 原始快照，显式指定自己的模型目录，无需格式还原。 |

准备命令是一键取得论文固定模型资产的推荐入口，平台启动和独立实验无需先执行这一步。
复现论文结果还需匹配后文说明的源码、数据划分、实验配置、随机种子和运行环境。

`docker/models.lock.json` 固定四类任务当前验收的 100 条候选及集成成员使用的资源。
五类模型需要**六套权重**，因为 ESM-C 同时使用过 300M 和 600M：

| 资源 | 固定版本（完整 revision、逐文件 SHA-256 和内容指纹见锁定清单） |
| --- | --- |
| ESM2 150M | `facebook/esm2_t30_150M_UR50D`，`a695f6045e2e32885fa60af20c13cb35398ce30c` |
| ProtBERT | `Rostlab/prot_bert`，`7a894481acdc12202f0a415dd567f6cfdb698908` |
| ProtT5-XL | 官方单文件 `973be27c52ee6474de9c945952a8008aeb2a1a73`，自动还原为原实验六分片 |
| ESM-C 300M | `EvolutionaryScale/esmc-300m-2024-12`，`7f10b20ae75017b2dbc884070e03434515709a8d` |
| ESM-C 600M | `EvolutionaryScale/esmc-600m-2024-12`，`e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c` |
| ESM3 Open Small | `EvolutionaryScale/esm3-sm-open-v1`，`47f0545b2b6daf26a93439a3cd610f4f7f3d5478` |

ProtT5 准备流程会下载固定版本的官方 `.bin`，再根据小型 `docker/prott5.restore.json`
配方在本地还原原实验文件，逐文件核对已有锁定清单。用户无需下载你提供的六分片模型包，
训练继续使用原格式，从而保留原加载过程及随机数行为。
完整模型约 3B 参数，平台加载其中约 1.2B 参数的编码器；详见[验证说明](prott5-equivalence.zh-CN.md)。

权重不提交 Git，也不写入应用镜像。安装后的六套资源约 22.7 GB；在线准备还会保留官方缓存。
ProtT5 官方 `.bin` 与还原文件各约 11.3 GB。下载缓存和安装副本建议至少预留 50 GB，
额外模型包、镜像、业务数据和旧版本备份另计。

### 目标服务器在线准备并启动

复现论文实验时，先配置 `.env`，为受限资源填写已获授权的令牌，再执行：

```bash
docker compose build
docker compose run --rm --no-deps backend python docker/download_models.py
docker compose run --rm --no-deps backend python docker/download_models.py --check-only
docker compose up -d --wait --wait-timeout 300
```

准备命令覆盖全部六套权重，复用合格安装和下载缓存，ProtT5 只需还原一次并保存在运行卷中。
准备阶段不启动数据库，也不执行数据库迁移。

已有固定版本的官方 ProtT5 快照时，可只读挂载并复用：

```bash
docker compose run --rm --no-deps \
  -v "$PWD/backup/prott5-official:/official-prott5:ro" \
  backend python docker/download_models.py --prott5-source /official-prott5
```

目录须包含固定版本的 `pytorch_model.bin`、`config.json`、`spiece.model`、
`special_tokens_map.json`、`tokenizer_config.json`，五个文件均会校验。
该选项不会为 ProtT5 发起网络请求，但仍会还原论文格式，其他缺失模型按正常流程准备；
不能与导出、导入或检查模式混用。直接加载原文件使用下节介绍的训练参数。

裸机 Python 环境可运行 `python docker/download_models.py --root backup/pretrained`，
训练时将 `--prott5_model` 指向 `backup/pretrained/models/prott5` 的绝对路径。
论文复现应使用这个准备后的入口；直接加载官方 `.bin` 原目录仍有随机数行为差异。

### 独立实验：原样使用官方 ProtT5

开展自己的训练或特征提取时，可以保留平台兼容的官方 ProtT5 完整快照，包括配置、
分词器和权重，无需还原格式。在已有 CLI 训练命令中加入以下参数，显式选择原始目录：

```text
--prott5_model /opt/prott5-official
```

显式绝对路径优先于已经安装的论文复现缓存；只填写仓库名时，自动解析仍可能选中已有本地模型。
Docker 中应将自行下载的目录只读挂载到 `/opt/prott5-official`，训练命令使用这个容器内路径；
裸机运行则填写模型在宿主机上的绝对路径。

| 参数 | 使用位置 | 行为 |
| --- | --- | --- |
| `--prott5-source DIR` | `docker/download_models.py` | 校验固定官方快照，再还原为论文原格式后安装。 |
| `--prott5_model DIR` | CLI 训练命令 | 直接加载指定的兼容快照，不执行格式还原。 |

直接加载 `.bin` 的随机数行为与论文原分片不同，因此这种独立实验选择不能据此承诺复现旧训练结果。
加载论文已有训练产物时，仍需满足其记录的模型身份校验。使用自己的兼容快照训练新模型，
无需执行论文模型准备命令。

### 可选：原服务器导出离线模型包

在实现本功能的源码目录运行，`--reference` 指向研究目录的原始登记文件。
脚本核对登记、验收索引、候选元数据及模型内容，再导出原文件；输出目录必须尚不存在。

```bash
python3 docker/download_models.py --export backup/pretrained-models \
  --reference ../PepPlatform/experiment_paper/current_results.json
```

模型包包含锁定清单、回执及模型原文件，缓存符号链接会复制成普通文件。
将整个 `backup/pretrained-models` 目录复制到目标服务器的同名目录；不要只复制权重文件。
研究目录及其原始模型不会被修改，模型包无需上传 GitHub。

### 可选：目标服务器导入模型包并启动

先按前文配置 `.env`。导入命令独立运行，不启动数据库，也不执行数据库迁移：

```bash
docker compose build
docker compose run --rm --no-deps \
  -v "$PWD/backup/pretrained-models:/model-package:ro" \
  backend python docker/download_models.py --import-from /model-package
docker compose run --rm --no-deps backend python docker/download_models.py --check-only
docker compose up -d --wait --wait-timeout 300
```

导入全程无需联网，逐文件校验后才启用模型，全部六套通过才返回成功。
模型保存在已有 `pep_runtime` 卷的 `/var/lib/pepplatform/pretrained`，镜像中的
`/app/models` 指向其模型入口。公开快照保留 `snapshots/<revision>` 层级，分片模型保留
原目录内容与指纹。原有运行卷备份已覆盖这些文件；模型包也应另行保存。
再次执行会校验并复用已有文件，损坏的安装副本会保留为 `.previous-*` 后重装；
准备失败可修复模型包后重试，不能将部分成功当成准备完成。

默认 GPU 检查会在 CUDA 上逐个加载模型、运行短序列、核对参数实际设备并释放模型。
CUDA 不可用或依赖版本不符时明确失败，需要预留足够主机内存和显存：

```bash
docker compose run --rm --no-deps backend \
  python docker/download_models.py --check-only --smoke-test --device cuda
```

### 官方源、镜像与离线模式

下载只请求模型锁和固定 ProtT5 还原配方列出的文件。ProtT5 确定性转换的输出须匹配
原锁定清单，不会自动切换训练阶段、版本或下载源。模型锁本身保持不变，因此旧模型包仍可导入。
是否需要镜像取决于服务器的网络。`.env` 默认 `HF_ENDPOINT=https://huggingface.co`；
需要第三方镜像时可显式改为 `HF_ENDPOINT=https://hf-mirror.com`。
`HF_TOKEN` 会发送给所选端点；受限资源仍需提供方授权，镜像不保证包含全部固定版本。
若镜像不支持 Xet，可设置 `HF_HUB_DISABLE_XET=1`；超时由
`HF_HUB_DOWNLOAD_TIMEOUT` 和 `HF_HUB_ETAG_TIMEOUT` 控制。

六套验证完成后，可在 `.env` 中设置 `HF_HUB_OFFLINE=1` 和
`PEPPLATFORM_OFFLINE_MODE=1`，然后重新创建后端容器。原模型包导入和 `--check-only`
始终不联网。普通平台启动不会自动下载全部模型，已有本地挂载也仍可按下文使用。

模型文件一致是复现条件之一。锁定清单同时记录论文参考依赖环境；完整数值复现仍需
冻结源码、数据划分、实验配置、随机种子和原 CPU/CUDA 环境。CPU/GPU 短序列检查
只证明资源在所选设备上可加载，不等于重新训练 100 条候选后的数值验收。
2026-11-06 的 Docker 验收已完成六套空缓存下载、GPU/CPU 断网加载，以及 ProtT5 的
随机数、63 条序列特征和三步合成训练对照。实际镜像、网络配置和通过范围见
[当前环境验收记录](paper-runtime-validation.json)。

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

### 模型下载的磁盘空间

六套已安装模型约占 21.1 GiB，Hugging Face 下载缓存还会保留一份源文件。
建议为缓存、安装文件和重试预留至少 100 GB。检查 Docker 数据目录所在磁盘的空间；
仓库所在磁盘的空闲空间并不等于 Docker 卷的可用空间。

全新部署可把运行卷放到大容量磁盘。先在该磁盘创建目录，将下面示例中的
`/mnt/data/pepplatform-runtime` 替换为实际路径，保存为 `compose.model-storage.yml`：

```yaml
volumes:
  pep_runtime:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/pepplatform-runtime
```

在所有 Compose 命令中叠加此文件；CPU 用户同时保留 CPU 覆盖文件：

```bash
docker compose -f docker-compose.yml -f compose.model-storage.yml run --rm --no-deps \
  backend python docker/download_models.py
```

此示例用于新建运行卷。已有部署迁移前需备份并迁移完整运行目录，再采用新的卷名称。

### 模型下载的容器代理

Docker 服务代理、构建代理和后端运行时代理需要分别配置。只在宿主机设置代理，
不会自动把代理传入执行下载命令的后端容器。需要代理时，可保存以下本地覆盖文件
`compose.download-proxy.yml`，并在本地 `.env` 中设置
`PEP_DOWNLOAD_PROXY=http://host.docker.internal:7890`，将端口替换为实际值：

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

代理需监听容器可访问的宿主机地址；仅监听宿主机 `127.0.0.1` 的代理无法通过
此桥接网络示例访问。下载命令增加 `-f compose.download-proxy.yml`，
需要自定义存储或 CPU 方案时再叠加相应文件。代理只改变连接路径，
模型仍由固定端点提供并接受原锁定清单校验。

如果代理只接受宿主机回环连接，或桥接连接的 TLS 请求持续失败，可先用相同镜像确认
宿主机网络可用，再为一次性模型准备增加 `compose.download-host.yml`：

```yaml
services:
  backend:
    network_mode: host
```

此时将本地 `PEP_DOWNLOAD_PROXY` 改为宿主机实际回环地址，例如
`http://127.0.0.1:7890`，并执行：

```bash
docker compose -f docker-compose.yml -f compose.download-proxy.yml -f compose.download-host.yml \
  run --rm --no-deps backend python docker/download_models.py
```

存储和 CPU 覆盖文件仍按需叠加。此宿主机网络覆盖仅用于一次性模型准备；
完整 Web 服务继续使用其正常 Compose 网络。

### 代理的 TLS 兼容模式

若同一镜像的 Hugging Face 默认 TLS 请求持续出现握手超时或 `SSLEOFError`，
而 TLS 1.2 对照请求能够成功，可在本地 `.env` 中设置 `PEP_HF_TLS12=1` 后重跑准备命令。
此开关仅影响模型准备脚本：保留 HTTPS 证书和主机名校验，将连接限制为 TLS 1.2，
并关闭独立 HTTP 客户端的 Xet，改用标准 HTTP 下载。默认值为 `0`，使用正常 TLS/Xet 配置。
该选项不改变模型端点、revision、文件哈希或 ProtT5 还原要求，离线导入和检查不受影响。
