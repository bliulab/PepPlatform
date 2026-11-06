# Docker 部署与运维

[English](deployment.md) · [在线手册](https://bliulab.github.io/pepplatform-docs/zh/install/docker)

## 首次部署

先按宿主机硬件选择一篇完整指南。各指南包含环境配置、首次登录、基础训练验收、
可选模型准备、更新、备份恢复和故障排查；已有服务从所选指南的更新章节开始。

| 部署方式 | 适用环境 | 完整操作指南 |
| --- | --- | --- |
| CPU | 无 NVIDIA GPU，或选择仅用 CPU | [CPU 部署与运维](deployment-cpu.zh-CN.md) |
| GPU | 兼容的 NVIDIA GPU、驱动及 NVIDIA Container Toolkit | [GPU 部署与运维](deployment-gpu.zh-CN.md) |

本页集中说明共享配置和模型原理。所有部署及运维 Compose 命令均放在对应硬件指南中，
无需在共享示例中手动替换硬件参数。CPU、GPU 使用相同数据卷和冻结模型资产。
项目支持 Linux amd64、Docker Engine 和 Docker Compose v2.35 或更新版本；
此版本为项目支持下限。

核心依赖基线：

| 组件 | 默认论文 GPU 版 | 显式 CPU 版 |
| --- | --- | --- |
| Python | 3.10.19 | 3.10.19 |
| Torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 | 2.5.1+cpu / 0.20.1+cpu |
| CUDA / cuDNN | 12.1 / 9.1（90100） | 不使用 |
| Transformers / Tokenizers | 4.48.1 / 0.21.4 | 相同 |
| ESM / SentencePiece | 3.2.0 / 0.2.1 | 相同 |
| Hugging Face Hub / Safetensors | 0.36.0 / 0.7.0 | 相同 |

服务健康、模型资源可用和论文数值复现分别需要验收。Web 服务可以先使用无需预训练权重的路线；
预训练路线需要对应模型；完整论文复现还需匹配源码、数据划分、实验配置、随机种子和硬件。

### CPU 版部署

进入 [CPU 完整指南](deployment-cpu.zh-CN.md)。每条操作均明确保留基础文件和 CPU 覆盖文件，
包括模型准备、更新、备份恢复和可选代理。CPU 覆盖文件选择 CPU wheel 并清除 GPU 请求，无需 NVIDIA Container Toolkit。
CPU 是兼容部署选项，不承诺与论文 GPU 训练数值逐位一致。

### GPU 版部署

进入 [GPU 完整指南](deployment-gpu.zh-CN.md)。宿主机需具备兼容 NVIDIA GPU、驱动和已配置好的
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)。
构建只检查依赖版本；独立运行环境检查会验证 CUDA 实际执行。切换 CPU/GPU 方案时重新构建后端。

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
配方在本地还原原实验文件，逐文件核对已有锁定清单。无需另行取得原六分片模型包，
训练继续使用原格式，从而保留原加载过程及随机数行为。
完整模型约 3B 参数，平台加载其中约 1.2B 参数的编码器；详见[验证说明](prott5-equivalence.zh-CN.md)。

权重不提交 Git，也不写入应用镜像。安装文件合计 22.68 GB（21.13 GiB）；当前在线准备保留源缓存后约 45.37 GB（42.25 GiB），额外开销另计。
详见[磁盘预算与省空间路径](#模型下载的磁盘空间)。

### 目标服务器在线准备并启动

按所选硬件执行完整步骤：[CPU 在线准备](deployment-cpu.zh-CN.md#在线准备) · [GPU 在线准备](deployment-gpu.zh-CN.md#在线准备)。
两篇指南也分别给出复用官方 ProtT5 快照的命令；`--prott5-source` 只省去 ProtT5 下载，仍会还原论文格式。
准备容器不启动数据库、不执行数据库迁移，重复执行会复用合格安装和下载缓存。

裸机 Python 环境可运行 `python docker/download_models.py --root backup/pretrained`，
训练时将 `--prott5_model` 指向 `backup/pretrained/models/prott5` 的绝对路径。
论文复现应使用准备后的入口；直接加载官方 `.bin` 原目录仍有随机数行为差异。

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

完整导入与设备检查命令：[CPU 离线导入](deployment-cpu.zh-CN.md#离线导入) · [GPU 离线导入](deployment-gpu.zh-CN.md#离线导入)。
导入逐文件校验，全部六套通过才返回成功；模型保存在 `pep_runtime` 卷的 `/var/lib/pepplatform/pretrained`。
镜像中的 `/app/models` 指向安装入口，运行卷备份已覆盖这些文件。外部模型包仍应另行保存。

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

下载缓存位于运行卷中的 `HF_HOME=/var/lib/pepplatform/huggingface`，重建容器后仍保留。
它与 `pretrained` 中的安装副本是独立文件；后者在本地安装完成后可独立加载。

如已有本地模型快照，可创建可选的 `compose.models.yml`：

```yaml
services:
  backend:
    volumes:
      - ./models:/app/models:ro
    environment:
      PEPPLATFORM_OFFLINE_MODE: "1"
```

完整快照放到路线对应目录，例如 `models/esm2`、`models/protbert`、`models/prott5` 或
`models/EvolutionaryScale/esmc-600m-2024-12`。此挂载会覆盖镜像中 `/app/models` 的默认入口，
应确认所需模型均在挂载目录中。挂载本身不下载模型，也不授予受限模型访问权限。
组合启动命令见 [CPU 可选配置](deployment-cpu.zh-CN.md#可选配置与覆盖文件)或 [GPU 可选配置](deployment-gpu.zh-CN.md#可选配置与覆盖文件)，
后续操作保留相同持久覆盖文件。需要在线下载时移除离线设置，并遵循提供方的许可证和授权要求。

邮件配置使用 `PEP_SMTP_HOST`、`PEP_SMTP_PORT`、`PEP_SMTP_USER`、`PEP_SMTP_PASSWORD` 和 `PEP_SMTP_FROM`。
587 端口通常使用 STARTTLS，465 端口自动使用隐式 TLS，发件人应与认证邮箱一致；QQ 邮箱密码填邮箱授权码。
修改 `.env` 后，使用所选指南的环境应用命令重新创建服务；普通重启不会重新加载环境文件。

## 更新与持久化

保持仓库目录和 Compose 项目名稳定，保留三个密钥和持久覆盖文件。`postgres_data` 保存数据库；
`pep_runtime` 保存上传、训练产物、版本化输入、预训练安装和 Hugging Face 缓存。
部署使用单后端进程，重启会取消进行中的实验。

完整升级步骤：[CPU 更新](deployment-cpu.zh-CN.md#更新与持久化) · [GPU 更新](deployment-gpu.zh-CN.md#更新与持久化)。

## 备份与恢复

先结束训练，停止两个应用服务，保持数据库运行，再取得一致的数据库及运行文件备份。
备份环境文件、覆盖文件、源码版本及外部模型；恢复使用匹配源码和空目标卷。
完整步骤：[CPU 备份恢复](deployment-cpu.zh-CN.md#备份与恢复) · [GPU 备份恢复](deployment-gpu.zh-CN.md#备份与恢复)。

升级失败时，数据库、运行文件与匹配源码或镜像一起恢复；只回退代码不会撤销数据库迁移。

## 故障排查

先按 [CPU 排查步骤](deployment-cpu.zh-CN.md#故障排查)或 [GPU 排查步骤](deployment-gpu.zh-CN.md#故障排查)查看状态及日志。
下载需区分 Docker 基础镜像、构建依赖和运行时模型三层网络，配置其中一层不会自动改变其他层。

如果 Linux 主机的代理只监听 `127.0.0.1`，构建容器无法通过自己的回环地址访问它。
主机已设置 `http_proxy` / `https_proxy` 时，可创建本地 `compose.build-network.yml`：

```yaml
services:
  backend:
    build:
      network: host
  frontend:
    build:
      network: host
```

构建命令见相应指南的可选配置章节。此覆盖只改变构建网络，不改变运行时 Compose 网络。
代理地址和凭据不应写入 Dockerfile 或提交 Git。

### 模型下载的磁盘空间

预算应区分最终安装、准备期间保留的源文件和业务数据。以下按
[`models.lock.json`](../docker/models.lock.json) 与 [ProtT5 还原配方](../docker/prott5.restore.json)中的文件大小求和，
GB 为十进制，GiB 为二进制单位：

| 内容或路径 | 模型文件大小 |
| --- | ---: |
| 六套安装文件 | 22.68 GB / 21.13 GiB |
| 在线准备保留的 Hugging Face 源文件缓存 | 22.68 GB / 21.13 GiB |
| 当前在线流程：源缓存与安装副本合计 | 45.37 GB / 42.25 GiB |
| 离线包在另一块盘：目标运行卷的安装文件 | 22.68 GB / 21.13 GiB |
| 离线包与安装文件在同一块盘 | 45.37 GB / 42.25 GiB |

脚本把下载缓存复制成独立安装文件；安装暂存目录校验后通过重命名启用，正常首次安装不会因此额外再留一整份。
ProtT5 官方源 `.bin` 与还原分片各约 11.28 GB，已包含在上表，不能再重复叠加。
上述数字是正常流程的文件预算，不含额外 Xet 缓存、失败下载、损坏安装保留的 `.previous-*` 或其他模型。
因此应在该预算之外留余量。100 GB 可作为容纳额外缓存和修复副本的宽裕示例，**不是统一最低要求**。

Docker 镜像与构建缓存、数据库、上传数据、训练产物、模型包副本和备份都需单独预算。
备份完整运行卷时会包含其中的安装文件和缓存，权重的压缩收益可能有限。

更省空间的现有用法：

- 将离线包放另一块盘，导入只在目标卷写入安装副本；镜像构建依赖仍需另行准备。
- 六套安装及设备检查通过后，可按 [CPU 缓存清理](deployment-cpu.zh-CN.md#可选清理模型下载缓存)或 [GPU 缓存清理](deployment-gpu.zh-CN.md#可选清理模型下载缓存)回收固定源缓存，再离线复验。此方式降低长期占用，不降低首次在线准备峰值；共享 blob 和其他缓存会影响回收量。
- 单纯把缓存移到另一块盘不会降低总用量，但可分摊各磁盘的容量需求。当前脚本没有逐模型安装后自动清缓存的模式。

默认运行卷与缓存均位于 Docker 管理的存储中；仓库旁的空闲空间不能代表卷容量：

```bash
docker_root="$(docker info --format '{{.DockerRootDir}}')"
df -h "$docker_root" .
```

全新部署可将运行卷绑定到大容量磁盘。先创建实际目录，替换路径并保存为本地 `compose.model-storage.yml`：

```yaml
volumes:
  pep_runtime:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/pepplatform-runtime
```

检查该目录所在磁盘的空间。此示例用于**新建运行卷**；已有卷不会因更改选项自动迁移。
已有部署需先备份并迁移完整运行目录，再采用新的卷名称，避免把空目录当成原数据。
每次操作保留此持久覆盖文件，命令见对应硬件指南的可选配置章节。

### 模型下载的容器代理

Docker 服务代理、构建代理和后端运行时代理分别配置。宿主机代理不会自动传入下载容器。
需要时保存本地 `compose.download-proxy.yml`，并在 `.env` 设置
`PEP_DOWNLOAD_PROXY=http://host.docker.internal:7890`，替换实际端口：

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

代理需监听容器可访问的宿主机地址；仅绑定宿主机 `127.0.0.1` 时，上述桥接网络方式无法访问。
命令见 [CPU 可选配置](deployment-cpu.zh-CN.md#可选配置与覆盖文件)或 [GPU 可选配置](deployment-gpu.zh-CN.md#可选配置与覆盖文件)。
代理只改变连接路径，仍从所选端点取得固定模型并校验哈希。

若代理仅接受宿主机回环连接，或桥接网络的 TLS 持续失败，先用同一镜像验证宿主机网络连通性。
一次性准备可增加本地 `compose.download-host.yml`：

```yaml
services:
  backend:
    network_mode: host
```

将 `PEP_DOWNLOAD_PROXY` 改为实际宿主机回环地址，例如 `http://127.0.0.1:7890`。
组合命令仍见对应硬件指南，同时保留存储等持久覆盖文件。此宿主机网络覆盖仅用于一次性准备，
完整 Web 服务继续使用正常 Compose 网络。

### 代理的 TLS 兼容模式

若同一镜像的 Hugging Face 默认 TLS 请求持续出现握手超时或 `SSLEOFError`，
而 TLS 1.2 对照请求能够成功，可在本地 `.env` 中设置 `PEP_HF_TLS12=1` 后重跑准备命令。
此开关仅影响模型准备脚本：保留 HTTPS 证书和主机名校验，将连接限制为 TLS 1.2，
并关闭独立 HTTP 客户端的 Xet，改用标准 HTTP 下载。默认值为 `0`，使用正常 TLS/Xet 配置。
该选项不改变模型端点、revision、文件哈希或 ProtT5 还原要求，离线导入和检查不受影响。
