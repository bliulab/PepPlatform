# PepPlatform

[English](README.md) | 简体中文

PepPlatform 是面向肽段建模的平台，提供命令行训练与 Web 实验管理、模型推理。

## 支持的任务

| 任务 | `--task_type` |
| --- | --- |
| 肽段功能预测 | `pep_function` |
| 肽–蛋白互作 | `pep_protein_interaction` |
| 肽–HLA-I 互作 | `pep_hlai` |
| 肽–HLA-II 互作 | `pep_hlaii` |

## 核心能力

- **特征表征**：经典序列特征与预训练蛋白语言模型表征。
- **模型类型**：传统机器学习、序列网络和图模型。
- **建模流程**：训练、基于验证集的模型选择、评估与模型导出。

## Python 依赖

先获取源码，再选择安装方式：

```bash
git clone https://github.com/bliulab/PepPlatform.git
cd PepPlatform
```

使用 Python 3.10.19，按实际用途选择依赖清单：

| 文件 | 安装范围 |
| --- | --- |
| `requirements.txt` | CLI 训练和经典特征使用的基础依赖 |
| `requirements-web.txt` | 基础训练依赖，以及 Web 后端、数据库驱动 |
| `requirements-pretrained.txt` | 基础训练依赖，以及 Transformers、SentencePiece、Protobuf、ESM |
| `requirements-dev.txt` | Web、预训练扩展，以及 pytest、httpx、Ruff |

科学计算依赖与参考训练环境保持一致，包括 NumPy `2.2.6`、SciPy `1.15.3`、
scikit-learn `1.7.2`、XGBoost `3.1.3` 和 LightGBM `4.6.0`。
验证使用的 PyTorch 版本为 `2.5.1`；预训练路线还使用配套的
`torchvision==0.20.1`。默认使用论文的 CUDA 12.1 构建，CPU 安装方式单独列出；
参见[PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v251)。
模型权重、训练数据和 BLAST 参考数据库需要另行准备。
显式版本化 CLI 复用 Web 的元数据和运行器组件，因此还需安装 `requirements-web.txt`。

## CLI 快速上手

使用 **Python 3.10.19**，在仓库根目录执行以下命令。
环境准备详见 [CLI 安装指南](https://bliulab.github.io/pepplatform-docs/zh/install/cli-only)。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
python -m pip install -r requirements.txt
```

按 [数据格式规范](https://bliulab.github.io/pepplatform-docs/zh/datasets/format) 准备数据。
单序列任务的每个划分包含 `seqs.fasta` 和 `labels.csv`。
将下方的 `/path/to/sequence-data` 替换为你的数据集根目录。
以下示例使用 k-mer / one-hot 特征训练肽段功能模型：

```bash
python scripts/train_sequence_pipeline.py \
  --task_type pep_function --data_dir /path/to/sequence-data \
  --tokenizer kmer --encoder onehot --model cnn \
  --epochs 10 --artifact_dir output/pep-function
```

模型、配置和评估结果保存在 `output/pep-function/`。

肽–蛋白、HLA-I 和 HLA-II 互作任务使用 Pair 专用入口：

```bash
python scripts/train_pair_pipeline.py --help
```

完整参数与使用示例见 [CLI 参考](https://bliulab.github.io/pepplatform-docs/zh/cli/overview) 和
[训练配方](https://bliulab.github.io/pepplatform-docs/zh/cli/recipes)。

训练损失统一展示为 **BCE**、**WBCE**（正类加权 BCE）和 **ASL**。单序列 CLI 继续使用 `--multilabel_loss bce`、`balanced_bce`、`asymmetric_focal`。`auto` 保留路线默认值：通用序列网络使用 BCE，预训练池化 MLP 使用 WBCE（权重上限 20）；配对任务继续使用 `--loss_type bce|wbce`，默认 WBCE、权重上限 50。历史候选后缀 BalBCE 和 AFL 分别表示 WBCE 和 ASL。

Web 在第 3 步 **Custom → 分类损失（可选）** 中提供选择，默认折叠。参数及历史兼容说明见[分类训练损失](docs/classification-loss.zh-CN.md)。

统一的本地 CLI 支持训练、推理、本地模型集成和运行历史，无需启动 Web 服务：

```bash
python -m pepplatform_cli
python -m pepplatform_cli --help
```


## 数据与模型

数据集和预训练权重需另行准备。上述示例使用经典特征，无需预训练权重。
预训练路线需要对应的模型快照，准备方式见 CLI 安装指南。
合成示例仅用于验证使用流程。数据准备工具见[公共数据集 CLI](https://bliulab.github.io/pepplatform-docs/zh/datasets/public-databases-cli)。
PSSM/PSFM 生成还需要 BLAST+（`ncbi-blast+`）和对应路线的参考数据。

数据集和模型路径可通过 CLI 或 Web 配置指定，相对默认路径以项目目录为基准。
Web 运行数据目录可通过 `PEP_RUNTIME_ROOT` 调整。保存的 Protocol/schema 版本用于
识别产物格式，带日期的数据库迁移 ID 用于识别升级链。这些标识用于兼容性检查，
不会选择内部实验批次，也不要求存在某个带时间戳的实验目录。

预训练路线默认使用论文的 CUDA 12.1 构建及扩展依赖：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1" "torchvision==0.20.1"
python -m pip install -r requirements-pretrained.txt
```

预期版本为 Torch `2.5.1+cu121`、torchvision `0.20.1+cu121`、CUDA `12.1` 和 cuDNN `9.1`。
仅使用 CPU 时，显式选择 CPU wheel：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" "torchvision==0.20.1"
python -m pip install -r requirements-pretrained.txt
```

之后安装所需依赖清单。CPU 与 GPU 构建可能存在数值差异；复现已保存的运行还需要
对应记录中的数据、配置、随机种子和模型资源。

## Web 界面

Web 工作台提供数据与实验管理、结果对比、模型推理、集成和 AI 辅助实验规划。

新建 Web 数据集上传默认使用以下限制（1 MiB = 1,048,576 字节）：

| 环境变量 | 默认上限 |
| --- | --- |
| `PEP_DATASET_UPLOAD_MAX_ARCHIVE_MIB` | ZIP 文件 64 MiB |
| `PEP_DATASET_UPLOAD_MAX_EXTRACTED_MIB` | 解压总量 512 MiB |
| `PEP_DATASET_UPLOAD_MAX_FILE_MIB` | 单个解压文件 256 MiB |
| `PEP_DATASET_UPLOAD_MAX_ENTRIES` | 1,000 个文件和目录条目 |

在 `.env` 中设置正整数并重启后端即可调整。HTTP 请求体在 ZIP 上限之外预留
1 MiB 的 multipart 表单空间。超限返回 HTTP 413，损坏或不安全的 ZIP 返回
HTTP 400；ZIP 条目支持标准 Deflate 压缩或不压缩（Stored）。失败上传会被清理，
不按压缩比单独拒绝文件。
限制仅适用于新的 Web 上传，CLI、内置及已有数据集、训练设置和语义层计算保持原有行为。
调高 ZIP 上限时，也需检查反向代理的请求体上限；随附 Nginx 配置允许每个请求最多 512 MiB。

本地后端安装 `requirements-web.txt`；使用预训练路线时，再安装
`requirements-pretrained.txt`。所提供的 Docker 后端默认安装这两层依赖。
前端依赖在 `frontend/` 目录中通过 `npm ci` 单独安装。

Docker 部署先复制配置并编辑 `.env`：

```bash
cp .env.example .env
openssl rand -hex 32
```

为 `POSTGRES_PASSWORD`、`PEP_SESSION_SECRET` 和 `PEP_LLM_KEY_ENCRYPTION_SECRET`
分别生成一个独立值；后两个密钥需在重启和备份之间保持稳定。
首次启动前同时填写两个初始管理员字段，以创建登录账号。
注册和密码重置邮件还需要 SMTP 配置。完成后启动：

```bash
docker compose up --build -d --wait --wait-timeout 300
```

访问 <http://127.0.0.1:8080>。服务栈使用 PostgreSQL 16、Python 3.10、Node.js 22
前端构建和 Nginx 1.30。数据库状态、上传数据、运行产物和下载的 Hugging Face 模型
保存在 Docker 卷中。后端默认使用论文的 CUDA 12.1 PyTorch 构建并请求 NVIDIA GPU，宿主机需配置 NVIDIA Container Toolkit。
无 GPU 时显式使用[CPU Compose 覆盖方案](docs/deployment.zh-CN.md#cpu-版部署)，所有 Compose 命令
都保留两文件前缀。本地模型快照仍可通过可选挂载提供。

需要与论文候选实验保持一致时，构建后端后运行 `docker compose run --rm --no-deps backend
python docker/download_models.py`。该命令覆盖五类模型、六套权重，包括 ESM-C 两种规模；
ProtT5 从官方固定版本下载单文件，再自动还原原实验格式，无需用户取得原六分片模型包。
镜像、已有官方下载及可选离线导入见[锁定模型准备说明](docs/deployment.zh-CN.md#准备与论文候选实验一致的模型)。

这项准备流程用于论文复现；做自己的实验时，也可以原样使用兼容的官方 ProtT5，
通过训练参数 `--prott5_model` 指定其绝对目录。准备参数 `--prott5-source` 仍会执行还原，
两种参数及 Docker 路径的区别见上述指南。

本地开发使用前文的 Python 环境，安装 Web 依赖。在一个终端配置 AI 密钥加密所需的
密钥并启动后端：

```bash
python -m pip install -r requirements-web.txt
export PEP_LLM_KEY_ENCRYPTION_SECRET='replace-with-a-stable-random-value-at-least-16-characters'
python -m uvicorn pepplatform.web.app:app --host 127.0.0.1 --port 8081
```

另一个终端使用 Node.js 22，依次运行 `cd frontend`、`npm ci` 和 `npm run dev`，
然后访问 <http://127.0.0.1:5173>。后端会加载项目根目录中已有的 `.env`；
请在其中配置初始管理员或 SMTP 以完成登录。

仓库内的 [Docker 部署与运维指南](docs/deployment.zh-CN.md) 包含首次登录、模型挂载、
更新、备份恢复和故障排查步骤。
在线使用方式见 [Docker 部署指南](https://bliulab.github.io/pepplatform-docs/zh/install/docker) 或
[本地 Web 开发指南](https://bliulab.github.io/pepplatform-docs/zh/install/local-web)。

## 开发与测试

CPU 开发和测试环境显式安装 CPU 版 Torch 与 torchvision：

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1" "torchvision==0.20.1"
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check --select E9,F63,F7,F82 .
```

完整 lint 报告使用 `python -m ruff check .`，其中也包含现有的风格告警。
前端测试与构建在 `frontend/` 中分别执行 `npm run test:unit` 和 `npm run build`。

## 使用文档

[中文手册](https://bliulab.github.io/pepplatform-docs/zh/quick-start/) · [English guides](https://bliulab.github.io/pepplatform-docs/quick-start/)

文档源码维护在独立的 [pepplatform-docs 仓库](https://github.com/bliulab/pepplatform-docs)。

## 许可证

[MIT](LICENSE)。数据集与预训练模型分别遵循其提供方的许可证。
