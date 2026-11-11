# PepPlatform

[English](README.md) | 简体中文

PepPlatform 是面向肽段建模的平台，提供命令行训练与 Web 实验管理、模型推理。

也可以按[部署指南](docs/deployment.zh-CN.md)自行运行 Web 服务。

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

交互启动时先选择中文或 English，导航、训练向导、集成、推理、运行历史及配置摘要均使用
所选语言。可以用 `python -m pepplatform_cli --lang zh` 或 `--lang en` 跳过语言选择。
旧的 `scripts/train_pipeline.py`、`train_sequence_pipeline.py`、`train_pair_pipeline.py`
交互入口也支持此选项。任务只选择一次；已通过参数指定任务时直接进入后续配置。

配置过程中，`b` 返回上一输入项，`q` 取消当前流程；在主菜单中则退出。返回后已填内容会作为
默认值保留；可用 `-` 清空可选文本字段。执行前会展示输入路径、seed、loss、语义设置、选择协议和非交互复现命令，
并提供“运行／修改／取消”，默认“修改”。选择“运行”后才创建运行记录并启动计算；
已有非交互命令不会增加询问。模型名、参数名及专用术语保留英文，训练日志和机器可读 JSON
维持原格式；语言与导航状态不进入训练元数据或复现参数。


## 数据与模型

数据集和预训练权重需另行准备。上述示例使用经典特征，无需预训练权重。
预训练路线需要对应的模型快照，准备方式见 CLI 安装指南。
合成示例仅用于验证使用流程。数据准备工具见[公共数据集 CLI](https://bliulab.github.io/pepplatform-docs/zh/datasets/public-databases-cli)。
PSSM/PSFM 生成还需要 BLAST+（`ncbi-blast+`）和对应路线的参考数据。

数据集和模型路径可通过 CLI 或 Web 配置指定，相对默认路径以项目目录为基准。
保存的 Protocol/schema 版本用于识别产物格式和兼容性检查，
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

Web 数据集使用 ZIP 文件，条目支持 Deflate 压缩或不压缩（Stored）。
ZIP 默认上限为 64 MiB，实际限制取决于所用服务的配置，详见[上传限制](docs/deployment.zh-CN.md#数据集上传限制)。

本地后端安装 `requirements-web.txt`；使用预训练路线时，再安装
`requirements-pretrained.txt`。所提供的 Docker 后端默认安装这两层依赖。
前端依赖在 `frontend/` 目录中通过 `npm ci` 单独安装。

Docker 部署先按宿主机硬件选择完整指南，再执行其中的配置和启动命令：

| 方式 | 宿主机要求 | 指南 |
| --- | --- | --- |
| CPU | 无 NVIDIA GPU，或选择仅用 CPU | [CPU 部署与运维](docs/deployment-cpu.zh-CN.md) |
| GPU | 兼容的 NVIDIA GPU、驱动和 NVIDIA Container Toolkit | [GPU 部署与运维](docs/deployment-gpu.zh-CN.md) |

部署完成后访问 <http://127.0.0.1:8080>。使用预训练路线前，还需准备对应模型权重。

需要固定论文资产时，执行所选指南的 [CPU 模型准备](docs/deployment-cpu.zh-CN.md#可选准备六套预训练模型)或
[GPU 模型准备](docs/deployment-gpu.zh-CN.md#可选准备六套预训练模型)。五类模型包含六套权重，
ProtT5 会从固定官方单文件还原原实验格式。安装量约 22.68 GB；当前在线准备保留缓存后约 45.37 GB，
额外开销另计。模型清单、磁盘预算、下载源及两种 ProtT5 用法见[共享参考](docs/deployment.zh-CN.md#准备与论文候选实验一致的模型)。

这项准备流程用于论文复现；做自己的实验时，也可以原样使用兼容的官方 ProtT5，
通过训练参数 `--prott5_model` 指定其绝对目录。准备参数 `--prott5-source` 仍会执行还原，
两种参数及 Docker 路径的区别见上述指南。

本地启动和首次登录步骤见[本地 Web 开发指南](docs/deployment.zh-CN.md#本地-web-开发)。

“记住账号”、浏览器保存密码与登录会话的区别见[账号登录与密码保存](docs/deployment.zh-CN.md#账号登录与密码保存)。

仓库内的 [Docker 部署入口与共享参考](docs/deployment.zh-CN.md) 提供 CPU/GPU 指南导航、模型与配置说明。
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
