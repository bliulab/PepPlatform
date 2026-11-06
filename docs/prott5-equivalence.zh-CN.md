# ProtT5 官方单文件与原实验六分片验证

[English](prott5-equivalence.md)

2026-11-06 实测确认：固定版本的官方单文件与原六分片的完整参数、平台特征输出均一致，
**六分片本身不是数值复现的必要条件**。但额外检查发现两种格式加载后的 Torch 随机数状态
不同，可能影响新分类器的初始化。现在准备命令会下载官方单文件，再自动还原原实验文件，
从而沿用原格式的加载行为。

[参数与特征 JSON 报告](prott5-equivalence-report.json) 保存了逐张量比较、有效配置、环境、
代码哈希和输出哈希。报告中的 `PASS` 指参数和特征等价，不代表完整训练过程已复现。

## 按使用目的选择模型

复现论文时，使用准备后的原格式模型入口、已验证的原六分片或离线模型包，并同时匹配
冻结实验配置和运行环境。

独立实验可以原样使用平台兼容的官方 ProtT5 快照。通过训练参数 `--prott5_model` 显式指定
快照的绝对路径，它会优先于已安装的本地候选目录；准备参数 `--prott5-source` 仍会执行还原。
直接加载原文件可用于自己的新实验，论文已有产物仍保留其模型身份校验。
两种使用流程及 Docker 路径说明见[部署指南](deployment.zh-CN.md#准备与论文候选实验一致的模型)。

## 固定资源身份

| 项目 | 固定值 |
| --- | --- |
| 官方仓库 | `Rostlab/prot_t5_xl_uniref50` |
| 官方 revision | `973be27c52ee6474de9c945952a8008aeb2a1a73` |
| 官方文件 | `pytorch_model.bin`，11,275,562,724 字节 |
| 官方文件 SHA-256 | `c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9` |
| 原六分片集合 SHA-256 | `c1e7b7328276917e729bf11585ac6b4b7965aa33251edc882ee2fad7d47d307d` |
| 共同规范化张量指纹 | `104950410ca044c20dd0dc2aa5515591cfdab660ffb2cff3e2de8975a5ee6315` |
| 验收版本 | `function_policy_completion_20261022T0620Z_aupr_trapezoidal_v2` |

509 个独立张量的名称、形状、FP32 类型和每一个字节全部一致，包括未用于特征提取的解码器。
官方文件有 512 个名称，多出的 `encoder.embed_tokens.weight`、`decoder.embed_tokens.weight`、
`lm_head.weight` 均是 `shared.weight` 的共享权重别名；验证先证明各别名一致，再去重比较。

| 参数组 | 独立张量数 | 参数数 | 逐位一致 |
| --- | ---: | ---: | --- |
| 共享嵌入 | 1 | 131,072 | 是 |
| 编码器，不含共享嵌入 | 194 | 1,208,010,752 | 是 |
| 解码器，不含共享嵌入 | 314 | 1,610,688,512 | 是 |

原始配置的差别来自序列化元数据和省略的默认值。平台对两种格式均显式使用 `T5EncoderModel`，
实际加载类、有效配置和加载后的编码器状态全部一致。三个分词器源文件也逐字节一致。

## 实际输出

使用原实验 Python 环境：Torch `2.5.1+cu121`、Transformers `4.48.1`、Tokenizers `0.21.4`、
SentencePiece `0.2.1`、Safetensors `0.7.0`、Hugging Face Hub `0.36.0`。
执行设备为 CPU，4 线程，seed 42，启用确定性算法及 Hugging Face 离线标志。

先核对冻结登记、验收索引、候选元数据和数据哈希，再从 Function、PPI、MHC-I、HLA-II
各取前 8 条登记验证记录，肽和配对序列共形成 7 组、56 条序列探针。
HLA-II 异构候选的肽端实际使用 ESM2；这里额外将其肽序列送入 ProtT5 作探针。
另加 7 条归一化、U/Z/O/B 替换、补齐、截断和 511 残基边界探针，并检查 4 个非法输入。

**9 个批次、63 条序列探针**的 token ID、attention/special-token 掩码、FP32 最后层隐状态、
FP32 池化向量、FP16 残基特征和残基掩码全部逐位一致，最大绝对误差为 **0**。
本次没有重新训练 100 条候选，也没有重新计算完整测试集指标。

## 直接加载的差异与自动还原

[随机数诊断脚本](../tools/verify_prott5_loader_rng.py) 单独核对 Python、NumPy 和 Torch CPU
在加载、一次前向计算后的随机数状态，再按冻结元数据初始化 Function 候选的 CNN。
脚本也比较显式重新设定种子后的初始化，作为受控对照。

最初的加载检查发现：两种格式的 Python/NumPy 状态相同，但 Torch CPU 状态不同。
六分片加载推进了 Torch 随机数状态，官方单文件加载则保留了初始状态。
当前训练入口在加载编码器之前设置种子，因此“参数、特征相同”仍不足以保证后续神经网络
初始化相同。[分类器初始化 JSON 报告](prott5-loader-rng-report.json) 记录这项独立检查。

原候选 CNN 的实测初始权重确实不同；在受控对照中，显式重设种子后才逐位一致：

| 检查项 | 两种格式之间一致 |
| --- | --- |
| 加载后的 Torch CPU 随机数状态 | 否 |
| 按原种子位置初始化的 CNN 权重 | 否 |
| 显式重新设定种子后的 CNN 权重 | 是 |

本次修复了本地快照检查：官方配置省略 `relative_attention_max_distance` 时，按 T5 标准默认值
128 校验；显式错误值和缺失的 XL 维度仍会拒绝。原模型文件与配置均未修改。

准备命令现在接受固定官方单文件作为输入，先在本地还原全部原实验文件，再用于训练。
`docker/models.lock.json`、训练产物身份检查和原种子位置均保持不变。
用户只下载官方资源，六分片缓存由程序自动生成；上文直接加载 `.bin` 的差异报告继续保留。

## 自动还原验收

[自动还原 JSON 报告](prott5-restore-report.json) 单独检查完整文件身份、加载后的编码器状态、
加载及前向计算后的随机数状态、原候选 CNN 初始化和三步合成数据训练，并复查上述 63 条
序列探针。检查使用准备后的模型入口，不重新训练全部候选。完整验收需要原登记研究资产；
普通用户准备模型无需研究目录或原六分片，源码已包含小型还原配方。

2026-11-06 自动还原验收结果为 **PASS**：全部 13 个文件匹配原锁定清单，各阶段随机数状态
一致，CNN 的 26 个状态张量初始化逐位一致，三步训练损失与训练后权重也逐位一致。
63 条特征探针最大绝对误差为 0。断网容器使用只读模型挂载，成功加载并检查了全部六套权重，
其中 ProtT5 使用本次还原结果。两个仓库各通过 96 项相关测试，已有离线模型包仍兼容。

上文保留的是此前 CPU 还原验收记录。后续 Docker 验收已从空缓存下载全部六套权重，
并在新 GPU、CPU 镜像中通过断网加载及 ProtT5 对照：13 个文件匹配原锁定哈希，
各设备内的随机数、63 条序列特征、CNN 初始化和三步训练均一致。
两个仓库各通过 115 项相关测试，实际环境及范围见[当前环境验收记录](paper-runtime-validation.json)。
还原验收脚本增加 `--device cuda` 可进行 GPU 对照，CPU 对照使用 `--device cpu`。

## 复跑

使用原实验环境并保留登记实验资产。先将固定 revision 下载到独立目录：

```bash
python - <<'PY'
import os
from huggingface_hub import hf_hub_download

for filename in [
    "config.json", "special_tokens_map.json", "spiece.model",
    "tokenizer_config.json", "pytorch_model.bin",
]:
    hf_hub_download(
        "Rostlab/prot_t5_xl_uniref50", filename=filename,
        revision="973be27c52ee6474de9c945952a8008aeb2a1a73",
        local_dir="backup/prott5-official-check",
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        token=False,
    )
PY

python tools/verify_prott5_equivalence.py \
  --official backup/prott5-official-check \
  --sharded /path/to/original/models/prott5 \
  --reference /path/to/PepPlatform/experiment_paper/current_results.json \
  --output backup/prott5-feature-verification --threads 4 --samples-per-task 8

python tools/verify_prott5_loader_rng.py \
  --official backup/prott5-official-check \
  --sharded /path/to/original/models/prott5 \
  --reference /path/to/PepPlatform/experiment_paper/current_results.json \
  --output backup/prott5-rng-verification --threads 4

python tools/verify_prott5_restore.py \
  --official backup/prott5-official-check \
  --sharded /path/to/original/models/prott5 \
  --restored /path/to/prepared/models/prott5 \
  --reference /path/to/PepPlatform/experiment_paper/current_results.json \
  --output backup/prott5-restoration-verification --threads 4
```

网络需要时可在下载前设置 `HF_ENDPOINT=https://hf-mirror.com`；两项检查均会在加载前核对
固定官方文件 SHA-256 与原分片哈希。原模型只读，输出目录必须尚不存在，结果包含 JSON
和代码快照；特征验证另保存 Markdown 报告及比对张量。诊断发现 `DIFFERENT` 时退出码为 1，
执行错误另记为 `ERROR`。
