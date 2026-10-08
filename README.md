# ODAA-AES：论文初始代码

**论文题目：** An Ordinal-Aware Dual Adversarial Alignment Framework for Cross-Prompt Automated Essay Scoring in English Language Education

本项目为上述论文的 **ODAA-AES 初始代码工程**，采用 Python 与 PyTorch 实现自动作文评分模型，支持在未见题目上进行严格的源域训练（source-only）与跨题目评估。项目包含数据预处理、模型构建、训练、预测、消融实验、参数敏感性分析、跨语料测试和结果统计等功能。

## 1. 研究任务

跨题目自动作文评分（Cross-Prompt Automated Essay Scoring）旨在使用已知作文题目的训练数据，预测新作文题目的人工评分。ODAA-AES 使用分层预训练语言模型编码作文，并联合以下三个学习模块：

- **GPAA（Global Prompt Adversarial Alignment）**：通过梯度反转层和题目判别器，减弱表示中的源题目特定特征。
- **OCAA（Ordinal-Conditioned Adversarial Alignment）**：利用连续序数锚点及高斯软权重，进行评分水平条件化的题目对齐。
- **OSPL（Ordinal Structure-Preserving Learning）**：通过作文对的排序约束和距离约束，保持作文评分的有序结构。

整体结构为：作文分段 → DeBERTaV3 编码 → 注意力池化 → 共享表示 → 评分预测；GPAA、OCAA、OSPL 在训练阶段施加联合约束，推理时使用编码器与评分预测头。

## 2. 项目目录

```text
ODAA_AES_Code/
├── README.md                        # 项目说明
├── README_中文.md                    # 中文项目说明
├── EXPERIMENT_NOTES.md               # 模型与实验配置说明
├── requirements.txt                  # Python依赖
├── pyproject.toml                    # 安装配置
├── configs/
│   ├── asap.yaml                     # ASAP实验配置
│   ├── toefl11.yaml                  # TOEFL11实验配置
│   └── asap2_template.yaml           # 第三语料配置模板
├── odaa_aes/
│   ├── data.py                       # 数据加载、作文分段及数据划分
│   ├── model.py                      # 编码器、GPAA、OCAA、OSPL及损失函数
│   ├── train.py                      # 训练、源域验证及模型保存
│   ├── metrics.py                    # QWK、MAE、RMSE、Spearman等指标
│   ├── analysis.py                   # 统计分析与评分误差分析
│   ├── cli.py                        # 命令行入口
│   └── config.py                     # 参数配置加载
├── scripts/
│   ├── prepare_asap.py               # ASAP原始数据预处理
│   ├── run_source_subsets.py         # 不同来源题目数量实验
│   ├── run_parameter_sweep.py        # 超参数与编码器敏感性实验
│   ├── umap_features.py              # UMAP特征可视化
│   ├── inference_profile.py          # 推理效率分析
│   └── plot_results.py               # 结果曲线生成
├── tests/test_core.py                # 核心功能测试
└── examples/synthetic_debug.csv     # 程序流程调试样例
```

## 3. 环境配置

推荐环境：Python 3.10 或 3.11；正式训练推荐 NVIDIA GPU 和适配 CUDA 的 PyTorch。

进入工程目录后执行：

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q tests
```

默认预训练模型为 `microsoft/deberta-v3-base`。首次执行需下载对应 Hugging Face 权重，或在配置文件的 `model.backbone` 字段中填写本地模型目录。

## 4. 数据准备

### 4.1 ASAP 数据集

ASAP 为论文的主要评估数据集，包含8个作文题目。数据获取地址：

https://www.kaggle.com/competitions/asap-aes

将官方 `training_set_rel3.tsv` 置于 `data/raw/` 下，执行：

```bash
python scripts/prepare_asap.py \
  --input data/raw/training_set_rel3.tsv \
  --output data/asap.csv
```

生成的 CSV 使用以下字段：

| 字段 | 含义 |
|---|---|
| `essay_id` | 作文唯一编号 |
| `essay_set` | 作文题目编号（1—8） |
| `essay` | 作文文本 |
| `domain1_score` | 人工评分 |

ASAP 各题目的评分范围在 `configs/asap.yaml` 中定义，并用于分数归一化及反变换。

### 4.2 TOEFL11 数据集

TOEFL11 获取地址：https://catalog.ldc.upenn.edu/LDC2014T06

准备本地 `data/toefl11.csv`，包含以下四列：

```text
essay_id,prompt_id,essay,score
```

`score` 可使用 `Low`、`Medium`、`High`，也可转换为 `0`、`0.5`、`1`。请根据数据授权要求使用与保存语料。

### 4.3 其他作文语料

第三语料可参照 `configs/asap2_template.yaml` 配置作文ID、题目ID、文本、评分列与评分区间。开展跨语料实验时，应提前检查重复作文和评分体系差异。

## 5. 主要实验参数

项目的 ASAP 初始配置如下，具体以 `configs/asap.yaml` 为准：

| 参数 | 设置 |
|---|---|
| 预训练编码器 | DeBERTaV3-base |
| 每段最大长度 | 384 tokens |
| 分段重叠 | 64 tokens |
| 每篇最多段数 | 6 |
| 共享表示维度 | 256 |
| 序数锚点数 K | 5 |
| 高斯带宽 σ | 0.15 |
| Dropout | 0.20 |
| 优化器 | AdamW |
| 编码器学习率 | 2 × 10⁻⁵ |
| 新增层学习率 | 1 × 10⁻⁴ |
| Weight decay | 0.01 |
| Batch size | 8 |
| 梯度累积步数 | 4 |
| 有效Batch size | 32 |
| 最大训练轮数 | 20 |
| Warmup比例 | 10% |
| 梯度裁剪 | 1.0 |

配置中还包含 `alpha`、`beta`、`gamma` 等损失权重、配对采样设置、模型选择范围及随机种子，均可根据实际实验方案调整。实验使用的最终配置会随运行记录保存。

## 6. 训练 ODAA-AES

### 6.1 单个目标题目

下面以 ASAP 的题目1作为外层测试题目、随机种子42为例：

```bash
python -m odaa_aes train \
  --data data/asap.csv \
  --config configs/asap.yaml \
  --target 1 \
  --seeds 42 \
  --out runs/asap_trial
```

训练过程采用源域内验证：外层目标题目的作文不参与参数训练、验证、早停与超参数选择，仅用于最后一次测试。

### 6.2 ASAP 全部8个题目

```bash
python -m odaa_aes train \
  --data data/asap.csv \
  --config configs/asap.yaml \
  --target all \
  --out runs/asap_full
```

默认随机种子见配置中的 `training.seeds`，可以使用 `--seeds 42,43,44,45,46` 明确指定。

### 6.3 结果汇总

```bash
python -m odaa_aes report \
  --root runs/asap_full \
  --config configs/asap.yaml \
  --out reports/asap_full
```

运行会保存各目标题目的指标、预测结果及训练信息，例如：

```text
runs/asap_trial/target_1/seed_42/
├── model_state.pt
├── metadata.json
├── train_history.csv
├── predictions.csv
└── results.json
```

主要指标包括 QWK、归一化 MAE、RMSE、Spearman 相关系数，适用时还会输出分类准确率。

## 7. 消融实验

### 7.1 GPAA、OCAA、OSPL 模块消融

```bash
python -m odaa_aes ablate \
  --kind factorial \
  --data data/asap.csv \
  --config configs/asap.yaml \
  --target all \
  --out runs/ablation
```

程序依次执行三个模块的全部8种启用组合。输出目录按 `G1_C0_S1` 等形式命名，分别表示 GPAA、OCAA、OSPL 是否启用。

### 7.2 条件对齐策略对比

```bash
python -m odaa_aes ablate \
  --kind conditioning \
  --data data/asap.csv \
  --config configs/asap.yaml \
  --target all \
  --out runs/conditioning
```

包含无条件分支、硬评分分组、非序数软分配、打乱评分关联的软分配以及有序锚点软分配等策略。

## 8. 参数敏感性与来源数量实验

### 8.1 序数锚点数量 K

```bash
python scripts/run_parameter_sweep.py \
  --kind k --data data/asap.csv \
  --config configs/asap.yaml --target 1 \
  --out runs/k_sweep
```

### 8.2 高斯带宽 σ

```bash
python scripts/run_parameter_sweep.py \
  --kind sigma --data data/asap.csv \
  --config configs/asap.yaml --target 1 \
  --out runs/sigma_sweep
```

`--kind` 还支持 `backbone`（编码器对比）、`anchors`（锚点布局）和 `mapping`（评分映射方法）。跨题目报告时，应对所需目标题目重复运行并统一汇总。

### 8.3 不同训练题目数量

```bash
python scripts/run_source_subsets.py \
  --data data/asap.csv \
  --config configs/asap.yaml \
  --target 1 --sizes 1,3,5,7 \
  --max-combinations 6 \
  --out runs/source_subsets
```

单源训练情况下自动停用源题目间的对抗判别器，使用源数据内部验证。

## 9. 跨语料测试

下面以 ASAP 为源语料、TOEFL11 为目标语料：

```bash
python -m odaa_aes cross-corpus \
  --source-data data/asap.csv \
  --source-config configs/asap.yaml \
  --target-data data/toefl11.csv \
  --target-config configs/toefl11.yaml \
  --seed 42 \
  --out runs/asap_to_toefl
```

训练与模型选择只在源语料上完成，目标语料仅用于最终推理和评价。反向实验可交换源语料与目标语料及其配置。

## 10. 模型预测

对不带人工评分的作文文件进行预测：

```bash
python -m odaa_aes predict \
  --checkpoint runs/asap_trial/target_1/seed_42 \
  --input data/unlabelled_asap.csv \
  --no-labels \
  --output predictions/unlabelled_scores.csv
```

输入需要包括作文编号、题目编号及作文文本；题目对应的评分区间需在配置中定义。

## 11. 统计分析与图表

### 11.1 误差与分组分析

```bash
python -m odaa_aes groups \
  --predictions runs/asap_trial/target_1/seed_42/predictions.csv \
  --config configs/asap.yaml \
  --out reports/errors
```

### 11.2 配对统计检验

在两个模型分别汇总结果后，可对相同目标题目进行配对比较：

```bash
python -m odaa_aes paired-stats \
  --a reports/model_a/per_prompt_seed_metrics.csv \
  --b reports/model_b/per_prompt_seed_metrics.csv \
  --out reports/paired_stats.json
```

### 11.3 UMAP 特征可视化

```bash
python -m pip install umap-learn
python scripts/umap_features.py \
  --data data/asap.csv \
  --config configs/asap.yaml \
  --encoder-only runs/encoder_only/target_1/seed_42 \
  --odaa runs/asap_trial/target_1/seed_42 \
  --out figures/figure4_umap.png
```

其中 `--encoder-only` 需要提供已训练的编码器基线权重目录。

### 11.4 推理效率

```bash
python scripts/inference_profile.py \
  --checkpoint runs/asap_trial/target_1/seed_42 \
  --data data/asap.csv \
  --out reports/inference_profile.json
```

## 12. 实验记录与结果管理

为便于管理论文实验，建议按数据集、模型、目标题目、随机种子分目录保存模型权重、预测文件、训练记录及最终指标。正式结果应由对应实验运行及预测文件生成，并保留实际使用的配置文件。

`examples/synthetic_debug.csv` 仅用于检查数据接口与运行流程，不参与论文的正式评价。ASAP、TOEFL11原始作文数据与模型预训练权重需由使用者自行在授权条件下准备。

更多参数和模块对应关系见 `EXPERIMENT_NOTES.md`。

---

**项目名称：** ODAA-AES 论文初始代码  
**实现框架：** PyTorch / Transformers  
**研究方向：** Cross-Prompt Automated Essay Scoring · Ordinal Learning · Adversarial Alignment
