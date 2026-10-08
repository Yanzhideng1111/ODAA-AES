# ODAA-AES 模型实现与实验配置说明

本文件说明项目中的算法模块、超参数、验证协议和实验输出，便于配置、运行及归档论文实验。

## 1. 模块与代码对应

| 论文模块 | 代码位置 | 功能 |
|---|---|---|
| Rubric分数归一化 | `odaa_aes/data.py` | 根据题目评分区间转换到[0,1] |
| 分层作文编码器 | `odaa_aes/model.py` | DeBERTaV3分段表示与注意力池化 |
| 评分预测 | `odaa_aes/model.py` | Sigmoid回归头与Huber损失 |
| GPAA | `odaa_aes/model.py` | GRL和源题目判别器 |
| OCAA | `odaa_aes/model.py` | Gaussian序数锚点和条件题目判别器 |
| OSPL | `odaa_aes/model.py` | 成对评分顺序与距离约束 |
| 联合训练 | `odaa_aes/train.py` | 损失组合、优化器与学习率调度 |
| Source-only验证 | `odaa_aes/train.py` | 外层未见题目测试、内层源题目验证 |
| 消融 | `odaa_aes/cli.py` | 模块组合和条件对齐策略 |
| 统计与校准 | `odaa_aes/analysis.py` | 评分误差、配对置信区间等 |

## 2. 配置说明

实验配置以 `configs/asap.yaml`、`configs/toefl11.yaml` 为入口，关键参数包括：

- 编码器、分段长度、重叠和最大段数：`model.backbone`, `max_tokens`, `overlap`, `max_segments`
- 共享特征和序数锚点：`hidden_size`, `num_anchors`, `sigma`, `anchor_strategy`
- 模块开关：`gpaa`, `ocaa`, `ospl`, `conditioning`
- 训练设置：`epochs`, `batch_size`, `accumulation_steps`, `lr_backbone`, `lr_new`, `warmup_ratio`
- 联合目标权重：`alpha`, `beta`, `gamma`, `ordinal_distance_weight`, `ordinal_margin`
- 成对采样：`pair_mode`, `max_pairs`, `min_delta_score`, `ties_in_distance`
- 选择与评估：`hyperparam_grid`, `checkpoint_strategy`, `score_transform`, `rounding`, `seeds`

论文未完整列出数值的部分采用工程可调整的初始设置，例如 `alpha=0.20`、`beta=0.20`、`gamma=0.10`、`ordinal_margin=0.5` 和 `seeds=[42,43,44,45,46]`。开展正式实验时应将这些配置设置为实际采用的参数并保存运行记录。

## 3. 重要方法细节

- **分段编码**：默认每段384 tokens，重叠64，最多6段；长作文可配置 `first` 或 `head_tail` 截取。
- **GPAA**：源题目类别预测与编码器梯度反转；仅一个源题目时不启用多源对抗模块。
- **OCAA**：每个锚点使用对应的源题目判别器，依据归一化成绩与锚点距离分配软权重；也可切换硬分箱或非序数控制组。
- **OSPL**：从批内抽样作文对，建立与评分差距相关的顺序和距离损失；配对数和同分样本处理在配置中定义。
- **分数输出**：预测归一化分数后，根据目标的公开评分区间反变换；默认使用预设舍入规则。
- **训练/测试隔离**：外层目标作文仅最终测试使用；超参数与最终轮数由源题目内部验证选择。
- **跨语料实验**：模型只在源语料训练和选模，目标语料用于最后评价；应核对评分区间、语料授权和文本重复情况。

## 4. 实验数据及输出

建议保留以下文件以支持结果统计和论文表格制作：

```text
model_state.pt       # 模型权重
metadata.json        # 题目与实验参数
train_history.csv    # 训练过程
predictions.csv      # 每篇作文的预测与人工评分
results.json         # QWK、MAE、RMSE等指标
```

论文性能数值应由实际训练和评估输出计算。预训练模型权重、ASAP或TOEFL11作文文本不包含在本工程中，需按相应来源和许可准备。

## 5. 可执行的实验入口

| 实验 | 命令 |
|---|---|
| 主模型 | `python -m odaa_aes train` |
| 模块消融 | `python -m odaa_aes ablate --kind factorial` |
| 条件对齐 | `python -m odaa_aes ablate --kind conditioning` |
| 跨语料 | `python -m odaa_aes cross-corpus` |
| 模型预测 | `python -m odaa_aes predict` |
| 汇总结果 | `python -m odaa_aes report` |
| 配对统计 | `python -m odaa_aes paired-stats` |
| 分组误差 | `python -m odaa_aes groups` |
| 超参数测试 | `python scripts/run_parameter_sweep.py` |
| 来源数量 | `python scripts/run_source_subsets.py` |
| UMAP图 | `python scripts/umap_features.py` |

完整命令与数据格式请参见 `README.md`。
