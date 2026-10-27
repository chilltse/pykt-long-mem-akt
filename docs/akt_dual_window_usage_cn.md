# akt-dual-window 使用说明

模型直接继承 `pykt/models/akt.py` 的 AKT，复用原始编码、投影、gamma、FFN 和预测头；原始 AKT 文件不变。新增的只有每层 K/V 缓存、双窗口选择及相应前向。参数名和 checkpoint 与原始 AKT 一致。

## A/B/C 对照入口

在仓库根目录、已安装本项目依赖的 Python 环境下运行：

```bash
python examples/experiment_akt_dual_window.py \
  --dataset-name assist2009 --fold 0 \
  --output examples/exp_akt_dual_window \
  --epochs 20 --seeds 42,43,44
```

默认比较 `akt,short-only,akt-dual-window`，B=128、W=128、M=256、k=32。每个 seed 的三组使用相同初始权重、学生划分、学生顺序、评分位置和优化器配置，各自按验证集 logloss 选取最佳 epoch，最后评估测试集。

- `akt`：直接调用原始 AKT，每个 B 长度片段独立前向，片段之间没有缓存。
- `short-only`：M=0，缓存连续短期历史。
- `akt-dual-window`：短期全量历史与长期逐 Head Top-k 共同进行原始 AKT 注意力。

使用 `data_config.json` 的 `train_valid_original_file` 和 `test_original_file`，必须每行一个学生的完整有序、未填充序列；训练/验证/测试学生不得重叠。可用 `--data-dir /path/to/data` 覆盖数据目录，或用 `--data-config /path/to/config.json` 指定配置。所有组评分位置完全相同：遵守原文件 selectmasks，只排除每个学生第一个 Token；保留后续 Block 的首个 Token，从而能报告边界效果。这里的基线指标不能直接和旧入口的切片首题排除口径混用。

学生之间可打乱，同一学生内部始终按 Block 顺序。训练 batch 为一个完整学生，Block 内矩阵并行；逐 Block 反传，梯度按该学生有效标签数归一化，学生结束时更新一次参数。缓存始终 detach，且不跨学生、epoch 或 optimizer step 保留。

结果保存在输出目录（必须为新目录）：

- `comparison.json`：各组测试集 AUC、Accuracy、Logloss，以及 Block 边界指标。
- 每组的 `metrics.json`、`predictions.csv`、`training.json`、`best.pt`。
- `config.json`：实验参数、模型参数与评分口径。

边界指标只统计第 2 个及以后 Block 的首个 Token。Long 使用统计仅针对 Knowledge Retriever，包含候选数、实际选择数、每个 Query/Head 平均选择数，以及使用 Long 的 Query/Head 比例。

## 分阶段训练

先单独运行 `--variants short-only`。然后从其 `best.pt` 开启双窗口训练：

```bash
python examples/experiment_akt_dual_window.py \
  --dataset-name assist2009 --variants akt-dual-window --seeds 42 \
  --init-checkpoint examples/exp_short/short-only_seed42/best.pt \
  --long-window 256 --output examples/exp_dual_256
```

收敛后用双窗口 `best.pt` 继续，设置 `--long-window 1024`，W 和 k 不变。checkpoint 只加载模型权重，优化器重新创建；模型结构参数必须和 checkpoint 一致，每个学生都重建缓存。若要保持从头训练的公平 A/B/C 对照，不传 `--init-checkpoint`。

## 模型接口与测试

通过 `init_model("akt-dual-window", ...)` 可创建模型。训练使用上述专用入口，旧的随机切片训练入口会拒绝该模型。

`predict_block(q_data, target, pid_data, student_ids=[...], memory=None, valid=None)` 返回预测、Rasch 正则项、新 memory、Long 使用统计。后续同一学生 Block 显式传回 memory；新序列/新 epoch 传 None。批量接口要求各行学生 ID 唯一且顺序固定，各行有独立有效性和绝对位置；padding 必须位于行尾。普通 `forward` 将每行视作独立完整学生序列，调用之间不保留缓存。

```bash
python -m pytest tests/test_akt_dual_window.py -q
```

测试覆盖文档指定窗口、Top-k、原始 AKT 首 Block 一致性、未来信息泄漏、缓存投影/截断/梯度隔离、学生与 padding 隔离，以及三组训练评估入口的端到端冒烟测试。冒烟测试只验证实现可运行，不代表真实数据集的性能提升。
