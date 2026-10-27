# akt-dual-window 使用说明

模型直接继承 `pykt/models/akt.py` 的 AKT，复用原始编码、投影、gamma、FFN 和预测头；原始 AKT 文件不变。当前版本遵循 `dual_window_akt_gated_brief_cn.md`：Q/K 按 Head 做 L2 归一化，相似度乘 sqrt(d_k)；Short 独立计算 AKT 距离衰减，Long 独立做纯相似度 Top-k attention。每层每 Head 的静态 gate 在 out_proj 前融合两路输出，初始 Long 比例为 0.05，无 Long 时完全回退 Short。V 不归一化。

## A/B/C 对照入口

在仓库根目录、已安装本项目依赖的 Python 环境下运行：

```bash
python examples/experiment_akt_dual_window.py \
  --dataset-name assist2009 --fold 0 \
  --output examples/exp_akt_dual_window \
  --epochs 20 --seeds 42,43,44
```

默认比较 `akt,short-only,akt-dual-window`，B=128、W=128、M=256、k=32。每个 seed 的三组的公共 AKT 参数使用相同初始权重、学生划分、学生顺序、评分位置和优化器配置，各自按验证集 logloss 选取最佳 epoch，最后评估测试集。

- `akt`：直接调用原始 AKT，每个 B 长度片段独立前向，片段之间没有缓存。
- `short-only`：Q/K 归一化，topk=0，缓存连续历史但只读取 Short。W/M 和门控模型相同。
- `akt-dual-window`：Q/K 归一化，Short/Long 独立 attention，再用每层每 Head 的 gate 融合。

`short-only` 与门控组用于隔离 Long 的效果；`akt` 是另列的未归一化原始模型对照，不把归一化带来的变化归因于 Long。

使用 `data_config.json` 的 `train_valid_original_file` 和 `test_original_file`，必须每行一个学生的完整有序、未填充序列；训练/验证/测试学生不得重叠。可用 `--data-dir /path/to/data` 覆盖数据目录，或用 `--data-config /path/to/config.json` 指定配置。所有组评分位置完全相同：遵守原文件 selectmasks，只排除每个学生第一个 Token；保留后续 Block 的首个 Token，从而能报告边界效果。这里的基线指标不能直接和旧入口的切片首题排除口径混用。

学生之间可打乱，同一学生内部始终按 Block 顺序。训练 batch 为一个完整学生，Block 内矩阵并行；逐 Block 反传，梯度按该学生有效标签数归一化，学生结束时更新一次参数。缓存始终 detach，且不跨学生、epoch 或 optimizer step 保留。

结果保存在输出目录（必须为新目录）：

- `comparison.json`：各组测试集 AUC、Accuracy、Logloss，以及 Block 边界指标。
- 每组的 `metrics.json`、`predictions.csv`、`training.json`、`best.pt`。
- `config.json`：实验参数、模型参数与评分口径。

边界指标只统计第 2 个及以后 Block 的首个 Token。原 `long_usage` 保持 Knowledge Retriever 汇总口径；新增：

- `gate_values`：每层每 Head 的 sigmoid(gate_bias)。
- `layer_long_usage`：所有 encoder/retriever 层的 Long 使用计数与 query 比例。
- `fraction_queries_using_long`：任一层/Head 实际选择了 Long 的位置占全部输入位置比例。
- `long_positions`：上述位置中参与评分的 AUC、Accuracy、Logloss；预测 CSV 增加 `has_long` 列。

这些日志写入训练/验证/测试报告；训练阶段不额外计算 AUC，`long_positions` 的预测指标由验证和测试阶段提供。topk=0 和原始 AKT 组没有实际使用 Long 的位置，该子集 count 为 0。

## 分阶段训练

先单独运行 `--variants short-only`。然后从其 `best.pt` 开启双窗口训练：

```bash
python examples/experiment_akt_dual_window.py \
  --dataset-name assist2009 --variants akt-dual-window --seeds 42 \
  --init-checkpoint examples/exp_short/short-only_seed42/best.pt \
  --long-window 256 --output examples/exp_dual_256
```

收敛后用双窗口 `best.pt` 继续，设置 `--long-window 1024`，W 和 k 不变。checkpoint 只加载模型权重，优化器重新创建；模型结构参数必须和 checkpoint 一致，每个学生都重建缓存。若要保持从头训练的公平 A/B/C 对照，不传 `--init-checkpoint`。

新增 gate 后 checkpoint 不再与原 AKT 完全同构。入口兼容原 AKT 和旧双窗口权重，只允许缺少 gate，其余缺失、多余键或形状不匹配均报错。直接调用时使用 `load_compatible_state_dict(state_dict)`；新 checkpoint 可用标准 `load_state_dict` 严格加载。旧缓存不兼容，必须丢弃并重新生成 normalized K/V 缓存。模型需要重新训练或微调后再评估收益。

## 模型接口与测试

通过 `init_model("akt-dual-window", ...)` 可创建模型。训练使用上述专用入口，旧的随机切片训练入口会拒绝该模型。

`predict_block(q_data, target, pid_data, student_ids=[...], memory=None, valid=None)` 返回预测、Rasch 正则项、新 memory、Long 使用统计。后续同一学生 Block 显式传回 memory；新序列/新 epoch 传 None。批量接口要求各行学生 ID 唯一且顺序固定，各行有独立有效性和绝对位置；padding 必须位于行尾。普通 `forward` 将每行视作独立完整学生序列，调用之间不保留缓存。

```bash
python -m pytest tests/test_akt_dual_window.py -q
```

测试覆盖文档指定窗口、Top-k、归一化 AKT 参考实现的一致性、两路独立性、gate 极值与回退、Long 的 Q/gate 梯度、FP32 attention、未来信息泄漏、缓存投影/截断/梯度隔离、学生与 padding 隔离，以及三组训练评估入口的端到端冒烟测试。冒烟测试只验证实现可运行，不代表真实数据集的性能提升。
