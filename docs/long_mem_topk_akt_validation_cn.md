# long_mem_topk_akt 验证报告

这是合成小数据流程验收，不是实际 KT 数据集上的效果实验。

## 环境与数据

Python 3.7、PyTorch 1.13.1+cu117、Faiss CPU 1.7.3、CPU 单线程；CUDA 不可用。

6 名合成学生，每人 13 个交互，train/valid/test 各 2 名，学生完全不重叠。原计分 maxlen=4，min_seq_len=3，每个 split 共 18 个计分标签。所有组的测试 uid/position/target 已逐项核对一致。

原 AKT 基线以 seed=1234 初始化，仅在 train 上做一次小规模 epoch 优化；每个消融组从这份相同权重开始，以 seed=42 微调 1 epoch。d_model=16、n_blocks=2、4 heads、dropout=0、R=2、M=5、k=2；一般组 L=4，akt_long L=8。每组均使用 18 个训练标签。按 valid 选择模型，再测试；不会用 test 选择参数。

## 实测结果

| 组 | AUC | NLL | ACC | 更新步数 | 训练秒 | 测试秒 | bank 峰值字节 | index 峰值字节 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| akt | 0.604938 | 0.687032 | 0.500000 | 6 | 0.0840 | 0.0585 | 0 | 0 |
| recent | 0.555556 | 0.691867 | 0.500000 | 6 | 0.1324 | 0.0404 | 1728 | 0 |
| exact | 0.580247 | 0.691244 | 0.555556 | 6 | 0.1006 | 0.0438 | 3168 | 0 |
| hnsw_ip | 0.580247 | 0.691244 | 0.555556 | 6 | 0.1041 | 0.0528 | 3168 | 10448 |
| akt_long | 0.555556 | 0.690881 | 0.500000 | 4 | 0.1930 | 0.0904 | 0 | 0 |
| dense_oracle | 0.555556 | 0.692265 | 0.555556 | 6 | 0.1083 | 0.0853 | 3168 | 0 |

HNSW 的 raw-IP Recall@k（相对当前投影 Exact）为 1.000000；同快照 dense attention 保留质量均值为 0.850228。二者衡量不同对象，高原始点积召回不代表保留了全部 AKT 最终注意力质量。

此规模只有 5 条远期记录，不能据此推断大历史 ANN 的性能。上述计时极短，包含冷启动/维护/统计开销，不能作为稳定加速证据。长输入组更新次数不同，已明确列出。各配置没有开展真实数据调参，不能据此得出模型优劣结论。

## 测试

综合回归 55 项通过：新模型 25 项、原 LongMemAKT 29 项、已有配置集成 1 项。之后针对 FP32 autocast 防护重跑 24 项核心测试，并单独通过固定种子的端到端消融测试（包括标准 wandb_predict 和 load_model 加载）。Faiss 仅产生 distutils 弃用警告。

覆盖：原 AKT output/qtest/梯度兼容，多层完整前缀 recent 等价，Exact 全候选与 dense 等价，位置平移和存储顺序不变性，当前/未来响应及问题因果扰动，学生混合 batch 隔离，历史 detach 与 Q/K/V/gamma 梯度，多块优化，重复提交/乱序/未来记录拒绝，FIFO/ANN 刷新和容量边界，padding、零容量、BF16，split 重放，以及六组端到端训练/测试。

复现：

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_long_mem_topk_akt.py tests/test_long_mem_akt.py tests/test_opera_integration_config.py -q
# 仅生成合成消融（测试临时目录中保留完整 checkpoint 和逐条预测）
OMP_NUM_THREADS=1 python -m pytest tests/test_long_mem_topk_akt.py::test_cli_matched_ablation -q --basetemp=/tmp/topk-akt-smoke
```

原始数值：[JSON](experiments/long_mem_topk_akt_smoke.json)、[CSV](experiments/long_mem_topk_akt_smoke.csv)。真实数据三种子实验、GPU FP16、GPU 内存曲线和长期规模性能尚未运行。完整参数和正式运行入口见 [实现说明](long_mem_topk_akt_implementation_cn.md)。
