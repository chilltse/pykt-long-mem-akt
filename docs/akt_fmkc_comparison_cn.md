# Yousician：AKT 与 AKT-FMKC 一键对比

在仓库根目录运行：

```bash
conda activate pykt
python examples/compare_yousician_akt.py
```

默认读取 `data/yousician/yousician_ukulele.json`，生成学生级互斥 train/valid/test 划分，使用 fold 0，各训练最多 30 epochs，按验证 AUC 保存最佳模型，再评估测试集。结果位于 `saved_model/yousician_comparison/comparison.json`，包含 AUC、accuracy、BCE、事件数、最佳 epoch 和 checkpoint 路径。训练仍沿用共享训练器的 early stopping。

- `akt`：完整四字段 KC 组合映射为连续整数 ID（dense ID），使用原 AKT embedding 和 Rasch 分支。
- `akt_fmkc`：四个字段分别 embedding、拼接并投影为 KC vector，使用现有 field drift 分支；默认 `response`。
- 两者使用同一训练词表、学生划分、seed、block/cache/window 设置和 backbone 超参数。验证和测试未见过的组合使用 ID 0；vector 路径仍可组合训练见过的字段。
- 两者均按学生依时间顺序训练；cache 保存原始历史 block，前向时重新计算，只对新 block 计算 BCE 和正则项。每个学生、每轮和每个数据划分重新建立 cache。
- 保留旧名称 `yousician_akt` 作为兼容入口。新实验使用 `akt_fmkc`。

自定义训练：

```bash
python examples/compare_yousician_akt.py --fold 1 --num_epochs 50 --block_size 128 --cache_blocks 2 --seq_len 256 --save_dir saved_model/compare_fold1
```

需满足 `block_size * cache_blocks >= seq_len`。复用已处理数据可传 `--prepared_config saved_model/yousician_comparison/data_config.json`。默认重新预处理到本次输出目录，不修改仓库全局数据配置。每次实验建议指定独立 `--save_dir`，避免覆盖同参数结果。

该比较保留两个模型各自的 embedding/drift 参数化，参数总数不强制一致；它不是只替换输入张量的单变量消融。`--field_dim` 和 `--drift` 只影响 `akt_fmkc`。
