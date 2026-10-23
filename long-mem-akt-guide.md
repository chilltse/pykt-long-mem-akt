# LongMemAKT：E1 完整长期残差

当前只保留 E1：有历史块时输出 `sigmoid(AKT_logit + residual_logit)`，不使用预测门控。首个有效块使用 AKT；非有限残差回退到 AKT。

## 数据与训练

采用标准 pykt fold 划分：指定 fold 为 valid，其余训练 fold 全部为 train，原始测试文件中 fold=-1 为 test。按学生隔离，无额外数据子集。

一个训练命令内部先训练 AKT（valid AUC 选择 checkpoint），再冻结 AKT，训练长期残差（valid logloss 选择 checkpoint）。两个阶段共用 train/valid，`epochs` 为每阶段最多轮数，连续 10 轮无改善早停。保留 E1 原训练语义，未改成联合训练。test 只在显式评估时读取用于预测。

## 记忆与残差

每名学生独立初始化 Value memory，共享 Key 槽；完整轨迹按连续非重叠块处理。块内 AKT 保持严格因果预测，整块读取进入该块时的记忆，观察完该块后才写入。

残差头输入为归一化问题表示、短期表示、长期读取表示及 AKT logit。最后一层零初始化。长期写入保留因果编码、时间衰减、erase/add 更新；这些写入机制不属于已删除的预测门控。

TBPTT 默认跨 2 块，残差训练要求至少 2 块；组间截断并重放边界写入，让后续块损失监督写入器。每批一名学生，避免轨迹或状态混用。

默认 `score_protocol=legacy` 保留 pykt 原分段计分掩码；`student` 按完整学生轨迹计分，跳过第一个交互。两种协议不要直接混比。

## 使用

在 `examples` 目录运行：

```bash
PYTHONPATH=.. python wandb_long_mem_akt_train.py --dataset_name bridge2algebra2006 --fold 0 --block_size 200 --epochs 20 --output saved_model/long_mem_e1
PYTHONPATH=.. python wandb_long_mem_akt_train.py --stage evaluate --source saved_model/long_mem_e1 --block_size 200 --output saved_model/long_mem_e1_test
PYTHONPATH=.. python evaluate_long_mem_akt_ablation.py --source saved_model/long_mem_e1 --output saved_model/long_mem_e1_comparison
```

默认 CPU，GPU 加 `--device cuda`。输出目录必须尚不存在。模型直接保存在输出目录的 `qid_model.ckpt`，配套 `config.json`、`roles.json`、`long_mem_manifest.json` 和训练日志，不再生成多阶段子目录。

也支持标准预测入口 `wandb_predict.py --save_dir saved_model/long_mem_e1 --use_wandb 0`。训练保留专用入口，以自动完成 E1 的两步优化和学生级记忆处理，内部复用 pykt `train_model`。

消融仅比较同一 checkpoint 的 E0（冻结 AKT）与 E1（完整残差），输出 `metrics.json` 和两份预测 CSV。旧选择性模型 checkpoint 不兼容，请重新训练。
