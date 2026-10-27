# LongMemTopKAKT V2 实现与验收

日期：2026-10-25。审计起点 commit：`9d1046a3ca0e013009530fbe252bea12390fbdf1`；工作区原先已有未提交修改，本次未覆盖或撤销。仓库及已检查的上级目录没有 AGENTS.md。

## 实现范围

- 保留 `legacy_all_layers` 默认模式及原 checkpoint 结构。显式传入 `architecture_variant=v2_last_retriever_residual` 才启用 V2。
- QA 编码器、Question Attention、之前的 Retriever 调用原 TransformerLayer；最后 Retriever 的 current 分支直接调用 `akt.attention`，保留原 dropout、pdiff、首行处理和局部位置语义。
- 最后 Retriever 的 memory 使用独立 scores、FP32 safe softmax、detached candidate-context distance，支持 `legacy_mul` / `log_additive`。recent 全读、remote 当前投影 exact Top-k，按 query chunk gather，然后按真实 int64 时间排序。
- 每 head 的 `memory_alpha` 精确零初始化；`tanh(alpha)` 在 head 输出处相加，仅调用一次原 out_proj。块首 current 输出为零但 memory 可非零。
- 仅缓存最后 Retriever 输入 x 与 QA 编码器最终 y；沿用现有 detached Records、recent→remote 迁移、FIFO、学生隔离和提交事务。feature_version 保留特征生成时版本。
- `training_stage=gate_only` 冻结原参数并关闭全部 dropout，保持 gate 的反向图；`last_layer` 只解冻最后 Retriever 与预测头，前置网络保持 eval；`full` 全量训练。gate 独立学习率 `gate_lr`，其他可训练参数使用 `learning_rate`。
- `load_baseline_state_dict` 只补新增的 gate，其余键集合和 shape 严格验证；普通 V2 checkpoint 仍 strict 加载。训练产物记录显式 architecture/stage/decay/gate_lr、可训练参数名称、权重、优化器、种子与 epoch 边界状态。

没有修改 baseline embedding 索引、数据划分或计分逻辑。`freeze_backbone=True` 仍代表全部冻结，gate-only 应使用 `training_stage`。

## 测试与复运行

使用已有 Python 3.7 / PyTorch 1.13.1+cu117 环境，CPU、单线程：

```bash
OMP_NUM_THREADS=1 /home/remote/u8016457/miniconda3/envs/pykt/bin/python -m pytest \
  tests/test_long_mem_topk_akt_v2.py tests/test_long_mem_topk_akt.py \
  tests/test_long_mem_akt.py tests/test_opera_integration_config.py -q

PYTHONPATH=. OMP_NUM_THREADS=1 /home/remote/u8016457/miniconda3/envs/pykt/bin/python \
  examples/validate_long_mem_topk_akt_v2.py --output /tmp/longmem-v2-new-run
```

最终联合运行结果：**86 passed，2 warnings，98.22 秒**。两条 warning 均为已有 Faiss/distutils 弃用提示。CLI `--help` 已确认新增参数可用，`git diff --check` 通过。

输出目录必须尚不存在。V2 专项 31 项：非空记忆零 gate 等价（2 种 decay × separate_qa × kq_same × qid/pdiff，n_blocks=2）、空 memory、容量和 Top-k 边界、单候选/全 padding 反向有限性、因果性、int64 大位置平移、学生交错、重复提交拒绝、gate-only 更新且 backbone 不变、读取 V 梯度、仅一次 out_proj、checkpoint 白名单、训练保存恢复 replay、显式 memory 数学参考和 tail_mass 正反例。另有旧 Top-k 25 项、其他相关回归 30 项。

ANN 快照投影测试不属于本次 V2 验收：V2 显式拒绝 hnsw_ip；旧模式保留原实现，不宣称修复其快照一致性。未新增重排或 oversample。

## 合成小数据 B0 / Z0 / gate-only

完整 resolved config、各 seed 指标、gate/gradient、分支范数、资源统计见 [JSON](experiments/long_mem_topk_akt_v2_smoke.json)。脚本先用 3 个训练 step 生成小 AKT baseline，再载入 V2 做 Z0 和 3 个 gate-only step。训练/验证学生不重叠，所有配置使用同一个 mask，排除每块首题，每次验证 36 个计分位置。seed 为 42/43/44，未使用测试集选配置。

| seed | Z0 最大预测误差 | B0/Z0 NLL | 训练后 NLL | B0 AUC | 训练后 AUC |
|---|---:|---:|---:|---:|---:|
| 42 | 0 | 0.7227643042 | 0.7227663314 | 0.422840 | 0.422840 |
| 43 | 0 | 0.6943907078 | 0.6943922434 | 0.456250 | 0.456250 |
| 44 | 0 | 0.7103298455 | 0.7103297670 | 0.331169 | 0.334416 |

FP32 等价容差为 atol=1e-6、rtol=1e-5；这批合成输入观测误差为 0，不推导所有环境下逐位等价。gate 梯度非零，backbone 参数完全不变。另记录同一训练后 checkpoint 的 gate=0 与 memory off 干预。

本次 gate-only 没有稳定改善；随机生成的数据只验证工程流程，不能作为实际预测性能结论。V2 验证 replay 约 0.059–0.080 秒，bank 峰值 2304 字节，进程 CPU 累计 RSS 峰值约 415–417 MB；没有 GPU 测量。组件计时是粗粒度诊断，不能据此宣称检索加速。

## 真实训练入口

下面是参数模板，`AKT_SOURCE` 必须是真实原 AKT 产物目录，且 config 中 dataset/fold 与指定数据一致；`DATA_CONFIG` 必须是实际数据配置。仓库 `examples/exp_bridge2006/{base,candidate,model}` 标记为 long_mem_akt，不可冒充原 AKT checkpoint。

```bash
PYTHONPATH=. /home/remote/u8016457/miniconda3/envs/pykt/bin/python \
  examples/wandb_long_mem_topk_akt_train.py \
  --akt_source "$AKT_SOURCE" --data_config "$DATA_CONFIG" \
  --dataset_name bridge2algebra2006 --fold 0 --output "$OUTPUT" \
  --architecture_variant v2_last_retriever_residual \
  --training_stage gate_only --gate_lr 0.001 \
  --memory_decay_mode legacy_mul --retrieval_backend exact \
  --block_len 200 --recent_capacity 200 --remote_capacity 8192 --remote_topk 32
```

本轮没有完成真实数据 B0/B1/G1–G4/F1 的匹配预算实验，也没有确认与该数据划分配套的原 AKT checkpoint；因此不报告真实数据收益。有限联合微调可改 `--training_stage last_layer --learning_rate ...`，共享学习率应按 baseline 微调学习率选择。B1 可使用相同 V2/last_layer 配置加 `--memory_mode off`，对齐可训练原层和有效交互预算。

## 明确限制

- candidate_context 不是全历史距离；删候选后 tail_mass 可升可降。最新 memory 候选距离为零，单候选权重恒为 1，没有额外年龄衰减。
- `dense_oracle` 仅表示最后 Retriever 在相同缓存特征上读取全部 remote，不是全层全历史 AKT。V2 暂未提供同快照 dense retention/输出差异诊断，report 显式标记不可用；整段 replay 对照也不能混作同快照比较。
- 未实现 ANN、rerank、local_anchored_gap、prior_margin、全局 additive。V2 只支持 qid/qid_pdiff。
- cached encoder 特征仍可能在联合训练时过时；重投影不能解决 encoder 漂移。只保存 epoch 边界训练状态，不支持块中断恢复。
- 零初始化仅保证条件满足时的零步等价，不保证训练后性能下界。大规模训练前应先用真实原 checkpoint 和同计分 mask 复现 Z0。
