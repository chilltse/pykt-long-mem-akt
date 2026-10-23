# LongMemAKT 当前实现说明

本文依据当前工作区源码整理，描述实际实现，不是待实施方案。模型类只有 `LongMemAKT`，注册名称仍为 `long_mem_akt`。

## 1. 总体结构

模型以原始 AKT 作为块内预测器，用按学生维护的固定槽位记忆保存跨块信息，再通过选择性 logit 残差修正基础预测。

```text
学生完整序列 → 按 block_size 顺序切块
                       │
          ┌────────────┴─────────────┐
          │                          │
    原始 AKT 块内预测         读取此前块的记忆 M_b
       zS、short、qe                 long_read
          └────────────┬─────────────┘
                       ↓
                 残差头输出 delta
                       ↓
      AKT MC 方差 U + 收益头预测 gain → 门控 alpha
                       ↓
              sigmoid(zS + alpha × delta)
                       ↓
    当前块交互编码、时间加权摘要 → 更新 M_{b+1}
```

整个块读取同一个入口记忆状态。当前块真实答案只在因果块内历史及块后写入中使用，不提前写入长期记忆用于预测本块。

旧 `gate/short/long/concat` 模式及 `ablation/vector_gate` 参数已删除；旧 checkpoint 不支持加载。下文 E0–E3 是同一个模型的评估干预，不是恢复旧架构。

## 2. 输入、分块与计分

`LongMemAKTDataset` 读取原始 split CSV，每行必须是一个学生的完整有序序列，不能是已经切好的片段。

- `uid`：字符串学生 ID，保留前导零。
- `cseqs`：知识点 ID，必需；`qseqs`：题目 ID，缺失时使用零。
- `rseqs`：0/1 作答结果；`tseqs`：时间戳，缺失时为 NaN。
- `smasks`：复现旧数据切片的计分位置；`allowed`：原始 `selectmasks` 允许的位置。

Runner 固定 `batch_size=1`，不混合学生，按原顺序切成至多 `block_size` 长的块。每个学生开始时重新初始化记忆。

两个计分协议：

| 协议 | 计分规则 |
|---|---|
| `legacy`（默认） | 按数据配置的 `maxlen` 重现旧分段计分：每段首题不计分，长度不足 `min_seq_len` 的尾段不计分，再与原始选择掩码取交集 |
| `student` | 只去掉学生整条序列的第一题，并遵守原始选择掩码 |

`legacy` 的分段长度来自数据配置，不一定等于模型的 `block_size`。`legacy` 只是计分协议，不代表支持旧模型。

**不计分不等于丢弃历史**：所有 valid 交互仍参与块内历史和块后记忆更新，计分掩码只控制监督目标与指标。

## 3. 块内基础 AKT

`_forward_short()` 复用原始 AKT 的 embedding、注意力网络和输出头，不另建基础模型类：

```text
qe, interaction = AKT.encode_embeddings(concept, response, question)
short = AKT.model(qe, interaction, difficulty)
zS = AKT.out(concat(short, qe))
pS = sigmoid(zS)
```

交互编码器可见当前位置及此前交互；检索路径只看严格早于当前位置的交互，当前位置题目信息可见。源码不额外 shift 输入。有效位置先压紧再进入 AKT，避免 padding 影响距离计算。

基础训练阶段保留 AKT 的难度正则项。后续阶段 AKT 参数冻结且保持 eval，基础预测为确定性预测；MC 仅是额外的不确定性估计，最终基准不是 MC 概率均值。

## 4. 跨块长期记忆

### 4.1 状态和读取

设槽位数为 S、模型维度为 D、Value 维度为 V（默认等于 D）：

- 共享可学习 Key：`K ∈ R^(S×D)`。
- 可学习初始 Value：`M_init ∈ R^(S×V)`。
- 学生运行状态：`M_b ∈ R^(S×V)`，从 `M_init` clone，学生之间不共享可变状态。

Key 和初始 Value 均采用 Kaiming normal 初始化。运行记忆作为函数参数传入，不存成持续修改的模块状态。

```text
w_t = softmax(query_proj(qe_t) K^T / sqrt(D))
long_read_t = Σ_j w_tj M_bj
```

这些寻址权重也用于块后槽位摘要。

### 4.2 写入编码与时间摘要

默认 `write_encoding="causal"`：共享 AKT 的交互 embedding，加块内位置 embedding，经独立的因果 Transformer，再经 `write_z` 投影到 V 维。它不是复用预测器的隐藏状态。

类内部还支持 `independent`，直接投影交互 embedding；当前训练 CLI 未提供此开关，使用默认 causal。

对块内有效交互 t，令 Δt 为距离块末的时间：

```text
lambda = softplus(raw_lambda)
a_t = exp(-lambda × Δt)
C_j = Σ_t w_tj
Cw_j = Σ_t w_tj a_t
summary_j = Σ_t w_tj a_t write_z_t / max(Cw_j, 1e-8)
```

时间戳完整、非负且单调不减时，使用 `(最后时间戳 - 当前时间戳) / time_unit`，默认 `time_unit=1000`；否则使用距块末的有效事件数。时间差先用 float64 计算，避免大时间戳的精度损失。关闭时间权重时，有效位置 `a_t=1`。`raw_lambda` 初始为 -5。

这里的 `a_t` 是写摘要权重，与预测门控 `alpha` 不同。

### 4.3 块级 erase/add 更新

```text
x_j = concat(summary_j, M_bj, log(1 + C_j))
erase_j = sigmoid(Linear(x_j))
add_j = tanh(Linear(x_j))
rho_j = 1 - exp(-softplus(raw_kappa) × C_j)
M_(b+1),j = M_bj × (1 - rho_j × erase_j) + rho_j × add_j
```

`raw_kappa` 初始为 0。更新强度使用未加时间权重的访问量 `C_j`；没有 valid 交互的块不更新记忆。读取只能使用此前已完成块的状态。

## 5. 残差修正与选择性融合

### 5.1 残差头

```text
features_t = concat(
    LN(qe_t),
    LN(short_proj(short_t)),
    LN(long_proj(long_read_t)),
    zS_t
)
delta_t = Linear(GELU(Linear(features_t)))
pC_t = sigmoid(zS_t + delta_t)
```

输入维度 `3D+1`，隐藏层默认 64。LN 使用无额外可学习仿射参数的 `F.layer_norm`。残差头最后一层权重和偏置全部零初始化，所以初始候选预测与基础预测相同。

只有已经完成至少一个有效历史块才允许残差介入。第一块即使有块内历史，也不启用长期修正。

### 5.2 收益头

收益头输入 `[features, delta]`，维度 `3D+2`，结构为 `Linear → GELU → Linear`，隐藏层默认 32，输出无 sigmoid 的有符号标量 `gain`。

监督目标是每个计分交互的 Brier 改善量：

```text
gain_target = (y - pS)^2 - (y - pC)^2
```

正值表示完整残差改善了该样本的平方误差；负值表示恶化。训练使用 MSE，且只使用存在历史块的计分样本。真实 y 仅用于训练收益头，不作为推理输入。

### 5.3 MC Dropout 不确定性

只开启 AKT 中的 `nn.Dropout`，默认运行 M=20 次：

```text
U_t = (1/M) Σ_m (pS_t^(m) - mean_m(pS_t^(m)))^2
```

采用 float64 Welford 算法计算总体方差，最终转回概率 dtype。随机种子由 `sha256(mc_seed:uid:block_index)` 派生，默认 `mc_seed=42`。采样无梯度，完成后恢复外部 RNG 和各子模块训练状态，不写长期记忆。

### 5.4 双门控与最终预测

平滑门函数：

```text
x = clamp((v - tau) / width, 0, 1)
smoothstep(v, tau, width) = x² × (3 - 2x)

alpha_t = has_history_t
          × smoothstep(U_t, tau_U, width_U)
          × smoothstep(gain_t, tau_G, width_G)
p_t = sigmoid(zS_t + alpha_t × delta_t)
```

默认推理先判断 `U > tau_U`，只在满足条件的位置执行残差头和收益头。注意：长期读取和特征构造仍会执行，有历史块时 MC 仍需运行；这不是省掉整个长期分支计算。

无历史、低不确定性或收益不足时，输出保持基础预测。源码仍保留新分支非有限数值检测与隔离，并在诊断中记录 `nonfinite`。这些是当前选择性计算和数值处理，不是旧模式或旧 checkpoint 兼容路径。

## 6. 数据角色与训练流程

`--stage train` 依次执行 `split → base → candidate → benefit → calibrate`。

### 6.1 学生级互斥划分

保留指定 fold 作为 dev、原始测试学生作为 test。从其余训练 fold 的学生池随机划分：约 60% fit、20% selector_fit、20% calibration；小数据情况下后两者各至少 2 人，训练学生池至少需要 6 人。

所有角色按 UID 互斥，角色列表和源 CSV 的 SHA256 一并保存。角色不重叠不代表必然有足够长序列：收益训练还要求至少两个学生有可计分的历史块。

### 6.2 各阶段训练范围

| 阶段 | 使用数据 | 可训练参数 | 目标与选择方式 |
|---|---|---|---|
| base | fit，dev 验证 | 仅 AKT | BCE 加难度正则，按 dev AUC 选模型 |
| candidate | fit，dev 验证 | 长期读写、投影、残差头等非 AKT/非收益头参数 | 有历史时使用完整残差，按 dev logloss 选模型 |
| benefit | selector_fit 内再次按学生划分训练/验证 | 仅收益头 | 预测 Brier 改善量，按验证 MSE 选择，10 轮不改善停止 |
| calibrate | calibration | 不训练权重 | 搜索 E2/E3 门控阈值和宽度 |
| evaluate | test | 全部冻结 | 最终模型评估 |

candidate 从 base 权重严格初始化，并检查阶段结束后 AKT 权重未改变。收益训练前冻结候选模型，收集特征与完整候选输出。

内部 `phase` 决定前向路径（base/candidate/inference），`set_stage()` 决定参数冻结和模块模式。这两者分离，使冻结后的 candidate 路径仍可用于特征收集和消融评估。

### 6.3 TBPTT

学生内部按 `tbptt_blocks` 个块组成一组；默认 2，candidate 必须至少为 2。组内保留跨块计算图，每组按计分样本数归一化损失，梯度裁剪阈值为 5。

组末保存待写摘要，在优化后 detach；下一组重新执行该边界写入，使下一组预测损失仍能监督写入参数，又不复用旧计算图。

### 6.4 校准

E2（仅不确定性门控）与 E3（双门控）分别校准。当前搜索网格为：

- 不确定性阈值：有历史样本 U 的 80%、90%、95% 分位数。
- 不确定性宽度：最大 U 的 0.1 或 0.5 倍，下限 `1e-8`。
- E3 收益阈值：0 或 0.01；收益宽度：0.01 或 0.05。
- 另加入 `tau_U=0.25` 的不介入策略。

根据实际融合概率计算指标，在 logloss 不超过基础模型加容忍量的策略中，选择 Brier 最低的策略。`--logloss_tolerance` 默认 0。

这只是 calibration 集上的选择准则，不保证测试集改善。选中不介入策略时，E3 与 E0 指标可能完全相同，并不自动意味着脚本失效。

## 7. 当前默认参数

| 参数 | CLI 默认值/约束 |
|---|---|
| `d_model` / `num_slots` | 128 / 32 |
| `n_blocks` / `num_attn_heads` | 2 / 8；n_blocks 是网络层数，不是学生序列块数 |
| `d_ff` / `final_fc_dim` | 256 / 512 |
| `dropout` | 0.1 |
| `block_size` | 必须显式指定 |
| `tbptt_blocks` | 2 |
| `correction_hidden` / `benefit_hidden` | 64 / 32 |
| `mc_samples` / `mc_seed` | 首阶段未指定时 20 / 42，后续继承 |
| `epochs` / `benefit_epochs` | base、candidate 各至多 20；benefit 至多 100 |
| `learning_rate` / 优化器 | 0.001 / Adam |
| `seed` / `fold` | 42 / 0 |
| `score_protocol` | legacy |
| `device` / `threads` | cpu / 4 |

`d_model` 必须能被注意力头数整除。当前只支持 `emb_type="qid"`。模型类还有时间摘要等内部参数，但并非全部暴露为 CLI 参数。

## 8. 训练与评估命令

从仓库根目录进入 `examples`：

```bash
cd examples

# 完整训练流水线
PYTHONPATH=.. python wandb_long_mem_akt_train.py --stage train \
  --dataset_name bridge2algebra2006 --block_size 200 \
  --epochs 20 --output saved_model/long_mem_akt_v2

# 可选：只评估最终双门控模型
PYTHONPATH=.. python wandb_long_mem_akt_train.py --stage evaluate \
  --source saved_model/long_mem_akt_v2/model --block_size 200 \
  --output saved_model/long_mem_akt_v2/evaluation

# 一键 E0–E3 对比，不需要先单独 evaluate
PYTHONPATH=.. python evaluate_long_mem_akt_ablation.py \
  --source saved_model/long_mem_akt_v2/model \
  --output saved_model/long_mem_akt_v2/ablations
```

使用 GPU 时给相应命令加 `--device cuda`。已有新流水线生成的最终 `model/` 可直接消融，不必重新训练。输出目录必须尚不存在；若示例路径已存在，改用新的实验或结果目录，不要覆盖原结果。

## 9. E0–E3 消融的含义

四组复用同一最终 checkpoint、基础 AKT、残差权重、测试学生和计分位置，各次评估独立重置学生记忆。

| 组别 | 有历史位置的预测 | 用途 |
|---|---|---|
| E0 | `sigmoid(zS)` | 基础 AKT |
| E1 | `sigmoid(zS + delta)` | 完整长期残差、不使用门控 |
| E2 | `sigmoid(zS + smoothstep(U) × delta)` | 仅不确定性门控，推理跳过收益头 |
| E3 | `sigmoid(zS + smoothstep(U) × smoothstep(gain) × delta)` | 完整双门控模型 |

无历史时四组均使用基础输出。默认读取已保存的 E2/E3 校准策略；`--recalibrate` 在 calibration 学生上重新选择策略，不使用 test 调参，也不修改源 checkpoint。

这是**同一训练结果的推理机制消融**，不是独立重训四个架构，也不提供跨种子显著性结论。E0 是最终模型内被冻结的基础 AKT。

结果包含 AUC、accuracy、Brier、logloss、相对 E0 的变化、介入率、平均 alpha、错转对/对转错计数和耗时。AUC/accuracy 越大越好；Brier/logloss 越小越好；对应差值为负表示改善。介入率表示 `alpha>0`，不等同于预测类别改变率。

计时包含共享评估器中的记忆维护和 CSV 输出，E0 也经过该流程，因此不能直接当作优化后的裸 AKT 推理时延。

## 10. 保存产物与源码索引

```text
saved_model/long_mem_akt_v2/
├── roles.json
├── base/
├── candidate/
├── benefit/
├── model/
│   ├── qid_model.ckpt
│   ├── config.json
│   ├── roles.json
│   ├── selective_manifest.json
│   ├── calibration.json
│   └── target_indices.json
└── ablations/
    ├── summary.md / summary.csv
    ├── metrics.json
    ├── students.csv
    ├── E0_predictions.csv … E3_predictions.csv
    ├── calibration.json
    ├── target_indices.json
    └── run.json
```

加载校验版本、权重、配置和角色摘要；数据加载再校验源 CSV。最终推理还校验收益头准备标志及校准阈值。不能只拿旧权重文件替换进新目录使用。

逐题 CSV 保留计分标志，残差组额外记录基础概率、候选概率、U、gain、alpha、delta、历史标志、是否执行候选、是否介入及原因。未执行的候选预测留空，不应视为零概率。

| 文件 | 职责 |
|---|---|
| [long_mem_akt.py](../pykt/models/long_mem_akt.py) | 唯一模型类、短期预测、记忆读写、残差、收益与 MC 门控 |
| [long_mem_akt_dataset.py](../pykt/datasets/long_mem_akt_dataset.py) | 原始学生序列及计分掩码 |
| [long_mem_akt_runner.py](../pykt/models/long_mem_akt_runner.py) | 顺序分块、TBPTT、评估与逐题诊断 |
| [long_mem_akt_workflow.py](../pykt/models/long_mem_akt_workflow.py) | 数据角色、产物校验、收益训练及校准；不是新模型类 |
| [wandb_long_mem_akt_train.py](../examples/wandb_long_mem_akt_train.py) | 完整训练及单次评估入口 |
| [evaluate_long_mem_akt_ablation.py](../examples/evaluate_long_mem_akt_ablation.py) | 同 checkpoint 的 E0–E3 比较 |
| [test_long_mem_akt.py](../tests/test_long_mem_akt.py) | 因果性、状态隔离、梯度、MC、阶段冻结及命令行流程测试 |
