# LongMemAKT 最小改动执行手顺：选择性长期修正与平滑融合

> 根据用户上传的 long_mem_akt_implementation_cn.md 制定（2026-10-21）。
> 本文取代此前“改回真实 AKT、保存不断增长的块摘要库、默认全量残差”的计划。
> 当前证据来自实现说明，不是对实际仓库源码的直接审计。执行者须核对源码再修改。
> 交付目标：保留当前结构，新增可关闭的 selective_residual 模式；本文件不代表代码已修改或实验已运行。

## 1. 本次确定的行为

默认使用原 AKT。仅当：
1. 原 AKT 的 MC Dropout 不确定性超过阈值；
2. 已有该学生的真实旧块状态；
3. 样本外训练的收益头预计长期修正有益；
才平滑引入长期修正。其他位置严格返回原 AKT。

公式：
```text
zS = 原 AKT 的确定性 logit
delta = Correction(qe, short_query_output, long_read, zS)
zC = zS + delta
pC = sigmoid(zC)
gain_hat = Benefit(qe, short_query_output, long_read, zS, delta)

alpha = has_history
        * smoothstep(U; tau_U, width_U)
        * smoothstep(gain_hat; tau_G, width_G)

zFinal = zS + alpha * delta
pFinal = sigmoid(zFinal)
```

short 是 AKT 当前查询历史的瞬时输出，不是跨块持久 hidden state。
memory 是已有的显式长期运行状态。
不采用长期分支独立预测替换 AKT，不混合两套隐藏表示后重用原输出头。

没有逐样本“不降性能”保证。保证仅限于 alpha=0 的位置严格回到基线；整体收益必须由校准与独立测试验证。

## 2. 最小改动范围

### 保留

- self.akt、encode_embeddings、Architecture 和原 self.akt.out。
- 当前 q/c/PID 语义、严格因果 mask、padding 压紧处理。
- keys、query_proj、memory_init、long 读取。
- summarize、因果写编码器、时间权重和 erase/add write。
- 学生级顺序加载、非重叠 block、显式 memory、legacy/student 计分协议。
- 原有 gate/short/long/concat 模式，保持旧 checkpoint 可评估。
- 原有 TBPTT pending 写入方案，先不改变训练图拓扑。

### 新增或修改

| 文件 | 最小修改 |
| --- | --- |
| pykt/models/long_mem_akt.py | 可复用确定性短期前向；新增 correction_head、benefit_head、smoothstep、MC 方法与新模式 |
| pykt/models/long_mem_akt_runner.py | 阶段控制、历史标记、收益数据生成、校准和新增导出 |
| examples/wandb_long_mem_akt_train.py | 新增模式和阶段参数；显式 block_size |
| pykt/models/train_model.py | 分阶段保存、适合本阶段的验证/早停，保留旧流程 |
| pykt/models/init_model.py | 传递新配置，旧配置默认不启用新行为 |
| examples/wandb_predict.py | 加载选择器/校准参数，选择性推理与日志 |
| tests/test_long_mem_akt.py | 增加基线回退、MC副作用、收益/融合检查 |
| 新增一个校准入口（按仓库惯例命名） | 收益头训练与阈值选择；不在通用训练循环里堆所有阶段 |

不要为了本任务重构 dataset、改变学生拆分、修改现有标签计分集合或重命名全套模型。

## 3. 先检查真实源码和 checkpoint

- [ ] 确认实现说明与实际源码一致，尤其 short 原头路径和 padding 处理。
- [ ] 保留用户尚未提交的 block_size=10 调试修改，不覆盖；正式命令显式传 200。
- [ ] 保存原配置及基线预测样本索引。
- [ ] 定位已训练独立 AKT checkpoint，或用 short 模式重新训练一个。
- [ ] 用同一原 AKT 权重验证 adapter 与原 short 路径一致。

重要：旧 gate checkpoint 内的 self.akt 可能已被融合训练改动，不能直接称为“原始 AKT 强基线”。可作为另一个实验初始化，但正式保护基线优先使用已验证的独立 short checkpoint。

## 4. 拆出纯短期前向，保持原计算不变

建议私有接口，具体名称遵循仓库：
```python
base = _forward_short(c, q, r, valid)
# base.qe                [B,T,D]
# base.interaction       原共享交互表示
# base.short             [B,T,D] 瞬时查询输出
# base.logits            [B,T]
# base.reg_loss          原 AKT 正则
```

base.logits 必须严格来自：
```python
self.akt.out(torch.cat([base.short, base.qe], -1)).squeeze(-1)
```

- 不对原 short 先做 short_proj/LayerNorm 后再求 zS。
- 不做额外 response shift。
- 保留有效位置压紧处理。
- predict_block 仍不写 memory。
- 扩展返回 info 时保留旧 runner 所需 query/weights/interaction 等 cache，不能破坏 summarize。
- logits/probability API 沿用仓库规范，不引入重复 sigmoid。

新分支冻结 self.akt；包装模型 train() 时确保 self.akt.eval()。
训练 candidate 时可以 no_grad 计算 base，但不能 no_grad 整个长期分支。

## 5. 新增小型候选修正头，不再复用旧隐藏融合

复用已有长期读取：
```text
weights = address(qe)
long_read = weights @ memory  # [B,T,V]
```

新增支路可复用 long_proj；short_proj 仅用于新支路，不改原 logits：

```text
qR = LayerNorm(qe)                       # 新支路内使用
sR = LayerNorm(short_proj(short))
lR = LayerNorm(long_proj(long_read))
input = concat(qR, sR, lR, zS[...,None])
delta = Linear(GELU(Linear(input))).squeeze(-1)
zC = zS + delta
```

起点：隐藏层 64；第一版 correction 不用 dropout，减少候选自身随机性。
最终 Linear 权重和 bias 全零，前层正常初始化。

- 共享到新分支的长期参数允许训练；旧 gate 不执行、不进入新模式优化器。
- 没有真实旧块时 delta 强制为 0。
- candidate 模式训练时先不乘不确定性/收益门控，避免冷启动饿死分支。
- selective_residual 推理才乘 alpha。
- 零初始化第一步上游梯度可为零；下一步末层非零后再检查写入梯度。

新增模式建议：
- residual_candidate：用于训练候选和生成样本外收益数据。
- selective_residual：需要已训练 benefit 和 calibration，正式选择性输出。
旧四种 ablation 保留；未知模式报错，缺 selector/calibration 不能静默全量融合。

## 6. “有长期历史”不用 memory 数值猜测

当前 memory_init 可学习且非零，它是共享初始状态，不代表个人旧证据。

runner 每名学生维护：
```python
completed_valid_blocks = 0
has_history = completed_valid_blocks > 0
```

预测使用进入块时的 has_history。成功处理一个含有效交互的完成块后递增。
空块不递增，pending 时也要保持“此前已有实际完成块”的语义。
首块 alpha=0、delta=0，不允许初始共享 memory 改变基线。

这是因果可用性条件，不是新增人工预测特征。不要再加入覆盖度、练习计数等路由输入。

## 7. MC Dropout：只扰动 AKT，不更新长期状态

定义 U 为原 AKT M 次随机前向所得概率的总体方差：
```text
U = mean(p_mc ** 2) - mean(p_mc) ** 2
U = clamp_min(U, 0)
```

M=20 是初始配置，不是论文复现承诺；校准和测试固定相同 M 与 dropout 范围。
pS 永远使用普通 eval 前向，不替换成 MC 均值。

实现要求：
1. 确定性 base 先计算并保留。
2. MC 只调用 _forward_short，不调用完整 predict_block，不写 memory，不生成摘要，不调用 selector。
3. 开启原 AKT 实际使用的 dropout；核对 nn.Dropout、functional dropout、attention 内 dropout。
4. 如用 self.akt.train()，先检查没有需要保持 eval 的 BatchNorm 等层；否则用显式上下文控制。
5. 保存并恢复每个子模块原 training 标记，不能只恢复最外层标记。
6. no_grad 执行；退出后 AKT 保持冻结/eval，长期写编码器恢复原模式。
7. 不替换确定性 forward 的 qe/interaction/cache 为某个随机样本。
8. 不让 MC 采样消耗训练随机序列影响其他 dropout；用局部 RNG 上下文并测试状态恢复。
9. 复现种子由配置 seed、稳定 uid 摘要和 block 索引构成；不要用跨进程不稳定的 Python hash(uid)。
10. MC 仍使用原因果 mask，当前/未来答案不可见。
11. 全 padding 返回零 U，不进入无效 attention。

纯短期低不确定样本也需要 MC 才能判断；因此这是存储轻量的策略，不承诺推理轻量。
可用流式累计均值/方差避免保存 [M,B,T]。初版循环前向，向量化作为后续性能优化。

训练 candidate 阶段不必跑 MC。生成 selector 数据、校准和正式推理时再跑。

## 8. 收益头：学习样本外收益，不能把更自信当作更有用

输入用现成可学习表征，不添加手工特征：
```text
benefit_input = concat(qR, sR, lR, zS[...,None], delta[...,None])
gain_hat = BenefitMLP(benefit_input)  # 有正负，不用 sigmoid
```

隐藏层可从 32 起步。第一版用线性输出和 MSE；正则在验证数据选。
benefit 所用表征和投影在该阶段全部冻结。

监督：
```text
gain_target = (y - pS)^2 - (y - pC)^2
```
target 在 [-1,1]；正值表示完整候选本次 Brier 更好。
学习目标是条件平均收益，不是“这次一定对”的标签。

仅选 scored 且 has_history 的样本；可在预先固定的不确定性候选区间训练，以聚焦部署分布，但不要根据 y 筛样本。
第一版最简单：全部 has_history 样本训练，校准时只在高 U 区间筛选。

禁止：
- 直接比较 pC 是否更接近 0/1 决定采用。
- 用训练 candidate 的同一学生上的拟合收益训练 selector 并当作样本外证据。
- 把 test 标签用于收益头训练、阈值选择或 early stopping。
- 对当前预测使用当前真实 y 计算在线 gain。

## 9. 平滑门控与数值回退

```python
def smoothstep(v, threshold, width):
    if width <= 0:
        raise ValueError("width must be positive")
    x = ((v - threshold) / width).clamp(0, 1)
    return x*x*(3 - 2*x)
```

```text
aU = smoothstep(U, tau_U, width_U)
aG = smoothstep(gain_hat, tau_G, width_G)
alpha = has_history * aU * aG
zFinal = zS + alpha * delta
```

- tau_G>=0；允许 tau_G 很高而完全不介入。
- 无历史、U<=tau_U 或 gain_hat<=tau_G，严格输出 pS。
- 对 active mask 之外直接复制 zS，不依赖 0*NaN；有 NaN 时记录错误并回退，不能静默污染整块。
- no-history 可以连 MC 都跳过，因为最终必定回退。
- 第一关后可只对 U>tau_U 的有效位置执行 candidate 和 benefit。
- 但寻址/interaction 若 summarize 需要，仍按原结构生成；“不参与预测”不代表停止记忆维护。
- 每块照常写入所有已观察 valid 交互，不能只记住高 U 交互。
- smoothstep 的平滑是对输入分数的数学性质；MC 的采样误差仍可能影响 U，不能宣称完全无抖动。

关键解释：benefit 预测完整修正的收益，而实际 alpha 可能小于1。因此最终校准必须对实际 zFinal 计算损失，不能只看 gain_hat 是否准确。

## 10. 分阶段训练与数据角色

建议显式保存学生 UID 划分，所有角色互不重叠：
- fit：AKT 与 candidate 的训练学生。
- dev：candidate 早停与基础超参数选择。
- selector_fit：固定预测器后训练 benefit。
- calibration：选择不确定性阈值和门控宽度。
- test：最终一次评估。

selector_fit 内部可再留学生子集早停。数据不足时采用按学生交叉拟合；不能用随机交互切分替代。
若只有60名学生，只用于流程 smoke test，不据此宣称路由收益稳定。

### 阶段 A：基线

训练或加载独立原 AKT。原 short/eval 数值一致后冻结。
若 score_protocol=legacy，所有后续比较保持 legacy；student 作为另一套完整实验。

### 阶段 B：候选

- mode=residual_candidate。
- self.akt 冻结且 eval；长记忆、写入、读取、残差允许训练。
- candidate BCEWithLogits 在与原协议一致的计分目标上计算。
- 无历史块不能监督长期修正；允许基线常量损失用于报告，但 backward 前检查有可训练有效项。
- 保留原 TBPTT；要求 tbptt_blocks>=2，避免 independent/causal 写编码器在全部边界 detach 时没有未来监督。
- 当前组内梯度足以先训练；组间 write encoder 不受下一组监督是已有明确限制，先不为了本任务重写 replay。
- 如统计发现组内没有足够有监督的未来块，再单独修正重算，不用 retain_graph=True。
- 冻结 AKT 的 difficulty L2 只是常量，对新增参数无用；可保留报告，但不要误认为它训练长期分支。
- candidate 早停优先验证 logloss/Brier，保存策略写明；旧模式保持原 AUC 早停行为。
- 最后一个学生块无未来写入监督，正常。

### 阶段 C：收益头

- 冻结 AKT、candidate、keys、writer、projection。
- 固定参数，从每个 selector_fit 学生的初始状态重新顺序构建 memory。
- 生成 pS/pC/U 和收益头输入，严格只用预测前状态。
- 当前标签只用于离线 gain_target。
- 训练 benefit，冻结预测器防止监督目标移动。
- 大数据可流式生成或缓存 CPU 文件；禁止把完整 autograd 图缓存到磁盘/内存。

### 阶段 D：校准

冻结所有模型，用 calibration 学生重新构建 memory：
- 预定义小规模 tau_U 分位数候选，例如 80/90/95百分位，仅为搜索起点。
- 对 gain 阈值、平滑宽度使用少量候选，不大规模追逐校准噪声。
- 始终包含完全不介入策略。
- 对每个组合计算真实 pFinal 的全量 Brier、logloss、AUC、介入率和平均 alpha。
- 预先指定主目标（建议 Brier），并设置 logloss 不能出现不可接受退化的约束。
- 不强求固定介入比例。
- 按学生评估不确定范围；选完阈值若要宣称统计保证，需要独立确认或正式多重选择处理，普通 bootstrap 不能自动抵消调参偏差。

不默认联合微调全部模型；这会改变 zS 和收益目标，破坏“保留已验证 AKT”的基线保证。

## 11. Checkpoint 与配置

新增配置示例，具体 flag 按仓库实际解析器注册：
```yaml
ablation: selective_residual
stage: inference
freeze_akt: true
block_size: 200
tbptt_blocks: 2
mc_samples: 20
mc_seed: 42
correction_hidden: 64
benefit_hidden: 32
residual_zero_init: true
memory_backend: existing_erase_add
# tau_U/width_U/tau_G/width_G 来自 calibration 文件，无默认假最优值
```

保存：
- base checkpoint 身份/校验摘要；
- candidate 权重与版本；
- benefit 权重；
- 校准阈值、宽度、MC范围/M/seed策略；
- score_protocol、block_size、时间单位；
- 各数据角色 UID 清单或不可逆校验摘要。

读取选择性 checkpoint 必须校验这些参数匹配。改变 block_size、MC范围、候选权重后应重新校准。
旧 checkpoint 缺新参数时只能进入旧模式或明确初始化训练新分支；不能静默进入选择性推理。

不要硬编码假 save_dir。执行者完成 CLI 后在仓库 README 写可复制的真实命令。沿用 examples 目录运行规范，所有新阶段命令显式指定 block_size。

## 12. 导出和监控

在现有 student_block_predictions.csv 基础上增加：
```text
base_probability, candidate_probability,
short_uncertainty, predicted_gain,
alpha, residual_logit, has_history,
candidate_evaluated, intervention, fallback_reason
```

- probability 保持表示最终预测。
- candidate 未执行的位置写空值，不伪造为实际候选结果。
- 导出 target 仅供离线评估，不用于在线逻辑。
- intervention 定义为 alpha>0；同时统计实际概率变化，避免很小 alpha 被夸大。
- 原 memory/梯度/吞吐诊断保留。
- 新增低 U 回退比例、收益不足回退比例、全局与介入子集损失差、MC耗时。
- log 中区分旧 gate 与新 alpha，不复用同一指标名造成歧义。

## 13. 必须新增的验证

1. selective alpha=0 时与原 short eval logits 一致，包括 PID/no-PID、padding。
2. 任意训练后的 memory_init，首块仍严格回退。
3. 在门控阈值上下的微小扰动下，固定 delta 的输出连续；smoothstep 范围正确。
4. 高 U 但 gain 不足回退；gain 高但 U 不足也回退。
5. MC 不改变 memory、不执行写入、不改变 cache、不改变参数/梯度，模式及 RNG 正确恢复。
6. 固定 seed 的 U 可复现；关闭所有 dropout 后 U近零（允许浮点误差）。
7. 修改当前/未来 response 不改变当前/过去 base、U、candidate、gain、final。MC 比较使用相同随机样本。
8. correction零初始化时首步基线相等；后续测试确认下一块损失训练当前 writer/encoder。
9. 不确定性筛选只影响采用预测，所有 valid 历史仍按原路径进入记忆。
10. candidate阶段冻结AKT；benefit阶段只有benefit有梯度。
11. tbptt_blocks=1 在新candidate训练模式明确拒绝或实现另有验证的重算，不能默默训练无梯度写编码器。
12. 选择器数据划分 UID 不重叠；同一目标 ID 的基线与最终输出对齐。
13. checkpoint恢复包含门控参数，预测一致；旧四模式仍通过旧测试。
14. 单块学生/全padding/无计分目标无NaN、无无效backward。
15. active位置含NaN按明确策略报诊断并回退，其他位置不受影响。

## 14. 最小实验次序

先只跑：
- E0：冻结原 AKT。
- E1：同一AKT＋候选残差，全量候选输出（仅用于测候选能力）。
- E2：高不确定时采用平滑候选，去掉收益筛选（测第二关必要性）。
- E3：双条件平滑介入（正式方案）。

所有实验相同目标和基线权重。E2/E3 分别校准，不能给某个方案更多测试集调参机会。
报告原错改对/原对改错、全量概率损失、介入覆盖率、学生级稳定性、总耗时。
如果E1在高U样本上没有可重复收益，先停止复杂selector调参，检查记忆是否有信息。
如果E3退化，允许完全回退，不强迫使用长期分支。

## 15. 轻量化是第二个小补丁，不与融合改造同时上线

当前 memory 已是固定 [S,V]，没有上一版计划的不断增长摘要库。优先保留它，避免混淆“选择性融合效果”和“记忆模型替换效果”。

待E3验证后再做：
1. 独立配置 write_n_blocks=1，不把 AKT n_blocks 一起改小。
2. 比较 value_dim=64、num_slots=16 等容量，记录成本与性能。
3. 如仍需移除erase/add，增加独立 memory_backend="normalized_aggregate"：
```text
weighted_count = summarize已计算的加权质量
numerator = summary * weighted_count[...,None]
A_new = rho * A_old + numerator
C_new = rho * C_old + weighted_count
M_new = A_new / clamp_min(C_new, eps)
```
- rho为每槽可学习的保留系数，明确是按完成块衰减，不冒称真实时间遗忘率。
- A/C从零开始，无历史回退保持。
- 全padding块不衰减不更新。
- state新增A/C，需要runner/tree_detach和pending调整，因而不是本次零成本替换。
- 使用weighted_count，不能把time-weighted summary与raw_count相乘后声称还原原分子。
- 该方案会丢掉部分自适应擦除能力，不保证比原writer准确。
- 更换backend之后重新训练candidate、benefit并校准。

MC多次AKT前向可能比Writer贵得多。先测量再优化，不把删erase/add当作推理性能瓶颈的必然解法。

## 16. 执行清单

- [ ] 审计当前源码，确认上传说明与真实实现一致。
- [ ] 保持旧四模式不变，增加纯短期前向。
- [ ] 加残差候选，零初始化，验证严格回退。
- [ ] 实现MC上下文及无副作用检查。
- [ ] 接入candidate阶段，保留当前TBPTT并明确>=2约束。
- [ ] 用独立学生数据生成收益监督，训练小benefit头。
- [ ] 加smoothstep和双条件final路径。
- [ ] 在独立calibration选择实际融合参数。
- [ ] 接入保存、恢复、CSV、计时和失败回退。
- [ ] 跑关键正确性测试及小数据smoke test。
- [ ] 跑E0/E1/E2/E3；报告是否需要长期介入，允许结论为无收益。
- [ ] 只有正式路径验证后再考虑轻量writer补丁。

交付应包含：实际diff、测试命令与结果、四阶段可执行命令、数据角色说明、checkpoint兼容规则、实验表。不要声称保证每次修正有效；不要把smoothstep的数学平滑当成预测准确性的证明。

