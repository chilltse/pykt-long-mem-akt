# LongMemTopKAKT V2：交给 Agent 的具体修正指南

日期：2026-10-25。状态：编码与实验规格；尚未运行真实仓库，不承诺性能提升。

依据：用户的 `akt(4).py` 与 `longmemtopkakt.txt`。后者是 SOP 和伪代码，不是完整实现或实验日志。必须核实真实仓库，不假定其中“55 项测试”和历史指标已经复现。

## 0. 固定主方案

**原 AKT 完整块内路径 + 仅最后一个 Knowledge Retriever 的独立记忆 attention + 每 head 零初始化残差门控。**

current 分支完整保留原 AKT 的 scores、context distance、乘法衰减和 dropout。memory 分支只读此前完成块，独立计算自己的初始概率、距离及最终 softmax。两路在每 head 输出处融合，随后执行一次原 out_proj，再接原残差、LayerNorm、FFN、预测头。

recent 全读，remote Top-k；两者只在 memory 分支内联合归一化。current 不能参与 memory 的概率分母和距离；memory 不能影响 current 的概率分母和距离。

初始化 eval 等价不保证训练后的性能下界。本版是对此前方案的修正，不等价于全历史 AKT，不以增加候选数量为性能保证。

## 1. 撤销或修正旧 SOP 的内容

| 旧内容 | 修正要求 |
|---|---|
| tail_mass 系统性低估 | 删除；重新归一化后可升可降 |
| local_anchored_gap | 不进入 V2 主路径；旧实验开关默认关闭 |
| positive_mul_log_pen | 暂不实现/禁用，避免机制叠加 |
| 共用 decayed_logits 后拆两个 softmax | 从初始 scores 起拆成两条完整计算路径 |
| 全局 log_additive | 禁止；current 永远保持原 legacy_mul |
| 所有层/所有 Retriever 注入 | 主路径只最后一个 Retriever |
| prior_margin 保证 baseline | 不使用，不能替代原路径隔离 |
| 零 gate 保证训练后不退化 | 删除，只保证符合条件的零步前向等价 |
| oversample 一定改善召回质量 | 改为实验假设，不写成普遍单测断言 |
| 同时更改距离、衰减、融合、ANN | 改为逐项消融 |

反例：初始概率 `[A=0.8,i=0.1,B=0.1]`，A 在 i 前、B 在 i 后。删 A 并归一化使 tail_mass(i) 从 0.1 变 0.5；删 B 则使其变 0。稀疏距离偏差方向不确定。

原距离 `D=sqrt(tail_mass*gap) <= sqrt(gap)`。旧 SOP 的远期线性外推可以远超此尺度，是新的遗忘假设，并非恢复遗漏概率质量的校准。

## 2. 先审计实际仓库

1. 读取 AGENTS.md、模型注册、实际 AKT/记忆模型、缓存、索引、训练/评估器、数据切块。
2. 记录 git commit、checkpoint、配置、种子与计分 mask。
3. 复现 memory off：相同输入、相同有效位置，原 checkpoint 结果不变。
4. 分别测零步开启记忆和微调后退化，不混为一个现象。
5. 明确 dense_oracle：是“同缓存特征上读全部远期”，还是“所有层整段全历史计算”。分开命名，前者不能代表后者。
6. 保留旧模式的复现入口，新增明确模式 `v2_last_retriever_residual`，不要静默改变旧 checkpoint 数学语义。

先支持普通 qid，覆盖 pdiff、separate_qa、kq_same、n_blocks>1。avgpool/linear 等未适配变体明确报错。

## 3. 修改位置与缓存内容

根据实际 Architecture 中的 layer role 找到最后一个 Knowledge Retriever。参考代码 blocks_2 交替执行 Question Attention / Retriever；n_blocks>1 时不要硬编码只改 index=1。

所有此前层完全沿用原块内计算，不读取 recent/remote：包括 QA 编码器、所有 Question Attention、此前 Retriever。

```python
# 概念接口，按仓库实际 API 落地
y = original_qa_encoder(current_block)
x = original_question_embedding(current_block)
for layer in blocks_2:
    if layer is last_knowledge_retriever:
        x, pending = layer.forward_with_memory(
            query=x, key=x, values=y,
            abs_pos=abs_pos, snapshot=snapshot,
        )
    else:
        x = original_layer_forward(x, y)
preds = original_prediction_head(x, q_embedding)
```

在最后 Retriever **输入处**捕获：

- key_feature：该层输入 x_i，不是最终 d_output。
- value_feature：QA 编码器最终 y_i。
- student_id、record_id、abs_pos、feature_version。

缓存 detached 的投影前特征，取到候选后用当前 W_K/W_V 重投影，使读取投影仍可训练。索引可另存 detached projected Keys。旧特征本身的 encoder 过时不会因重投影自动消失。

当前 y_i 包含 r_i，只能供当前块后续合法位置和未来块读取；当前 Retriever 保持 i<t。整块预测完成前不向跨块 bank 提交。

## 4. current 路径：一项都不能被 memory 改写

令本块合法历史 C_t，原投影得到 Q、K、V：

`s_cur = Q_t @ K_cur.T / sqrt(Dh)`

`p_cur = masked_softmax(s_cur, current_only_mask)`

`D_cur(t,i) = sqrt((pos_t-pos_i) * sum_{j in C_t, pos_j>pos_i} p_cur(t,j))`

`theta = softplus(original_gammas)`；普通模式 phi_t=1；pdiff 模式 `phi_t=exp(sigmoid(pdiff_t))`。

`z_cur = s_cur * clamp(exp(-theta * phi_t * D_cur), 1e-5, 1e5)`。

`O_cur = dropout(masked_softmax(z_cur)) @ V_cur`。

距离 detach，theta 和 scores 仍可导。复用已验证的原 attention 优于重新拼装一个近似路径。current 的分母、距离、首题处理均不得读取 memory。

本块首题没有本块响应历史时 O_cur=0，但 memory 可非空。不要沿用旧“每块首行把两路输出都清零”的逻辑。原 out_proj 的 bias 继续原样使用。

## 5. memory 路径：候选内 context-aware attention

候选 M_t 为 recent 与远期 Top-k 的并集，均为此前完成块，按 abs_pos 升序，去重，屏蔽 padding 和非法记录。每 head/query 单独选候选。

复用最后 Retriever 的当前 Q/K/V 参数，kq_same 配置沿用原模型，不新增随机 Q/K/V 作为主路径。

`s_mem = Q_t @ K_mem.T / sqrt(Dh)`

`p_mem = masked_softmax(s_mem, memory_only_mask)`

`D_mem(t,i) = sqrt((pos_t-pos_i) * sum_{j in M_t, pos_j>pos_i} p_mem(t,j))`。

距离 detached。真实 pos 保留跨度，但不恢复未召回记录的概率质量；名称使用 `candidate_context`，不能称 exact full-history distance。

已知局限必须注明：memory 最新候选的 tail_mass=0，即使其绝对年龄很大，D_mem 也为 0。只有一条候选时权重恒为 1。该机制没有保证所有旧证据的绝对幅度衰减；不要偷偷添加年龄衰减来“修复”，如以后添加应作为独立消融。

memory 只实现两种模式：

```python
if memory_decay_mode == "legacy_mul":
    z_mem = s_mem * exp(-theta * phi_t * D_mem).clamp(1e-5, 1e5)
elif memory_decay_mode == "log_additive":
    z_mem = s_mem - theta * phi_t * D_mem
O_mem = dropout(safe_masked_softmax(z_mem, valid_mem)) @ V_mem
```

首次实验使用 legacy_mul；另做仅 memory 改 additive 的消融。原 gamma 在乘法机制下训练，其数值不保证适合 additive。state_dict 可加载只证明参数结构兼容，不证明函数语义或效果兼容。

pdiff 调制指数中的惩罚，不乘 raw IP；原始 gammas 经 softplus 形成正 theta，避免符号反转。

## 6. 融合：每 head 零初始化有符号残差系数

每 head 一个新参数 alpha_h，初始化精确为 0：

`g_h = tanh(alpha_h)`

`O_h = O_cur,h + g_h * O_mem,h`。

拼接 heads 后，**只运行一次原 out_proj**，接原残差/LayerNorm/FFN/预测头。不对 memory 单独执行带 bias 的 out_proj 再相加。

tanh 可以为负，本方案称其为有符号记忆残差系数，不是 0–1 信任概率。不要再把 memory 输出层也零初始化，否则可能导致 gate 和分支双方梯度同时被截断。

第 0 步 gate 可有梯度，而 memory-only 路径被零 gate 阻断的梯度是预期。验证读取参数梯度时，先把 gate 人工置为小非零值。

零步等价条件：current 原样、前置网络无 memory、gate=0、memory 输出无 NaN/Inf、相同输入/参数/eval 模式/精度。0*NaN 仍是 NaN。

FP32 eval 使用 atol=1e-6、rtol=1e-5，报告实际最大误差，不称容差通过为逐位严格相等。训练态新增 dropout 会消耗随机数，不能只设置同一 seed 就要求全模型逐元素等价；训练态等价测试关闭 dropout 或隔离随机流。

## 7. 安全 softmax 和形状

不能仅在 softmax 全 -inf 后用 where 掩盖 NaN。采用以下结构：

```python
def safe_masked_softmax(scores, valid):
    if scores.size(-1) == 0:
        return torch.zeros_like(scores, dtype=torch.float32)
    scores = scores.float()
    has_any = valid.any(dim=-1, keepdim=True)
    logits = scores.masked_fill(~valid, float("-inf"))
    logits = torch.where(has_any, logits, torch.zeros_like(logits))
    probs = torch.softmax(logits, dim=-1)
    return probs.masked_fill(~valid, 0.0)
```

合法 scores 本身 NaN/Inf 应触发错误诊断；invalid Values 清零，避免 0*NaN。softmax/cumsum/sqrt/exp 用 FP32。位置先 int64 相减再转 FP32。

Q `[B,H,L,Dh]`，远期候选 K/V 概念形状 `[B,H,L,k,Dh]`。recent 可用共享矩阵计算，不要复制完整历史 Value 成 `[B,H,L,M,Dh]`。按 Query chunk gather 控制显存。

## 8. 缓存事务和 exact 检索

默认 L=200、R=200、M=8192、k=32，均为起始配置。recent 保存此前最近 R 条，remote 保存更早最多 M 条，二者不重叠。

每块 snapshot 固定；forward 返回 pending；训练后提交有效 detached 记录，recent 溢出迁入 remote，远期 FIFO 淘汰。R=0 时记录可直接迁入 remote。

exact 搜索：no_grad 下用当前 W_K 重投影 remote features、用 Q.detach() 扫描 Top-k；选 ID 后重新以当前可导 Q/K/V 算 attention。exact 分块扫描仍为 O(LMd)，不能宣传为快速 ANN。

按学生顺序处理块，学生间可 shuffle；同 batch 不包含同学生有依赖的两个块。epoch / split 切换清空，eval 固定参数从合法历史 replay。loss_mask 与 valid_mask 分开，warm-up 可以有效但不计损失。

每块顺序：snapshot → forward → loss/backward → optimizer.step → commit。pending 的 feature_version 标记生成时参数版本，不是提交时版本。断点首版只支持 epoch 边界，块中间恢复需完整状态或重放前缀。

## 9. 分阶段训练，不同时打开全部改动

### A：零步验证

复现 baseline，再用 gate=0、任意合法非空记忆验证 V2 等价。旧 all-layer cache 的整段等价审计可以单独做；不要要求新的双路残差结构等于全历史 AKT。

### B：gate-only 诊断

冻结全部原模型参数，只训练 alpha。backbone 保持 eval，包括 memory 的 dropout 关闭，固定表示。

**冻结参数不等于把整个 forward 包在 no_grad 中。** gate 之后的 out_proj、LayerNorm、FFN、预测头仍需保留反向计算，使 loss 回传 gate。

gate_lr 可从 1e-3 起，通过有限验证集预算选择。记录 gate、gradient、O_cur/O_mem 范数和注入比率。失败只说明现成表示加有限标量混合不够，不证明历史无效。

### C：有限联合微调

先解冻最后 Retriever 的 Q/K/V、gamma、out_proj、FFN/LayerNorm 和原预测头，保持之前网络冻结。共享参数学习率可从 baseline 微调学习率的 0.1 倍开始，gate 独立参数组；最终值由验证集选。

给 baseline 同样的可训练层范围与训练交互预算，排除继续训练本身收益。full finetune 后置，单独报告特征漂移与缓存年龄；每 epoch 清空不能消除 epoch 内漂移。

有限微调阶段若前置网络冻结且关闭其 dropout，缓存的输入特征更加稳定。必须显式控制子模块 train/eval，不能调用整体 model.train() 后无意重新打开冻结层 dropout。

## 10. ANN 和二次重排后置

先比较 exact k=32、128、小规模全部 remote。不要第一次就 oversample=4 后再压回 32，以免混入候选集合改变导致的距离变化。

如以后实现重排：以当前参数重算 pool 的 Keys/scores，使用与声明的 memory 模式一致的 theta/pdiff；明确距离近似。删选后距离可能变化，不保证最终 ranking 完全一致。测试排序实现正确，不测试“任何数据必定提升”。

ANN 使用每学生/head 独立 inner-product 索引，不无声改成 cosine/L2。快照必须含 W_K、b_K；kq_same=False 时也含 W_Q、b_Q。

- 插入使用快照 K 投影。
- 查询从投影前 x_t 使用快照 Q 投影；不能把已投影 Q 再投影，不能非共享模式用 W_K 替代 W_Q。
- 返回 ID 后用当前参数可导精算。
- 投影快照一致不等于 encoder features 版本一致；记录 index_age/feature_age。
- 保留 FIFO 删除、active ID 过滤、索引刷新、短历史 exact fallback。
- 把搬运、建图/刷新、重投影成本计入总耗时。

若 exact/dense 都差，先查模型/训练，不优先归因 HNSW。ANN 不加速就如实报告。

## 11. 文件修改职责与配置

旧 SOP 文件名须先核实，不存在则映射到实际模块：

| 模块 | 任务 |
|---|---|
| long_mem_topk_akt.py | V2 模式、最后 Retriever 注入、捕获输入、注册 alpha、保持旧 API |
| akt_candidate_attention.py | 两条完整 scores→distance→softmax 路径、安全空行、head 融合、单次 out_proj |
| akt_memory_bank.py | 最后一层缓存、事务读写、迁移、学生/split 隔离、版本 |
| akt_memory_index.py | exact 当前投影，后置 ANN 完整 Q/K+bias 快照 |
| 训练入口 | gate-only / last-layer / full 参数组与明确 train/eval 控制 |
| 评估入口 | zero-step、同位置指标、缓存 replay、记忆关闭干预 |
| tests | 等价/因果/空行反向/学生隔离/冻结与梯度 |

加载原 checkpoint 仅允许缺失新增 gate 键，其余缺失、多余、shape 不匹配报错；不要无条件 strict=False 吞掉错误。保存 resolved config、模型版本和可训练参数清单。

```yaml
architecture_variant: v2_last_retriever_residual
memory_scope: last_retriever_only
local_decay_mode: legacy_mul
memory_decay_mode: legacy_mul # 另做 log_additive 消融
memory_distance_mode: candidate_context
memory_fusion: head_tanh_zero_residual
recent_capacity: 200
remote_capacity: 8192
remote_topk: 32
retrieval_backend: exact
oversample_factor: 1
rerank_with_decay: false
local_anchored_gap: false
prior_margin: false
training_stage: gate_only
```

## 12. 必须通过的测试

### P0：正确性

- [ ] memory off 复现旧 checkpoint，相同计分位置。
- [ ] gate=0、非空 recent/remote、两种 memory decay、n_blocks>1 均 eval 等价 baseline。
- [ ] 改变记忆数量/内容，gate=0 时 current 的 scores/D_cur/O_cur 不变。
- [ ] R/M/k=0、k>M、全 padding、块首、空 memory、单候选时 forward/backward finite。
- [ ] gate 非零但 memory 空时只剩原 current 分支。
- [ ] 当前/未来 response 改动不影响当前预测；未来 question 不影响此前预测。
- [ ] 学生交错与分别运行 eval 等价；乱序、重复 commit 拒绝。
- [ ] padding/删除/未来/他人记录不参与检索 attention。

### P1：梯度与状态

- [ ] gate-only 在非退化样本下 alpha 有梯度并更新，backbone 参数完全不变。
- [ ] 没有整体 no_grad 把 gate 梯度切断。
- [ ] alpha=0 时 memory-only 梯度可为零；设非零 alpha 后读取梯度正确。
- [ ] 历史无 grad_fn，多块训练无旧图泄漏/重复 backward。
- [ ] 原 current dropout 未遗漏，冻结子模块模式符合配置。
- [ ] gate/config/checkpoint 可保存恢复，eval replay 可复现。

### P2：数学与索引

- [ ] tail_mass 上升/下降反例均成立，不写系统性低估假设。
- [ ] 候选按时间排序，abs_pos 整体平移不改变结果。
- [ ] additive 单候选权重恒为 1；只有固定其他候选 logits 时，才验证某候选随 D 增大权重下降。
- [ ] 重排按给定公式选择正确；不要求任意输入都优于无重排。
- [ ] ANN kq_same=False、bias=True 的快照投影等于手工计算。
- [ ] 同候选/同参数/同模式的聚合结果等于显式参考实现。

## 13. 实验矩阵与诊断

| ID | 配置 | 目的 |
|---|---|---|
| B0 | 原 checkpoint，不微调 | 原成绩 |
| B1 | 原 baseline，匹配微调层和预算 | 排除继续训练收益 |
| Z0 | V2 gate=0，不训练 | 零步等价 |
| G1 | gate-only，recent，无 remote，legacy memory | 近期诊断 |
| G2 | G1 + exact remote k=32 | 远期增量 |
| G3 | G2 k=128 | 候选预算 |
| G4 | G2 仅 memory 改 additive | 衰减影响 |
| F1 | 验证集所选 V2，last-layer 微调 | 有限适应 |
| F2 | F1 改 full finetune，可选 | 全局适应与漂移 |
| A1 | 已确定配置 exact→ANN | 加速与误差 |

不能用测试集选择配置。主结果至少 3 个种子，报告均值/离散程度；若 bootstrap，按学生重采样以尊重交互相关性。

对齐学生划分、有效计分位置、训练有效交互数和调参预算，同时报告 steps/wall time。首题 baseline 不计分时主对照也不计分，可另报块首子集；不要仅以 fixed_steps 宣称完全公平。

诊断指标：

- AUC/NLL/ACC，按历史长度和块内位置分层。
- gate 与 gradient、O_cur/O_mem 范数、`||g*O_mem||/(||O_cur||+eps)`。
- memory 内 attention 熵、recent/remote mass、候选年龄。
- 原始/衰减后 memory logits 分布，识别大量接近零分数的候选。
- 同一训练后 checkpoint：正常 memory、gate=0、清空 memory 的预测/BCE 差异；这是依赖性干预，不是独立训练 baseline。
- 打乱同学生历史 Key-Value 对应关系作诊断，保持时间合法，不当作部署模型。
- exact/ANN Recall@k、同缓存特征 dense memory attention 保留质量、输出差异。
- 总耗时/显存/CPU内存，以及检索、刷新、重投影、聚合的分项成本。

memory 内 softmax 总和为 1，remote mass 不是整个模型的总注意力份额；实际影响取决于 gate、Values、out_proj 和下游网络。不同衰减模式的 dense 目标不同，不能把 retention 数字跨模式直接解释为恢复原 AKT。

## 14. 失败处理与交付标准

1. Z0 不等价：停止大规模训练，先修路径隔离、mask、bias、NaN；不要调参掩盖。
2. G1 好/G2 差：分析远期候选与 Value 质量、年龄、信息冗余；不要首先增加 ANN 复杂度。
3. gate-only 无收益：不推导长期历史无效，允许受控 F1 并检查表示质量。
4. additive 恢复指标但 gate/注入范数趋零：可能仅抑制了 memory，不宣称学会更好利用历史。
5. exact 好/ANN 差：定位索引召回、过时和工程实现。
6. 无稳定增益：保留负结果，不继续盲目叠加距离外推或复杂路由。

完成标准：实际代码 diff、可运行命令、resolved configs、P0/P1/P2 测试结果、B0/Z0/训练后指标和资源报告。缺数据/checkpoint/GPU 时完成独立实现与单元测试，明确阻塞，不编造实验。

## 15. 可直接复制的 Agent 总指令

请实际修改仓库实施本指南。先阅读 AGENTS.md 和真实模型/训练/评估接口，保留旧模式复现入口，新增 v2_last_retriever_residual。

优先实现最后 Retriever 单点注入、current/memory 从初始 scores 起完全隔离、head-wise tanh 零门控、单次原 out_proj、安全空行、exact 检索、gate-only/last-layer 参数组、checkpoint 缺失键白名单、Z0 与 P0/P1 测试。

禁止 local_anchored_gap、全局 additive、额外 Logit 分支、未验证的性能下界承诺；禁止将相同缓存特征的 dense oracle 称为全层全历史 AKT；ANN 和二次重排后置。

按“审计→零步等价→单点记忆→小数据训练→逐项真实数据实验→ANN”推进。各阶段报告已改内容、通过/失败检查与下一步依据。实际命令按真实 CLI 生成，不照抄旧 SOP 中未核实的脚本或参数。不要顺带改变 baseline 的 embedding 索引、数据划分或计分位置。
