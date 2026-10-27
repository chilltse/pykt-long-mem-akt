# AKT 跨块记忆与远期检索：Coding 执行方案

版本：V1，2026-10-24。状态：已确定实现规格，尚未实现或验证性能收益。

## 0. 目标、依据与不可混淆的结论

基于用户上传的 `akt(4).py` 实现：**所有注意力层加入近期连续缓存，只有 Knowledge Retriever 加入远期稀疏检索；在联合候选上沿用 AKT 的 context-aware monotonic attention，然后继续原网络和原预测头。**

源文件 SHA256：`f84d7bd21e6843749127f7440dd8bfc632a377f5aa724cd9a3d54954b6e7c1e0`。

本方案替代先前“外部 Long Branch + 固定时间衰减 + Logit 残差”设计。不要混入旧文档的实现约束。

核心目标：保持短块训练，跨块复用历史表示，在更远历史中读取少量候选，控制活跃显存与读取成本。这里的 seq_len 是计算块长度，不再等于直接可读取的全部历史长度。

明确限制：

- 保留 AKT 层结构和距离函数形式，不保证与全历史 AKT 数值等价。
- 候选删除会改变距离分布、softmax 分母和最终输出。
- 历史 detach 后是截断训练，不是全历史反向传播。
- 全历史点积后 top-k 只能作为正确性基准，不能宣传为次线性检索。
- ANN 只近似原始点积候选，不保证找到 AKT 最终权重最大的历史。
- 增长的外部存储不等于固定资源无限记忆；V1 有明确容量上限。
- 是否提升 AUC、是否更快必须实验测量。

参考思想及边界：

1. AKT：https://arxiv.org/abs/2007.12324 。以用户代码为计算规格，尤其保留平方根距离、head-specific gamma、可选 pdiff。
2. Transformer-XL：https://arxiv.org/abs/1901.02860 。借鉴 segment recurrence / 历史表示复用；不额外移植其位置编码。
3. Memorizing Transformers：https://arxiv.org/abs/2203.08913 。借鉴历史 K/V 与近似检索；本文的联合候选 AKT attention 是适配方案，不声称为该论文原实现。

## 1. 冻结设计决策

| 项目 | V1 决策 |
|---|---|
| Embedding | 原 Rasch / qid / separate_qa 逻辑不变 |
| QA 编码器 blocks_1 | 当前块 + 此层近期缓存 |
| blocks_2 的 Question Attention | 当前块 + 此层近期缓存 |
| blocks_2 的 Knowledge Retriever | 当前块 + 此层近期缓存 + 此层远期候选 |
| 层参数 | 复用原 W_Q/W_K/W_V、gammas、out_proj；不新增长期预测头 |
| 注意力融合 | 三类位置去重后统一 softmax，不独立 softmax 后相加 |
| 衰减 | 原始 AKT logits 乘 context-aware 指数因子，随后 softmax |
| 距离 | 真实 abs_pos 差 × 候选集合累计概率，再 sqrt；距离 detach |
| 检索粒度 | 每个学生、每个 Retriever、每个 head、每个 Query |
| 检索度量 | 未归一化投影 Key 的 inner product；不偷换为 cosine |
| 候选 | 远期 Top-k 原始相似度；候选全部参与 AKT，无二次 top-k |
| 训练 | 加载 AKT checkpoint，新增记忆功能后联合微调，历史 detach |
| 写入 | 当前块预测完毕后统一提交；读取使用块起始快照 |
| 存储上限 | 每学生每 Retriever 最多 M 条远期记录，FIFO |
| 不做 | MC dropout、router、额外 gate、RL、压缩槽、全层远期 ANN |

默认配置是起点，不是已调好的超参数：

```yaml
memory_mode: retrieval    # off | recent | retrieval | dense_oracle
block_len: 200
recent_capacity: 200
remote_capacity: 8192
remote_topk: 32
retrieval_backend: exact  # 先完成 oracle，后实现 hnsw_ip
retrieval_query_chunk: 32 # 限制 gather/精确扫描的临时显存
index_refresh_blocks: 8   # 训练时每学生处理若干块刷新该学生索引
freeze_backbone: false
memory_detach: true
```

完整交付必须包括 exact 与 ANN 两个后端及成本报告；没有 ANN 时只算功能原型，不能标为低成本检索最终交付。

## 2. 代码映射与兼容范围

开始编码前阅读实际仓库、AGENTS.md、模型注册、数据集、训练/评估器和 checkpoint 格式。本文件只提供建议目录，禁止假设实际仓库一定存在相应 API。

保留 `AKT.forward(q_data, target, pid_data=None, qtest=False)` 默认行为和返回契约。建议新增 `forward_with_memory(..., memory_snapshot, abs_pos, valid_mask)` 返回预测、正则及待提交记录；旧入口显式走原路径。

新增模块可以包含：

- `akt_memory.py`：记忆版 Architecture / 模型入口，复用旧参数命名。
- `akt_memory_bank.py`：学生隔离、快照、近期/远期迁移、生命周期。
- `akt_candidate_attention.py`：矩形 / 每 Query 候选 attention。
- `akt_memory_index.py`：ExactIPIndex 与 HNSWIPIndex。
- `student_block_iterator.py`：按学生内部时间顺序组织样本。
- 对应训练脚本、测试与 profiling 脚本。

普通 `qid` 和 `qid` + `pdiff`、`separate_qa` 开/关、`kq_same` 开/关必须覆盖。`avgpool` / `linear` 注意力变体不在本次范围：memory 模式下明确报不支持，不能静默改变其行为。

不要顺带修复/重解释 Rasch embedding 索引、reset 或其他既有模型问题。若发现异常，单独记录，避免 baseline 同时变化。

## 3. 每层缓存必须取对位置

对一次注意力调用，把输入记为 `u_q, u_k, u_v`；缓存该次调用的 **投影前 key/value 输入**，另保存用于搜索的投影 Key。

| 层 | key_feature | value_feature |
|---|---|---|
| QA 编码层 j | 进入该层前的 y | 同一个 y |
| Question Attention j | 进入该层前的 x | 同一个 x |
| Knowledge Retriever j | 进入该 Retriever 前的 x | QA 编码器最终输出 y |

必须在每层输入处捕获，不得统一用最终 d_output。当 n_blocks > 1 时，各 Retriever 的 x 不相同，缓存不能共用。Retriever 的 y 可以通过受控共享存储减少重复，但第一版先保证逻辑清楚。

近期和远期都保存 detached features。读取候选后用**当前层当前参数**重新计算 K/V；索引 Key 只负责候选检索。这使 W_K/W_V 在历史读取时仍可训练，也减少投影过时；它不能消除旧特征本身的编码器过时。

```python
LayerMemoryRecord = {
    "record_id": int,        # 学生内稳定 ID，可用 abs_pos
    "abs_pos": int,
    "key_feature": Tensor[D],
    "value_feature": Tensor[D],
    "feature_version": int,  # 生成特征时的 optimizer step
}
```

索引附加信息：`indexed_key[H, Dh]`、`index_version`，其 ID 映射必须与 record_id 一致。

不保存历史计算图；不得只保存旧投影 K/V 然后声称历史路径仍训练了当前 W_K/W_V。

## 4. 时间、掩码与迁移协议

每个 batch 至少包含 student_id、block_id、q_data、target、可选 pid_data、valid_mask、abs_pos、当前 loss_mask。Padding 不写记忆，也不参与 attention、距离、损失。

每个学生块内 abs_pos 严格递增；跨块也严格递增。时间单位是交互索引，不是秒，V1 不引入真实时间衰减。

块起始时：

- 每层 recent 保存该学生此前最近 R 个有效交互。
- 每个 Retriever 的 remote 保存比 recent 更早的最多 M 条记录。
- 两者不能重叠。

整块完成后，把本块有效记录追加 recent；溢出的旧记录按时间迁入对应 Retriever 的 remote，其他层溢出记录直接丢弃。remote 超过 M 后删除最老记录，同时更新索引有效 ID。

如果 L > R，本块较早记录可在块提交时直接进入 remote；整个 forward 期间依然不可被同块远期检索读取。

因果规则：

- QA / Question self attention：key_pos <= query_pos。
- Retriever：key_pos < query_pos。
- 远期和近期只能来自此前完成块；当前块内部按原 mask 可见。
- 不同学生严格隔离；同 batch 不放同学生两个互相依赖块。
- 不能在处理后面的 Query 时修改前面 Query 使用的候选池。

当前 Response 可以进入该位置的 QA 编码，但不能进入该位置的 Retriever Value 读取。对多层模型必须实测未来响应泄漏，不能只检查最后一层 mask。

## 5. 联合候选与 context-aware attention

### 5.1 原始投影

沿用当前层投影和 head 划分：

`Q = q_linear(u_q)`，若 kq_same 则用 k_linear；`K = k_linear(u_k)`；`V = v_linear(u_v)`。

索引的粗检索距离使用 Q·K，精排使用 Q·K/sqrt(Dh)。同 head 的常数缩放不改变 Top-k 排名。不对 Q/K 做 L2 normalize。

### 5.2 候选构造

对每个 query/head：

1. 取 current 的所有合法位置。
2. 取 recent 的所有合法位置。
3. Retriever 才检索 remote_topk 个合法 record_id。
4. 按 record_id 去重，按 abs_pos 升序排列。
5. 从特征库 gather remote 特征，以当前 W_K/W_V 投影选中记录。
6. 得到每 Query/head 的 K/V、candidate_abs_pos 和 candidate_valid。

ANN 搜索输入为 Q.detach()；搜索 ID 不可导。取到候选之后重新计算的 s 使用有梯度的 Q、当前投影参数。不得把最终 attention 全包进 no_grad。

概念形状：Q `[B,H,L,Dh]`，候选 K/V `[B,H,L,C,Dh]`，位置/有效位 `[B,H,L,C]`。实际不应完整复制所有 local K/V 到每 Query；本地用矩阵乘法，remote 分 query chunk gather。完整复制会抵消显存收益。

### 5.3 原公式在候选集合上的计算

对合法候选 i，令 `s_i = Q dot K_i / sqrt(Dh)`。

```python
with torch.no_grad():
    p = safe_masked_softmax(s.float(), valid)
    # 候选已按原时间升序；sum - cumsum 排除 i 本身
    tail_mass = (p.sum(-1, keepdim=True) - p.cumsum(-1)).clamp_min(0)
    gap = (query_abs_pos[..., None] - candidate_abs_pos).clamp_min(0)
    distance = (tail_mass * gap.float()).clamp_min(0).sqrt()

theta = F.softplus(gammas)  # 正数，每个 head 独立
factor = exp(-theta * distance)
# pdiff 路径：exponent 再乘 exp(sigmoid(current_query_pdiff))
factor = factor.clamp(1e-5, 1e5)
adjusted = s.float() * factor
weights = safe_masked_softmax(adjusted, valid)
weights = dropout(weights)
output = weighted_sum(weights, V)
```

pdiff 是当前 Query 的标量，广播到候选维；不得误用候选历史的 pdiff 替代现有语义。指数和概率相关运算用 FP32，最后转换输出 dtype。

原代码在无历史首行将 attention 置零。新版删除“每个 Block 第一行无条件清零”，替换为所有合法候选为空时 weights/output 为零。out_proj 若有 bias 仍沿用原参数，不额外强制整个 Transformer 输出为零。

safe_masked_softmax 对全 invalid 行返回全零，不对全 -inf 行直接 softmax 后保留 NaN；invalid logits 填充要在 FP32 操作。

### 5.4 必须写入实现说明的近似

候选集合的 p 不等于全历史 p。真实位置能保留跨度，却不能补回被删除历史的概率质量。V1 不添加“乘一个 gap 就恢复完整 context”的虚假修正。

AKT 衰减乘在有符号 logits 上：负分数可被推近零。因此原始点积 Top-k 并非最终 attention 的严格 Top-k。不能以候选相似度高为理由断言召回充分。

## 6. 检索后端的明确实现规范

### 6.1 ExactIPIndex：必须先实现的正确性基准

对 remote 的 detached key features，用当前 W_K 构造精确搜索 Key，在 no_grad 中分 Query/Memory chunk 扫描并合并 Top-k。索引刷新语义不影响此后端：搜索使用当前投影参数。

选中后再用可导路径重新投影 K/V 并计算 attention。时间依然 O(LMd)，分块只降低峰值显存，不降低扫描复杂度。

### 6.2 HNSWIPIndex：低成本候选检索版本

使用支持 inner-product 的 HNSW 实现，例如 Faiss IndexHNSWFlat + METRIC_INNER_PRODUCT；编码时核对安装版本的官方接口。按 `(student_id, retriever_layer_id, head_id)` 隔离索引，可用 wrapper 管理 ID 映射。默认建图连接数 16，efConstruction 100，efSearch=max(64, 2*k)，均可配置并计入实验。

- 检索的是未归一化 K；不能默认使用 L2/cosine 索引。
- 批量查该块 Queries，不要逐 token 执行 Python/CPU 同步。
- 学生历史很短时允许退回 exact，阈值通过 benchmark 决定。
- CPU 索引需要记录 CPU↔GPU 搬运和检索耗时。
- FIFO 删除采用 active ID 标记；过量召回后过滤已失效 ID，不足时扩大搜索或 exact fallback。
- tombstone 超过 20% 时重建；索引容量限制和重建成本不能隐瞒。
- 若索引返回 ID 不存在/不属于该学生/位置不合法，报错或安全拒绝，不读入。

训练期间每个学生每 8 个已处理块、或失效记录过多时，使用当前 W_K 重新投影 active remote features 并重建索引；新迁入记录使用当时参数插入。保存 index_version 和候选重新打分差异诊断。该策略只控制过时程度，不保证与 exact 同召回。

eval 固定模型参数，从空缓存顺序 replay，天然避免跨 optimizer step 的投影漂移；不要沿用训练索引。

如果大量学生/head 的小索引开销大于收益，报告结果并保留 exact，不强行宣称 ANN 加速。不要擅自改成跨学生共享私有学习历史的全局索引。

## 7. Forward 与训练事务

MemoryBank 在模型外，由 runner 按学生管理；forward 只读快照并返回 pending_records，避免 autograd 中途原地改历史。

```python
snapshot = bank.snapshot(student_ids)
preds, reg, pending = model.forward_with_memory(
    q_data, target, pid_data, abs_pos, valid_mask, snapshot
)
loss = masked_bce(preds, target, loss_mask) + reg
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), configured_clip)
optimizer.step()
bank.commit(pending_detached_features=pending, student_ids=student_ids)
```

pending features 是本次 forward（optimizer.step 之前）生成的，标记正确版本；commit 后索引可使用 step 后的 W_K 建立搜索 Key。不要误标为“已用新编码器重编码”的特征。

训练首版：加载已训练原 AKT checkpoint，全部原模型参数允许联合微调；使用相同初始化 checkpoint 比较 recent / retrieval。可用 baseline 学习率的 0.1 倍作为微调初值，报告实际选择与验证集依据。不强制零初始化 out_proj、不新增 gate。

历史 detach 不影响当前块内部的正常梯度。当前块损失通过 Query、当前投影层、本块合法 Values 回传；不回传到历史 features 的生成图。

数据流约束：

- 学生之间可以 shuffle，每个学生内部 block 顺序不可 shuffle。
- batch slot 换学生必须按 student_id 取新 bank，不可继承 slot 的旧状态。
- 建议先单卡 batch=1 验证，再做学生间 batch 并行。
- 多卡必须将同学生路由至固定 rank，或明确迁移全部状态；V1 默认单卡。
- 每个 epoch 从空 bank 开始；切换 train/val/test 清空。
- 测试允许合法前缀 warm-up 和逐块真实响应写入，但必须与 baseline 使用同一评估协议。
- 同学生时间切分，不能把未来测试块预填缓存。
- loss_mask 与 valid_mask 分开：warm-up 可以有效但不计损失，padding 两者均无效。
- 断点首版只保证 epoch 边界恢复；块中间恢复必须重放前缀或保存完整 iterator/bank/version，不得默默丢失历史后继续。

## 8. 分阶段 coding 手顺

### M0：冻结 baseline

- [ ] 记录源文件 hash、运行配置、checkpoint 和样本批次。
- [ ] 记录普通 qid、pdiff、separate_qa、kq_same 的实际支持情况。
- [ ] 旧 forward 不修改数学行为，memory=off 可加载旧 checkpoint。

### M1：泛化 attention，但不接历史

- [ ] 支持独立 query/key 长度和 abs_pos mask。
- [ ] 将原始方形 arange 距离改成绝对 query/key 位置差。
- [ ] 将首行清零改为全无合法历史行清零。
- [ ] 在无缓存、无 padding、dropout=0 时对照原 attention 的输出与梯度。
- [ ] 引入 padding 安全处理，并说明 baseline 原实现没有有效 padding mask。

### M2：所有层近期缓存

- [ ] 正确捕获每层 key/value 输入；只保存 detached 有效位置。
- [ ] 加入 R 个跨块近期记录，暂不检索远期。
- [ ] 实现快照/commit/迁移、学生排序迭代器。
- [ ] 用足够大的 R 保留全部历史，在固定参数 eval 下验证切块与整段前向等价。

### M3：Retriever exact 远期检索

- [ ] 实现 remote 特征库和 ExactIPIndex。
- [ ] 实现每 Query/head 的远期候选、时间排序、联合 attention。
- [ ] 不新增参数预测头，不添加 post-softmax exp 衰减。
- [ ] 测试 k 覆盖全部 remote 时等于“同一缓存特征上 dense Retriever”。
- [ ] 注意：该测试不能称为整个模型等于 full-history AKT，前置编码层仍受 R 限制。

### M4：ANN 后端

- [ ] 实现 HNSW inner-product 查询、stable ID、删除过滤与刷新。
- [ ] 比较 ANN 与 exact 原始 Top-k 的 Recall@k。
- [ ] 对错误 ID、空索引、历史不足、混合学生 batch 实施边界测试。
- [ ] 实测索引维护和数据搬运开销，保留 exact fallback。

### M5：微调与评估接入

- [ ] 学生内顺序训练、跨 epoch 清空、split 隔离。
- [ ] 验证当前读取路径梯度和历史 detach。
- [ ] 小数据 overfit/smoke run 后，完成公平对照与成本测量。
- [ ] 输出运行命令、完整配置、指标 JSON/CSV、已知限制。

## 9. 必须通过的验收测试

### 9.1 数值与结构

1. memory=off 的旧 API 输出与旧模型一致，checkpoint 参数不丢失。若使用新统一实现，FP32 初始容差 atol=1e-6、rtol=1e-5；不同设备可记录并解释合理误差，不得大幅放宽掩盖逻辑错误。
2. 无缓存候选全覆盖时，新 attention 与旧 attention 一致；dropout=0 对照输入/参数梯度。
3. 固定权重 eval，所有层 recent 容量覆盖完整前缀，切块结果与整段结果一致，包含 n_blocks>1。
4. remote_topk 覆盖全部 remote，候选实现与同缓存特征的 dense Retriever 一致。
5. candidate 存储顺序随机打乱，按 abs_pos 排序后结果一致。
6. 所有 abs_pos 同加一个常数，结果不变；不能使用每块重新从零计数。

### 9.2 因果与状态

7. 改变 r_t 或未来响应，不改变时刻 t 的预测；改变过去已观察响应允许影响 t。
8. 改变当前块未来 Question ID，不改变此前预测。
9. 块首已有历史时能读取，不能因 zero_pad 被无条件清零。
10. 两个学生分别运行与交错 batch 运行 eval 结果一致；互换/清除另一学生记忆不影响当前学生。
11. 乱序 block、重复提交、record_id 冲突、pos 倒退必须报错。
12. recent/remote 无重复；FIFO 迁移和删除后的索引不能返回已删除记录用于 attention。

### 9.3 数值、梯度和资源

13. M=0、R=0、k=0、M<k、全 padding、最后短块、某 head 无候选均无 NaN。
14. FP16/BF16 下初始/最终 softmax、sqrt、exp 安全；大位置差先在整数/FP32 运算。
15. 历史 features 无 grad_fn；有历史且非退化样本时当前 Q/K/V 参数收到有限梯度。gammas 用有非零距离的多候选样本测试。
16. 连续多个块训练无重复 backward / graph retained / inplace 错误。
17. bank 达容量后 CPU/GPU 占用不再随块数无限增长（索引 tombstone 也受控）。
18. profiler 不应出现无意构造 `[B,H,L,M,Dh]` 的全历史 Value 广播张量。

## 10. 实验矩阵和失败判定

| 编号 | 配置 | 要回答的问题 |
|---|---|---|
| E0 | 原 AKT，L=200，无记忆 | 原始基线 |
| E1 | 原 AKT，更长输入，如 400/800，资源允许时 | 直接增大窗口是否足够 |
| E2 | 所有层 recent，L=200、R=200，无 remote | 连续缓存本身的收益 |
| E3 | E2 + Retriever exact remote，k=32/64 | 稀疏远期读取的增量 |
| E4 | E3 的 ANN 替代 | 真正检索加速的收益/损失 |
| E5 | 小规模 dense oracle | 候选近似误差诊断 |

E0/E1 使用相同学生/交互划分，明确每位置可见历史；不能因块长不同改变计分样本。E2/E3/E4 相同 checkpoint、训练交互预算和调参预算；同时报告固定训练步数和实际 wall time，至少主结果用 3 个种子。

指标：AUC、NLL/BCE、ACC；按历史长度及距上次相关 KC 练习间隔分层。只在有 KC 定义时使用 KC 间隔，不凭题目 embedding 臆造标签。

诊断至少包括：

- ANN 原始相似度 Recall@k，相对当前投影 exact 结果。
- 同一层、同一缓存特征上，候选保留的 dense AKT 最终 attention mass（dropout 前）。
- 候选与 dense 输出 L2 差异；当前 Query 最终预测差异。
- 远期 attention mass、远期候选年龄分布、读取命中比例。
- index_age / feature_age、刷新次数、fallback 比例。
- 每块总耗时，检索/重投影/attention/索引更新分别耗时；GPU 峰值 allocated/reserved、CPU RSS、bank/index bytes。

dense oracle 的两种含义分别标注：A）相同局部缓存表示下的全远期读取；B）所有层整段全历史计算。A 测候选误差，B 还包含编码上下文差异，不能混为一种。

失败判定：

- E3 没有稳定优于 E2：远期检索价值未证实，先分析候选质量，不直接增加 gate/router。
- E4 总耗时不低于 E3：ANN 在该规模无工程收益，如实保留 exact 或扩大数据规模再测。
- 候选最终 attention mass 很低：不要把 ANN recall 高误当作 AKT 召回好。
- 索引过时明显损害结果：缩短刷新间隔/降低微调学习率并报告额外成本，必要时固定编码器做诊断。

## 11. 后续方向（禁止偷偷加入 V1）

V1 候选距离有偏差，后续可研究按时间区间的历史摘要，用于估计遗漏的 softmax 质量；这涉及新的估计器和验证，不能用简单平均 Key 宣称精确恢复。也可研究全层远期检索或更适合 AKT 最终分数的候选策略。先完成上述矩阵再扩展。

## 12. 可直接复制给代码模型的执行 Prompt

请基于本仓库实际 AKT 实现和用户提供的 akt(4).py，完成本文的 AKT 跨块近期缓存 + Retriever 远期检索方案。先阅读仓库约束和实际训练接口，再按 M0→M5 实现，不只输出示例伪代码。

必须遵守：

1. 保留原 Rasch embeddings、各层 Q/K/V 与 gamma、残差、LayerNorm、FFN、原预测头及旧 forward API；不得添加 Logit 长期分支。
2. 所有注意力层缓存此前 R 条有效记录；只有 Retriever 保存/检索更远历史。按学生/层/head 隔离检索。
3. 缓存各层投影前 key/value 输入并 detach，不使用最终 d_output 代替各层输入；搜索 Key 与特征版本明确记录。候选选定后用当前参数重投影以保留读取梯度。
4. 三类合法位置合并去重，按 abs_pos 排序，在联合集合上使用原 AKT sqrt(tail_mass*gap) 和 logits 乘指数因子的公式。支持原 pdiff。承认候选集合近似，不改成固定时间衰减。
5. 修复跨块矩形 mask、块首无条件清零、Padding 和全无效行；保留当前 Response 不可被当前 Retriever 读取的约束。
6. forward 只读块起始快照并返回待提交记录。按学生内部顺序训练，预测后提交，recent 溢出再进入 remote；epoch/split 重置。保持历史无计算图。
7. 先实现 ExactIPIndex oracle，再实现 HNSW inner-product 后端与刷新/失效 ID 处理；必须计入索引维护和搬运成本，不能用 dense top-k 冒充快速索引。
8. 先跑数值/因果/缓存等价测试，再进行小数据训练和 E0–E5 对照。没有完成实验就标记未运行，不编造指标或性能承诺。
9. 所有新模块/运行命令根据实际仓库落地。新增外部索引依赖按环境支持选择并记录版本，未安装时给出清晰错误和 exact fallback。
10. 最终交付实际改动清单、运行命令、配置、通过/未通过的测试、指标与成本报告、明确的近似边界。不要顺带改 baseline embedding 语义或数据划分。

交付完成标准：旧 AKT 可复现，近期缓存等价性测试通过，远期检索与 ANN 生命周期正确，训练/评估无泄漏，实验和成本可复现；性能提升不是实现验收的预设事实。
