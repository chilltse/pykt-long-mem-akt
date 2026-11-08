# AKT_FMKC 增量推理与学生状态持久化：修改手顺

日期：2026-11-02

依据：本次消息中的 `AKTFMKC` 和附件中的 `AKT / Architecture / TransformerLayer / MultiHeadAttention / attention`。这是针对所给代码的实现计划，不是已经完成或通过测试的补丁。未提供训练 runner、数据分块器和现有缓存代码，它们的实际行为仍需核对。

## 1. 先确定要实现什么

**目标：保存每个学生各层需要的历史表示；新事件到来时只计算新位置，并与当前模型在完整历史上的因果前向结果一致。**

不改 FMKC 的表达方式、不添加新训练参数、不改预测头；给现有模型增加增量执行路径与可持久化状态。

这里“完整历史前向”是：输入该学生截至当前的全部有效事件，仍使用原来的 `seq_len=L` 局部注意力 mask。它不是取消 L 限制的全局 Attention，也不是把输入先裁剪成最近 L 条。

你的直觉正确的部分：旧表示中已经编码了更早的上下文，应当复用，而不是每次把最近 L 条当作一段全新的序列重算。

需要精确化的部分：应保存各层参与下一步计算的 K/V，不是只保存最后的 `output` 或 `features`。这些 K/V 就是工程上需要的学生状态。

## 2. 你这份代码目前到底在算什么

### 2.1 FMKC 只负责构造两个输入流

对于第 t 个事件，令 `u_t` 为各字段 embedding 的拼接：

```python
u = cat(field_embeddings[i](attrs[..., i]))
c = base_head(u)
g = response_embedding(response)
```

三种 drift 的具体语义如下：

| drift | question 输入 | interaction 输入 |
| --- | --- | --- |
| none | c | c + g |
| shared | c + μ × question_drift_head(u) | question + g |
| response | c + μ × question_drift_head(u) | c + g + μ × wrong/correct_drift_head(u) |

`pid=0` 时 μ 必须保持为零，包括加载 checkpoint 后。保留现有显式乘 `(pid != 0)` 的处理。`response` 模式的 interaction 不应擅自改成 `question + g + ...`，那会改变现有公式。

虽然名字包含 FMKC，这份代码可见的输入主干是字段 embedding 拼接加线性头；不要在本次缓存改造中自行增加二阶 FM 项。

### 2.2 Backbone 有三类需要缓存的 Attention

初始 `x=question`、`y=interaction`。

| 模块 | 代码位置 | Q / K 来源 | V 来源 | 当前位置可见性 |
| --- | --- | --- | --- | --- |
| 作答编码器 | `blocks_1[k]` | 该层输入 y | 该层输入 y | 可看当前真实作答 |
| 题目侧自注意力 | `blocks_2[2*k]` | 该层输入 x | 该层输入 x | 可看当前题目 |
| 知识检索 | `blocks_2[2*k+1]` | 前一子层输出 x | 所有 blocks_1 完成后的 y | 只看过去 |

后续题目侧层的 x 已可能融合更早的作答信息，不能简单认为每一层 x 都只含题目。

最终：

```python
output = backbone(question, interaction)
features = cat([output, question], dim=-1)
pred = sigmoid(out(features))
```

`qtest=True` 返回的是拼接后的 `features`，不是完整的可递推状态。预测位置 t 的表示不会看到 `response_t`，而作答编码器在位置 t 的 y 会看到它。

### 2.3 你的 seq_len 是逐层局部 mask

当前代码中的限制是：

```python
distance = query_position - key_position
# mask=1: 0 <= distance < L
# mask=0: 1 <= distance <= L
```

因此：

- 自注意力读取“当前 + 前 L-1 个位置”。
- 知识检索读取“前 L 个位置”，不含当前。
- 即使每层只看 L 个位置，高层读取的旧向量也已经包含其低层可见的更早上下文。
- 只保留最近 L 个原始输入再从头计算，会改变窗口左边界附近的表示，影响后续层。

**但这不等于无限长期记忆。** 固定层数加有限局部窗口的依赖范围仍有限。逐层缓存保留的是现有计算图允许的信息，不会把普通局部 AKT 自动变成能记住全部人生历史的递归 memory 模型。

## 3. 修改范围与验收基准

建议按实际仓库位置新增或修改以下文件；下面的路径是建议路径，不表示已确认仓库具有这些文件。

| 文件 | 修改内容 |
| --- | --- |
| `pykt/models/akt_fmkc.py` | 拆出输入编码；增加预测/提交真实事件接口 |
| `pykt/models/akt.py` | 增加逐步执行入口，复用原投影、LayerNorm、FFN、输出层 |
| `pykt/models/akt_cache.py` | cache 数据结构、位置、裁剪、设备迁移、序列化 |
| `tests/test_akt_fmkc_incremental.py` | 与原 forward 的逐位置一致性测试 |
| 服务层状态存储模块 | 学生隔离、版本、事件游标、原子提交和恢复 |

保留原 `forward()` 作为训练入口及正确性 oracle。第一阶段不重写旧方形 Attention；增加单 query 的独立实现，避免同时修改基准与被测实现。

所有一致性比较先在 `eval()`、FP32、无梯度、相同权重下进行。保持原参数名称与形状，旧 checkpoint 应继续 `strict=True` 加载成功。cache 不作为可学习参数，也不放进模型全局可变属性中。

## 4. 手顺一：拆出事件编码，不改变公式

建议抽取：

```python
encode_question(attrs, pid) -> question, context
encode_interaction(context, response) -> interaction
```

`context` 可以包含 u、c、μ 等该事件的临时中间量。抽取后，旧 `forward()` 仍调用同一套编码函数，避免训练与推理两套逻辑漂移。

修改要求：

1. 三种 drift 的结果保持一致。
2. 保留 PAD/UNK/MISSING 和 pid=0 的原定义。
3. 显式检查字段 ID、pid 范围；不要把越界 pid 默默映射成随机合法题目。
4. 候选题接口不接收真实 response；真实事件提交接口要求 response 为 0 或 1。
5. 第一版增量接口只处理一个学生的一条有效事件，不接收 padding。

特别注意：现有 forward 用 `target == -1` 判断 padding，并把对应 attrs 清零。所以不能用 `target=-1` 表示“待预测题答案未知”。

测试旧 forward 的候选预测时，可以在临时分支给最后一个目标位置填 0，再换成 1，确认目标位置预测不变。该占位答案绝不能提交进真实缓存。正式接口应拆开 question 与 interaction，避免依赖假答案。

## 5. 手顺二：定义每个学生的缓存

建议缓存投影后的 K/V，形状 `[1, heads, retained_time, head_dim]`。缓存属于学生状态对象，不属于共享模型实例。

```python
StudentCache:
    model_version
    feature_version
    cache_schema_version
    seq_len
    next_position       # 下一个真实事件的位置，零起始
    history_revision
    last_event_cursor
    response_layers[]   # len = n_blocks
    question_layers[]   # len = n_blocks
    retrieval_layers[]  # len = n_blocks

LayerCache:
    key
    value
    positions           # 与 K/V 时间轴严格对齐
```

### 5.1 各缓存的来源

| 缓存 | K 保存什么 | V 保存什么 |
| --- | --- | --- |
| response_layers[k] | `blocks_1[k]` 输入 y 的 K 投影 | 同一输入 y 的 V 投影 |
| question_layers[k] | `blocks_2[2*k]` 输入 x 的 K 投影 | 同一输入 x 的 V 投影 |
| retrieval_layers[k] | `blocks_2[2*k+1]` 输入 x 的 K 投影 | 最终作答编码 y 的、该检索层自己的 V 投影 |

这里缓存的是该层 Attention 的输入投影，不是随意取这一层的输出。检索层的 K 与 V 来自不同流，但必须对应同一个事件位置。

虽然各检索层用同一个最终 y，不同层有不同 `v_linear`，不能直接共用投影后的 V。第一版允许重复存储，后续再优化。

### 5.2 L 有限时不必永远保存所有时间步

实现最简单的统一策略：每个缓存保存最近最多 L 个已完成事件。

- 自注意力使用时：从历史取最近 L-1 个，再拼当前项。
- 检索使用时：读取最近 L 个已完成事件。
- 提交当前事件后：每个缓存统一裁剪到最近 L 个。
- L=1 时，自注意力没有过去项，检索仍读取上一条。
- L=None 时不裁剪；存储与单步计算随历史增长。

已有各层向量保留了原来计算时的上下文，所以按上述 mask 裁剪 K/V 不等于重新编码最近 L 条输入。绝不能在裁剪后重新计算所有保留的旧向量。

## 6. 手顺三：实现单 query 的 AKT Attention

当前 `attention()` 假设 Q 和 K 长度相同，使用一个 `seqlen` 构造方形位置矩阵。增量计算中 Q 长度为 1，K 长度为历史保留数，因此不能直接复用其形状逻辑。

建议新增：

```python
attention_incremental(q, k, v, query_pos, key_positions,
                      allowed, gamma, dropout)
```

保持原运算顺序：

```python
scores = (q @ k.transpose(-2, -1)) / sqrt(d_k)
# 先应用 mask，计算用于距离估计的初始注意力
base = softmax(masked_scores, dim=-1) * allowed
cum = cumsum(base, dim=-1)
total = sum(base, dim=-1, keepdim=True)
distance = abs(query_pos - key_positions)
dist = sqrt(clamp((total - cum) * distance, min=0))
effect = clamp(exp(-softplus(gamma) * dist), 1e-5, 1e5)
weights = softmax(mask(scores * effect), dim=-1)
head_output = weights @ v
```

这只是公式伪代码；实际实现须对齐维度广播、dtype、无有效 key 行和 detach/no_grad。K/V 必须按时间从旧到新排列，原距离公式的 cumsum 依赖这个顺序。

你的 FMKC 固定 `emb_type=qid`，传给 backbone 的 `pid_embed_data` 是 None。第一版只支持这个路径，不额外实现 `pdiff`、avgpool 或 linear 变体。题目漂移已由 FMKC 编码计算，不能重复加入 Attention。

### 三个必修陷阱

**A. 不能继续使用“把 query 第 0 行清零”的逻辑。**

原 `zero_pad=True` 把完整序列首行的权重清零，因为第一个事件没有历史。增量每次只有一行，如果照搬，就会把每个事件都清零。改成：当该 query 没有有效历史 key 时，attention 聚合向量为零。

**B. 无历史不代表整个 TransformerLayer 输出为零。**

先生成全零的多头聚合向量，再经过原 `out_proj`；其 bias 可能非零。随后保留原残差、LayerNorm、FFN。不能直接把最终子层输出写成零。空 key 集合不要做 softmax，单独走这条分支。

**C. 使用真实相对位置。**

不能让当前 query 每次位置为 0、而缓存 key 编号为 0..L-1。保存绝对事件位置，再计算差值。附件虽定义了位置 embedding 类，但所给 Architecture.forward 没有使用它们；本次不要自行启用。

## 7. 手顺四：让 TransformerLayer 支持新位置

新增执行方法，复用现有参数：

```python
layer.forward_step(query_input, key_input, value_input,
                   old_cache, query_pos, mask_kind,
                   apply_pos=True)
```

内部步骤：投影当前 Q/K/V → 按 mask 组合旧 K/V 与当前项 → 调用单 query Attention → out_proj → 原残差/LayerNorm → 按 apply_pos 决定是否运行 FFN。

保留 `kq_same=True` 时 Q 使用 `k_linear` 的行为。题目侧偶数层 `apply_pos=False`，并不代表取消位置编码，而是在这份实现中跳过 FFN；不要误读参数名。

检索层需要特别接口或分支：预测时只使用历史 K/V，不需要当前 interaction；可暂存当前 K，等待真实作答编码出来后再补当前 V 并一起提交。

不要在读预测的过程中原地修改旧 cache 张量。第一版使用返回增量 delta 的方式最容易审计。

## 8. 手顺五：按两个阶段执行当前事件

### 8.1 predict：只读学生缓存，预测候选题

```python
predict_next(cache, attrs, pid) -> probability
```

位置 `t = cache.next_position`。

1. 计算当前 question；x 初始化为 question。
2. 对每组 k：先做 `blocks_2[2*k]` 的 self attention，读取该层过去缓存并包含当前题目；再做 `blocks_2[2*k+1]` 的 retrieval，只读取历史 K/V。
3. 计算 `sigmoid(out(cat([x, question])))`。
4. 丢弃该候选题的临时缓存，不推进 next_position，不写历史。

多个候选题共享同一个历史快照和位置 t，分别形成独立分支；不能把候选 A、B、C 当作连续发生的事件。第一版循环预测即可，正确后再批量化。

### 8.2 observe：真实作答到达后提交新状态

```python
observe(cache, attrs, pid, response, event_id)
    -> new_cache, pre_response_probability
```

建议第一版采用简单、安全的完整单步计算：

1. 从同一旧 cache 计算当前 question 与真实 interaction。
2. 顺序执行全部 `blocks_1`：每层读取过去缓存并包含当前 interaction；记录各层新增 K/V，得到最终 `y_t`。
3. 顺序执行全部 `blocks_2`：题目 self 层包含当前；retrieval 层只读旧历史，不能读取当前 `y_t`。
4. 对每个 retrieval 层，记录当前 x 的 K 投影和最终 `y_t` 的该层 V 投影，作为未来事件可读取的历史。
5. 原子提交三类缓存的所有增量，裁剪长度，next_position 加一，更新事件游标。

步骤 2 可以先算出 `y_t`，但步骤 3 的 retrieval 仍必须排除当前位置。计算先后顺序不能替代 mask。

这里返回的概率是“观察当前 response 之前的预测”，即使函数同时收到了 response，也必须与 predict_next 一致。真实 response 的影响应在下一题及以后的预测中体现。

第一版允许 observe 重复计算刚才 predict 已算过的当前题目侧路径，避免 pending state 复杂度。后续再缓存临时 delta，并校验模型版本、历史 revision、候选身份完全一致后复用。

## 9. 手顺六：初始化、恢复与存储

### 初次上线

从学生最早有效事件开始，按真实顺序逐条 observe，生成初始 cache。L 有限时缓存始终有界，重放过程不需要在 GPU 上构造完整 T×T 矩阵。可以分页读取原始事件，但不能在每页之间重置状态。

缓存丢失时同样重放，或从有效的历史状态快照继续。不要仅加载最近 L 个原始事件，声称重建出与完整重放相同的状态。

### 持久化内容

保存每层 K/V、位置、next_position，以及模型权重版本、特征/词表版本、cache schema、seq_len、事件游标和 history_revision。推理缓存 detach 后序列化，恢复时放回当前设备和约定精度。第一版不做 FP16 压缩。

### 服务一致性

- 共享模型只存参数；每个学生持有独立 cache。
- 同一学生的更新串行化，或通过 history_revision 做 CAS 原子提交。
- event_id 幂等；重复消息不能重复 observe。
- 历史事件改错、补插或顺序改变：从变更点之前的有效快照重放；不能只改最后一层 cache。
- 权重、词表、特征规则、L 变更：旧 cache 拒绝直接使用，重新构建。
- 原始事件先可靠落库，再更新缓存；缓存带游标，失败后能追赶重放。

## 10. 手顺七：核对训练分块语义

`student_block_input=True`、`loss_mask` 和“replaying cached blocks”的注释，说明训练可能已使用学生分块/历史重放。但仅凭当前两个文件，不能确定 runner 实际保存了什么，也不能确定重放范围是否足够。

必须检查：

1. 每块是否直接截取最近 L 条原始事件并重新编码？
2. 是否传递各层历史表示？是否 detach？
3. 是否有足够的原始上下文作为重放前缀？
4. 块边界是否重置局部位置或学生状态？
5. 权重每个 optimizer step 更新后，是否继续使用旧权重产生的缓存？

纯推理缓存等价于完整前缀局部 Attention，不自动证明它等价于当前 runner 的分块近似。如果 runner 只重算很短的原始前缀，必须测量块边界差异，并按计划采用的在线语义重新评估现有 checkpoint。

本次不把推理 cache 直接用于训练。训练时 dropout、反向传播和参数更新使缓存复用另有约束。若只是实现固定权重的等价推理，不必重新训练；若改变实际上下文策略，应先评估，必要时再微调，而不是先断言一定需要或不需要重训。

## 11. 手顺八：一致性测试必须先于服务接入

### 核心测试

取同一学生序列，在 eval/FP32 下：

```python
reference = model.forward(all_attrs, all_responses, all_pids)
cache = empty_cache()
for t in range(T):
    p_readonly = predict_next(cache, attrs[t], pids[t])
    cache, p_observe = observe(cache, attrs[t], pids[t], responses[t], ids[t])
    assert_close(p_readonly, reference[t])
    assert_close(p_observe, reference[t])
```

reference 保留相同 seq_len。对于小样本，额外逐个前缀重算确认旧 forward 的因果性。先检查输入编码，再定位每一层新位置输出，最后比较概率；不能只比 AUC。

初始可采用 `atol=1e-5, rtol=1e-4`，同时记录最大绝对误差。不同设备/精度需要另定容限；不得通过不断放宽容限掩盖固定位置的系统性偏差。

### 必测场景

| 场景 | 主要检查 |
| --- | --- |
| 第 1、2 个事件 | 无历史分支、out_proj bias、首行清零 |
| L=1、L=2、小 L 长序列 | 两种 mask 的 off-by-one 和裁剪 |
| T=L-1、L、L+1、2L+1 | 窗口边界一致性 |
| L=None | 不裁剪路径 |
| n_blocks=1 和实际 n_blocks=4 | 多层 K/V 来源正确 |
| drift 三模式、pid=0、合法未知字段 | 输入编码一致 |
| kq_same 两种设置 | Q 投影复用正确 |
| 改变当前 response | 当前概率不变，提交后缓存允许改变 |
| 多候选顺序交换、重复预测 | 不污染缓存，候选互不影响 |
| 保存/恢复后继续 observe | 与不中断运行一致 |
| 两个学生交错请求、事件重复投递 | 学生隔离与幂等 |
| 批量导入拆成不同 chunk | 内部逐条执行结果不随 chunk 边界变化 |
| 原 forward 右 padding | 比较每个学生有效位置，padding 不提交缓存 |

不能要求“改变 response 后下一题概率一定显著变化”：具体权重可能对该事件不敏感。应检查计算路径、缓存及与 reference 的一致性。

## 12. 性能预期与上线顺序

令 B=n_blocks，d=d_model，每层缓存最多 L 项。按三类 Attention、每类 B 层、每层一份 K 与 V 估算：

```text
缓存元素数 ≈ 6 × B × L × d
```

B=4、L=200、d=256、FP32 时约 4,915,200 字节，即 4.69 MiB/学生，不含元数据、临时张量和框架开销。1000 个活跃学生仅这些缓存约 4.58 GiB，所以不能默认全部常驻 GPU。

有限 L 下，单事件的注意力部分约 O(BLd)，另有各层投影与 FFN 的计算；不会随学生累计事件数持续增长。缓存读写也有成本，应和重算耗时一起测量。

建议上线顺序：

1. 单学生、单事件、CPU FP32，所有一致性测试通过。
2. 实际 checkpoint、小批真实历史，验证与原模型及现有 runner 的差异。
3. 实现 cache 序列化、版本拒绝、幂等更新和恢复重放。
4. 接业务后端；先使用单进程/明确学生锁，避免并发状态错误。
5. 再测 GPU、多候选 batching、活跃缓存 LRU 与冷状态存储。

## 13. 可直接交给编码代理的任务摘要

> 为当前 AKTFMKC 增加推理专用的逐层 K/V cache。保留训练 forward、全部可学习参数和 checkpoint key。以现有完整前缀 forward（保持 seq_len 局部 mask）为 oracle，实现单学生 predict_next 与 observe。缓存 blocks_1、blocks_2 偶数层、blocks_2 奇数层的各层投影 K/V；奇数层 K 来自题目侧 x，V 来自最终作答侧 y。自注意力允许当前加过去 L-1 项，检索只允许过去 L 项。新增非方形单 query 的 AKT 距离衰减实现，使用真实相对位置，修复增量路径首行清零语义，保留无历史时 out_proj bias 与残差/LayerNorm。候选预测绝不写状态，真实事件完成后原子追加所有缓存。验证序列化恢复、模型版本、幂等和多学生隔离。先逐条实现，再优化 batching；不要添加新 memory 架构或改变 FMKC 编码公式。训练 runner 尚未提供，另行检查分块重放与在线语义是否一致。

## 完成标准

- [ ] 旧 checkpoint strict 加载成功，旧训练 forward 不变。
- [ ] 每种 drift 的编码与原实现一致。
- [ ] 完整前缀与逐条增量预测在容限内一致。
- [ ] 跨 L 边界及重启恢复后一致。
- [ ] 当前真实答案不影响当前预测。
- [ ] 候选预测不改变学生状态。
- [ ] 有限 L 时缓存长度有界，不随累计历史增长。
- [ ] 原始事件、缓存版本、更新游标可追溯且可重建。
- [ ] 已确认训练 runner 的分块语义，并记录与目标线上语义的差异。

本计划最关键的选择：缓存当前模型各层的计算结果，以保持连续计算语义；不把最终 s 冒充完整学生状态，也不靠重新截取最近 L 条原始事件替代它。
