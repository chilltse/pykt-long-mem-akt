# Dual Window AKT：独立 Attention + 每 Head 静态门控

## 1. 改造目标与范围

基于当前 `AKTDualWindow` 修改 `_layer()`：Short 和 Long 分别计算 attention，在每个 head 内融合，再执行现有 `out_proj`、残差和 FFN。

- 两路共享该层现有 Q/K/V 投影；“独立”指 mask、softmax 和 attention 输出独立，不是复制两套 AKT。
- 保留现有学生隔离、绝对位置、滑动窗口、每层缓存和 block 顺序。
- 沿用现有所有 `_layer()` 调用的双窗口范围，包括 encoder 和 retriever；每层分别拥有自己的 gate。
- 本版约定：Short 保留 AKT context-aware 衰减；Long 使用纯相似度 Top-K attention，不额外施加 AKT/时间衰减，也不复制 Short 边界衰减。

## 2. Q/K 归一化与分数

先执行现有线性投影和分 head，再沿 `d_k` 维做 L2 归一化；不要对 V 归一化。当前 Q 不 detach。

```python
q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
k_current = F.normalize(k_current.float(), p=2, dim=-1, eps=1e-6)
# 拼接历史缓存的 normalized K 与当前 normalized K
scores = (q @ k_all.transpose(-2, -1)) * math.sqrt(attn.d_k)
```

本版采用固定 `sqrt(d_k)` 缩放余弦相似度，两路一致；不要再除以 `sqrt(d_k)`。这是明确的工程默认值，避免单位向量点积再除根号后使 softmax 过平，不声称复刻论文温度配置。分数、softmax 和距离计算用 FP32，乘 V 前转换为适当 dtype。

## 3. Short：只在 Short 内计算 AKT

保留 `window_masks()` 和 valid/peek 规则。Short 的两次 softmax 都只能使用 `short[:, None]`，Long 不得进入其归一化分母或 distance 计算。

```python
with torch.no_grad():
    p_s = masked_softmax(scores, short_mask)
    distance_s = ((p_s.sum(-1, keepdim=True) - p_s.cumsum(-1))
                  * absolute_age[:, None]).clamp(min=0).sqrt()

exponent_s = distance_s * (-F.softplus(attn.gammas)[None])
# 保留现有 pdiff 对 exponent_s 的修正（如启用）
a_s = masked_softmax(scores * exponent_s.exp().clamp(1e-5, 1e5), short_mask)
o_s = attn.dropout(a_s).to(v_all.dtype) @ v_all
```

`masked_softmax(scores, mask)` 改为显式接收 mask，不再闭包引用联合 `allowed`；全空行返回全零，不能出现 NaN。key 维保持原始时间顺序，以保证 cumsum 的含义。

## 4. Long：逐 Head、逐 Query 检索并独立 softmax

1. 对 `scores.detach()` 使用 Long mask，再沿最后一维 Top-K；每个学生、head、query 独立选择。
2. 将索引 scatter 为 `selected`，并与 `long[:, None]` 相交，去掉候选不足时选到的无效位置。
3. 在原始、未 detach 的 `scores` 上计算 Long softmax；只允许 `selected` 位置。

```python
a_l = masked_softmax(scores, selected)
o_l = attn.dropout(a_l).to(v_all.dtype) @ v_all
```

`topk=0` 或没有有效 Long 时，`a_l` 和 `o_l` 为零。不要把 `short_mask | selected` 用于任何一路 softmax。

先沿用完整时间轴上的稠密 scores/mask，减少改动；这版没有消除全量 QK 计算成本。之后若改为 gather Top-K，也必须用带梯度的分数重新计算或 gather，不能直接用 detached 的 Top-K values。

## 5. 梯度与缓存

- 保存 `normalized K.detach()` 和 `V.detach()`，保留现有逐学生压缩与 `W+M` 容量。
- 读取历史时使用 detached K/V；Long K/V 均不回传到旧 block。
- 当前 Q、Q 的归一化、最终 Long 分数、attention 权重及 gate 正常参与反向传播；不要把整个 Long 分支放进 `no_grad()`。
- Top-K 离散索引不参与梯度；只有选择索引时使用 detached scores。
- 保留 `block_size <= short_window`，因此 Long 一定来自历史缓存；当前 block 的 K/V 仍可通过 Short 路径学习。
- `kq_same=True` 时共享投影参数仍可能通过当前 Q 获得梯度；这不等于历史 K 获得梯度。

## 6. 每层、每 Head 的静态门控

在 `__init__` 为每个 attention 层注册 `nn.Parameter`，形状 `[heads]`。必须在创建 optimizer 前注册，不能在 `_layer()` 中临时新建。

```python
# 每层分别注册；初始 Long 比例 0.05
gate_bias = nn.Parameter(torch.full((heads,), math.log(0.05 / 0.95)))

g = gate_bias.sigmoid()[None, :, None, None]  # [1,H,1,1]
has_long = selected.any(-1, keepdim=True)    # [B,H,T,1]
g_eff = g * has_long.to(g.dtype)
o = (1 - g_eff) * o_s + g_eff * o_l
```

`g` 只依赖层与 head，不接收 token、题目或学生作为输入。`has_long` 仅用于无候选时回退 Short，不是动态学习门控。

融合位置：`o_s/o_l` 为 `[B,H,T,d_k]` 时融合 → 拼接 heads → 现有 `out_proj` → 残差/LayerNorm → 原 FFN。不要分别做两次 out_proj 或在预测概率层融合。

新增 gate 后不再与原 AKT 完全同构 state_dict；加载旧权重时只允许新增 gate 缺失并保留初始化，其余缺失/多余键需检查。旧缓存直接丢弃，重新生成归一化缓存。

## 7. 最小验收与实验

- **独立性**：固定同一层的输入 Q/K/V 和 Short，替换 Long 内容，dropout 关闭时该层 `a_s/o_s` 必须不变。不要要求多层端到端的 Short 表示不变，因为上游层可能已经融合 Long。
- **门控**：手动覆盖 `g=0` 时输出等于该层 Short；有 Long 且 `g=1` 时等于 Long；无 Long 时始终回退 Short。
- **梯度**：用至少两条有效 Long、不同 V 的样例验证当前 Q 和 gate 可获得梯度；历史缓存 K/V 的 `requires_grad=False`。不要用只有一条 Long 的 softmax 检验 Q 梯度。
- **因果与边界**：未来回答不影响过去预测；学生隔离、padding、空历史、Top-K 候选不足与跨 block 窗口边界均正确。
- **对照**：同样启用 Q/K 归一化和缩放，比较 `topk=0` 与门控版本；原未归一化版本另作对照，避免混淆归一化和 Long 的效果。
- **日志**：记录每层每 head 的 `sigmoid(gate_bias)`、有 Long 的 query 比例，以及这些位置的预测指标。新增参数后重新训练或微调，不能直接用旧模型推断收益。

完成标准：Short 的 AKT 距离只由 Short 决定；Long 逐 head 检索且历史 K/V 无梯度；两个独立输出由可学习、token 无关的每 head gate 融合。
