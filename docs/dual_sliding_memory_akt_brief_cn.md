# 双滑动窗口 Long-Memory AKT：精简 Coding 手顺

## 目标与参数

保持原始 AKT 的 Question/Response 编码、Q/K/V 投影及 Context-aware Attention。按学生时间顺序逐 Block 前向，**Block 内并行**，跨 Block 缓存。默认 `B=128`（Block 长度）、`W=128`（Short 历史）、`M=256`（Long 候选池，即 2 Blocks）、`k=32`（每个 Query 选取的远期 Token 数）。首版要求 `W>=B`。

对于当前预测位置 `t`（全局 0-based），历史分区为：

```text
Short(t) = [max(0,t-W), t)            # 全量短期 attention
Long(t)  = [max(0,t-W-M), max(0,t-W)) # 长期 Top-k 候选池
Selected(t) = Short(t) ∪ TopK(Long(t), k)
```

上述两窗口**按 Query 的位置逻辑滑动**，不是逐 Token 物理迁移。Long 候选池连续且无空洞；Top-k 后实际长期 attention 是稀疏的。

## 需要怎么改（按顺序）

1. **训练循环改为按学生顺序分 Block**：同一学生 Block 1→2→3→…，Block 内仍采用矩阵并行。不同学生记忆隔离；序列开始、学生结束或新 epoch 时清空缓存。不要打乱同一学生的 Block 次序。
2. **为 AKT 每个 Attention 层缓存对应的历史 K/V 和全局位置**：缓存的是该层实际 `k_linear(key_input)`、`v_linear(value_input)` 的输出（含 head 维度），不能拿最终的静态 Question Embedding 代替。历史 K/V 在跨 Block 使用时 `detach()`；当前 Block 保持梯度。保留最近 `W+M` 个已完成历史 Token，而不只是 `M`。
3. **每次处理新 Block，拼接历史缓存与当前 Block 的 K/V**：例如处理 Block 4 时，缓存 Block 1–3，当前为 Block 4。对本 Block 所有 Query 同时做 Q·Kᵀ；使用全局位置计算两个布尔 Mask。
4. **动态 Long Shadow + Top-k**：某条 Block 3 Token 只要对 Query `t` 满足 `W < t-i <= W+M`，便自动进入 Long 候选池；与此同时最旧的 Long Token 随左边界前进被屏蔽。对 `LongMask` 中的 QK 分数取 Top-k；不逐 Token 更新索引。ShortMask 对应的全部历史保留。
5. **在选中集合上执行原 AKT 注意力**：Short 全部 + Long Top-k 一起计算注意力（不要将两支各自 Softmax 后简单相加）；保留原始 `gamma` 衰减、Context-aware Distance、Response Value 路径。Attention 候选按真实全局时间排序，用绝对位置计算距离。**这是稀疏 AKT 近似**：累积注意力距离会因 Top-k 截断而改变。
6. **Block 完成后提交各层 K/V**：与已有历史追加，`detach`，只保留最近 `W+M` 个 Token，进入下一个 Block。对 `mask=0` 的 Knowledge Retriever 严格禁止读取当前及未来 Response；对原始 `mask=1` 自注意力保留允许当前位置 Question/QA 输入的语义，不能共用严格过去的 Mask。

## Mask / Top-k 核心逻辑（伪代码）

```python
# q_pos: [B]；kv_pos: [P]，包含历史及当前 Block；全局 0-based 位置
age = q_pos[:, None] - kv_pos[None, :]     # [B,P]
short_mask = (age >= 1) & (age <= W)      # Knowledge Retriever: 严格过去
long_mask  = (age > W)  & (age <= W + M)

# scores: [H,B,P]，来自当前层的 Q@K^T / sqrt(d_head)
long_scores = scores.masked_fill(~long_mask[None], -float('inf'))
indices = long_scores.topk(min(k, scores.size(-1)), dim=-1).indices
selected_long = torch.zeros_like(scores, dtype=torch.bool)
selected_long.scatter_(-1, indices, True)
selected_long &= long_mask[None]           # 排除 topk 中 -inf 的无效项
allowed = short_mask[None] | selected_long  # [H,B,P]
# 使用 allowed、原始全局位置与 AKT 的距离/衰减机制计算 attention
# 全屏蔽的 Query 需输出零向量，避免 softmax 全 -inf 产生 NaN
```

**实现约束**：`W>=B` 时，本 Block 内早期 Token 不会在本 Block 期间从 Short 过期；因此 Long Shadow 只来自已经计算完的历史缓存，无需将尚未生成的当前 Block K/V 提前存入长期索引。`scores` 和 Top-k 可按 Head 分别计算；所有 Batch 样本都必须有自己的位置与有效性 Mask。

## 最小测试（必须通过）

- `B=W=128, M=256`：查询第 `385/400/450/512` 条（1-based）时，Long 分别为 `1–256 / 16–271 / 66–321 / 128–383`；Short 分别为 `257–384 / 272–399 / 322–449 / 384–511`。
- 无重复、无候选资格空洞、无未来信息泄漏、学生间无串缓存；首个 Token 无历史时不出现 NaN。
- 第一个 Block、`M=0` 且无旧缓存时，输出应与原始 AKT 对应实现基本一致（固定随机数和权重，允许数值容差）。
- 改变未来答案，不应影响更早位置的预测；训练时当前 Query 与当前 Block 参数可收到梯度，历史缓存不反传。

## 训练顺序

1. 先实现 `Short-only`（`M=0`）并验证跨 Block 缓存与因果性。
2. 开启 `M=256, k=32` 的 Long Top-k，从头训练或从 Short-only checkpoint 继续训练。
3. 模型收敛后把 `M` 扩大至 `1024` 微调，`W` 和 `k` 保持不变；每个新训练序列按当前参数重新构建缓存，避免跨 optimizer step 长期复用过旧的 K/V。
4. 对照原始 AKT、Short-only AKT、双滑动窗口 AKT；额外报告 Block 边界的预测性能及 Long Top-k 的使用情况。
