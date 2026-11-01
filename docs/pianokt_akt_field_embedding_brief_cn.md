# PianoKT：AKT 六字段组合 Embedding 改造手顺

> Yousician 适配约定：本文的原始谱面位置 question_id 要求不适用于 Yousician。Yousician 以规范化四字段 KC vector 作为题目身份，相同 vector 共享题目 ID 与 μ，不引入歌曲、练习或事件位置身份；其余适配细节见 [Yousician 实现说明](yousician_akt_field_embedding_cn.md)。

请将现有 AKT 的离散 KC embedding 改为 **PianoKT 六字段组合 embedding**，保留 AKT 注意力、衰减和因果结构。这是组合属性版 AKT 扩展，不是原版实现。

## 1．数据输入

每个 onset 保留一条交互：

```python
attrs = [
    has_black_key,
    simultaneous_note_count,
    both_hands,
    onset_interval_bin,
    change_count,
    mean_move_bin,
]
question_id = 原始题目ID
response = 1 - reject_reason.event
```

- 不要将六字段展开成六个时间步。
- 属性只来自目标乐谱及预测前已知条件，不使用当前演奏结果。
- 缺失值单独编码，不能与有效值 0 混用；映射规则只在训练集拟合。
- 字段名对接实际 schema，例如 `onset_interval_bin_cs`；离散值先映射为合法索引。
- 保留真实题目 ID（例如乐谱版本＋onset 位置），不以完整六维组合替代。

## 2．替换题目 embedding

六字段分别查表，拼接后通过两个独立输出头：

```python
u = torch.cat([E_i(attrs[..., i]) for i in range(6)], dim=-1)
c = base_head(u)                    # [B, T, d_model]
d = question_drift_head(u)          # [B, T, d_model]
mu = question_scalar(question_id)   # [B, T, 1]

question_emb = c + mu * d
```

`E_i` 表示第 i 个字段的独立 embedding 表。`mu` 零初始化，添加可配置 L2 正则；未知题目的 `mu` 固定为 0。基础表示 c 与偏移方向 d 使用不同输出头。

## 3．替换 question-response embedding

```python
g = response_embedding(response)   # Embedding(2, d_model)

f0 = wrong_drift_head(u)
f1 = correct_drift_head(u)
r = response.unsqueeze(-1).float()

interaction_emb = c + g + mu * ((1 - r) * f0 + r * f1)
```

两个分支共享字段 embedding、`base_head` 和题目标量 `mu`。原有依赖离散 KC ID 的交互 embedding 路径应同步替换。padding 标签需先安全映射，不能直接把 -1 传入 embedding。

## 4．接入 AKT

- `question_emb` 接题目分支。
- `interaction_emb` 接历史交互分支。
- 预测事件 t 允许使用当前题目属性，但只能使用 t 之前的响应；保留各层的因果约束，防止间接泄露。
- 每个 onset 计算一次 BCE；padding 不参与损失。
- 不改动注意力、窗口、衰减或长短记忆结构。



## 5．保留三个消融配置


| 配置            | 题目表示       | 交互表示               |
| ------------- | ---------- | ------------------ |
| 无 drifting    | `c`        | `c + g`            |
| 共享 drifting   | `c + mu*d` | `question_emb + g` |
| 响应条件 drifting | `c + mu*d` | 上述 `f0/f1` 公式      |


三种配置使用相同的数据划分、主干和训练预算；未启用的偏移模块不参与训练。参数、编码映射和配置随 checkpoint 保存。

## 6．必要验证与交付

在 `eval()` 模式下验证：

- 修改当前或未来 response，不改变当前预测。
- 修改未来属性，不改变更早预测。
- 未见题目、未见完整属性组合可以正常预测；未知字段取值有明确回退编码。
- 一个 onset 对应一个预测和一个标签。
- padding 不影响有效位置的预测及损失。

提交修改说明、三个配置的运行命令，以及上述验证结果。