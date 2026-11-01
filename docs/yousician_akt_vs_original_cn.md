# Yousician 四字段 AKT 与原版 AKT：实现差异说明

本文对照当前仓库代码，说明模型、数据和训练流程的实际差异。这里的“原版”指本仓库 `pykt/models/akt.py` 中的标准 `AKT` 路径，主要以 `emb_type="qid"`、`separate_qa=False`、启用题目难度参数为比较基准；不泛指所有 AKT 论文实现或其他变体。

当前改进版是 **四字段组合 embedding + 原版 AKT 主干 + 逐层滑动窗口 + 学生独立历史 block 缓存**。它保留了 AKT 的核心计算结构，但不能描述成“只换了 embedding，其他行为完全一致”。窗口、监督位置和训练入口也发生了变化；题目身份继续沿用四字段 KC 组合。

## 1. 先区分三类变化

**模型表示变化**：将单个离散 KC 查表替换为四字段组合表示，并按文档实现三种 drifting 配置。

**上下文变化**：保留原有因果方向，新增每层 attention 的滑动窗口左边界；在相邻训练 block 之间补充同一学生的历史输入。

**训练与工程变化**：在现有 data_preprocess 中增加四字段适配器和可复用学生划分模块，保存每个 fold 的训练词表；训练按 dataset_name/fold 加载产物，接入共享数据工厂、train_model/evaluate 和标准 checkpoint。训练阶段不读取原始 JSON。

学生划分、缓存和损失选择属于训练系统；它们不是 AKT attention 层新增的可学习模块。

## 2. 输入数据：一个离散 KC 变为四个属性字段

### 2.1 原版接收什么

原版 `AKT.forward(q_data, target, pid_data)` 接收：

- `q_data`：形状 `[B, T]`，每条交互对应一个离散 KC 索引。
- `target`：形状 `[B, T]`，正确或错误响应。
- `pid_data`：形状 `[B, T]`，题目索引。

注意命名：原版构造参数 `n_question` 实际对应 KC 数量，而 `n_pid` 对应题目数量。

如果直接使用原有 Yousician 标准预处理，它会把 `prev_frets|prev_strings|frets|strings` 整个组合同时作为 question ID 和 KC，再由通用流程映射为整数。此时原版不会分别理解这四个字段。

### 2.2 新版接收什么

新版 `YousicianAKT.forward(attrs, target, pid_data, ...)` 将第一项改为 `[B, T, 4]`，字段顺序固定：

1. `prev_frets`：同一 exercise 内上一个有效事件的四弦 fret 表示。
2. `prev_strings`：上一个有效事件使用的弦集合。
3. `frets`：当前事件的四弦 fret 表示。
4. `strings`：当前事件使用的弦集合。

每个字段有自己的类别词表。例如 `0^2^0^0` 作为 frets 字段中的一个类别，而不是拆成四个时间步，也不是把数字直接当连续数值输入。**一条有效事件仍只有一个预测、一个响应标签。**

字段计算复用现有预处理：只保留 play 记录、pitch 差分累加、映射四弦 fret、strings 排序去重、无效帧跳过、每个 exercise 的 prev 重置为 `inf`，响应仍为 reject_reason=0 时取 1，否则取 0。属性计算不使用当前响应。

### 2.3 题目身份只由局部 KC vector 决定

```text
question_key = prev_frets|prev_strings|frets|strings
question_id = 该 question_key 在当前 fold 训练词表中的整数索引
```

四字段完全相同，就共享 question ID 和题目标量 μ，无论来自哪个学生、歌曲、exercise、片段或事件位置。四字段中任一字段不同，即为不同的原始题目身份。这与现有 Yousician 标准预处理的组合身份一致，不引入歌曲或谱面位置条件。

题目词表仍只由训练学生拟合；未见完整组合统一回退至索引 0、μ=0，但四字段 embedding 仍可组合预测。不同未见组合共享的是未知回退编码，不是把它们判定为相同原始 KC。

song_id、exercise_id 不参与编码，也不要求原始记录提供它们。时间、session、part 只用于恢复学生练习顺序，不输入 embedding 或难度标量；每条 exercise 记录起点仍重置 prev 为 inf。学生级历史、已确认的逐层局部窗口和缓存保持不变。相同 KC 共享题目表示，但学生历史不同，最终预测可以不同。

这是对 PianoKT 参考手顺中“使用原始谱面位置题目 ID”的明确覆盖：Yousician 采用局部四字段身份，不照搬曲目级题目定义。

## 3. Embedding：保留标量难度思想，替换属性表示和漂移方向

下面用 k 表示离散 KC，p 表示题目，r 表示响应，μ 表示题目标量。

### 3.1 原版默认路径

根据当前 `akt.py`，原版默认路径可写成：

```text
c = KCEmbedding(k)
g = ResponseEmbedding(r)
d = KCDifferenceEmbedding(k)
h = ResponseDifferenceEmbedding(r)
μ = QuestionScalar(p)

question    = c + μ*d
interaction = c + g + μ*(h+d)
```

这里有一个容易误读的实现细节：虽然 `qa_embed_diff` 的表很大，但当前 forward 使用 `target` 索引它；默认路径的 h 是按响应索引的漂移向量，不是四字段条件输出头。

### 3.2 新版公共表示

新版四个字段分别查表，拼接后投影：

```text
u = concat(E_prev_frets, E_prev_strings, E_frets, E_strings)
c = base_head(u)
g = response_embedding(r)
μ = question_scalar(p)
```

默认每个字段 embedding 宽度为 32，拼接后的 u 宽度为 128，base_head 将其映射到默认 256 维的 d_model。各输出头目前均为 Linear，没有额外加入多层非线性属性网络。

完整属性组合即使未在训练集中出现，只要各字段能够编码，就可以组合得到表示；未知字段值退回 UNK。这提供了组合泛化的结构条件，但不等于已经证明泛化性能更好。

### 3.3 三种 drifting 配置

**无漂移：`drift=none`**

```text
question    = c
interaction = c+g
```

不创建题目标量或漂移头。

**共享漂移：`drift=shared`**

```text
d = question_drift_head(u)
question    = c+μ*d
interaction = question+g
```

题目与交互共享同一偏移方向，不创建响应条件头。

**响应条件漂移：`drift=response`，训练入口默认配置**

```text
d  = question_drift_head(u)
f0 = wrong_drift_head(u)
f1 = correct_drift_head(u)

question    = c+μ*d
interaction = c+g+μ*((1-r)*f0+r*f1)
```

d、f0、f1 使用不同的 Linear 输出头，共享字段 embedding 和基础表示 c。交互分支的漂移方向同时依赖属性和响应，不能视为原版 `h+d` 的符号改写。

### 3.4 初始化与未知值

新版题目标量显式零初始化，未知题目索引为 0，其 μ 在 forward 中强制为 0。字段编码区分 PAD=0、UNK=1、MISSING=2，有效零值和 exercise 起点 `inf` 不与缺失混用。

原版也有题目难度零初始化，但 `reset()` 按参数第一维是否等于 `n_pid+1` 判断，并非只按模块名初始化题目标量；新版新增字段模块和输出头在父类初始化之后创建，不经过这个形状匹配重置逻辑。原版没有在 forward 中对某个“未知题目索引”强制标量恒为 0。

## 4. AKT 主干：哪些保持不变

新版继承 AKT，直接使用同一套 `Architecture`、`TransformerLayer`、`MultiHeadAttention` 和输出 MLP。

保留的结构包括：

- 交互 encoder：`n_blocks` 层，对历史 question-response 表示进行编码。
- knowledge retriever：`2*n_blocks` 层，交替执行题目自注意力与严格过去交互检索。
- Q/K/V 投影、多头划分、kq_same 配置。
- AKT 基于注意力累计分布和位置距离的衰减计算、可学习 gamma。
- 原有 residual、LayerNorm、FFN 及 dropout 的位置。
- 将 retriever 输出与当前 question embedding 拼接后，通过原 MLP 和 sigmoid 预测。

默认宽度、层数、头数与 `examples/wandb_akt_train.py` 的主要设置一致：d_model=256、d_ff=512、n_blocks=4、8 个头、dropout=0.2。

**保留衰减公式不代表衰减数值不变。** 输入 embedding 和允许参与 attention 的位置改变后，softmax 分布及其累计距离也会改变。

新版只支持 qid 主干，不支持 separate_qa=True，也未启用原实现中的 qid_pdiff、avgpool、linear 等可选路径。新版 μ 作用于 embedding，不额外调制 attention 衰减。

## 5. 滑动窗口：从“全部过去”变成“每层最多 L 个直接可见位置”

### 5.1 原版

在一次输入序列内部，原版使用三角因果 mask。题目自注意力和交互 encoder 可以包含当前位置；响应检索只能读取过去位置，没有额外的局部窗口左边界。

原版实际能看到多少历史，受输入片段长度限制；AKT 模型自身不会自动找回上一个片段。

### 5.2 新版

令 `L=seq_len`，新版在原三角 mask 上额外限制左边界：

```text
交互 encoder / 题目自注意力：max(0,t-L+1) ≤ j ≤ t
严格过去的响应检索：       max(0,t-L)   ≤ j < t
```

两类 attention 每行都最多直接关注 L 个键，但包含当前题目的自注意力占用一个当前位置。L 因而指“每层最多可见的键数量”，不是“L 条历史加当前题目”在所有分支中完全一致的定义。

默认 L=128。当前 block 的第一个、居中和最后一个事件都遵守同样的滑动规则；学生起始处历史不足时使用已有历史，不虚构补足。

### 5.3 已确认采用的多层语义

允许历史表示间接携带更早信息。例如第 10 条只直接读取第 7、8、9 条，但第 7 条的表示可能已汇总第 6 条。

因此本实现保证的是**逐层直接 attention 范围**，不保证超过 L 的原始事件对最终输出绝对没有影响。这是当前采用的方案，不执行逐目标独立窗口重算。

## 6. Block cache：补足片段边界，不增加新的记忆网络

### 6.1 默认运行方式

```text
block_size=128，cache_blocks=1，seq_len=128

前一个学生历史 block（最多128条） + 当前 block（最多128条）
                      ↓
          重新计算 embedding 与各层 AKT
                      ↓
          仅当前 block 的预测进入本次损失
```

`block_size` 决定一次更新包含多少新事件；`seq_len` 决定每层 attention 的窗口宽度；`cache_blocks` 决定保留多少历史输入 block。三者职责不同。

默认缓存一个 block 足够补足直接窗口；如果单独减小 block_size，必须满足 `block_size*cache_blocks >= seq_len`，否则入口报错。

### 6.2 保存的内容与生命周期

缓存保存编码后的 attrs、pid、response，存放在 CPU。它不保存 K/V，也不保存上一轮参数产生的隐藏表示。每次前向都使用最新模型参数重新计算。

每个学生、每个 epoch、每个数据 split 都重新建立缓存。学生内部按时间顺序推进，只随机化训练学生的遍历顺序；同一学生跨 exercise 可以保留历史，但预处理字段 prev 仍在每个 exercise 起点重置。

新版缓存由通用 StudentBlockDataset 管理，经 init_dataset4train 返回 DataLoader，每次产生一个学生的一个 block。原版模型没有这个跨 block 缓存。通用数据流程中的 window 测试文件也不能直接等同于这种训练期缓存协议。

### 6.3 BP 如何经过缓存

当前 block 的损失通过历史表示反传，更新历史路径所涉及的共享 embedding、attention 和其他模型参数。缓存数据是整数输入，本身不是要优化的参数。

缓存中输入的 detach 不会切断本次重算形成的计算图；但不会跨 optimizer step 保留上一个 block 的计算图。历史标签不再次计算 BCE，历史题目标量也不重复计入当前正则。

### 6.4 计算代价与边界

当前使用密集 attention 矩阵，窗口 mask 不会自动变成稀疏 attention 算法。默认输入可达 256 条，其矩阵规模仍随总输入长度平方增长，而不是仅按有效窗口边数计算。

历史重算有额外开销，不是 KV cache 的推理加速。缓存能避免后续 block 完全缺少历史，但有限缓存仍会截断多层间接上下文，所以改变 block 切分可能改变预测；不能宣称与整名学生序列一次前向完全等价。

## 7. 学生隔离和词表：训练协议的差异

新版在预处理阶段先按 user_id 固定约 20% test，再将其余学生分成 k 个验证 fold。训练指定 fold=i，选择该 fold 作为 valid，其余 fold 作为 train。默认 k=5、划分 seed=1024，保持 pyKT 的 fold 使用方式；学生 ID 的具体排列采用新的通用模块，不承诺与旧函数逐项相同。

**学生隔离不是新版独有能力。** 仓库通用 split_datasets.py 在切块前划分输入序列；Yousician 每名学生对应一条完整序列时，也能形成学生隔离。新模块显式以学生 ID 为划分单位，可复用于其他字段型数据集。训练时不重新划分。

另一个实质差异是词表拟合时机：通用流程先对 total_df 执行 id_mapping，再划分数据；新版在预处理阶段先划分学生，再为每个 fold 仅使用其训练学生拟合词表并保存。训练加载既有词表，不拟合。前者让编码表知道全体数据中出现过的 ID，并不等于泄露响应标签，但会改变未知题目的评测条件。

“学生隔离”也不代表每名学生训练一套模型。所有训练学生共享同一模型参数；隔离的是划分归属和缓存上下文。

## 8. 损失与优化：不应忽略的实验差异

原版通用训练路径将 AKT 输出的 `y[:,1:]` 与 shifted 标签对齐，因此输入片段第一位置不进入该路径的 BCE。新版对当前 block 的所有有效事件计算 BCE，包括学生首条和各 block 首条；学生首条没有历史，只能依据题目表示预测。

新版每个 block 的优化目标为：

```text
loss = 当前有效事件 BCE 的均值
     + l2 * 当前有效事件题目标量 μ² 的总和
```

只对新事件计损失，缓存事件不重复计数。这里正则项求和、BCE 求均值；因此改变 block_size 也会改变正则相对尺度，不应假设正则天然按事件归一化。

新版每次用一个学生的一个当前 block 更新参数，复用 wandb_train 的优化器配置（当前默认 Adam），默认 lr=1e-4，并将梯度范数裁剪为 5。原版通用 AKT 训练按 DataLoader batch 更新，当前 AKT 分支没有相同的梯度裁剪步骤。

新版入口默认 num_epochs=30，现已复用共享 train_model 的验证 AUC 选模和连续 10 epochs 无改善早停，不再使用独立的 BCE 选模循环。单类评估 AUC 返回 -1；针对该情况保证第一轮能保存 checkpoint。最佳模型重载后评估 test，并额外保存事件平均 BCE、AUC、accuracy。

验证和测试按既往真实响应形成上下文，属于在线历史已知的知识追踪评测，不是将模型预测递归填回历史的多步生成。

## 9. 工程流程、Padding 和 checkpoint

当前流程为：

```text
data_preprocess.py --dataset_name yousician
  → Yousician 原始格式适配器
  → 通用 prepare_student_sequences
  → 标准化学生序列 + 各 fold 编码/划分 + data_config 注册

train_yousician_akt.py --dataset_name yousician --fold 0
  → wandb_train.main
  → init_dataset4train / init_model
  → train_model / evaluate
```

四维属性需要独立 batch 协议，但不意味着另写一套训练系统。新版 DataLoader 接入 init_dataset4train，返回带 attrs、pid、response、loss_mask 的 block；共享 model_forward 和 evaluate 按模型名调用小型 batch 适配函数。原先直接读 raw JSON 的独立训练循环已移除。

通用模块分为两层：preprocess/student_sequences.py 负责学生级划分和各 fold 词表保存；datasets/student_block_dataset.py 负责读取预处理产物、编码和按学生组织历史 block。Yousician schema 解析只存在于 preprocess/yousician_field_preprocess.py。

训练无需原始 JSON，但需要预处理产物；更换 fold 不重拟合编码。StudentBlockDataset 当前只支持 num_workers=0、每次一个学生 block，不支持跨学生混合 batch。它不使用 padding；模型额外支持右侧 padding（response=-1），不支持左侧或内部 padding。padding 输出不能进入 BCE。

训练 CLI 默认 seq_len=128；直接实例化 Python 模型时 seq_len 默认 None，用来兼容此前未启用局部窗口的配置。直接调用模型时如果需要窗口，必须显式传参。

保存沿用 pyKT 约定：qid_model.ckpt 为 state_dict；config.json 保存模型和训练配置，以及当前 fold 的字段词表、题目数量、学生划分和缓存参数。支持通用 load_model；load_yousician_checkpoint 可同时读取邻近配置，并兼容旧实验 bundled checkpoint。新的训练不再生成旧格式，也不再宣称 checkpoint 包含 optimizer 状态。尚未提供断点续训。

原版 AKT 权重不能直接作为新版完整权重加载，因为 embedding 模块和参数形状已变化。

## 10. 如何解释实验结果

当前三种 drifting 配置之间，使用相同数据、seed、窗口、缓存、预算和选模规则，可以比较漂移设计的影响。但完整新版与原通用 AKT 跑分不同，不能直接归因于四字段 embedding，因为还同时改变了数据划分、词表拟合、可见历史、首事件监督和每次更新的 batch 组织。选模与训练循环已对齐共享入口。

如果后续要做归因清晰的原版对照，应另行对齐学生集合、事件过滤、题目 ID、词表拟合范围、窗口与缓存、损失选择、更新预算、梯度裁剪和选模指标，再比较离散 KC 表示与组合属性表示。该公平对照实验尚未实现或执行，本文不声称新版优于原版。

## 11. 代码索引与验证范围

- [原版 AKT 与共享主干](../pykt/models/akt.py)：原 embedding、encoder/retriever、可选窗口 mask、衰减和预测头。
- [四字段模型](../pykt/models/yousician_akt.py)：字段 embedding、三种漂移、未知题目处理、padding。
- [通用数据与缓存](../pykt/datasets/student_block_dataset.py)：预处理产物加载、编码和学生 block cache。
- [通用学生预处理](../pykt/preprocess/student_sequences.py)：学生 fold 划分和训练专属词表。
- [Yousician 适配器](../pykt/preprocess/yousician_field_preprocess.py)：原始 schema、局部 KC 组合身份与配置注册。
- [原始字段预处理](../pykt/preprocess/yousician_preprocess_kc_fmkc_standard.py)：四字段与响应生成，新增可选原始事件索引。
- [字段 batch 适配](../pykt/models/yousician_akt_runner.py)：供共享 train_model/evaluate 调用，不包含独立训练循环。
- [训练入口](../examples/train_yousician_akt.py)：默认配置、训练循环、选模和保存。
- [模型工厂](../pykt/models/init_model.py)：新版模型注册。
- [原通用训练流程](../pykt/models/train_model.py)：原 AKT shifted 损失、选模和 early stopping。
- [原通用数据划分](../pykt/preprocess/split_datasets.py)：ID 映射、数据划分和切块顺序。
- [验证测试](../tests/test_yousician_akt.py)：模型与缓存的回归测试。
- [运行说明](yousician_akt_field_embedding_cn.md)：三种消融的具体命令。

工程重构验证记录为 19 项测试通过，覆盖因果性、右 padding、未知值、学生隔离、逐层窗口、历史梯度和 checkpoint 等。完整前缀等价测试使用 seq_len=None，不能作为局部窗口加有限缓存与整段运行完全等价的证明。

已完成先预处理、再按 dataset_name/fold 调用共享训练入口的小样本训练与保存重载检查，尚未执行完整数据训练或公平对照实验。本文是实现说明，不是性能评估报告。
