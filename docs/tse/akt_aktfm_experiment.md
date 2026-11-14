# Task: Implement a Reproducible AKT vs AKT-FMKC Benchmark on the Yousician Ukulele Dataset

## 1. 项目背景与总体目标

当前工程为：

- Repository: `https://github.com/chilltse/pykt-long-mem-akt.git`
- Target branch: `feature/akt_fmkc_incremental_inference`
- Dataset: Yousician Ukulele
- Models: `akt` and `akt_fmkc`

请基于当前本地工程的实际代码，实现一套完整、可复现、科学严谨、支持一键执行的 AKT vs AKT-FMKC 对照实验框架。

本次研究主要回答三个问题：

**RQ1 — Predictive Performance**

在相同数据集、学生划分、训练协议和 AKT Backbone 下，当前 AKT-FMKC 与完整 KC 组合 ID 编码的普通 AKT 相比，是否具有更好的知识追踪预测性能？

**RQ2 — Long-tail and Natural Cold-start Generalization**

对于训练频率较低的完整 KC 组合，以及自然出现在验证集和测试集中的未见 KC 组合，AKT-FMKC 是否具有更好的预测性能和概率校准能力？

**RQ3 — Data Efficiency**

当可用训练学生数量分别为 60、120、240 和完整训练集时，AKT-FMKC 是否能比普通 AKT 更有效地利用训练数据？

不要预设 AKT-FMKC 一定优于普通 AKT。所有实验必须客观、对称、可复现，并保留不支持研究假设的结果。

本任务仅包含：

- Phase 1：普通预测性能对照。
- Phase 2：逐事件预测导出、长尾评估、自然冷启动评估、概率校准及统计检验。
- Phase 3：训练数据规模实验。

暂时不实施人工 Cold-KC Holdout、PID Dropout、Concat/MLP Encoder 消融、推荐系统或强化学习实验。

---

## 2. 首先检查当前工程

开始前完整检查当前 Git 状态、分支、已有实现及相关测试。

重点阅读：

```text
examples/compare_akt_fmkc.py
examples/compare_yousician_akt.py
examples/train_akt_fmkc.py
examples/train_yousician_akt.py
examples/wandb_train.py

pykt/models/akt.py
pykt/models/akt_fmkc.py
pykt/models/akt_fmkc_runner.py
pykt/models/init_model.py
pykt/models/train_model.py
pykt/models/evaluate_model.py

pykt/preprocess/yousician_field_preprocess.py
pykt/preprocess/yousician_preprocess_kc_fmkc_standard.py
pykt/preprocess/student_sequences.py
pykt/preprocess/vector_kc_preprocess.py

pykt/datasets/student_block_dataset.py
pykt/datasets/init_dataset.py

configs/kt_config.json
configs/data_config.json

docs/akt_fmkc_comparison_cn.md
docs/akt_fmkc_standard_cn.md
docs/yousician_akt_vs_original_cn.md

tests/test_vector_kc.py
tests/test_yousician_akt.py
tests/test_akt_fmkc_incremental.py
```

以上路径依据此前仓库版本列出。如果当前分支存在变动，以实际代码为准，记录差异，不要假设所有路径和实现保持不变。

优先复用现有：

- Yousician 数据适配器。
- Student-level split。
- Training-only vocabulary。
- StudentBlockDataset。
- StudentBlockCache。
- 共享训练器。
- 现有模型工厂。
- Shared evaluation adapter。
- Standard checkpoint loading。

不要重新实现 AKT Attention，不要重新实现一套与现有框架脱离的训练循环。

可以在现有模块中增加必要的可选接口，也可以新增独立 benchmark orchestration 和 evaluation 模块。

尽量将实验代码与生产 serving 代码分离。

不要修改：

- PianoKT 的生产推理逻辑。
- 增量 KV cache 算法。
- 当前模型默认推理行为。
- 现有 checkpoint 的加载兼容性。
- 原始数据文件。
- 其他数据集的默认预处理或训练行为。

不得覆盖现有实验结果，不得擅自删除文件、重置 Git 历史或 push 到远程仓库。

---

## 3. 确认 Yousician 数据集语义

尝试定位（这只是yousician数据集的一个子集）：

```text
data/yousician/yousician_ukulele.json
```

真正完整的yousician数据集在：『data/yousician_full/yousician_ukulele.json』

支持通过 `--raw_file` 指定实际路径。

不要自动选择含义不同的 JSON 文件，也不要将模拟数据误认为真实 Yousician 数据。

当前 Yousician 模型的 KC Vector 由以下四个字段构成：

```python
FIELD_NAMES = [
    "prev_frets",
    "prev_strings",
    "frets",
    "strings",
]
```

完整 KC identity：

```text
prev_frets|prev_strings|frets|strings
```

要求：

1. 仅保留现有预处理规则定义的有效 play 事件。
2. 响应标签沿用当前工程规则：`reject_reason == 0` 对应正确，否则对应错误。
3. 保持现有 pitch、fret、string、prev event、exercise 边界等语义。
4. 每个有效事件对应一个预测目标。
5. 不改变既有四字段 KC 提取方式。
6. 不利用当前 response 构造 KC 输入特征。
7. 不改变现有练习时间顺序。
8. 不跨学生拼接历史。
9. 日期、session、exercise part 等排序规则必须与现有适配器一致。
10. 检查排序键相同的记录是否存在歧义，并记录实际 tie-breaking 行为。

需要在预处理阶段生成数据审计报告：

```text
dataset_audit.json
```

至少包括：

```text
raw_file
raw_file_sha256
raw_record_count
raw_student_count
filtered_play_record_count
valid_student_count
valid_event_count
skipped_record_count
correct_count
incorrect_count
correct_rate
unique_kc_count
field_names
unique_values_per_field
sequence_length_min
sequence_length_max
sequence_length_mean
sequence_length_median
sequence_length_quantiles
```

同时输出：

- 每个学生的有效事件数量。
- KC frequency distribution。
- 各字段类别频率。
- 每名学生的正确率。
- 预处理前后事件数量变化。
- 跳过记录的原因及数量（在现有代码可以可靠获取的范围内）。

不必将所有原始练习记录重复写入审计文件。

---

## 4. 两个模型的定义

### 4.1 Baseline: AKT-ID

使用当前仓库的普通 AKT。

每个完整四字段 KC Vector 映射为唯一整数 ID。

```python
kc_id = vocabulary[canonical_kc_vector]
```

使用现有离散 question/KC embedding 和原版 Rasch-inspired drift。

不得通过增加字段级输入增强 baseline，否则不再是本实验定义的纯 ID baseline。

必须核查在 `student_fields_v1` 数据格式下：

```python
q_data = pid
pid_data = pid
```

以及共享 `StudentBlockDataset` 和 `forward_field_batch` 的实际传递方式是否正确。

### 4.2 Proposed: AKT-FMKC

使用当前已有的 `akt_fmkc` 实现。

```python
u = concat(
    E_prev_frets,
    E_prev_strings,
    E_frets,
    E_strings
)

c = base_head(u)
question = c + mu * question_drift_head(u)
```

使用：

```text
field_dim = 32
d_model = 256
drift = response
```

即四字段 embedding 拼接形成 128 维，再线性投影至 256 维。

保留现有 response-conditioned drift：

```python
interaction = (
    c
    + response_embedding
    + mu * (
        (1 - response) * wrong_drift_head(u)
        + response * correct_drift_head(u)
    )
)
```

本次不修改其网络结构。

### 4.3 实验结论的归因限制

必须在实验报告中明确：

AKT-ID 和 AKT-FMKC 的差异不仅包含基础 KC encoder，也包含 question/response drift 的参数化方式、embedding 初始化行为和参数规模。

因此，本次实验是两个完整编码架构的对照，而不是只改变一个变量的严格 embedding ablation。

不要将任何性能差异全部归因于 Linear Head 或字段参数共享。

---

## 5. 严格的数据划分与规模实验

### 5.1 主要实验：Fixed Evaluation Set + Nested Training Subsets

默认：

```python
split_seed = 1024
training_seed = 3407
subset_seed = 1024
kfold = 5
fold = 0
test_fraction = 0.2
train_student_sizes = [60, 120, 240, "full"]
```

首先对完整有效学生集合使用现有学生级划分机制。

保持：

```python
train_pool_ids
valid_ids
test_ids
```

三者严格互斥。

如果数据集包含 1,000 名有效学生，则默认预期：

```text
Train pool: 640
Validation: 160
Test: 200
```

以上人数仅为示例，必须以真实有效学生数为准。

然后从 `train_pool_ids` 中构造一个固定、确定性排列，生成嵌套训练学生集：

```python
S60 = ordered_train_pool[:60]
S120 = ordered_train_pool[:120]
S240 = ordered_train_pool[:240]
Sfull = ordered_train_pool
```

默认可以使用已有 split 后的训练学生顺序，或者显式使用固定 `subset_seed` 生成确定性排列，但必须只选择一种并记录实际规则。

必须保证：

```python
set(S60) <= set(S120)
set(S120) <= set(S240)
set(S240) <= set(Sfull)
```

所有规模共享完全相同的：

```text
Validation students
Validation events
Test students
Test events
Test event ordering
Response labels
```

每个规模下，AKT 和 AKT-FMKC 使用完全相同的训练学生、训练事件及其顺序。

禁止在不同训练规模下重新划分验证集或测试集。

如果训练池不足 240 名学生，明确报错或将不可用规模标记为 skipped。不能偷偷替换成较小规模。

### 5.2 每个规模独立构建训练词表

这是关键要求。

60 名训练学生对应的编码词表只能使用这 60 名学生构建。

120、240 和 full 同理。

不得先基于完整训练池构建词表，再将其直接用于 60、120 或 240 名训练学生。

每个规模必须重新拟合：

```python
field_vocabularies
question_vocabulary
train_kc_frequency
train_field_frequency
```

然后由同规模的 AKT 和 AKT-FMKC 共享这份 encoding metadata。

Validation 和 Test 不参与任何词表拟合。

保存：

```text
train_student_ids.json
valid_student_ids.json
test_student_ids.json
encoding.json
train_kc_frequency.json
split_manifest.json
```

对于不同规模，完整 KC ID 的整数映射可能不同。

因此，跨规模分析必须使用原始 canonical KC identity，而不能使用本地整数 PID 判断两个 KC 是否相同。

### 5.3 可选实验：Literal Dataset Prefix

额外支持：

```text
--scale_mode dataset_prefix
```

该模式允许从原始数据中选择前 60、120、240 名有效学生，分别进行学生级 Train/Valid/Test 划分。

需要明确定义“前 N 名学生”的顺序，例如：

```text
raw_first_occurrence
```

即原始 JSON 中首次出现的学生顺序，而不是预处理排序后的字典顺序。

可另外支持：

```text
sorted_user_id
```

每名学生的全部有效记录必须被完整保留。

这一模式仅作为补充实验，不作为默认 data efficiency benchmark。

由于不同规模可能对应不同的验证和测试学生，必须将其结果单独存储、单独报告，不得与 Fixed Evaluation Set 模式混合。

### 5.4 数据一致性检查

训练开始前执行：

```python
assert train_ids.isdisjoint(valid_ids)
assert train_ids.isdisjoint(test_ids)
assert valid_ids.isdisjoint(test_ids)
```

检查：

- 学生集合严格互斥。
- 训练集只包含被选中的训练学生。
- 词表只包含实际训练集中的 KC 和字段类别。
- AKT 和 FMKC 的事件身份及标签完全一致。
- 训练规模之间的验证和测试事件及标签完全一致。
- 学生内部事件顺序未改变。
- 训练集的事件统计与 frequency counter 一致。
- 不使用其他规模的 checkpoint、embedding 或优化器状态。

为 split、event sequence、encoding 和 frequency manifest 生成确定性的 SHA-256 fingerprint。

---

## 6. 统一训练协议

默认实验参数：

```yaml
dataset_name: yousician

models:
  - akt
  - akt_fmkc

fold: 0
kfold: 5

split_seed: 1024
training_seed: 3407
subset_seed: 1024

d_model: 256
d_ff: 512
n_blocks: 4
num_attn_heads: 8

field_dim: 32
drift: response

dropout: 0.2
learning_rate: 0.0001
l2: 0.00001

optimizer: adam

block_size: 128
cache_blocks: 1
seq_len: 128

num_epochs: 30
early_stopping_patience: 10
selection_metric: validation_auc

gradient_clip: 5.0
use_wandb: 0
```

核查现有代码中的实际参数，而不是仅记录 CLI 请求值。

特别注意：

- 当前 student-block 路径每次处理一个学生的一个 block。
- 历史 block 通过原始事件重放重新计算。
- 缓存历史仅提供上下文，不重复参与 BCE 和 PID 正则。
- 当前 block 的 `loss_mask` 确定有效预测位置。
- 每个学生、epoch 和数据 split 必须重新建立缓存。
- 两个模型采用相同的逐层 Attention Window。
- 两个模型都应用现有梯度裁剪。
- L2 PID 正则不等于 Adam weight decay，需要分别记录。
- 相同训练 seed 应产生相同的训练学生遍历顺序，但不同模型初始化和随机操作数量可能导致不同的随机数消耗，不能宣称具有相同初始化权重。
- 保持现有 early stopping 和 validation checkpoint selection 语义，不擅自替换。

所有模型均从头训练。

使用最佳验证 AUC 的 checkpoint 评估测试集，禁止使用最后一个 epoch 的权重代替最佳 checkpoint。

不得根据 Test AUC 选择学习率、训练轮数、模型版本或数据规模。

需要保存：

```text
epoch
train_loss
validation_auc
validation_accuracy
validation_bce（如可用）
best_epoch
optimizer_steps
supervised_events
elapsed_seconds
```

训练规模实验采用相同的 epoch 上限和 early stopping 规则，但不同规模的每 epoch optimizer steps 不同。

所以本次主要研究的是固定训练协议下的数据可用量效应，而非严格固定计算预算下的数据效率。

报告实际优化步数、训练耗时及累计监督事件数。

### 6.1 多种子扩展

当前默认只执行：

```text
fold = 0
seed = 3407
```

但 orchestration 应支持：

```text
--folds 0 1 2 3 4
--seeds 42 3407 2026
```

不要为了本次任务强制启动多 Fold、多 Seed 的昂贵训练。

### 6.2 Baseline UNK 与初始化审计

核查原版 `AKT.reset()` 和 dense model 初始化路径。

此前代码中存在根据参数第一维是否等于 `n_pid + 1` 进行零初始化的行为，这可能影响 dense question embeddings 或其他同形状参数。

必须检查：

- `q_embed.weight` 的初始化。
- `q_embed_diff.weight` 的初始化。
- `difficult_param.weight` 的初始化。
- 实际初始化为零的参数名称。
- UNK question ID 0 的向量初始值。
- UNK 是否在训练阶段获得监督。
- PID 0 的标量行为。
- FMKC 的 PAD、UNK、MISSING 和 PID 0 行为。
- 已见完整组合与未见完整组合的预测路径。

保存：

```text
initialization_audit.json
```

不要为了让 baseline 或 FMKC 获得更好的结果，静默修改上述行为。

如果发现实际 bug 或严重的不公平因素，应详细报告，并将任何修复设计为单独、显式、可追踪的实验变体。

自然冷启动实验中的结果必须结合 UNK 是否经过训练进行解释。

---

## 7. Phase 1：常规预测性能对照

每个训练规模执行：

```text
AKT-ID
AKT-FMKC
```

默认完整规模运行矩阵：

```text
60   × 2 models
120  × 2 models
240  × 2 models
full × 2 models
```

合计 8 次训练。

使用和现有 `compare_akt_fmkc.py` 一致的共享训练协议。

每个运行保存自己的：

```text
checkpoint
config.json
train.log
epoch_metrics.csv
validation_metrics.json
test_metrics.json
```

不要依赖只保存在终端中的结果。

对每个模型计算：

```text
ROC-AUC
Accuracy
Binary Cross Entropy / Log Loss
PR-AUC（明确记录采用 Average Precision 还是 PR 曲线面积）
Brier Score
ECE
Macro Student AUC
Macro Student BCE
Macro Student Brier Score
Macro Student Accuracy
```

另外记录：

```text
total_parameters
trainable_parameters
question_encoder_parameters
shared_backbone_parameters
drift_parameters
training_wall_time
best_epoch
total_epochs
optimizer_steps
number_of_train_students
number_of_train_events
number_of_validation_events
number_of_test_events
```

不将基础 embedding 参数量误称为整个模型的参数量。

参数统计需要根据当前真实模型实例进行。

### 7.1 指标定义

Accuracy 使用固定阈值：

```python
prediction >= 0.5
```

Brier Score：

```python
mean((predicted_probability - response) ** 2)
```

Log Loss 使用明确记录的概率裁剪规则，例如 `epsilon=1e-7`。

同时保留现有训练器 BCE 指标，若与独立计算的 Log Loss 存在数值差异，应说明裁剪或 dtype 原因。

ECE 默认采用 15 个固定等宽概率区间：

```python
bins = np.linspace(0.0, 1.0, 16)
```

最后一个 bin 必须包含概率 1.0。

```text
ECE = sum(
    bin_event_count / total_events
    * abs(bin_accuracy - bin_mean_probability)
)
```

保存 Reliability Diagram 对应的 bin 数据。

同时额外报告：

```text
positive_response_rate
mean_predicted_probability
```

Macro Student AUC 仅计算同时具有正负标签的学生，并报告有效学生和被排除学生数量。

其他学生级 macro 指标先按学生计算，再对学生取等权平均。

遇到空分组、单类分组或无法计算的指标时返回 null，并附上原因，不要将其解释为 0 或有效的 -1。

---

## 8. Phase 2：逐事件预测导出

这是本次任务的核心工程要求之一。

当前 `evaluate_field_model()` 主要汇总 predictions 和 labels，对完整实验分析而言不够。

请增加向后兼容的、可选的逐事件预测导出能力，或者新增独立 benchmark evaluator，复用现有 `forward_field_batch()`。

不需要修改生产 serving 推理接口。

每个有效监督事件保存以下字段：

```text
student_id
event_index
fold
seed
model_name
kc_vector
question_id
train_frequency
is_unseen_combination
has_unknown_field
response
predicted_probability
```

另外增加以下必要字段：

```text
run_id
scale_mode
train_student_count
split
kc_key
encoded_attrs
unknown_fields
unknown_field_count
has_missing_field
encoded_pid
history_length
block_index
is_first_student_event
is_first_kc_occurrence_for_student
encoding_fingerprint
```

字段语义必须明确。

### 8.1 原始事件身份

`event_index` 定义为：

学生完成现有预处理之后，完整有效事件序列中的零基索引。

同一个有效事件在不同模型、不同训练规模下具有相同的：

```python
(student_id, event_index)
```

并且原始 KC 和 response 必须一致。

不能使用：

- 当前 block 内局部位置。
- cache 拼接后的局部位置。
- 全局 DataLoader batch index。

作为原始 event_index。

每个有效事件必须且只能出现一次。

缓存历史事件不得重复导出。

### 8.2 KC 原始身份与整数 ID

必须保留：

```text
kc_vector: 原始四字段 KC Vector
kc_key: 原始规范化完整组合身份
question_id: 当前训练规模词表中的整数 ID
encoded_pid: 实际输入模型的 PID
```

对于未见完整组合：

```python
question_id == 0
train_frequency == 0
is_unseen_combination == True
```

但是不同未见组合的 `kc_key` 和 `kc_vector` 必须保持不同。

不能通过 PID 0 将所有未见完整组合错误合并成同一个原始 KC。

### 8.3 Unknown Field 与 Missing Field

`has_unknown_field` 依据当前训练规模的字段词表判断。

当前工程如果采用：

```text
PAD = 0
UNK = 1
MISSING = 2
```

则必须明确区分：

```text
未知但真实存在的字段取值
缺失字段值
Padding
```

不要将 Yousician 的有效 exercise-start token，例如 `inf`，错误视为 missing。

从原始 KC Vector 和当前 encoding 生成 unknown_fields。

AKT baseline 虽不直接使用字段级 embedding，也必须基于共享词表导出完全相同的 coverage metadata，以便进行配对分析。

### 8.4 History metadata

`history_length` 必须采用明确的定义，例如预测该事件时模型输入中实际可用的此前原始事件数量。

注意区分：

```text
此前累计观察的事件数
当前输入上下文中的历史事件数
每层 Attention Window 的直接可见范围
```

不能将有限 block replay 等价为完整学生历史，也不能将逐层直接窗口长度误称为模型的严格总感受野。

如果需要，分别导出：

```text
history_events_seen
input_history_length
seq_len
```

### 8.5 导出格式

优先使用：

```text
predictions_test.parquet
predictions_valid.parquet
```

使用支持分批写入的 Parquet Writer 和压缩编码，避免全部事件字典堆积在 Python 内存后一次性输出。

默认可使用 `--export_format auto`：

- 如果安装了可用的 PyArrow，采用 Parquet。
- 否则使用流式 `jsonl.gz`。
- 不要为了本实验强制修改整个工程的依赖或升级 PyTorch。

可选提供 CSV 导出，但不作为千万级数据的默认格式。

必须保存 schema 定义版本和 metadata。

学生 ID 保持字符串形式，避免整数转换导致身份冲突。

所有预测必须：

```python
model.eval()
torch.no_grad()
```

使用真实历史 response 推进。

不能使用当前 response 泄露当前预测。

这是 teacher-forced next-response prediction，不是 autoregressive Monte Carlo rollout。

### 8.6 预测配对检查

在比较两个模型前，基于：

```python
(student_id, event_index)
```

进行严格 one-to-one join。

验证：

```text
无重复身份
无缺失事件
无额外事件
response 一致
kc_vector 一致
kc_key 一致
train_frequency 一致
unknown flags 一致
fold 一致
seed 一致
train_student_count 一致
```

任何不一致都必须报错。

不得使用 inner join 静默删除无法配对的事件。

---

## 9. 长尾 KC 分析

对每个训练规模，只使用当前实际训练事件统计：

```python
train_frequency[kc_key]
```

预先固定频率分组：

```text
G0: 0
G1: 1–10
G2: 11–50
G3: 51–100
G4: 101–500
G5: 501–1000
G6: >1000
```

所有规模和模型采用相同分组边界。

对每个分组计算：

```text
event_count
unique_kc_count
unique_student_count
positive_count
negative_count
correct_rate

ROC-AUC
Accuracy
PR-AUC
Log Loss
Brier Score
ECE
mean_predicted_probability
```

需要额外报告：

```text
group_fraction_of_all_test_events
group_fraction_of_all_test_kcs
```

对两个模型输出配对性能差异：

```text
delta_auc = fmkc_auc - akt_auc
delta_accuracy = fmkc_acc - akt_acc
delta_logloss = fmkc_logloss - akt_logloss
delta_brier = fmkc_brier - akt_brier
delta_ece = fmkc_ece - akt_ece
```

明确正负方向：

- AUC、Accuracy、PR-AUC 越高越好。
- Log Loss、Brier Score、ECE 越低越好。

不要因为罕见组 AUC 很高，就忽略该组可能只有少量学生或事件。

输出：

```text
frequency_group_metrics.csv
frequency_group_comparison.csv
```

同时产生：

- KC training frequency histogram。
- Long-tail AUC comparison plot。
- Long-tail Log Loss comparison plot。
- Long-tail Brier Score comparison plot。
- 每个频率分组的事件数和 KC 数量。

不要仅在某些表现较好的分组上报告结果。

---

## 10. 自然 Cold-start 分析

本阶段不人工删除任何训练 KC。

仅根据实际训练词表，将验证集和测试集划分为：

```text
Seen combination
Unseen combination, all fields known
Unseen combination, at least one field unknown
```

主要关注第二组。

其含义为：

完整 KC Vector 没有在当前训练数据中出现，但每个组成字段取值均曾在训练中出现。

该分组直接检验 FMKC 是否能够利用已知字段构造新组合表示。

额外单独记录 missing-field 情况，避免将缺失值直接解释为新类别。

对每组报告：

```text
events
unique_kcs
students
positive_rate
AUC
Accuracy
Log Loss
Brier Score
ECE
```

另外区分：

```text
First occurrence of a KC in a student's history
Repeated occurrences of a KC in a student's history
```

注意：

对模型训练而言未见的 KC，可能已经在某个测试学生的此前练习历史中出现过。

因此：

```text
Global training-unseen KC
```

不等于：

```text
Student-first-exposure KC
```

需要分别统计，避免将自然在线历史适应误认为纯零样本泛化。

输出：

```text
coldstart_metrics.csv
coverage_metrics.json
```

必须记录每个规模的：

```text
question_vocabulary_size
field_vocabulary_sizes
unseen_combination_rate
unseen_field_rate
fully_known_field_unseen_combination_rate
```

这些均以实际事件和实际 KC 种类为基础，分别报告事件比例与类别比例。

如果自然未见组合数量过少，应明确说明统计支持不足，不得夸大结论。

---

## 11. Student-level Paired Bootstrap

实现学生级配对 Bootstrap，而不是 event-level independent bootstrap。

原因：

同一学生的连续交互具有时间和行为相关性，不能把每个事件当作完全独立的实验样本。

默认参数：

```python
bootstrap_seed = 2026
n_bootstrap = 2000
confidence_level = 0.95
```

支持：

```text
--bootstrap_replicates 10000
```

具体流程：

1. 对两个模型的预测结果进行完整事件配对。
2. 提取唯一测试学生 ID。
3. 每次有放回地抽取与原始测试学生数量相同的学生。
4. 被选中的学生携带其所有相关事件。
5. 同一次抽样使用完全相同的学生及其重复权重评估 AKT 和 FMKC。
6. 计算配对性能差异。
7. 通过 bootstrap percentile 方法得到 95% CI。

至少支持：

```text
Overall AUC difference
Overall Log Loss difference
Overall Brier Score difference
Overall Accuracy difference
```

另外支持针对频率组和自然冷启动组的配对 Bootstrap。

对分组指标，可以从完整测试学生集合抽样，再在抽样后选择目标分组事件，或者使用明确说明的目标分组学生抽样协议。

默认采用前者，保持整体学生抽样单位一致。

如果某次抽样的目标分组没有正负两类标签，跳过该次 AUC 计算并记录原因。

报告：

```text
observed_difference
bootstrap_mean_difference
ci_lower
ci_upper
valid_bootstrap_replicates
invalid_bootstrap_replicates
student_count
event_count
```

注意：

- 相同学生重复抽中时必须正确保留重复权重。
- 不得简单对学生级 AUC 求平均后冒充 pooled AUC。
- 不得使用 bootstrap 样本中的排序作为新的模型选择标准。
- 不得将单 Fold、单 Seed 的 CI 解释为训练随机性下的整体置信区间。
- 本阶段 CI 仅描述固定已训练模型条件下，测试学生抽样带来的不确定性。

对于多个统计检验，预先指定：

```text
Primary metric: Overall ROC-AUC
Secondary metrics: Log Loss and Brier Score
Exploratory metrics: Frequency and cold-start subgroup metrics
```

如果为多个子组输出正式显著性检验的 p-value，需要明确检验方法，并采用 Holm correction 等方法控制多重比较。

否则可以将子组 CI 清楚标记为 exploratory / unadjusted，不强制输出不可靠的 p-value。

### 11.1 计算效率

Yousician 可能存在千万级事件。

不要为每次 bootstrap 复制整份预测数据。

优先实现：

- 按学生预聚合可加指标，例如 BCE、Brier、Accuracy。
- 通过学生 multiplicity 构造样本权重。
- 对 ROC-AUC 使用经过验证的 weighted AUC。
- 如果需要自定义高性能 AUC 算法，应正确处理 score ties，并与 sklearn 的 `roc_auc_score(sample_weight=...)` 对照测试。

可通过批处理和预排序优化。

保留精确算法作为 reference。

如果完整 2,000 次 AUC bootstrap 在当前资源下无法完成，应提供显式可控的运行配置和真实的未完成状态，不能静默使用事件抽样后仍宣称是完整学生级 Bootstrap。

---

## 12. 数据规模比较

按：

```text
60
120
240
full
```

分别计算完整性能指标。

同一规模的两个模型属于直接配对对照。

不同规模使用相同验证和测试学生，可以分析学习曲线。

输出：

```text
scaling_metrics.csv
data_efficiency_comparison.csv
```

至少包含：

```text
train_student_count
train_event_count
model_name
test_auc
test_accuracy
test_logloss
test_brier
test_ece
unknown_question_rate
unknown_field_rate
total_parameters
training_seconds
optimizer_steps
best_epoch
```

需要生成：

```text
AUC vs Number of Training Students
AUC vs Number of Training Events
Log Loss vs Number of Training Students
Brier Score vs Number of Training Students
ECE vs Number of Training Students
Unseen KC Rate vs Training Scale
Model Parameter Count vs Training Scale
```

额外进行以下分解分析：

### 12.1 所有测试事件上的性能变化

用于衡量总体数据效率。

### 12.2 共同已见 KC 上的性能变化

定义：

```python
common_seen_kcs = intersection(
    vocabulary_60,
    vocabulary_120,
    vocabulary_240
)
```

如果包含 full 实验，也纳入相应交集定义并记录。

使用原始 canonical KC key 计算交集。

在所有规模下固定相同的目标测试事件。

这样可以观察更多训练数据是否改善对同一批已见 KC 的学习。

### 12.3 共同未见 KC 上的性能变化

定义所有参评训练规模下均未见的测试 KC 集合。

进一步筛选在所有参评规模中字段取值均已知的 KC。

在该固定事件子集上比较两个模型。

若组为空，报告不可用。

### 12.4 各规模自身的低频和未知 KC 表现

这属于不同规模各自的 coverage 分析。

由于 KC membership 可能随训练规模改变，不能将其指标变化直接解释为固定目标集合上的性能变化。

必须在报告中区分这两类分析。

### 12.5 效应解释

本次规模实验默认只有一个嵌套训练学生排列和一个训练随机种子。

因此只能报告在该预先固定训练子集上的结果。

不得宣称所有随机抽取的 60、120、240 人训练集都会表现相同。

后续可通过不同 subset seeds 和 training seeds 扩展。

---

## 13. 实验产物目录

建议新增独立的 benchmark entrypoint：

```text
examples/benchmark_yousician_fmkc.py
```

具体模块组织可以根据工程结构调整，但功能必须完整。

输出目录建议为：

```text
saved_model/yousician_fmkc_benchmark/
└── run_<timestamp_or_id>/
    ├── experiment_config.json
    ├── experiment_manifest.json
    ├── dataset_audit.json
    ├── environment.json
    ├── git_commit.txt
    ├── git_status.txt
    ├── git_diff.patch
    ├── dataset_sha256.txt
    ├── prepared/
    │   ├── shared_student_sequences.json
    │   ├── split_manifest.json
    │   ├── train_60/
    │   ├── train_120/
    │   ├── train_240/
    │   └── train_full/
    ├── train_60/
    │   ├── akt/
    │   ├── akt_fmkc/
    │   ├── comparison.json
    │   ├── frequency_group_metrics.csv
    │   ├── coldstart_metrics.csv
    │   └── bootstrap_results.json
    ├── train_120/
    ├── train_240/
    ├── train_full/
    ├── summary/
    │   ├── overall_metrics.csv
    │   ├── scaling_metrics.csv
    │   ├── longtail_metrics.csv
    │   ├── coldstart_metrics.csv
    │   ├── bootstrap_comparison.csv
    │   ├── common_seen_metrics.csv
    │   ├── common_unseen_metrics.csv
    │   └── parameter_comparison.csv
    ├── figures/
    ├── execution_status.json
    └── report.md
```

每个模型目录内至少保存：

```text
config.json
qid_model.ckpt
train.log
epoch_metrics.csv
validation_metrics.json
test_metrics.json
initialization_audit.json
predictions_test.parquet（或 jsonl.gz）
predictions_valid.parquet（或 jsonl.gz）
parameter_counts.json
```

避免按照每个规模重复复制整个原始数据集或标准化学生序列。

优先复用一份共享只读的 canonical student sequences，以及各规模独立的 split、encoding 和 frequency manifest。

需要保证现有 loader 不会错误地从未选中的训练学生拟合词表或生成训练样本。

---

## 14. CLI 和一键执行

希望最终提供如下命令接口。

### 14.1 Dry-run

```bash
python examples/benchmark_yousician_fmkc.py \
  --raw_file data/yousician/yousician_ukulele.json \
  --train_student_sizes 60 120 240 \
  --fold 0 \
  --seed 3407 \
  --split_seed 1024 \
  --dry_run
```

Dry-run 仅进行：

- 数据文件检查。
- 数据审计。
- 划分与词表规划。
- 数据一致性检查。
- 实验矩阵输出。
- 设备及环境检查。

不运行正式模型训练。

### 14.2 Smoke test

```bash
python examples/benchmark_yousician_fmkc.py \
  --raw_file data/yousician/yousician_ukulele.json \
  --smoke_test
```

使用隔离的小型测试数据和较短训练流程验证：

```text
preprocess
→ split
→ train AKT
→ train AKT-FMKC
→ checkpoint reload
→ prediction export
→ pairing validation
→ metrics
→ bootstrap
→ report
```

Smoke test 产物必须与正式实验分离。

不得将 smoke test AUC 写入正式结果。

### 14.3 正式规模实验

```bash
python -u examples/benchmark_yousician_fmkc.py \
  --raw_file data/yousician/yousician_ukulele.json \
  --scale_mode fixed_holdout \
  --train_student_sizes 60 120 240 \
  --include_full \
  --fold 0 \
  --seed 3407 \
  --split_seed 1024 \
  --subset_seed 1024 \
  --kfold 5 \
  --num_epochs 30 \
  --block_size 128 \
  --cache_blocks 1 \
  --seq_len 128 \
  --d_model 256 \
  --d_ff 512 \
  --field_dim 32 \
  --n_blocks 4 \
  --num_attn_heads 8 \
  --dropout 0.2 \
  --learning_rate 0.0001 \
  --l2 0.00001 \
  --drift response \
  --bootstrap_replicates 2000 \
  --export_format auto \
  --save_dir saved_model/yousician_fmkc_benchmark
```

支持省略 `--include_full`，只运行 60、120、240 人的 6 次训练。

CLI 的具体命名可根据现有工程调整，但最终必须提供真实可执行的完整命令，而不是未实现的设计示例。

### 14.4 独立评估

支持使用已经训练完成的 checkpoint 重新导出逐事件预测及分析结果。

例如：

```bash
python examples/benchmark_yousician_fmkc.py \
  --evaluate_only \
  --run_dir <existing_run_directory>
```

以及仅使用已保存预测执行分析：

```bash
python examples/benchmark_yousician_fmkc.py \
  --analyze_only \
  --run_dir <existing_run_directory>
```

完成长尾分析或 Bootstrap 不应要求重新训练模型。

### 14.5 失败恢复

支持：

```text
--reuse_completed_runs
```

跳过已经完整结束并通过 manifest/checkpoint/metric 验证的模型运行。

如果运行中断，允许从已有完成的模型继续整个 benchmark。

区分：

```text
experiment-level resume
```

和：

```text
optimizer-level checkpoint resume
```

如果当前训练器没有保存 optimizer state，不得宣称支持模型训练断点续训。

不得在配置不一致时错误复用已有 checkpoint。

---

## 15. 可复现性与环境记录

保存：

```text
Git branch
Git commit SHA
Git dirty status
Git diff
Python version
PyTorch version
NumPy version
scikit-learn version
CUDA version
GPU model
Operating system
Execution device
CLI arguments
Resolved training parameters
Random seeds
Dataset SHA-256
Split fingerprints
Encoding fingerprints
Checkpoint SHA-256
```

记录 deterministic settings，包括当前训练代码是否支持完全确定性执行。

如果 GPU 算子无法保证 bitwise deterministic，明确说明，不要宣称已经实现跨设备完全一致。

保持现有设备机制。

如果当前工程只支持 CUDA/CPU，则不要将 MPS 支持作为已经实现的能力。需要先检查真实设备兼容性。

正式训练优先使用当前可用的兼容设备。

不要未经许可启动昂贵的云计算实例。

---

## 16. 测试与验收

新增针对 benchmark 的单元测试和集成测试。

至少覆盖：

### 数据划分

- Train/Valid/Test 无学生重叠。
- 60、120、240 训练学生集合嵌套。
- 不同规模的验证和测试集合完全一致。
- 不足目标学生数时正确处理。
- 未使用训练学生不参与当前规模词表构建。
- Validation/Test 中独有 KC 不进入任何训练词表。
- 不同未见 KC 的原始身份不会通过 PID 0 错误合并。
- 训练 frequency counter 与真实训练事件一致。

### 预测导出

- event_index 在跨 block 时正确累计。
- cache events 不重复导出。
- 每个有效事件仅导出一次。
- 首条事件正常导出。
- 学生缓存隔离。
- 当前 response 不影响当前预测。
- 两个模型输出对齐。
- 未见字段、缺失字段、有效 `inf` 起始 token 正确区分。
- 未知 PID 与原始 KC identity 分离。
- 导出概率有限且处于合法范围。

### Metrics

- Accuracy、BCE、Brier、ECE 与独立 reference 一致。
- ECE 正确处理概率 0 和 1。
- 单类目标返回 null 或定义明确的 unavailable 状态。
- macro student metrics 正确排除无法计算 AUC 的学生。
- 频率分组边界无重叠、无遗漏。
- 全部频率分组事件总数等于测试事件总数。
- cold-start 分组覆盖率正确。

### Statistical analysis

- 学生级抽样而不是事件级抽样。
- 每次 bootstrap 两个模型使用相同学生抽样。
- 重复抽中的学生获得正确权重。
- weighted AUC 与显式重复样本的 reference 结果一致。
- 单类 bootstrap 正确记录无效情况。
- 严格检查预测 join 的重复、缺失和标签冲突。
- 相同预测的两个模型产生零差异。

### Integration

- 至少完成一次真实执行的 synthetic smoke test。
- 训练后的 checkpoint 可以正常重新加载。
- evaluate_only 可以正常运行。
- analyze_only 无需 checkpoint 训练。
- 已完成运行可以安全复用。
- 现有 AKT-FMKC serving 和 incremental inference 测试不发生回归。

不要只编写测试代码而不执行。

需要报告实际执行的命令、测试通过数量、失败数量和原因。

---

## 17. 最终实验报告

自动生成中文 `report.md`。

包含：

1. Research Questions。
2. Dataset description。
3. Data audit。
4. Student split protocol。
5. AKT-ID 与 AKT-FMKC 的实际架构差异。
6. Hyperparameters。
7. Vocabulary construction。
8. Dataset scaling protocol。
9. Overall prediction results。
10. Frequency group results。
11. Natural cold-start results。
12. Calibration results。
13. Parameter efficiency。
14. Student-level paired bootstrap。
15. Data efficiency learning curves。
16. Limitations。
17. Reproducibility instructions。

必须根据实际产物生成报告。

不允许使用虚构 AUC、Accuracy、Brier Score 或置信区间填充模板。

如果正式训练尚未完成：

- 未完成结果标记为 pending / incomplete。
- 报告已完成的实验。
- 不得生成伪造的完整学习曲线。
- 不得提前得出 FMKC 优于 AKT 的结论。

即使 FMKC 的常规预测性能低于 AKT，也应真实报告，并分析长尾、覆盖率、概率校准和参数量是否存在不同表现。

不能将：

- 预测正确率提高。
- 概率校准改善。
- 结构化 embedding 共享。

直接解释成现实学生真实知识增长或因果学习迁移。

---

## 18. 工程执行顺序

请按以下顺序实际实施：

1. 检查当前 repo 和工作区，理解已有代码，不擅自覆盖用户改动。
2. 审计现有 AKT/FMKC 训练协议和初始化行为。
3. 新增共享 benchmark 配置和 nested student subset 机制。
4. 实现逐事件预测导出。
5. 实现配对一致性检查。
6. 实现整体预测指标、长尾和自然 cold-start 评估。
7. 实现学生级 paired bootstrap。
8. 实现规模实验汇总与可视化。
9. 编写并运行单元测试及 smoke test。
10. 如果真实 Yousician 数据在当前环境中可用，则执行真实数据 dry-run 并检查所有划分。
11. 在资源允许的情况下，执行预先定义的正式训练矩阵；若不能完整执行，则提供准确可运行的一键命令和已验证的实际完成状态。
12. 编写完整 README 或 benchmark usage 文档。

不要只提供伪代码或实验建议，需要实际完成工程改动。

---

## 19. 最终交付要求

任务结束时向我报告：

- 修改及新增了哪些代码文件。
- 为什么这样修改。
- 当前 baseline 和 FMKC 的训练协议是否公平。
- 是否发现旧代码中的初始化问题或数据泄露风险。
- 实际选择的学生划分及 scale subset 规则。
- 每个训练规模的学生数、事件数及词表大小。
- 执行了哪些测试，结果如何。
- 实际运行了哪些正式实验。
- 已验证的一键执行命令。
- 配置、日志、checkpoint、逐事件预测、统计分析和最终报告的真实路径。
- 哪些实验已经完成，哪些尚未完成。
- 任何未解决的科学限制或工程问题。

最终目标是形成一个完整、可独立复现、可以用于论文实验的 Yousician AKT-FMKC benchmark，而不是得到一个临时的 AUC 对比数字。

# Additional Requirements: Local Development, GitHub Deployment and HPC Execution

## 20. 开发环境与正式实验环境分离

需要特别注意：我的正式 GPU 训练环境是远程 HPC Slurm 集群，该集群无法直接运行 Codex。

因此，本任务必须采用以下工作模式：

```text
Local Mac + Codex
    → Implementation
    → Unit tests
    → Synthetic smoke tests
    → Git commit / push
    → HPC git clone / pull
    → Standalone Slurm execution
    → Result collection
```

Codex 只在本地负责代码开发、工程检查、测试和运行脚本的实现。

正式实验必须能够在没有 Codex、没有 ChatGPT、没有 OpenAI API、没有外部 LLM 服务的环境中独立完成。

不要假定本地存在支持完整 Yousician 训练的 GPU。

不要在本地擅自开始千万级交互的正式训练。

本地主要完成：

- 数据处理流程验证。
- 合成数据端到端测试。
- AKT 和 AKT-FMKC 小样本训练。
- Checkpoint 保存及重新加载。
- 逐事件预测导出验证。
- 长尾分组及自然冷启动分析验证。
- Brier Score、ECE、Bootstrap 验证。
- Slurm 脚本参数及依赖检查。
- Git 可提交代码准备。

完整真实数据训练将在 HPC 上执行。

---

## 21. 实验阶段必须支持独立执行

请不要将所有功能强制放进一个无法分阶段运行的 Python 进程。

Benchmark 必须至少支持以下独立执行阶段：

```text
prepare
train
evaluate
analyze
summarize
```

### 21.1 Prepare

负责：

- 读取原始 Yousician 数据。
- 生成数据审计报告。
- 建立共享 canonical student sequences。
- 固定验证和测试学生。
- 生成 60、120、240、full 的嵌套训练学生集。
- 为每个训练规模独立构建训练词表。
- 统计 KC frequency。
- 建立实验任务 manifest。
- 写入数据、划分及词表 fingerprint。

Prepare 阶段必须只执行一次，由所有训练任务共享其只读产物。

不得让每个 GPU Job 重复解析原始 JSON 或重新生成学生划分。

### 21.2 Train

每次只训练一个明确指定的模型及训练规模。

例如：

```bash
python examples/benchmark_yousician_fmkc.py \
  train \
  --run_dir <run_directory> \
  --train_student_count 120 \
  --model_name akt_fmkc \
  --fold 0 \
  --seed 3407
```

每个任务独立初始化模型、优化器、随机数状态。

训练产物存入该任务的专属目录。

不同训练任务不得共享可写的：

- Checkpoint 文件。
- Config 文件。
- 日志文件。
- 临时预测文件。
- Experiment-level 状态文件。

每个任务必须可以独立重跑，不影响其他已成功任务。

### 21.3 Evaluate

从对应模型的最佳 validation checkpoint 加载模型。

导出完整逐事件预测数据。

可以作为 GPU 训练任务最后一个步骤执行，也可以独立提交 evaluation Job。

必须支持：

```bash
python examples/benchmark_yousician_fmkc.py \
  evaluate \
  --run_dir <run_directory> \
  --task_id <task_id>
```

重新评估时不得重新训练模型。

### 21.4 Analyze

读取已保存的逐事件预测结果。

完成：

- Overall metrics。
- Frequency group metrics。
- Natural cold-start metrics。
- Calibration metrics。
- Student-level paired bootstrap。
- Common seen/unseen subset evaluation。

这一阶段应尽可能只依赖 CPU。

不得无必要占用 GPU 资源。

### 21.5 Summarize

汇总整个实验矩阵。

生成：

- CSV。
- JSON。
- Figures。
- Markdown report。
- 实验完成状态。
- 缺失及失败任务清单。

不要求模型重新训练。

---

## 22. 新增 Slurm 一键执行框架

请创建独立目录：

```text
scripts/yousician_benchmark/
```

至少包含：

```text
cluster.env.example
run_local.sh
submit_all.sh
prepare.sbatch
train_array.sbatch
analyze.sbatch
retry_failed.sh
collect_results.sh
```

如现有仓库已经包含 Slurm 工程规范或可复用脚本，优先遵循现有规范。

不要写死旧集群参数，除非已经从当前环境或工程配置中确认其有效性。

### 22.1 Cluster Environment Configuration

新增：

```text
cluster.env.example
```

至少提供可配置项：

```bash
PROJECT_ROOT=""
CONDA_ENV="pykt"
CONDA_SH=""

YOUSICIAN_RAW_FILE=""
BENCHMARK_OUTPUT_ROOT=""

SLURM_ACCOUNT=""
GPU_PARTITION=""
CPU_PARTITION=""
GPU_TYPE=""

TRAIN_CPUS=4
TRAIN_MEMORY="32G"
TRAIN_TIME="24:00:00"

PREPARE_CPUS=8
PREPARE_MEMORY="64G"
PREPARE_TIME="04:00:00"

ANALYZE_CPUS=8
ANALYZE_MEMORY="64G"
ANALYZE_TIME="12:00:00"

MAX_CONCURRENT_GPU_JOBS=2
```

这些数值只是可修改的示例资源配置，不代表已经确认适用于我的集群。

GPU 类型、partition、account、QoS、最大 wall time、GPU resource syntax 必须支持根据真实集群修改。

兼容常见的：

```text
--gres=gpu:1
--gres=gpu:<type>:1
--gpus=1
```

具体方式以集群文档及现有脚本为准。

不要擅自假定 GPU 类型为 A100，也不要写死 CUDA module 版本。

实际使用的配置文件为：

```text
cluster.env
```

该文件应允许包含本机路径，并加入 gitignore。

不要提交个人环境的绝对路径、访问凭证、集群认证信息或其他敏感数据。

必须提供清晰的配置说明。

---

## 23. Slurm Job Dependency 与 Job Array

正式实验采用以下 DAG：

```text
Prepare CPU Job
    ↓ afterok
GPU Training Job Array
    ↓ afterany
CPU Analysis and Summary Job
```

Prepare 只执行一次。

如果 Prepare 失败，不允许后续 GPU 训练正常开始。

若部分 GPU Array Task 失败，Analysis 阶段仍应生成真实的部分结果和失败任务清单，而不是伪造完整报告。

只有实际满足完整性要求的模型配对才允许进行对照统计。

提供整个实验矩阵的重新提交及失败任务恢复能力。

### 23.1 默认实验矩阵

固定：

```text
Fold = 0
Training seed = 3407
Split seed = 1024
Subset seed = 1024
```

实验矩阵：

```text
Task 0: Train 60 students, AKT
Task 1: Train 60 students, AKT-FMKC

Task 2: Train 120 students, AKT
Task 3: Train 120 students, AKT-FMKC

Task 4: Train 240 students, AKT
Task 5: Train 240 students, AKT-FMKC

Task 6: Full training pool, AKT
Task 7: Full training pool, AKT-FMKC
```

如果不指定 full，只生成 6 个任务。

不要在脚本中固定假定一定存在 8 个任务。

需要根据实际实验配置动态生成：

```text
task_manifest.json
```

或者：

```text
task_manifest.tsv
```

Task manifest 至少包含：

```text
task_id
model_name
train_student_count
fold
seed
prepared_config_path
output_dir
```

`SLURM_ARRAY_TASK_ID` 必须通过 task manifest 解析任务参数。

不得通过多个不一致的硬编码数组维护模型与规模映射。

支持未来扩展多个 Fold 和 Seed。

### 23.2 GPU 并发限制

使用可配置的 Job Array concurrency。

例如：

```bash
#SBATCH --array=0-7%2
```

但是动态数组范围应由 submit_all.sh 根据 manifest 在运行时传给 sbatch。

限制并行 GPU Job 数量，避免一次性占用大量集群 GPU。

每个任务默认申请一张 GPU。

禁止启动不必要的分布式训练。

### 23.3 资源配置

根据实际数据规模支持不同资源需求。

60 人实验、240 人实验和 full 实验可能需要不同的：

- Wall time。
- CPU memory。
- GPU memory。
- 输出存储容量。

可以先使用统一的保守资源配置，但必须允许用户覆盖。

若使用不同资源级别的 Job Arrays，必须保存对应的任务映射及提交记录。

不能让不同资源配置改变模型的科学实验协议。

---

## 24. HPC 单任务独立性

每个 Slurm Task 必须：

1. 激活正确的 Conda 环境。
2. 检查 Python、PyTorch 版本。
3. 检查 CUDA 是否可用。
4. 检查 Slurm 分配的 GPU 是否真实可见。
5. 检查数据文件、prepared manifest 和 encoding fingerprint。
6. 核对 Git commit 与实验 manifest。
7. 根据 task_id 获取实验参数。
8. 初始化独立模型和优化器。
9. 执行正式训练。
10. 保存最佳 validation checkpoint。
11. 导出逐事件预测结果。
12. 生成单任务 metrics 和状态文件。

正式 GPU 训练必须在 CUDA 不可用时明确失败。

不要静默回退到 CPU，导致长时间占用资源却无法按计划完成训练。

本地 smoke test 可以使用 CPU。

MPS 仅在当前代码已经验证支持时启用。

每个任务保存：

```text
task_status.json
```

包含：

```text
task_id
slurm_job_id
slurm_array_task_id
hostname
start_time
end_time
exit_code
status
git_commit
gpu_name
checkpoint_path
prediction_path
config_fingerprint
```

状态至少区分：

```text
pending
running
failed
completed
```

只有实际 checkpoint、预测文件、指标、fingerprint 等完整性检查通过，才允许标记 completed。

不能仅根据 Python 进程退出状态判断实验科学有效性。

---

## 25. HPC 日志与失败恢复

每个任务保存独立 Slurm 日志。

例如：

```text
slurm_logs/prepare_%j.out
slurm_logs/prepare_%j.err

slurm_logs/train_%A_%a.out
slurm_logs/train_%A_%a.err

slurm_logs/analyze_%j.out
slurm_logs/analyze_%j.err
```

日志目录必须在 sbatch 提交前建立。

脚本应使用：

```bash
set -euo pipefail
```

处理训练异常、信号终止、超时等情况。

如果任务被 Slurm 终止，状态记录应尽可能反映真实结果；仍需由后续状态核查辨别异常退出或遗留 running 状态。

不要静默吞掉异常。

支持：

```bash
bash scripts/yousician_benchmark/retry_failed.sh \
  --run-dir <existing_run_directory>
```

该脚本应根据：

- Task manifest。
- Task status。
- Checkpoint。
- Prediction export。
- 实际产物完整性。

识别需要重新执行的任务。

如果 AKT 已经完成而 FMKC 失败，只重跑 FMKC。

如果 checkpoint 完整但 prediction export 失败，优先仅重新执行 evaluation，而不是重新训练模型。

如果当前工程不支持 optimizer state 恢复，则训练中断后从头训练。

不要将 experiment-level retry 描述为 optimizer-level checkpoint resume。

---

## 26. 数据及实验产物存储策略

原始 Yousician 数据集已经位于本地或者 HPC 数据目录中，不应提交至 GitHub。

本地与 HPC 可以通过不同路径访问相同数据集。

例如：

```text
Local:
  /Users/.../yousician_ukulele.json

HPC:
  /home/remote/.../yousician_ukulele.json
```

具体路径通过命令行或 cluster.env 配置。

不要写死这些示例路径。

必须验证原始文件 SHA-256，避免本地与集群实际使用不同版本的数据。

如果 raw dataset 尚未存在于 HPC，请报告正确的传输命令，但不要自动将原始数据上传到 GitHub。

GitHub 仅保存：

- Python 源代码。
- 测试。
- Benchmark config。
- Shell scripts。
- Slurm scripts。
- 文档。
- 小型合成测试 fixtures。

以下内容不得提交：

```text
Raw Yousician datasets
Prepared student sequences
Model checkpoints
Full prediction exports
Large training logs
Temporary files
Private environment configurations
Secrets
```

检查并补充 `.gitignore`，但不要删除已有重要跟踪文件。

千万级数据应只保留一份 canonical prepared sequences。

各规模使用独立 split 和 encoding manifest。

Prepared 数据目录应在训练期间保持只读，避免并行 GPU Job 竞争写入。

检查现有 loader 是否会一次性解析或编码整个数据集，而不是仅加载当前任务需要的学生。

如果存在这一问题，优先实施向后兼容的、按学生选择性加载或编码机制。

不允许通过截断学生序列、减少验证测试数据来静默解决内存不足问题。

---

## 27. 一键提交脚本

最终用户应能够在 HPC 上执行类似以下命令：

```bash
git pull

cp scripts/yousician_benchmark/cluster.env.example \
   scripts/yousician_benchmark/cluster.env
```

完成一次环境配置后，运行：

```bash
bash scripts/yousician_benchmark/submit_all.sh \
  --train-student-sizes 60 120 240 \
  --include-full \
  --fold 0 \
  --seed 3407
```

该脚本需要自动：

1. 加载并验证 cluster.env。
2. 检查当前 Git 版本。
3. 检查实验配置。
4. 检查数据文件存在。
5. 创建唯一、不会覆盖旧结果的 run directory。
6. 保存完整 experiment config。
7. 创建 GPU task manifest。
8. 提交 Prepare Job。
9. 获取 Prepare Job ID。
10. 提交依赖 Prepare 的 GPU Job Array。
11. 获取 GPU Array Job ID。
12. 提交依赖训练结果的 CPU Analysis Job。
13. 保存全部 Slurm Job IDs。
14. 打印实际实验目录及任务状态查询命令。

提交命令应使用：

```bash
sbatch --parsable
```

并正确处理其返回结果。

在实际环境不支持预期的 Slurm 命令或选项时，清晰报错。

支持：

```bash
bash scripts/yousician_benchmark/submit_all.sh \
  --train-student-sizes 60 120 240 \
  --dry-run
```

该 dry-run 不应调用真实 sbatch 提交。

必须打印完整任务矩阵、资源配置和计划提交命令。

支持单模型、单规模提交用于诊断。

例如：

```bash
bash scripts/yousician_benchmark/submit_all.sh \
  --train-student-sizes 60 \
  --models akt akt_fmkc \
  --fold 0 \
  --seed 3407
```

最终提供的 CLI 必须与实际实现一致，不能仅提供理论上的接口。

---

## 28. 实验环境版本锁定

一旦提交正式实验，8 个训练任务必须使用相同的代码版本。

不能在 Job Array 排队期间直接修改同一工作目录中的模型源码，否则不同任务可能使用不同的 Git commit。

建议每次正式实验采用以下机制之一：

- 为实验 Git commit 建立 detached Git worktree。
- 建立不可变的代码 snapshot。
- 使用独立 checkout 目录，并在任务启动时验证源码版本及工作区状态。

优先选择符合当前工程和 HPC 使用习惯的最简方案。

所有任务保存：

```text
git_commit
code_fingerprint
environment_fingerprint
dataset_sha256
prepared_manifest_fingerprint
```

任一不一致时必须报错或明确拒绝将结果纳入统一比较。

不要在 HPC Job 内自动 git pull。

不要在训练开始后自动升级依赖。

---

## 29. 结果下载与归档

新增：

```text
collect_results.sh
```

支持：

```bash
bash scripts/yousician_benchmark/collect_results.sh \
  --run-dir <run_directory> \
  --mode summary
```

默认 summary 模式仅打包：

```text
experiment_config.json
experiment_manifest.json
dataset_audit.json
environment.json
initialization_audit.json
task_status.json
overall_metrics.csv
scaling_metrics.csv
longtail_metrics.csv
coldstart_metrics.csv
bootstrap_results.json
bootstrap_comparison.csv
parameter_comparison.csv
figures/
report.md
```

默认不包含：

```text
Raw dataset
Prepared student sequences
Large checkpoints
Full per-event prediction exports
```

支持额外选项：

```text
--include-logs
--include-checkpoints
--include-predictions
--mode full
```

打包文件应具有唯一实验 ID。

提供 SHA-256 checksum 和文件清单。

输出示例：

```text
archives/yousician_fmkc_<run_id>_summary.tar.gz
```

最终报告应打印可以在本地执行的 scp 或 rsync 下载命令。

集群 hostname 和目录必须由用户实际配置，不能编造真实服务器地址。

---

## 30. 本地开发及验收

本地提供：

```bash
bash scripts/yousician_benchmark/run_local.sh
```

默认仅执行：

- Python dependency check。
- Benchmark unit tests。
- Existing model regression tests。
- Synthetic end-to-end smoke test。
- Local one-command dry-run。
- Slurm submission command dry-run。
- Shell syntax check。
- Artifact integrity check。

不执行真实 sbatch。

不启动正式千万级数据训练。

对于 HPC 代码，至少验证：

```bash
bash -n scripts/yousician_benchmark/*.sh
```

以及所有 sbatch 脚本的 shell 语法。

如果本地存在 shellcheck，也可以使用。

GPU 和 Slurm 的实际运行兼容性只有在真实 HPC 执行后才算验证完成。

请清楚区分：

```text
Verified locally
Verified with synthetic data
Verified with real data
Verified on CUDA
Verified on Slurm
Not yet verified
```

不要将本地 smoke test 成功描述为真实 HPC 大规模实验已经通过。

---

## 31. Git 工作流

首先检查：

```bash
git status
git branch --show-current
git rev-parse HEAD
```

保护我已有的未提交代码。

建议在当前目标分支基础上创建独立实验开发分支：

```text
feature/yousician-fmkc-benchmark
```

不要擅自覆盖原有：

```text
feature/akt_fmkc_incremental_inference
```

所有实现及测试完成后：

- 提供变更摘要。
- 检查是否存在大文件。
- 检查是否意外包含数据、checkpoint 或凭证。
- 提供准确的 git add 和 git commit 命令。
- 提供准确的 git push 命令。
- 不未经明确授权自动 push。

GitHub 代码同步完成后，我会在 HPC 上 checkout 该分支并执行一键 Slurm 脚本。

---

## 32. 最终交付的验收标准

本次工程任务结束时，我希望能够做到：

### 本地

```bash
conda activate pykt

bash scripts/yousician_benchmark/run_local.sh
```

通过测试及 smoke test 后，将完成的代码 commit 和 push 到 GitHub。

### HPC

```bash
git fetch origin

git switch feature/yousician-fmkc-benchmark

git pull --ff-only
```

配置真实数据路径、Conda 环境和 Slurm 资源参数。

随后只需要执行：

```bash
bash scripts/yousician_benchmark/submit_all.sh \
  --train-student-sizes 60 120 240 \
  --include-full \
  --fold 0 \
  --seed 3407
```

即可启动完整实验流水线。

不需要 Codex。

不需要人工分别编写八条训练命令。

不需要手动对每个模型导出预测。

不需要手动统计长尾指标。

所有模型验证、checkpoint 选择、预测导出和统计分析都由明确的独立程序完成。

如果出现失败，能够根据 task manifest 精确恢复。

实验完成后可以通过：

```bash
bash scripts/yousician_benchmark/collect_results.sh \
  --run-dir <run_directory> \
  --mode summary
```

获得适合下载到本地分析的完整结果归档。

最终需要提交一份真实、经过本地验证的 HPC deployment guide，并明确列出尚未在真实 Slurm 集群验证的事项。