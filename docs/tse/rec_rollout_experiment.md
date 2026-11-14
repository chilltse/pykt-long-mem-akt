# PianoKT 实验开发任务：Five-Level μ-Probe + Model-Based Monte Carlo Segment Recommendation Benchmark

## 0. 任务定位与总体要求

请作为熟悉 Knowledge Tracing、AKT、Model-Based Planning、PyTorch、MIDI 数据处理和科研实验设计的高级研究工程师，基于我现有的两个仓库，实际实现一套完整、可复现、严谨的实验系统。

**本任务要求直接阅读代码、实现功能、运行测试并生成实验报告，不是只输出技术方案或伪代码。**

研究主题：

**Model-Based Piano Practice Recommendation via Knowledge Tracing and Counterfactual μ-Probe Evaluation**

核心思想：

给定学生的真实练习历史，冻结已经训练好的 AKT-FMKC 模型，对曲库中的每个候选 MIDI Segment 进行 Monte Carlo Rollout，模拟学生完成整段练习后的状态变化。

然后使用固定的 KC Probe Bank 测量练习前后模型预测正确率的变化。

**每一个 KC vector 都要在五个指定难度标量 μ 下分别接受 Probe 测试。**

最终比较 Full Segment MC、One-step Gain、Sweet Spot 和 Random 等方法。

本阶段不训练 RL、Q-learning、Actor、Critic，也不实现 PETS。

必须优先复用现有工程能力，保持结构简洁，不破坏当前项目。

---

# 1. 仓库与目标分支

## 1.1 PyKT

Repository：

当前工程

重点文件：

- `pykt/models/akt_fmkc.py`
- `pykt/models/akt_fmkc_runner.py`
- `pykt/models/akt_cache.py`
- `pykt/models/akt_state_store.py`
- `pykt/serving/encoder.py`
- `pykt/serving/engine.py`
- `pykt/serving/rollout.py`
- `pykt/serving/schemas.py`
- `pykt/serving/config.py`
- `docs/akt_fmkc_incremental_usage.md`

## 1.2 PianoKT

本地路径在：/Users/tse/my_code/pianokt

Repository：

https://github.com/chilltse/pianokt.git

Branch：

`feature/3_layer_kc`

重点文件：

- `backend/pianokt_backend/pykt_integration.py`
- `backend/pianokt_backend/pykt_recommendation.py`
- `backend/pianokt_backend/pykt_catalog.py`
- `backend/pianokt_backend/pykt_history.py`
- `backend/pianokt_backend/pykt_runtime.py`
- `backend/pianokt_backend/pykt_sync.py`
- `backend/pianokt_backend/pykt_verify.py`
- `backend/pianokt_backend/alignment/feature_extraction/reference.py`
- `backend/pianokt_backend/alignment/feature_extraction/versions/v1_2_0.py`
- `docs/PYKT_RECOMMENDATION.md`
- `docs/PYKT_INTEGRATION.md`

如果可以用本地优先用本地的。不然的话，首先检查当前仓库 HEAD、工作区、分支、依赖、checkpoint、服务配置及数据来源。

如果本地代码已经更新，以实际代码为准，并记录差异。

不得覆盖未提交修改。

所有新增代码优先位于 PianoKT 独立实验模块。只有必要时才为 PyKT 增加只读实验接口。

禁止未经授权的生产部署、数据库写入、在线模型状态修改或 checkpoint 修改。

---

# 2. 核心研究问题

本实验需要回答以下问题。

**RQ1：完整 Segment MC 是否比简单难度匹配和独立 Onset 价值近似产生更好的模拟推荐决策？**

**RQ2：完整 Segment 内部的连续状态更新是否对最终 Probe Gain 具有重要影响？**

**RQ3：不同 μ 条件下的 Probe 预测变化是否不同？这些差异是否具有稳定性？**

**RQ4：MC 样本数量如何影响推荐排序、估值方差、Regret 和计算成本？**

**RQ5：当前 AKT 的模拟 Gain 是否可能只是历史更新、窗口截断、题目记忆或模型预测不一致性造成的？**

需要事先区分三个结论层次：

1. **Model-internal optimization**：算法是否更擅长优化当前冻结 AKT 定义的模拟目标。
2. **Model robustness**：这种优势是否对随机种子、Probe 选择、历史长度、模型版本等因素稳健。
3. **Real learning effectiveness**：推荐是否真正提高人类学生的未来演奏表现。

本次实验主要完成前两层。

不能将第一层直接解释为第三层。

---

# 3. 模型定义与 μ Override

这是本实验最重要的技术要求。

## 3.1 原始 AKT-FMKC 的编码机制

当前模型在 `encode_question()` 中使用：

```python
u = torch.cat(
    [emb(attrs[..., i]) for i, emb in enumerate(self.field_embeddings)],
    -1
)

c = self.base_head(u)

mu = self.question_scalar(pid) * (pid != 0).unsqueeze(-1)

question = c + mu * self.question_drift_head(u)
```

这里：

- `attrs`：六个 KC 字段的分类编码。
- `u`：拼接后的 KC embedding。
- `c`：由 KC embedding 形成的基础题目表示。
- `pid`：question identity。
- `mu`：模型训练得到的 question-specific latent scalar。
- `question_drift_head(u)`：与 KC 结构相关的 drift direction。

`u` 与 `mu` 不是同一变量。

不能错误地将 μ 当作第七个 KC 字段。

## 3.2 我的 Probe 定义

从实际 PianoKT 曲库中提取所有有效 KC vector，形成去重集合：

\[
\mathcal V=\{v_1,v_2,\ldots,v_K\}
\]

对于每一个固定的 KC vector，分别使用：

\[
\mathcal D=\{-0.5,-0.25,0,0.25,0.5\}
\]

构造：

\[
\mathcal P=\{(v_i,\mu_j)\mid v_i\in\mathcal V,\mu_j\in\mathcal D\}
\]

也就是说：

**同一个 KC vector 必须测量五次，唯一变化的因素是 μ。**

例如：

```text
KC vector = [False, 2, True, 30, 2, 1]

Probe 1: μ = -0.50
Probe 2: μ = -0.25
Probe 3: μ =  0.00
Probe 4: μ = +0.25
Probe 5: μ = +0.50
```

以上只是结构示例，实际向量必须来自真实有效的 labeler 输出。

不要从真实曲库中搜索五个不同难度的 KC 来代替这个实验。

不要根据 checkpoint 中原始 μ 接近哪个值来选择五组不同 PID。

不要修改训练词表来生成新的题目身份。

## 3.3 需要实现的只读推理接口

请实现类似：

```python
predict_next_with_mu_override(
    cache,
    attrs,
    pid,
    mu_override
) -> float
```

或者经过工程审查后选择更合理的同等接口。

要求：

1. 模型处于 `eval()`。
2. 使用 `torch.inference_mode()` 或等价的无梯度方式。
3. 不修改模型参数。
4. 不修改 `question_scalar.weight`。
5. 不修改 checkpoint。
6. 不修改传入的 StudentCache。
7. 不将 Probe 作为作答历史写入。
8. 不执行 `observe()`。
9. 不推进真实或模拟历史的 `next_position`。
10. 使用原始模型的 question encoding、attention 和 prediction head。
11. 只能改变当前 Probe 的 question-specific μ。
12. 其他所有参数与模型原始预测完全一致。

对于 `drift="response"`，μ 在历史交互编码中也可能出现。

**这里只覆盖当前被查询 Probe 的 μ，不重新编码已有历史，也不改变已发生的 Segment 内部模拟事件。**

对于 `pid=0` 的未知组合，原模型会将 learned μ 置零。

主实验优先要求 `pid>0`、字段均已知，避免混淆未知题目退化机制与 μ Override。

未知组合可在独立敏感性实验中分析，但不得混入主结果。

## 3.4 数值等价性必须验证

这是强制验收项。

对于任意有效已知 KC vector：

```python
mu_original = model.question_scalar(pid)
```

验证：

```text
predict_next_with_mu_override(cache, attrs, pid, mu_original)
≈
model.predict_next(cache, attrs, pid)
```

在相同设备和精度下，误差应处于合理浮点容差范围。

建议在 CPU FP32 下以 `atol=1e-6` 作为初始测试标准，按实际数值实现调整并说明理由。

如果无法达到等价性，不能继续执行正式 Probe Gain 实验。

还要验证：

```text
mu_override = 0
```

的前向计算符合原始 question drift 为零时的数学定义。

对于 `drift="none"`，μ Override 不应该改变预测输出。

如果实际 checkpoint 不具备有效的 μ drift 机制，则明确报告当前模型不支持有意义的五档 μ 实验，不要伪造差异。

## 3.5 五档 μ 的解释边界

μ 是 learned latent scalar，不保证其大小与真实演奏难度严格单调对应。

所以本实验将五个值称为：

**Five-Level Question Drift Probes**

不要默认称为五个经过验证的真实难度等级。

请额外统计：

- learned μ 的实际 min/max/mean/std。
- 五个指定 μ 是否处于训练分布内。
- 每个 KC 的预测概率随 μ 改变的方向。
- \(p(\mu)\) 是否单调。
- 单调性违反比例。
- μ 对不同 KC 预测的敏感程度。

这些是模型诊断，不改变五档 Probe 的正式定义。

---

# 4. 从 PianoKT 曲库构建 Probe Bank

## 4.1 曲库数据来源

复用：

```text
pykt_catalog.py
pykt_integration.py
ReferenceCandidates
CatalogCandidateProvider
```

读取 PianoKT 实际已发布的 Segment MIDI。

不使用曲名匹配代替稳定 Segment identity。

必须保留：

```text
song_id
segment_id
segment_index
song_version
midi_sha256
kc_labeler_version
```

如果现有曲库要求 `standalone_reference` provenance sidecar，继续遵守，不得绕过。

如果某些 Segment 必须依赖完整原曲上下文提取 KC，却缺少可信证据，跳过并报告原因。

## 4.2 KC Labeler

使用已经存在的 KC labeler v1.2.0。

原始字段：

```python
KC_FEATURE_NAMES = [
    "has_black_key",
    "simultaneous_note_count",
    "both_hands",
    "onset_interval_bin_cs",
    "change_count",
    "mean_move_bin",
]
```

必须保留原始 bool/int/null 等 JSON 类型。

使用现有经过验证的字段映射转换到 PyKT checkpoint 所需的模型字段名称。

不得新建另一套不兼容编码。

## 4.3 Probe Bank 生成规则

收集所有有效 Segment 中的 onset KC vectors。

按照六维向量的**类型敏感 canonical identity** 去重。

同一 KC vector 在多个 Segment 中出现时，主 Probe Bank 只保留一份。

主实验使用等权重 KC macro-average。

如果共有 K 个 distinct KC vectors，正式 Probe 数量应为：

\[
5K
\]

**主实验原则上保留全部有效、已知词表覆盖的 KC，不人为限定为 100 个。**

如果数量过大导致资源不足：

- 提供 `full` 模式，保留完整集合。
- 提供 `budgeted` 模式，通过事先固定的、可复现的规则选择代表性子集。
- 记录抽样规则、seed、coverage、实际 KC 数和 Probe 数。
- 所有推荐方法必须共用相同的 Probe Bank。
- 严禁在看到测试 Gain 后修改 Probe 集合。

输出：

```text
probe_bank.json
probe_coverage.json
probe_bank_digest
```

每项 Probe 应至少包含：

```text
probe_id
kc_vector
attrs
pid
mu_override
kc_identity
```

主实验需要预先报告已知 PID 覆盖率。如果已知组合覆盖率很低，则这项实验的结论仅适用于已知组合，必须明确这一限制。

---

# 5. Probe Score 和 Learning Gain

设学生真实历史为 \(H\)，KC 向量为 \(v\)，指定的 μ 为 \(d\)。

定义：

\[
p(H,v,d)=P_\theta(y=1\mid H,v,\mu=d)
\]

注意这是模型预测的正确概率，而不是实际 Accuracy。

## 5.1 单档 Probe Score

对于固定 μ：

\[
M_d(H)=\frac1K\sum_{v\in\mathcal V}p(H,v,d)
\]

## 5.2 五档宏平均

\[
M(H)=\frac15\sum_{d\in\mathcal D}M_d(H)
\]

也就是：

\[
M(H)=\frac1{5K}
\sum_{v\in\mathcal V}\sum_{d\in\mathcal D}p(H,v,d)
\]

这就是本实验的主要评价函数。

## 5.3 Segment 后的 Gain

给定候选 Segment：

\[
S=(q_1,\ldots,q_L)
\]

使用 KT MC 模拟其完整作答过程，得到模拟后历史 \(H'_S\)。

定义：

\[
G(H,S)=
\mathbb E_{\mathrm{MC}}[M(H'_S)-M(H)]
\]

五档分别为：

\[
G_d(H,S)=
\mathbb E_{\mathrm{MC}}[M_d(H'_S)-M_d(H)]
\]

主推荐分数：

\[
G(H,S)=\frac15\sum_{d\in\mathcal D}G_d(H,S)
\]

五档使用相同的 K 个 KC vector，不能因为某个档的表现不好而改变 KC 集合。

## 5.4 必须保存的指标

每个学生、Segment、μ 档都记录：

```text
p_before
mean_p_after
gain
gain_std
gain_sem
n_rollouts
```

同时保存宏平均结果。

对于同一次模拟轨迹，需要先完成所有 Probe 预测并形成 trajectory-level Gain，再进行跨轨迹平均和置信区间计算。

不能把同一轨迹中的 5K 个 Probe 当作 5K 个独立学生样本。

---

# 6. Segment Monte Carlo Rollout

## 6.1 正式模拟流程

当前学生状态：

\[
H_0
\]

Segment KC sequence：

\[
S=[v_1,\ldots,v_L]
\]

对于每次模拟：

```python
state = clone(H0)

for v in segment.kc_vectors:
    p = original_kt_predict_next(state, v)

    y = sample_bernoulli(p)

    state = original_kt_observe(
        state,
        v,
        response=y
    )

after_probe = evaluate_all_mu_probes(state)
```

其中：

**Segment 内部所有 onset 必须使用原始 learned μ 和正常 PID 编码。**

不允许对 Segment 里的 onset 使用五档 μ Override。

μ Override 只用于模拟之前和之后的 Probe Measurement。

每次 rollout 都从同一份初始 StudentCache 开始，不能继承上一次模拟的历史。

不能将模拟 response 写入任何持久化用户状态。

## 6.2 复用现有 PyKT 能力

当前服务已有：

```text
/kt/rollouts/evaluate
```

但它不直接支持 `mu_override` 的 forecast evaluation。

请比较以下两个方案：

A. 在原有 rollout API 中增加实验专用的只读 Probe 机制。

B. 在本地实验进程直接使用 `AKTFMKC`、`StudentCache`、原始 `predict_next()` 和 `observe()`，模拟后直接执行 μ Override Probe。

**优先选择 B，除非经过代码审查证明 A 的侵入性更小。**

因为当前实验主要是离线科研，不需要通过 HTTP 为同一状态连续查询成千上万个 Probe。

不要为了实验修改生产 HTTP API。

但必须保证新路径与现有 `run_rollout()` 对相同输入、相同 response trajectory 的状态和预测计算一致。

## 6.3 计算优化

由于 Probe 数量为 \(5K\)，不能简单针对每个 Probe 重复模拟整段 Segment。

正确做法是：

1. 模拟一次 Segment，获得一个最终 state。
2. 固定这个 state。
3. 对全部 \(5K\) 个 Probe 进行只读 inference。
4. 计算该轨迹的各档 Gain。
5. 重复下一条轨迹。

根据实际模型实现，考虑：

- 编码缓存。
- KC embedding 预计算。
- Probe mini-batching。
- GPU/CPU 批处理。
- 内存上限。
- 避免重复构造 student state。
- 避免在每个 μ 上重复计算相同的 KC 基础表示。
- 严格保持预测数值等价。

可优化实现，但不能为加速改变原始推理数学语义。

---

# 7. 四种推荐算法

所有算法必须使用同一学生状态、同一候选曲库、同一 Probe Bank。

## 7.1 Random

从有效 Segment 中随机选择。

固定 seed，支持多次重复。

## 7.2 Sweet Spot

复用当前 PianoKT 已有的难度匹配思路。

不推进历史，只在当前学生状态下预测候选 Segment 中各 onset 的成功概率。

计算：

```text
mean_predicted_probability
percentage_p_above_0_5
```

主 Sweet Spot baseline 使用：

```text
abs(mean_probability - 0.7)
```

越小越优。

同时保留 `percentage_p_above_0_5` 作为敏感性分析。

这只是固定的启发式 baseline，不代表 0.7 是经过验证的最佳练习难度。

不能根据最终测试结果反向修改 Sweet Spot 的中心值。

## 7.3 One-step Expected Gain

对每个 Segment 的每个 onset，均从相同初始学生状态 \(H\) 出发。

不使用该 Segment 前面 onset 的模拟结果更新当前 onset 的起点。

对于 onset \(q_i\)：

\[
p_i=P(y=1\mid H,q_i)
\]

构造：

\[
H_i^+=H\oplus(q_i,1)
\]

\[
H_i^-=H\oplus(q_i,0)
\]

计算：

\[
g_i=
p_iM(H_i^+)+(1-p_i)M(H_i^-)-M(H)
\]

主要使用：

\[
Score_{\mathrm{one-step}}(S)=\frac1L\sum_i g_i
\]

此外报告 sum 形式及按练习时长归一化的版本。

注意这种方法是**独立 onset 一步估值启发式**，不是 ExRec 训练后的长期 Q 函数。

并且 One-step mean 与 Full Segment total Gain 并不是完全相同的评价目标，因此需要同时报告它们的时间/长度归一化敏感性结果。

## 7.4 Full Segment MC

对整个 Segment 按实际 onset 顺序推进 KT 状态。

对于每条 trajectory，完成全部 onset 后计算一次固定 Probe Gain。

通过 MC 得到：

\[
Score_{\mathrm{MC}}(S)=\widehat G(H,S)
\]

按照期望 Gain 排序候选 Segment。

不能默认所有 onset 正确。

不能只模拟单步就声称进行了 Full Segment Rollout。

不能将每个 onset 的长期 Q 值直接相加。

---

# 8. 学生数据与历史状态

## 8.1 优先真实学生历史

使用 PianoKT development/test 环境中实际已完成、READY 且经过现有同步机制验证的练习记录。

尊重：

```text
learner_key
attempt ordering
source_environment
kc_labeler_version
model_version
feature_version
history_revision
```

只能读取当前 checkpoint 对应的合法历史。

如果多个 checkpoint 来自同一个学生，统计时必须保留同一学生归属关系。

不能把多个 checkpoint 当作不同独立学生。

## 8.2 历史快照

对于某一时间点 \(t\)，仅使用：

\[
H_t
\]

不允许使用 \(t\) 之后真实发生的练习记录。

如需重建不同历史截点，请在本地独立缓存中重放历史。

禁止回滚或重写线上 StudentStateStore。

## 8.3 数据不足时

如果真实学生很少，可以完成以下两种运行模式：

- `mock`：使用合成数据验证实验工程正确性。
- `real-checkpoint`：使用真实 checkpoint 和实际可获取的学生历史运行模型实验。

必须区分两类结果。

不能把 synthetic student 的模拟结论称为真实教育效果。

如果缺少真实 checkpoint、学生历史或曲库访问凭据，明确报告缺失条件，不允许静默采用 mock 数据代替正式实验。

---

# 9. 公平实验协议

## 9.1 候选集合

对每个学生，四种方法必须看到完全相同的有效 Segment 集合。

可以根据事先定义的业务规则统一排除：

- 不可解析 MIDI。
- 无有效 KC。
- 缺少可靠 provenance。
- 超过预先设置的练习时长上限。
- 超过资源限制且无法完整模拟的片段。
- 与 checkpoint 编码不兼容的片段。

不能先根据 Sweet Spot 区间筛选，再让 Full MC 只在筛选后的集合中搜索。

候选过多时，使用预先固定的、与预测 Gain 无关的采样规则。

## 9.2 时间长度控制

不同 Segment 可能包含不同数量的 onset。

必须记录：

```text
duration_seconds
onset_count
raw_gain
gain_per_second
gain_per_onset
```

主要分析可使用时长接近的 Segment。

同时提供宽松时长条件下的 sensitivity analysis。

不能让长片段仅因包含更多练习机会而占据不公平优势，同时又声称这种优势来自更合适的教学结构。

## 9.3 训练、选择与评估隔离

区分：

```text
development students
benchmark students
```

如果存在真实未来标签，另保留 held-out future data。

Probe Bank、难度规则和超参数必须在最终 benchmark 之前固定。

不允许查看测试结果之后重新设计 Probe，使 Full MC 看起来更好。

---

# 10. Monte Carlo 计算预算

先支持：

```yaml
selection_mc: 32
evaluation_mc: 256
reference_mc: 512
```

这些是实验目标值，不是每次 API 请求的 trajectory 数。

按照实际接口和资源限制进行安全分批。

也要支持：

```yaml
mc_budgets: [4, 8, 16, 32, 64, 128]
```

用于比较不同预算的效果。

推荐阶段与最终评估阶段必须使用独立 RNG streams。

建议由以下信息生成稳定的种子：

```text
experiment_seed
student_checkpoint_id
segment_id
phase
trajectory_index
```

使用稳定的序列化和哈希规则，不得依赖 Python 进程随机化的内置 `hash()`。

允许使用 Common Random Numbers 降低候选间比较方差，但必须记录具体耦合方式，且不得引入跨阶段样本泄漏。

如果同一个 Segment 在不同算法中被选中，可复用同一独立评估结果作为配对比较，但不能复用 selection 阶段的模拟结果。

---

# 11. 实验 A：推荐方法比较

每个学生 checkpoint 下：

1. 获取当前学生状态。
2. 固定全部有效候选。
3. 固定 Five-Level μ Probe Bank。
4. 分别运行四个推荐算法。
5. 每个算法选择 Top-1 Segment。
6. 对每个被选中的 Segment 执行独立评估 MC。
7. 汇总所有学生。

主要指标：

\[
\operatorname{MeanGain}=
\frac1U\sum_u G(H_u,S_u)
\]

还需要：

```text
median_gain
gain_std
95_percent_confidence_interval
top1_selection_stability
runtime_p50
runtime_p95
simulated_onset_count
probe_evaluation_count
```

所有方法都必须报告同样口径的指标。

## 11.1 Independent Evaluation

Full MC 是直接优化当前 KT Probe Gain 的方法。

因此，不能直接复用它选曲时的 MC 结果来报告效果。

必须使用独立的 evaluation seed 和 trajectories。

报告应明确：

这种评估降低 MC selection noise 带来的乐观偏差，但仍然属于同一个 KT 模拟器内部的评价。

它不能证明真实教学效果。

---

# 12. 实验 B：Reference Oracle 与 Regret

为了判断 32 次 MC 是否足够，需要对候选集合构造高预算参考估值。

对于每个 Segment：

\[
G_{\mathrm{ref}}(H,S)
\]

使用更高的 MC 次数，例如 512 或更多。

选择：

\[
S^*_{\mathrm{ref}}=
\arg\max_S G_{\mathrm{ref}}(H,S)
\]

然后定义：

\[
Regret=
G_{\mathrm{ref}}(H,S^*_{\mathrm{ref}})
-
G_{\mathrm{ref}}(H,S_{\mathrm{method}})
\]

由于参考分数仍是随机估计，不能把它称为真实 Oracle 或真实最优解。

如果最高几个候选差异小于 MC 误差，应自适应增加这些候选的参考采样次数。

必要时使用独立评估样本重新比较最终 Top-1 与参考候选，避免高预算参考最大值本身的 winner's curse。

若计算预算不足以覆盖全部候选，必须标记为：

`restricted_reference`

不得报告成全曲库全局最优。

输出：

```text
mean_regret
median_regret
epsilon_optimal_rate
ranking_spearman
```

其中 epsilon-optimal 应根据预先指定的、具有实际意义的 Gain 容差判断，并给出对不同 epsilon 的敏感性分析。

---

# 13. 实验 C：五档 μ Probe 的变化

这是本次实验重点。

对每个推荐方法选择的 Segment，分别计算：

\[
G_{-0.5},G_{-0.25},G_0,G_{0.25},G_{0.5}
\]

以及：

\[
G_{\mathrm{macro}}
\]

输出完整表格：

```text
student_checkpoint
method
segment_id
mu_level
p_before
p_after_mean
gain
gain_std
```

分析：

1. 不同 μ 档的初始预测概率。
2. 不同 μ 档的最终预测概率。
3. 不同 μ 档的 Gain。
4. 不同学生能力水平的 Gain。
5. 每个 KC vector 的五档预测响应曲线。
6. 五档结果的单调性。
7. μ 与 Gain 的关系是否随不同 KC 变化。
8. Five-Level Macro Gain 是否被极少数 KC 主导。

尤其需要关注：

**某个 Segment 的总体 Gain 很高，究竟是所有五档都提高，还是只有某一档发生了大幅变化？**

同时检查分档预测响应是否受 Sigmoid 饱和效应影响。

不能仅凭某档 Gain 大，就断言这一档对应的真实演奏难度最适合学生。

---

# 14. 实验 D：顺序效应和模型状态更新诊断

## 14.1 No-op Control

不追加任何 onset。

要求：

\[
M(H')-M(H)\approx0
\]

如果不为零，检查推理状态污染、随机性或计算错误。

## 14.2 Forced Correct / Forced Wrong

分别对同一 Segment 模拟：

```text
all_correct
all_wrong
sampled_responses
```

观察 Probe Gain。

前两种只是诊断，不属于正常推荐评分。

## 14.3 Onset 顺序

比较：

```text
original_order
reversed_order
shuffled_order
```

保持 onset KC 多重集合一致，仅改变顺序。

这些重排可能不构成真实可演奏 MIDI，因此只能用于模型状态敏感性诊断。

需要统计不同顺序的 Gain 差异及其 MC 不确定性。

## 14.4 Attention Window

记录：

```text
history_length
seq_len
cache_length
rollout_length
```

检查长历史和长 Segment 是否使旧记录被窗口截断，从而人为改变 Probe 预测。

需要比较多个历史长度区间。

不能将窗口移动产生的变化直接解释为知识增长。

## 14.5 Exact KC Overlap

由于 Probe Bank 来自曲库，某些 Probe KC vector 可能与正在模拟的 Segment 完全相同。

这可能产生题目重复或短期熟悉效应。

因此必须报告：

```text
probe_overlap_ratio
gain_on_exact_overlapping_kcs
gain_on_nonoverlapping_kcs
```

主评价仍使用固定 Probe Bank，保证各候选可比较。

上述 overlap 分析只能作为诊断分组，不能对不同候选采用不同的主评分目标。

如有足够数据，额外构建一个事先固定、按歌曲来源隔离的 Probe 集合，用于检查跨曲目迁移。

注意 KC identity 与实际音乐题目不完全相同；KC 不重叠不等于严格的认知技能独立，KC 相同也不等于完全相同的旋律。

---

# 15. 实验 E：静态贝叶斯负对照

这是验证理论正确性的辅助实验。

实现一个非常简单的 toy student model：

- 学生潜在能力固定。
- 回答提供能力信息。
- 不存在练习引起的真实学习转移。
- 使用正确的贝叶斯更新。
- Probe 的预测分布与模型自洽。

验证：

\[
\mathbb E[M(H')\mid H]=M(H)
\]

在足够多 MC 样本下应近似成立。

该实验用于表明：

**单纯的信息更新，不应系统性制造正的期望 Probe Gain。**

然后检查 AKT 是否出现持续正 Gain。

如果 AKT 出现正 Gain，只能说明它的预测更新不是这个静态贝叶斯模型的鞅过程，不能立即证明它学到了真实因果学习动力学。

---

# 16. 实验 F：真实未来表现验证（有数据时执行）

若可取得实际学生练习历史及之后的真实作答：

```text
H = 练习之前的历史
S = 实际完成的 Segment
Y_future = 后续真实作答
```

使用 H 和 S 的 KC sequence，通过 MC 模拟完成 S 后，对未来题目预测正确率。

与真实 future response 比较：

```text
Brier Score
Log Loss
Calibration
```

真实 future Probe 如果仅提供 KC vector 而没有实际 μ 干预条件，则使用正常 checkpoint learned μ 对其进行预测评价。

**不能将人为 μ Override 得到的五档预测，直接与未经对应干预的真实题目标签当作五组独立 Accuracy 进行比较。**

同时不能将观察数据中“做过 A 的学生后续表现更好”直接解释为 A 的因果学习收益，因为可能存在学生能力、动机和练习选择偏差。

如果数据不足，明确标记：

`not_evaluated_insufficient_real_outcomes`

---

# 17. 统计分析

所有策略比较应采用 paired design。

对每个独立学生计算方法间 Gain 差值。

如果同一学生拥有多个 checkpoint，采用 student-level cluster bootstrap。

至少报告：

```text
mean_difference
95% confidence interval
number_of_students
number_of_checkpoints
number_of_trajectories
```

建议 Bootstrap 重复次数为 2000。

不能把一次模拟中的不同 Probe 当作独立学生样本。

不能把同一个学生的多条 trajectory 当作独立真实学生。

若同时进行多个 μ 档及多个方法的显著性检验，应区分主要预注册比较与探索性比较，并在适当时进行多重比较校正。

主要比较预先指定为：

**Full Segment MC vs One-step Gain**

次要比较：

**Full Segment MC vs Sweet Spot**

Random 作为基础参考。

同时报告实际 effect size，不仅报告 p-value。

---

# 18. 代码结构

请优先在 PianoKT 中创建独立、轻量的实验模块。

建议：

```text
backend/pianokt_backend/experiments/
    __init__.py
    mu_probe.py
    probe_bank.py
    candidate_loader.py
    student_checkpoints.py
    estimators.py
    mc_runner.py
    metrics.py
    diagnostics.py
    reporting.py
    cli.py
```

不要机械照搬此结构。如果职责可以合并，应采用更简洁的实现。

必须复用现有：

```text
PyKT checkpoint loader
KCEncoder
ReferenceCandidates
CatalogCandidateProvider
StudentCache
AKTFMKC.predict_next
AKTFMKC.observe
```

不要复制原模型实现。

如果实验专用 μ Probe 需要读取原模型的内部 question encoder，尽量通过单独的只读函数共享原始计算路径。

必须有测试证明：

在未使用 override 时，新路径与原始路径数值等价。

---

# 19. CLI 与配置

提供统一实验 CLI，例如：

```bash
PYTHONPATH=backend python -m pianokt_backend.experiments.cli \
  --config configs/kt_mc_mu_probe.yaml \
  --mode offline \
  --output output/kt_mc_mu_probe/run001
```

至少支持：

```text
--mode mock
--mode offline
--mode dev
--dry-run
--student-limit
--candidate-limit
--selection-mc
--evaluation-mc
--reference-mc
--seed
```

`mock` 只验证逻辑。

`offline` 使用本地可信数据和真实模型。

`dev` 只能读取经过明确确认的 development/test 数据源。

禁止使用 production credentials 运行可能修改用户状态的路径。

实验默认不写数据库。

---

# 20. 推荐默认配置

```yaml
experiment:
  seed: 3407
  student_limit: 100
  candidate_limit: 100

probe:
  mode: five_level_mu_override
  mu_levels:
    - -0.50
    - -0.25
    - 0.00
    - 0.25
    - 0.50
  kc_source: published_catalog
  use_all_distinct_known_kc: true
  require_known_pid: true
  require_known_fields: true
  weighting: uniform_per_distinct_kc

selection:
  methods:
    - random
    - sweet_spot
    - one_step_gain
    - full_segment_mc

  sweet_spot_center: 0.7
  mc_trajectories: 32

evaluation:
  mc_trajectories: 256
  reference_mc: 512
  seed: 9001
  bootstrap_replicates: 2000
  confidence_level: 0.95

diagnostics:
  noop: true
  forced_correct: true
  forced_wrong: true
  onset_order: true
  history_window: true
  probe_overlap: true
  mu_monotonicity: true
  static_bayesian_control: true

safety:
  modify_model_weights: false
  persist_rollouts: false
  write_database: false
  production_deployment: false
```

这些是初始配置，必须在运行前验证资源与模型兼容性。

支持轻量 smoke preset。

对于计算昂贵的全量实验，优先实现可恢复的断点续算、增量结果落盘和确定性运行。

断点续算不得产生重复结果或混用不同模型版本。

---

# 21. 输出与可复现性

每次实验输出：

```text
output/kt_mc_mu_probe/<run_id>/
    config_resolved.yaml
    manifest.json
    probe_bank.json
    kc_coverage.json
    mu_distribution.json
    student_checkpoints.json
    candidate_inventory.csv
    selection_scores.csv
    selected_segments.csv
    probe_predictions.csv
    probe_gain_by_mu.csv
    evaluation_metrics.csv
    reference_regret.csv
    diagnostics.json
    figures/
    report.md
```

如 Probe 很大，可以使用 Parquet 等更合适的格式保存完整预测数据。

Manifest 至少包含：

```text
pianokt_commit
pykt_commit
checkpoint_sha256
model_version
feature_version
kc_labeler_version
probe_bank_digest
catalog_digest
student_snapshot_digest
all_random_seeds
runtime_configuration
```

不允许把认证 token、数据库密码或真实用户身份信息写入报告。

学生应使用匿名实验 ID。

## 必须生成的图表

1. 四种算法的平均模拟 Gain 对比。
2. 五档 μ 的 Before / After Prediction。
3. 五档 μ 的 Gain 曲线。
4. μ 与预测概率的关系。
5. μ 单调性诊断。
6. MC 次数与 Regret。
7. MC 次数与推荐排序稳定性。
8. 计算耗时与估计效果。
9. 不同学生初始能力区间的 Gain 分布。
10. Probe 与 Segment KC overlap 对 Gain 的影响。

必须清楚标记模拟结果、真实测量结果和未完成验证。

---

# 22. 单元测试与验收

至少完成以下测试。

### μ Probe

- 固定 KC vector，五档 μ 均产生独立预测调用。
- μ Override 不修改 checkpoint。
- μ Override 不修改 StudentCache。
- learned μ override 与原始 predict_next 数值等价。
- μ=0 的编码语义正确。
- `drift=none` 时结果正确。
- 未知 PID 不会被错误地赋予训练词表身份。
- Probe 数量正确等于 `5 × known_distinct_KC_count`。
- 同一 KC 的五档仅有 μ 不同。

### Monte Carlo

- Rollout 使用原始模型 learned μ。
- Rollout 不使用 Probe 的 μ Override。
- 每次 trajectory 从相同状态开始。
- Bernoulli outcome 按当前状态预测概率采样。
- 每一步 observe 后下一步预测使用更新的模拟历史。
- 模拟历史不写入真实 StudentStateStore。
- 不同候选不会互相污染。
- Probe 预测只读。
- 同一 seed 结果可复现。
- 选择阶段和评价阶段随机样本独立。

### 推荐对照

- 四种方法使用同一 candidate set。
- Sweet Spot 不提前过滤 MC 的候选集合。
- One-step 正负结果加权正确。
- One-step 不意外推进后续 onset 的初始状态。
- Full MC 正确模拟完整顺序。
- Segment 长度和时长归一化正确。
- Reference Regret 计算正确。
- 独立评估不复用 selection 样本。

### 模型一致性

- 原始 PyKT golden regression 通过。
- 当前 PianoKT PyKT integration tests 通过。
- 模型版本检查通过。
- KC feature ordering 一致。
- checkpoint 冻结。
- No-op Gain 为零。
- μ Probe 的 query 不推进 `next_position`。

### 报告

- CSV/JSON 与统计指标一致。
- 数据不足时明确报告。
- 不伪造 Probe、学生或实验结果。
- 所有报告可根据 manifest 和命令复现。
- 生产推荐逻辑没有变化。

---

# 23. 开发执行顺序

请按顺序实际执行。

## Phase 1 — Audit

核查两个仓库的最新代码，确认：

- 模型内部 μ 的实际作用。
- checkpoint drift 模式。
- StudentCache 的只读保证。
- MIDI KC 标注链路。
- 实际可用曲库。
- 实际可用学生数据。
- 当前 rollout 的资源限制。

记录检查结果。

## Phase 2 — μ Probe

先实现最核心的：

```python
predict_next_with_mu_override()
```

完成原模型数值等价测试。

随后实现：

```python
evaluate_mu_probe_bank(state, kc_vectors, mu_levels)
```

保证只读、高效、可批处理。

这是后续实验的基础，必须先通过测试。

## Phase 3 — Full Segment MC

实现完整 Segment 状态模拟与五档 Probe Gain。

先对单学生、单 Segment 跑通。

检查模拟之前和之后持久化学生状态完全一致。

## Phase 4 — Baselines

实现 Random、Sweet Spot、One-step、Full MC。

## Phase 5 — Independent Evaluation

实现独立 MC、Reference Regret、稳定性和计算预算测试。

## Phase 6 — Diagnostics

完成 μ 单调性、No-op、forced outcomes、onset order、历史窗口和 Probe overlap。

## Phase 7 — Reporting

生成完整报告与图表。

## Phase 8 — Validation

运行现有相关回归测试及新增测试。

如果实际 checkpoint 可用，完成真实模型的 smoke test。

如果 development 数据源可用，运行小规模真实曲库与真实学生历史实验。

资源不足时不能声称完成全规模 Benchmark。

---

# 24. 最终必须交付

请最终提供：

1. 仓库审查结果与实际 commit。
2. 新增和修改的文件。
3. μ Override 的实现原理。
4. 与原始 predict_next 的数值等价测试结果。
5. 实际 Probe Bank 的 KC 数、Probe 数和覆盖率。
6. learned μ 分布与五档设定的关系。
7. 四种推荐方法的实现说明。
8. 完整测试命令与结果。
9. 实际运行的实验配置。
10. 各推荐方法的 Gain、Regret、稳定性和计算成本。
11. 五档 μ 的 Probe Before / After / Gain。
12. 模型偏差诊断结果。
13. 实验报告位置与复现命令。
14. 未完成项及原因。
15. 对下一步是否值得使用多 Segment Planning 的判断。

所有报告必须区分：

- 已实现。
- 通过单元测试。
- 通过真实 checkpoint 测试。
- 使用真实 MIDI 曲库。
- 使用真实学生历史。
- 验证真实学生学习效果。

以上状态不能互相替代。

如果本地环境具备条件，请直接完成真实小规模实验，不要只停留在 mock。

如果条件不具备，也应完成可运行的实现和严格的 mock / regression 验收，并提供准确的缺失条件。

---

# 25. 科研结论的边界

本实验的主要评分目标是：

\[
G(H,S)=
\mathbb E_{\mathrm{KT}}
[M(H'_S)-M(H)]
\]

其中：

\[
M(H)=
\frac1{5K}\sum_{v\in\mathcal V}
\sum_{\mu\in\mathcal D}
P_\theta(y=1\mid H,v,\mu)
\]

这衡量的是：

**冻结 AKT 模型在一组固定 KC × μ Counterfactual Probes 上预测成功概率的期望变化。**

它不自动等于真实知识增长、长期技能掌握度或者因果教学收益。

Full Segment MC 直接优化这个评分，因此在同一模型环境中表现更好并不令人意外。

研究重点应该是：

- 相比简单 One-step 和 Sweet Spot，它额外获得了多少价值？
- 这种优势是否超过 MC 估计噪声？
- 这种优势能否在较低计算预算下稳定获得？
- 对五档 μ 是否存在系统性的不同作用？
- 这种优势是否源于真实的历史条件依赖，而不是模型更新伪影？
- 它是否具有跨 Probe、跨模型和真实未来表现的稳健性？

**本实验的成功不是要求 Full MC 一定最好，而是建立一个科学、可重复、能够检验或证伪这些假设的 Benchmark。**

最终目标是决定：当前 PianoKT 是否值得直接采用 KT + Full Segment MC 进行推荐，以及未来是否有必要进一步研究 Q-learning、PETS 或多 Segment 序列规划。

请现在开始实际检查代码并实施，不要只返回方案。
