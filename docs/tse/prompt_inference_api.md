# Implement AKT-FMKC Stateful Inference Service with Persistent Student Cache and Ephemeral Monte Carlo Rollouts

## 1. 项目背景与目标

仓库：https://github.com/chilltse/pykt-long-mem-akt

工作分支：`feature/akt_fmkc_incremental_inference`

请先仔细阅读该分支现有的模型、增量推理、缓存、预处理和测试代码，然后直接实现一个可在本地运行、未来可接入 PianoKT 的 AKT-FMKC 在线推理服务。

假设已经存在训练好的 `qid_model.ckpt` 和配套配置文件。Checkpoint 的实际路径由环境变量提供，不能硬编码，也不要重新训练模型。

本次开发的核心目标是：

1. 复用训练时的 KC 编码规则，将原始 KC 向量转换为 `attrs` 和 `pid`。
2. 为每个学生管理独立、可恢复、可持久化的 AKT KV Cache。
3. 支持不修改学生状态的单步及批量候选预测。
4. 支持真实练习事件的增量更新和幂等写入。
5. 支持从真实学生状态创建临时模拟分支。
6. 支持 Monte Carlo Rollout，连续模拟一整段音乐的事件序列，并保留模拟后的临时 Cache 供后续预测。
7. 所有模拟操作绝对不能修改真实学生的事件日志、持久化 Cache 或模型权重。
8. 暴露类型明确的 FastAPI HTTP 接口，同时提供 Python 内部调用 API。
9. 添加自动化测试、真实 checkpoint smoke test、API 文档及可直接执行的示例。

请直接完成实现、运行可运行的测试并汇报修改结果，不要停留在设计说明或伪代码。

## 2. 必须先阅读的代码

重点检查：

- `pykt/models/akt_fmkc.py`
- `pykt/models/akt_cache.py`
- `pykt/models/akt_state_store.py`
- `pykt/models/akt_fmkc_runner.py`
- `pykt/preprocess/vector_kc_preprocess.py`
- `pykt/preprocess/student_sequences.py`
- `tests/test_akt_fmkc_incremental.py`

当前核心接口包括：

- `load_fmkc_checkpoint(path, device)`
- `model.empty_cache(model_version, feature_version)`
- `model.predict_next(cache, attrs, pid, model_version=..., feature_version=...)`
- `model.observe(cache, attrs, pid, response, event_id, model_version=..., feature_version=...)`
- `StudentStateStore.recover(student)`
- `StudentStateStore.predict_next(student, attrs, pid)`
- `StudentStateStore.observe(student, attrs, pid, response, event_id)`
- `StudentCache.to()`
- `StudentCache.state_dict()`
- `StudentCache.from_state_dict()`

应优先复用现有实现，避免复制一套新的 Attention 或 KV Cache 逻辑，也不要改变 AKT-FMKC 的数学定义。

特别注意：`model.observe()` 返回新的 Cache 和观察前的预测概率，不等于 `StudentStateStore.observe()` 的持久化操作。这是实现临时 Rollout 的关键。

## 3. 建立统一的 Inference Engine

实现类似以下 Python 服务层：

`AKTFMKCInferenceEngine`

建议提供：

- `encode_kc(kc_vector) -> EncodedQuestion`
- `get_student_state(student_id) -> StudentStateInfo`
- `predict_next(student_id, kc_vector) -> Prediction`
- `predict_candidates(student_id, candidates) -> Predictions`
- `observe(student_id, event) -> ObservationResult`
- `observe_batch(student_id, events) -> BatchObservationResult`
- `fork_student(student_id) -> BranchHandle`
- `simulate_step(branch_id, kc_vector, response) -> SimulationResult`
- `predict_branch(branch_id, candidates) -> Predictions`
- `run_rollout(student_id, sequence, mode, seed, ...) -> RolloutResult`
- `discard_branch(branch_id)`
- `rebuild_student(student_id)`，作为受保护的管理能力

具体类名允许按照现有代码风格调整。

推理引擎必须能够直接由 Python 调用，不依赖 HTTP 请求对象。FastAPI 只应作为外层适配器。

## 4. Checkpoint 与 KC 编码

模型在服务启动时加载一次，并设置 `eval()`，使用 FP32 和 `torch.no_grad()` / `torch.inference_mode()` 中与当前实现兼容的方式运行。

必须使用 checkpoint 对应的：

`metadata["data_config"]["encoding"]`

编码规则必须与 `vector_kc_preprocess.py` 完全一致，包括：

- 使用现有 `category(value)` 函数生成 typed JSON scalar token。
- 使用原有 `json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))` 生成 question identity。
- 使用 `encoding["fields"]` 编码每一维 KC。
- 使用 `encoding["questions"]` 编码 `pid`。
- 维护 PAD=0、UNK=1、MISSING=2 的既有约定。
- 完整 KC 组合未见过时，按原有规则使用 `pid=0`。
- 不允许根据请求自动扩展词表或修改 checkpoint 的 embedding。
- 对所有输入进行维度、字段顺序、类型、有限数值和范围校验。

从 `encoding["field_names"]` 获取训练时的字段名称和顺序，避免服务端自行假定为固定六维。

当前 PianoKT 的六维 KC 通常为：

1. `has_black_key`
2. `simultaneous_note_count`
3. `both_hands`
4. `onset_interval_bin_cs`
5. `change_count`
6. `mean_move_bin`

请求里的 `kc_feature_names` 必须与 checkpoint 实际字段顺序一致。若不同，返回明确错误，而不是偷偷重新排序或继续推理。

对 OOV 类别和未见过的 question identity，可以返回诊断信息，例如 `unknown_fields` 和 `unknown_question`，以便分析分布偏移。

Checkpoint 必须来自可信来源。不要向外暴露不受限制的 checkpoint 文件加载接口。

## 5. 真实学生状态管理

优先复用当前 `StudentStateStore` 的 SQLite 事件日志和 Snapshot 机制。

当前阶段支持单进程本地服务，不需要提前实现分布式数据库，但应通过接口抽象将存储层和推理层分开，以便以后替换为 Postgres / Redis / GCS 实现。

必须保证：

- 每个 `student_id` 拥有独立状态。
- 不同学生共享模型参数，但不共享 Cache。
- 服务重启后可以恢复真实学生 Cache。
- `predict_next` 和 `predict_candidates` 不写入新事件。
- 真实 `observe` 必须持久化。
- 事件 ID 幂等，同一个事件重复提交不能重复增长历史。
- 同一 `event_id` 携带不同 payload 时必须拒绝。
- 事件按真实发生顺序处理，不能因为并发或异步 Worker 重试而默默改变历史顺序。
- 返回 `history_revision`、`next_position`、Cache 长度、模型版本和特征版本等诊断信息。
- 支持新学生空 Cache。
- 版本不兼容时不得静默使用旧 Snapshot。
- 需要重新构建历史时，应明确调用 rebuild/replay。

`observe_batch` 要支持 PianoKT 的一个 attempt 包含数十到数百个 onset 的场景。

实现前请检查现有 SQLite 事务能力。至少要预先验证完整批次，保证重试幂等并明确处理部分成功。不能在尚无事务保证时声称整个 batch 是原子操作。

增加 `expected_revision` 等并发保护机制，避免同一个学生的两个 Worker 乱序写入。

## 6. 必须区分三种推理语义

### A. Read-only Prediction

输入当前真实学生状态和一个候选 KC。

执行 `model.predict_next()`。

返回预测概率。

不改变真实 Cache，也不创建任何新的历史事件。

支持一次预测多个候选题目。多个候选都必须基于相同的历史状态，而不是依次将前一个候选加入 Cache。

### B. Persistent Observation

输入学生刚刚实际完成的 KC、二值 response 和唯一 event_id。

使用 `StudentStateStore.observe()`。

真实状态更新并持久化，返回事件发生前的预测概率及更新后的 revision。

不得把预测概率误认为真实 response。

### C. Ephemeral Hypothetical Observation

输入临时分支 Cache、KC、假设 response。

直接调用 `model.observe()` 取得新的 Cache，替换对应临时分支的内存状态。

绝对不能调用真实状态存储的写入方法，不能写 SQLite `events`、`snapshots`，不能把模拟 event_id 混入真实日志。

允许同一真实 Cache 派生多个相互独立的模拟分支。

## 7. 临时分支管理：Branch Manager

增加内存中的 `RolloutBranchManager`。

每个模拟分支至少包含：

- 随机生成且不可预测的 `branch_id`
- 所属 `student_id`
- 基于真实状态的 `base_revision`
- 当前模拟的 `StudentCache`
- 模拟事件数 `simulated_steps`
- `model_version`
- `feature_version`
- 创建时间、最近访问时间、过期时间

创建分支时，先恢复真实学生状态，再复制或安全共享不可变缓存。需要避免 Tensor 被意外原位修改造成分支互相污染。

支持以下操作：

- 创建分支
- 在指定分支上执行模拟事件
- 在指定分支上进行一次或多次只读预测
- 从同一基础状态再次创建独立分支
- 删除分支
- TTL 自动过期
- 内存与分支数量上限

默认 TTL 可以设为 15 分钟，通过配置覆盖。

临时分支仅在当前进程内有效。进程重启后失效，不允许意外恢复，也不写入持久化数据库。

返回给 HTTP 客户端的只能是 `branch_id` 和必要的元数据，不要把完整 KV Tensor 直接转换成 JSON。

为了防止其他用户访问模拟分支，必须将 branch 绑定到已认证主体/学生权限。开发模式和未来生产模式的认证边界需要明确。

## 8. Monte Carlo Rollout

实现对一个有序 KC 序列的多步模拟。

输入包含：

- 真实 `student_id`
- 按演奏顺序排列的 KC 事件序列
- `mode`：`forced` 或 `sample`
- `num_trajectories`
- 可选 `seed`
- 可选 `forecast_candidates`
- 可选是否保留单条轨迹的临时 branch

### forced 模式

每个事件明确给出假设 `response=0/1`。

对当前分支逐个调用 `model.observe()`。

最终得到模拟完成整个序列之后的 Cache，并允许继续预测下一批候选 KC。

此模式用于反事实分析，例如：

- 假设整首曲子全部演奏正确。
- 假设某些高难度音符出错。
- 比较不同反馈序列对最终预测的影响。

### sample 模式

对每条 Monte Carlo 轨迹：

1. 从同一个真实学生 Cache 创建独立分支。
2. 对序列中第 t 个 KC，调用 `predict_next()` 得到 `p_t`。
3. 使用独立的随机数生成器采样 `response_t ~ Bernoulli(p_t)`。
4. 调用 `model.observe()` 将采样结果加入临时分支。
5. 继续下一个 KC。
6. 最后在该分支状态上预测 `forecast_candidates`。

支持固定 seed 复现实验，并确保多条轨迹不会互相污染。

不要将 `num_trajectories` 次模拟错误地串联成一条长历史。

应输出：

- 每条轨迹的每步预测概率和采样 response（可选）
- 每条轨迹的累计正确事件数或正确率
- 每条轨迹的最终模拟位置
- 指定后续候选 KC 的最终预测概率
- 多轨迹的均值、标准差及可计算的分位数
- 基础真实 revision
- 实际使用的随机 seed
- 模型版本与特征版本

支持有上限的中间输出以控制响应大小。

不要直接把 AKT 的 KV Cache 解释成具有显式含义的 KC mastery 向量。当前模型输出的是预测概率。

在代码中区分预测概率、采样结果、实际演奏结果以及 Rollout 的评估统计量。

未来可由 ExRec 或曲目推荐策略调用这些接口，本次不需要实现 ExRec 优化算法。

## 9. HTTP API

使用 FastAPI 和 Pydantic 实现清晰、可校验的 JSON API。

建议路由：

- `GET /health`
- `GET /kt/model/info`
- `GET /kt/students/{student_id}/state`
- `POST /kt/students/{student_id}/predict`
- `POST /kt/students/{student_id}/observe`
- `POST /kt/students/{student_id}/observe-batch`
- `POST /kt/branches`
- `POST /kt/branches/{branch_id}/step`
- `POST /kt/branches/{branch_id}/predict`
- `DELETE /kt/branches/{branch_id}`
- `POST /kt/rollouts/evaluate`

可按 REST 最佳实践调整路由，但语义必须保持一致。

预测请求应支持 `kc_feature_names` 和多个 `{candidate_id, kc_vector}`。

真实观察请求应支持 `{event_id, kc_vector, response}`。

批量观察请求应支持 `attempt_id`、有序 `events` 和可选 `expected_revision`。

Rollout 请求应支持 `mode`、`sequence`、`seed`、`num_trajectories` 和 `forecast_candidates`。

响应统一提供结构化错误、模型版本、学生状态版本以及对应的预测结果。

提供 `/docs` OpenAPI 文档。

本地开发可以使用受控的开发鉴权配置，但不要把无鉴权的状态修改或管理接口作为生产默认。

## 10. 与 PianoKT 训练数据语义保持一致

PianoKT 的真实交互通常来自 `events_data`：

- `kc_feature_names`
- `kc_vectors`
- `reject_reason.event` 或 `responses`
- `onset_seconds`

请核实当前 `vector_kc_preprocess.py` 对 response 的映射。

当前代码使用 `response = int(reject_reason == 0)`，因此服务层应保持该二值标签定义。

不要在不同层重复或矛盾地转换标签。

PianoKT Worker 未来负责从 MIDI 对齐结果生成 KC 向量和正确性标签，KT 服务不负责解析或对齐 MIDI 文件。

真实事件按 attempt 时间和内部 onset 顺序提交。

推荐候选题目没有真实 response，应通过预测或模拟接口处理，不能误用真实观察接口。

## 11. 模块结构

优先按现有项目结构组织，建议增加独立的 serving 包，例如：

- `pykt/serving/encoder.py`
- `pykt/serving/engine.py`
- `pykt/serving/branch_manager.py`
- `pykt/serving/rollout.py`
- `pykt/serving/schemas.py`
- `pykt/serving/api.py`
- `pykt/serving/config.py`

尽量保持 `akt_fmkc.py` 和原有训练流程不变。

如果发现必须修改模型底层，请说明原因并增加兼容性测试。

## 12. 配置与运行

支持环境变量：

- `FMKC_CHECKPOINT_PATH`
- `KT_DEVICE`，默认为 `cpu`
- `KT_STATE_DB_PATH`
- `KT_MODEL_VERSION`
- `KT_FEATURE_VERSION`
- `KT_BRANCH_TTL_SECONDS`
- `KT_MAX_BRANCHES`
- `KT_MAX_ROLLOUT_STEPS`
- `KT_MAX_TRAJECTORIES`

版本应与 checkpoint 和训练 encoding 绑定。请设计合理的模型及特征 fingerprint，避免不同 checkpoint 错误地复用旧状态。

服务启动时校验 checkpoint、model type、metadata、field names、encoding 和设备兼容性。如果加载失败，启动应该失败并打印明确原因。

不得把 checkpoint、真实学生记录和模拟状态提交到 Git 仓库。

提供可以直接运行的本地启动命令，以及一个调用 Python Engine 的示例脚本。

## 13. 必须提供的测试

新增自动化测试，至少覆盖：

1. 编码结果与现有预处理完全一致。
2. 已知 KC 与未知 KC 的编码。
3. 单学生无历史时的预测。
4. 真实 observe 后 revision 增长。
5. 重复真实事件幂等。
6. SQLite 重启恢复状态后预测一致。
7. 多学生完全隔离。
8. 批量候选预测不修改状态。
9. 创建分支后，真实状态不改变。
10. 模拟 observe 后只有临时分支前进。
11. 不同模拟分支互不影响。
12. 删除或过期分支后不可继续使用。
13. forced Rollout 结果与手动逐步调用 `model.observe()` 一致。
14. seeded Monte Carlo Rollout 可复现。
15. 多条轨迹始终基于同一原始学生状态。
16. Rollout 后再做真实 observe，不会受到模拟历史影响。
17. Rollout 不向 SQLite 写入模拟事件和 Snapshot。
18. 版本、字段顺序、非法输入、过期 branch 和并发冲突能够正确报错。

所有测试应能使用合成的小型 AKTFMKC 模型运行，不依赖真实 checkpoint。

再提供一个单独的 smoke test，使用环境变量中的真实 checkpoint，进行端到端预测、真实观察及模拟。

## 14. 性能与部署约束

第一版只要求 CPU、单进程服务可用，但设计上应为将来使用独立 KT Service 做准备。

请注意：

- 保持模型权重共享。
- 防止同一学生并发写入乱序。
- 限制 Rollout 步数、轨迹数和内存使用。
- 避免不必要的 Cache 复制。
- 检查 `seq_len` 的语义与模拟长序列的关系。
- 不需要重算整段真实历史才能预测一个新增事件。
- 不允许 Rollout 更新模型权重。
- 不进行梯度更新。
- 不能假定当前 SQLite 适用于多实例 Cloud Run 生产环境。
- 明确记录当前版本的单进程限制。

## 15. 最终交付

请直接修改代码，运行测试，并给出：

1. 新增和修改的文件清单。
2. 架构和状态生命周期说明。
3. 如何设置 checkpoint 环境变量。
4. 如何启动 FastAPI 服务。
5. 如何初始化一个学生并执行预测。
6. 如何提交真实 onset 并持久化 Cache。
7. 如何创建临时 branch、模拟多步交互并预测后续题目。
8. 如何一次调用 Monte Carlo Rollout。
9. 完整的 curl 示例和 Python SDK 调用示例。
10. 测试命令、实际测试结果及尚未解决的限制。

不要只创建路由占位符；每个接口必须真正调用现有 AKT-FMKC 增量推理能力。

优先保证状态正确、训练/推理编码一致、模拟隔离与测试可靠，再考虑性能优化。