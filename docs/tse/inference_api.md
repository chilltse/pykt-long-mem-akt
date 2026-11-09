# AKT-FMKC 本地在线推理服务

实现入口是 `pykt.serving.AKTFMKCInferenceEngine`，HTTP 只是 FastAPI 适配器。无需重新训练，不修改模型数学定义。需要 Python 3.10+、已安装的 pyKT/PyTorch 环境，以及可信来源的 checkpoint。

## 安装和启动

在仓库根目录：

```bash
conda activate pykt
pip install -e '.[serving,serving-test]'
export FMKC_CHECKPOINT_PATH='/实际导出目录/qid_model.ckpt'
# 标准导出必须同时保留该目录下的 config.json。
export KT_DEVICE=cpu
export KT_STATE_DB_PATH='/实际可写目录/kt_state.sqlite3'
# 显式开启仅用于本机调试的鉴权模式：
export KT_DEV_AUTH=true
python -m uvicorn pykt.serving.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

访问 <http://127.0.0.1:8000/docs> 获取请求和响应的完整 OpenAPI schema；`GET /health` 检查启动结果。模型只在启动时加载一次，转换成 FP32/eval，禁用参数梯度；增量推理使用现有模型的 `torch.no_grad()`。模型类型、词表范围/格式/宽度、字段顺序、设备和空 Cache 推理不合法都会使启动失败。

正常鉴权模式默认关闭开发绕过；请配置服务器信任的 Worker token 和学生权限：

```bash
export KT_DEV_AUTH=false
export KT_API_TOKEN='替换为至少32字符的随机Worker令牌'
export KT_PRINCIPAL='pianokt-worker'
export KT_ALLOWED_STUDENTS='student-001,student-002'
# 只有受信任且拥有全部学生权限的 Worker 才使用 '*'。
# 管理接口默认禁用，启用时设置独立且至少32字符的令牌：
export KT_ADMIN_TOKEN='替换为不同于Worker令牌的随机管理令牌'
```

所有 `/kt/*` 接口需要 `Authorization: Bearer <KT_API_TOKEN>`。分支绑定服务器配置的主体及学生权限，不能通过请求参数冒充主体。管理重建接口只接受独立管理令牌，同时检查学生 allowlist；开发模式也不会自动开启管理权限。`/health` 和 `/docs` 不需要鉴权。

该静态 Worker token 模式是单个受信任服务身份的边界，**不是多租户终端用户认证系统**。正式对外部署前需要 TLS、令牌管理及可信网关/认证依赖，依据真实身份映射学生权限；不要公开 `KT_DEV_AUTH=true` 服务。Python Engine 属于受信任进程内部接口，默认主体 `python`，不承担 HTTP 认证。

## 状态语义和生命周期

- `predict_next` / `predict_candidates`：从同一个真实状态预测所有候选，不追加事件，不依次串联候选。
- `observe` / `observe_batch`：事件先完整校验，随后在 SQLite `BEGIN IMMEDIATE` 事务内计算并提交日志、原始 payload 身份、attempt 身份、概率及最终 Snapshot。任何推理/提交异常使本批新增内容全部回滚。
- `fork_student`：恢复真实 Cache 后复制所有 Tensor，分配不可预测 branch ID。分支只存在于进程内存。
- `simulate_step` / `run_rollout`：直接调用 `model.observe` 返回新 Cache，绝不调用真实存储写接口。模拟 response 是假设/采样结果，绝不能当作实际练习标签提交。
- Snapshot 可丢弃并通过事件日志重建。服务重启后真实状态可恢复，分支全部失效。恢复旧版单事件操作中断后的真实日志可能先补写真实 Snapshot，这属于恢复，不会写任何模拟事件。

真实状态返回 `history_revision`、`next_position`、`cache_length`、模型和特征版本。`seq_len` 限制每层保留的 KV 数量，不限制累计事件位置；长序列超过窗口后仍按现有增量模型前进。`seq_len=None` 的模型保留全部历史，临时模拟仍受内存预算限制。Cache 不是显式 KC mastery 向量；输出是下一事件正确概率。

Engine 用进程锁串行化操作，确保一个请求的候选和多条轨迹使用同一基础状态。SQLite 存储独立于 Engine，通过 `StateRepository` 协议及 `store_factory` 注入，为未来替换存储留出边界。目前只支持**单进程、一个数据库一个 Store、一个 Uvicorn worker**；不要将本地 SQLite 当作多实例 Cloud Run 数据库。

## 编码和标签

`metadata["data_config"]["encoding"]` 是唯一词表来源；字段数和顺序来自 `field_names`。HTTP 必须提交完全一致的 `kc_feature_names`，不自动重排。

编码直接复用 `category()`，再使用 `json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))` 构造 question identity：PAD=0（真实事件不用）、UNK=1、MISSING=2，未知完整组合 pid=0。`null` 和空字符串编码为 MISSING；JSON bool、整数、浮点数和字符串保留类型区别。数值必须有限，容器类型不合法。字段是类别而非连续量；checkpoint 未携带业务数值上下界，因此合法但未见的类别返回 UNK，而不杜撰六维字段的数值范围。编码 ID 在启动时严格与 embedding 范围校验。

预测返回 `attrs`、`pid`、`unknown_fields`、`unknown_question`，服务不会增长词表。观察以原始 JSON payload 进行幂等校验，所以两个不同 OOV 向量即使映射到同样 IDs 也不能冒用同一 event ID。

`response` 必须是严格二值整数，拒绝布尔值、字符串和小数。PianoKT Worker 将 `reject_reason.event` 按 `int(reason == 0)` 转为 response；如果已经是 responses，则直接提交，不在服务内再次反转。Worker 按 attempt 时间及内部 onset 顺序组织事件。服务不解析 MIDI、不自行按 event_id 排序，也无法从没有时间戳的请求判断业务时间先后。

## 写入顺序、重试和版本

HTTP 单条和批量真实写入**要求** `expected_revision`（比可选字段更严格，以避免 Worker 无条件写入）。先读取 `/state` 获取当前 revision，再提交新增事件；冲突返回 409，包含当前状态。Worker 应核对权威事件顺序后重试，不能只是刷新 revision 后盲目追加旧事件。

完整重复请求忽略旧 expected_revision，返回事件首次写入时的概率，不追加历史。重复 event ID 的原始 payload 不同、同 attempt ID 改变内容/顺序，均返回 409。批次内 event IDs 必须唯一；历史重试必须连续且顺序一致。允许以当前日志结尾的重复前缀加新事件继续提交，但其 expected_revision 必须是当前值；夹杂历史旧事件、反转顺序或旧前缀后追加均拒绝。未持久化失败的原子批次可原样重试。

Python 的 `expected_revision=None` 仅供明确串行的内部调用；并发调用必须传 revision。事件 ID 在每个学生内唯一，建议 `attempt_id:onset_index`。

模型版本包含实际 FP32 权重、窗口及注意力结构的 SHA256；特征版本包含完整编码字典的 SHA256。`KT_MODEL_VERSION` / `KT_FEATURE_VERSION` 仅添加可读前缀，不能覆盖 fingerprint。数据库绑定两种版本，即使 Snapshot 缺失也不能跨模型混用日志。版本改变或已有无版本日志时启动失败，必须显式使用新数据库，从权威原始 KC/response 历史按新编码 replay。`rebuild_student` 只在当前模型/编码下重建 Snapshot，不做词表迁移。

## HTTP 示例（完整可执行）

仓库提供 [examples/fmkc_inference_curl.sh](../../examples/fmkc_inference_curl.sh)，依赖 curl 和 jq。它从 `/kt/model/info` 读取字段名，默认用合法缺失值构造任意宽度的演示向量；可用 `KT_DEMO_VECTOR` 提供真实向量。

```bash
# 服务以开发模式运行时：
KT_DEMO_STUDENT=demo-001 bash examples/fmkc_inference_curl.sh
# 对应六维模型才使用如下向量，字段顺序以模型返回为准：
KT_DEMO_VECTOR='[false,1,false,30,1,1]' \
  KT_DEMO_STUDENT=demo-002 bash examples/fmkc_inference_curl.sh
```

脚本依次演示初始化空学生状态、只读候选预测、真实 onset 持久化、幂等重试、批量 attempt 写入、fork、三步假设观察、分支预测、删除分支、sample Monte Carlo、多步 forced rollout、保留最终分支并继续预测。正常鉴权模式下 demo 学生必须包含在 `KT_ALLOWED_STUDENTS` 内。

最小单条观察的请求结构为（此处字段仅用于说明，替换成 checkpoint 的真实字段）：

```json
{
  "kc_feature_names": ["field_a", "field_b"],
  "expected_revision": 0,
  "event": {"event_id": "attempt-001:0", "kc_vector": [false, 1], "response": 1}
}
```

主要路由：

- `GET /kt/model/info`、`GET /kt/students/{student_id}/state`
- `POST /kt/students/{student_id}/predict`：`kc_feature_names` + 多个 `{candidate_id,kc_vector}`
- `POST /kt/students/{student_id}/observe`：上面的单事件结构
- `POST /kt/students/{student_id}/observe-batch`：`kc_feature_names,attempt_id,expected_revision,events`
- `POST /kt/branches`：`{student_id}`
- `POST /kt/branches/{branch_id}/step`：`kc_feature_names,kc_vector,response`
- `POST /kt/branches/{branch_id}/predict`：与真实候选预测相同的请求结构
- `DELETE /kt/branches/{branch_id}`
- `POST /kt/rollouts/evaluate`：下面的 Python 参数加 `kc_feature_names`
- `POST /kt/students/{student_id}/rebuild`：独立管理员 token，仅重放当前版本历史

输入错误为 422，状态/幂等冲突为 409，鉴权/授权为 401/403，分支不存在、不可访问或过期为 404，分支资源上限为 429。错误包含 `error.code`、`error.message`、model/feature version；状态冲突同时包含 `state`。无权限请求不泄露学生状态。

## Python API 和 Monte Carlo

完整可执行脚本 [examples/fmkc_inference.py](../../examples/fmkc_inference.py) 自动从训练词表提取一个合法向量并生成独立 demo 学生：

```bash
python -m examples.fmkc_inference
```

Python 调用不需要启动 HTTP：

```python
import json
from pykt.serving import AKTFMKCInferenceEngine

engine = AKTFMKCInferenceEngine.from_env()
try:
    tokens = json.loads(next(iter(engine.encoder.encoding['questions'])))
    kc = [None if t is None else json.loads(t) for t in tokens]
    student = 'demo-python'
    candidates = [{'candidate_id': 'next-onset', 'kc_vector': kc}]
    state = engine.get_student_state(student)  # 新学生自动为空，无需注册写入
    prediction = engine.predict_candidates(student, candidates)
    result = engine.observe(student, {'event_id': 'attempt-1:0',
                            'kc_vector': kc, 'response': 1},
                            expected_revision=state['history_revision'])
    branch = engine.fork_student(student)
    for response in [1, 0, 1]:
        engine.simulate_step(branch['branch_id'], kc, response)
    forecast = engine.predict_branch(branch['branch_id'], candidates)
    engine.discard_branch(branch['branch_id'])
    rollout = engine.run_rollout(
        student, sequence=[{'kc_vector': kc}] * 8,
        mode='sample', seed=42, num_trajectories=16,
        forecast_candidates=candidates, include_steps=True)
    forced = engine.run_rollout(
        student, sequence=[{'kc_vector': kc, 'response': 1}] * 8,
        mode='forced', seed=42, retain_branch=True,
        forecast_candidates=candidates)
    engine.predict_branch(forced['branch']['branch_id'], candidates)
    engine.discard_branch(forced['branch']['branch_id'])
finally:
    engine.close()
```

forced 每步必须带 response；sample 每步不能带 response，服务根据当步预测用独立 `random.Random` 采样 Bernoulli 标签。每条轨迹都从同一个真实 Cache 开始，使用按 seed 和轨迹序号派生的 RNG，不修改全局 Python/torch RNG。相同模型、状态、环境、seed 和请求可复现；CPU 与其他设备不承诺逐位一致。

返回每条轨迹的正确事件数/率、最终位置、可选逐步概率与 hypothetical_response，以及后续候选预测。聚合返回均值、总体标准差（ddof=0）、线性插值 q05/q50/q95，并返回实际 seed 与 base_revision。统计描述假设演奏轨迹，不是学习收益的因果估计。`retain_branch=true` 仅允许单条轨迹，返回的 branch 可继续模拟和预测。保留分支也计入最大模拟步数；否则工作 Cache 在请求结束后释放。

## 资源配置

环境变量及默认值：

- `FMKC_CHECKPOINT_PATH`：必需；可信 checkpoint 路径，不接受 HTTP 上传/任意路径加载。
- `KT_DEVICE=cpu`，`KT_STATE_DB_PATH=kt_state.sqlite3`。
- `KT_MODEL_VERSION`、`KT_FEATURE_VERSION`：可选可读前缀。
- `KT_BRANCH_TTL_SECONDS=900`：绝对生命周期，不因访问续期；HTTP 后台清理最多每30秒一次，所有分支访问也惰性清理。纯 Python 调用在分支操作时清理，也可调用 `engine.branches.expire()`（需持有 engine.lock）。
- `KT_MAX_BRANCHES=64`、`KT_MAX_BRANCH_BYTES=268435456`：保留分支总 Tensor 字节数以及模拟投影预算。运行前保守预留新旧 Cache；这不是整个 Python/PyTorch 进程 RSS 硬上限。
- `KT_MAX_ROLLOUT_STEPS=512`、`KT_MAX_TRAJECTORIES=32`、`KT_MAX_ROLLOUT_WORK=8192`（步数×轨迹数）。
- `KT_MAX_CANDIDATES=256`、`KT_MAX_BATCH_EVENTS=512`、`KT_MAX_OUTPUT_STEPS=2048`（逐步输出和候选预测输出各自受此上限约束）。超出逐步输出上限可用 `include_steps=false`。

请求 schema 另有绝对数量上限：字段数256，候选/批次/序列4096，轨迹1024；配置不会突破这些限制。部署时还应在网关限制请求体大小。持久事件日志不自动裁剪，真实历史和数据库容量需由部署方管理。Rollout 计算期间全局锁会阻塞其他请求，这是首版以确定性和正确性优先的取舍。

## 测试和交付文件

```bash
python -m pytest tests/test_fmkc_serving.py tests/test_akt_fmkc_incremental.py -q
# 真实 checkpoint 使用临时数据库，绝不写入 KT_STATE_DB_PATH 中的真实学生数据：
FMKC_CHECKPOINT_PATH='/实际目录/qid_model.ckpt' \
  python -m pytest tests/test_fmkc_serving.py -k real_checkpoint_smoke -q
```

合成小模型测试覆盖编码一致、输入与词表校验、读操作无写入、真实观察/幂等/OOV冲突、批次完整预验证和推理失败事务回滚、SQLite 重启、学生隔离、并发冲突、分支生命周期/所有权/内存限制、forced 与模型逐步调用一致、seeded sample、轨迹基础状态一致、模拟不写数据库/不改权重、模拟后真实观察、HTTP 路由、默认拒绝无鉴权，以及 checkpoint 加载。真实 smoke test 没有环境变量时明确跳过，不用合成权重冒充真实 checkpoint。

新增 `pykt/serving/`：encoder、config、schemas、errors、storage 协议、branch_manager、rollout 统计、engine、api 和包入口；新增服务测试、Python 示例、curl 示例及本说明。修改 `akt_state_store.py` 添加版本绑定和事务批次方法，保留原单事件接口兼容；修改 `setup.py` 添加 serving 可选依赖，`.gitignore` 排除 checkpoint/SQLite 状态。`akt_fmkc.py`、Attention 和训练编码实现保持不变。

本次实现验证结果：服务、增量推理、vector KC、Yousician AKT 及 Opera 配置五组测试合计 **110 passed, 1 skipped**。跳过项是真实 checkpoint smoke test（当前未设置 `FMKC_CHECKPOINT_PATH`）。Python 示例使用合成 checkpoint 和临时数据库执行成功；curl 示例通过 `bash -n`，对应 HTTP 路由均经 TestClient 验证。`git diff --check` 和新增 Python 模块编译检查通过。依赖带有 Starlette/httpx 与已有 matplotlib/pyparsing 的弃用提示，不影响断言结果。

本机验证使用已有 Conda `pykt` 环境，新增 HTTP 依赖安装在临时目录，没有修改该环境的全局依赖：

```bash
PYTHONPATH=/tmp/kt-serving-deps:. /Users/tse/miniconda3/envs/pykt/bin/python -m pytest \
  tests/test_fmkc_serving.py tests/test_akt_fmkc_incremental.py tests/test_vector_kc.py \
  tests/test_yousician_akt.py tests/test_opera_integration_config.py -q
```

其他机器按前述 `pip install -e '.[serving,serving-test]'` 安装后使用普通 `python -m pytest` 即可，不需要此临时路径。
