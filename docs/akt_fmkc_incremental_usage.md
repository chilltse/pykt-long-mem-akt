# AKTFMKC 增量推理使用说明

已实现 CPU FP32 验证的单学生、单事件路径，使用现有参数与预测头，未新增 checkpoint key。原训练 forward 复用抽出的编码公式，并增加有效字段/pid 的范围校验；越界输入现在明确报错。

## 模型接口

```python
from pykt.models.akt_fmkc_runner import load_fmkc_checkpoint
from pykt.models.akt_cache import StudentCache
import torch

model, metadata = load_fmkc_checkpoint('path/to/model.ckpt', device='cpu')
model.eval()
versions = dict(model_version='checkpoint-sha256', feature_version='vocab-and-rules-v1')
cache = model.empty_cache(**versions)
# attrs 为单事件的一维整数字段向量；用真实编码替换以下示例。
attrs, pid = [1] * model.num_fields, 0
probability = model.predict_next(cache, attrs, pid, **versions)
cache, pre_response_probability = model.observe(
    cache, attrs, pid, response=1, event_id='event-001', **versions)

torch.save(cache.state_dict(), 'student-cache.pt')
restored = StudentCache.from_state_dict(
    torch.load('student-cache.pt', weights_only=True),
    device=next(model.parameters()).device)
model.validate_cache(restored, **versions)
```

每次调用必须提供当前权重及特征版本；版本由调用方管理，不能更换权重却沿用同一版本号。`seq_len`、schema、层数、张量形状、位置、设备和精度也会校验。权重或编码规则变化后应重放历史。第一版只允许 eval/FP32；不用于训练，不自动切换模型模式。

`predict_next` 不接受答案，不修改 cache。多个候选分别传同一个 cache。`observe` 返回新 cache 与作答前概率，调用方只有成功后才能替换旧快照。底层接口拒绝紧邻重复 event_id；任意历史重复事件的幂等由下面的持久化存储负责。字段 0 是 PAD，增量接口拒绝；UNK=1、MISSING=2 和 pid=0 可用。

cache 的 dataclass 禁止属性赋值，但 PyTorch 张量本身仍可变；调用方应将快照视为只读。序列化/迁移会复制并 detach 张量。有限 L 下每层最多保留 L 个位置，`L=None` 保存全部历史。

## 服务侧持久化

```python
from pykt.models.akt_state_store import StudentStateStore

store = StudentStateStore('student-state.sqlite', model, **versions)
p = store.predict_next('student-123', attrs, pid)
cache, p_before = store.observe('student-123', attrs, pid, 1, 'event-001')
# 同一学生、同一事件 ID、同一内容重复调用：不再次提交，返回原概率。
cache, same_p = store.observe('student-123', attrs, pid, 1, 'event-001')
store.close()
```

SQLite 保存原始编码事件、顺序游标、事件 ID、作答前概率及 CPU 状态快照。先提交事件日志，再提交所有层的快照和概率；中间失败后 `recover` 或下一次请求会从快照游标追赶日志。每个学生独立，实例内用锁串行化；快照更新还检查 revision，竞争失败明确报错，由调用方重新读取后重试。建议一个数据库使用一个服务实例；不同模型版本不应同时写同一数据库。调用顺序就是事件顺序，业务方应先处理乱序消息。

`store.rebuild(student)` 丢弃该学生快照并从最早日志重放，也用于明确切换权重版本后的重建。若词表 ID 含义变化，必须从原始业务数据重新编码，不能直接重放旧编码日志。历史纠错、补插和重排应先在业务来源更正，再将完整正确序列导入新的状态库；当前 API 不提供原地历史修改。分页导入只需逐条 `observe`，页间保留同一学生状态。不要只导入最近 L 条来替代全量重建。

## 训练 runner 的已确认语义

`StudentBlockCache` 缓存最近 `cache_blocks` 个原始事件块，长度约为 `block_size * cache_blocks`，仅保存 detach/CPU 的输入数据。`iter_student_blocks` 将它们拼接到当前块前；`forward_field_batch` 每次重新调用完整 forward，`loss_mask` 只选择新块位置计算 BCE 和正则。每块位置从零开始，没有各层 K/V 传递。训练每次优化更新后，下个块重新使用当前权重编码原始前缀，因此没有复用旧权重 K/V 的问题。

已有 loader 要求原始重放长度至少覆盖 L，但这并不保证覆盖多层有效依赖范围。新在线路径等价于完整前缀的局部 Attention；短前缀重放在左边界会丢上下文。本次不修改训练策略。

可复现的合成检查：seed=42，字段大小 `[5,6]`，17 个事件，d_model=8，n_blocks=4，heads=2，field_dim=3，dropout=0，L=2；每块 2 条、前缀重放 2 条。短前缀重放与完整 forward 的最大概率误差为 `0.0022577643`；逐条增量为 `5.9604645e-08`。这只是合成权重上的检查，不代表实际 checkpoint 的业务指标。上线前仍需用实际 checkpoint 和真实历史评估块边界及整体指标差异；GPU/其他精度尚未验证。

## 验证

```bash
/Users/tse/miniconda3/envs/pykt/bin/python -m pytest \
  tests/test_akt_fmkc_incremental.py tests/test_vector_kc.py \
  tests/test_yousician_akt.py tests/test_opera_integration_config.py -q
```

测试包含三种 drift 的原公式比较，1/4 层，两种 Q/K 投影，L=1/2/3/None，逐前缀因果性、逐层 K/V 对照、非零输出投影 bias、候选只读、当前答案隔离、右 padding、严格加载 checkpoint、不同导入 chunk、保存恢复、重复事件、多学生隔离、版本拒绝以及日志提交后失败恢复。
