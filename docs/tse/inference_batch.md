# Batch Candidate Prediction

`AKTFMKCInferenceEngine.predict_candidates(student_id, candidates)` 现在在编码后按
`(tuple(attrs), pid)` 去重，然后执行真正的 PyTorch Tensor Batch，最后恢复原始顺序、
candidate_id 和各条编码元数据。不同原始未知值若编码相同，仅计算一次。

```python
import logging
from pykt.serving import AKTFMKCInferenceEngine

logging.basicConfig(level=logging.INFO)
engine = AKTFMKCInferenceEngine.from_env()  # FMKC_CHECKPOINT_PATH、KT_STATE_DB_PATH
try:
    # 每个向量的字段顺序必须与 engine.encoder.field_names 一致。
    vector = [None] * len(engine.encoder.field_names)
    result = engine.predict_candidates('student-001', [
        {'candidate_id': 'a', 'kc_vector': vector},
        {'candidate_id': 'b', 'kc_vector': vector},
    ])
    print(result['predictions'])
finally:
    engine.close()
```

日志包含 count、unique_kc、total_ms、device、mode=tensor_batch。
总耗时覆盖校验、编码、锁等待、状态恢复、推理和结果构造；结果转回 Python
会等待设备计算完成。REST 的结构、鉴权、唯一 candidate_id 规则、空请求拒绝规则、
max_candidates 限制不变（默认 256）。超过请求限制时业务调用方自行切分请求；
若要求多个请求处于完全相同状态，调用方需避免其间 observe，或使用现有只读分支。

## 实现和复杂度

训练 forward、参数/checkpoint、距离衰减公式、observe 和单步 predict_next 的语义保留。
增量 forward_step 只扩展 batch 维度；每个候选位于同一 next_position，候选之间不互相注意。
检索 Attention 广播复用历史 K/V；自注意力使用 expand 共享历史视图，再与每个候选当前
K/V 拼接为临时张量。不会写入 StudentCache，也不会重新投影历史。

设唯一候选数 U、窗口长度 L、模型宽度 D、层数 K，Attention 计算量为 O(K U L D)，
另有投影/前馈计算；不构造候选间 U×U Attention。默认每个 Tensor Batch 最多 256 个
候选，临时自注意力 K/V 占用 O(B L D)，scores 占用 O(B H L)。模型编码输出占用
O(U D)。这是真正的 batch 加内部分块，不是分块后逐条调用 predict_next。
长无界历史仍会增加内存；底层 model.predict_candidates 可指定较小 batch_size。
FP32 的矩阵运算批次变化可能产生末位差异，验收使用绝对误差 1e-5。

## 验证与 Benchmark

```sh
conda run -n pykt python -m pytest tests/test_batch_predict.py tests/test_akt_fmkc_incremental.py tests/test_fmkc_serving.py -q
RUN_BATCH_BENCHMARK=1 conda run -n pykt python -m pytest tests/test_batch_predict.py::test_benchmark -q -s
```

Benchmark 使用固定随机种子的小型合成模型（D=8、2 层、2 heads）、64 条历史、
64 个唯一编码 KC、CPU 单线程、Tensor Batch 上限 256。四条路径分别为逐条、
去重后逐条、未去重 Tensor Batch、去重后 Tensor Batch。计时包含结果转换和顺序恢复，
不包含服务编码、SQLite 恢复以及预先确定唯一集合的成本。用于隔离计算收益，
不代表真实 checkpoint 的端到端吞吐；服务端到端耗时请参考上述日志。

2026-11-09 本地结果（PyTorch 2.14.1，单位 ms，单次预热后测量）：

- 100 个候选 / 64 唯一：逐条 57.167；去重逐条 37.588；Tensor Batch 2.979；去重 Tensor Batch 1.040。
- 1000 个候选 / 64 唯一：逐条 569.970；去重逐条 36.441；Tensor Batch 9.805；去重 Tensor Batch 1.052。
- 5000 个候选 / 64 唯一：逐条 2707.823；去重逐条 32.258；Tensor Batch 40.082；去重 Tensor Batch 1.164。

5000 个候选时，单独去重约 83.9 倍、单独 Tensor Batch 约 67.6 倍；
去重后的 64 个候选再使用 Tensor Batch 约 27.7 倍。具体收益取决于模型、历史长度和重复率。

回归结果：105 passed、26 skipped（24 CUDA 条件测试、1 真实 checkpoint 测试、
1 默认关闭的 Benchmark）；另行启用 Benchmark 后 1 passed。本机 CUDA 不可用，
未设置 FMKC_CHECKPOINT_PATH，因此未声称真实模型/GPU 性能。
覆盖 FP32 绝对误差 ≤1e-5、重复 100 次、候选顺序、缓存全部字段与 K/V 精确不变、
空/长历史、窗口边界、UNK/MISSING、pid=0、不同 batch_size、版本与范围检查。
