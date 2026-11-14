# 任务：为 AKT-FMKC 增加高效 Batch Candidate Prediction

工程：`pykt-long-mem-akt`

分支：`feature/akt_fmkc_incremental_inference`

请先阅读当前 `pykt/models/akt_fmkc.py`、`pykt/serving/engine.py`、`pykt/serving/api.py`、相关缓存实现及测试，确认现有功能后进行最小增量修改。

## 目标

支持在同一个学生当前状态 S 下，对大量候选 Onset KC 进行批量预测：

`predict_candidates(student_id, candidates)`

返回各个候选的正确概率，不修改学生状态。

## 实现要求

1. 保留原有 checkpoint、模型公式、训练代码以及单步 `predict_next()` 语义。
2. 所有候选共享同一个只读 StudentCache，不进行 `observe()`。
3. 在编码之后按 `(attrs, pid)` 去重，重复的 KC 只计算一次，输出时恢复原始候选顺序。
4. 优先实现真正的 PyTorch Tensor Batch 推理，共享历史 K/V，避免逐候选重复执行完整 Python 推理路径。
5. 如果真正 Batch 需要较大范围地重写 Attention，请先提供复杂度和风险分析，不要牺牲原模型正确性。分批调用可以作为兼容路径，但必须明确区分分批和真正的 Tensor Batch。
6. 保留严格的字段范围校验、UNK/MISSING、pid=0、模型版本和 StudentCache 版本检查。
7. 不改变现有 REST 接口语义。大于请求限制的输入可由业务调用方分批处理。
8. 输出推理数量、唯一 KC 数量、总耗时与设备信息，用于 Benchmark。

## 验收测试

创建 `tests/test_batch_predict.py`，覆盖：

- Batch 与逐个 `predict_next()` 一致，FP32 最大绝对误差不超过 1e-5。
- 同一 KC 重复 100 次与单次预测概率一致。
- 改变候选输入顺序不改变各候选结果。
- 预测前后 history_revision、next_position 和学生 K/V Cache 完全不变。
- 覆盖空历史、较长历史、UNK、pid=0、缓存长度边界及不同 Batch Size。
- CPU 路径正确；GPU 可用时增加设备测试。
- Benchmark：比较 100、1000、5000 个候选的逐条与批量耗时，单独报告去重收益和 Tensor Batch 收益。

不要加入曲目推荐、Sweet Spot、ExRec、MC Gain 等业务逻辑。

完成后给出变更文件、测试结果、性能数据和本地调用示例。