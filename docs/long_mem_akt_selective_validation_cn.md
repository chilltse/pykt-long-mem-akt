# LongMemAKT 替换验证

本轮删除旧四模式及兼容入口，覆盖原模型与原训练脚本。
重复入口 `examples/long_mem_akt_selective.py` 已删除。
旧版使用说明以 README 当前命令为准。

验证结果：32 项测试全部通过（27.05 秒），包含一键训练、最终评估、旧命令拒绝和旧 checkpoint 拒绝。
`git diff --check` 通过。原有实验输出未修改。

测试覆盖：AKT 基线、因果性、写入梯度、TBPTT、padding、学生隔离、时间权重、
MC 随机状态恢复、收益训练、平滑选择、数值隔离、checkpoint 校验、旧参数拒绝，
以及单条 train 命令完成全部阶段后执行 evaluate。

测试入口：

```bash
python -m pytest -q tests/test_long_mem_akt.py tests/test_opera_integration_config.py
```

真实数据效果需重新训练后评估；旧 checkpoint 不兼容，已有实验文件未删除。
