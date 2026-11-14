# 实验执行记录（2026-11-13）

## 范围与状态

这不是完整科研验收报告。已完成可运行离线原型、mock闭环和真实checkpoint的μ接口检查。
尚未取得可用于主实验的已发布曲库provenance清单与已验证READY学生快照；未将本地训练JSON冒充真实线上历史。

PyKT HEAD: 357574dd458338f3c8eece9e33cc6d9bf8ec5026
PianoKT HEAD: 73d4427c1d7ce403736caa06845b38fb59e7546b
两个工作区原有未提交修改均保留。生产HTTP API未改变，无数据库、持久化学生状态或checkpoint写入。

## 已实现

- PyKT当前查询μ覆盖，共享原始question encoder、attention和prediction head。
- PianoKT backend/pianokt_backend/experiments/{benchmark,cli,analysis}.py。
- 固定五档、已知PID和字段、类型敏感KC去重；完整顺序MC、四种策略、独立评估和参考估值。
- 稳定SHA256种子、阶段隔离、每checkpoint断点恢复、输入与实现digest检查。
- No-op、强制正误、顺序重排、overlap、μ单调性与饱和诊断、静态贝叶斯负对照。
- 4/8/16/32/64/128预算扫描、三次排序重复；学生聚类配对bootstrap；CSV和五张图。

## 真实checkpoint检查

`tse/qid_model.ckpt`, drift=response, CPU FP32。
332个已知KC在空历史下覆盖learned μ：与原predict_next最大绝对误差0。
μ min=-0.0546477809548378, max=0.04178301990032196,
mean=0.0052909161895513535, std=0.013181059621274471。
五档中只有0处于上述范围；±0.25与±0.5是分布外干预，不能当作验证过的真实难度。
332是训练词表KC数量，不是实际曲库Probe Bank数量。

## 执行与复现

```bash
PYTHONPATH=.:../pianokt/backend MPLCONFIGDIR=/tmp/kt-mpl \
 /Users/tse/miniconda3/envs/pykt/bin/python -m pianokt_backend.experiments.cli \
 --mode mock --output /tmp/kt-final-smoke --selection-mc 4 --evaluation-mc 8 --reference-mc 16
```

已保存成功运行：docs/tse/kt_mu_smoke_v3。2个synthetic KC、10个Probe、1个synthetic学生。
结果见evaluation_metrics.csv、probe_gain_by_mu.csv、paired_comparisons.csv、results.json与figures。
单学生不提供总体置信区间。预算扫描与诊断明细在匿名checkpoint JSON中。
旧kt_mu_smoke是较早实现结果；kt_mu_smoke_v2是中断运行，不应引用。

验证命令：

```bash
PYTHONPATH=.:../pianokt/backend /Users/tse/miniconda3/envs/pykt/bin/python -m pytest \
 tests/test_mu_probe.py tests/test_akt_fmkc_incremental.py tests/test_batch_predict.py \
 tests/test_fmkc_serving.py ../pianokt/backend/tests/test_mu_benchmark.py -q -p no:cacheprovider
```

结果116 passed, 26 skipped（包括CUDA条件跳过）。

## 未完成与结论边界

真实曲库/真实历史主实验未运行，未来真实结果未评价。
直接dev与CatalogCandidateProvider接入、严格历史版本/顺序证明、自适应参考近似并列增采样、
历史长度/模型/Probe敏感性、完整十图和各算法选曲成本计数仍未完成。
现有总览runtime仅为独立evaluation时间，不能据此比较四方法选曲成本。
离线输入的verified标记依赖可信导出者，不是自动数据验证机制。
参考仅针对固定候选子集，不是真实oracle。
目前不足以判断Full MC值得上线或进一步采用多Segment Planning；应先补齐真实数据闭环与预算稳健性。

## 最终回归补充

PianoKT相关集成回归在临时目录补齐MIDI、数据库客户端和云SDK依赖后：52 passed, 1 skipped。
依赖仅装入/tmp/kt-experiment-deps；运行时在sys.path末尾追加，避免覆盖原环境NumPy。

一键mock入口：`bash scripts/run_mu_benchmark.sh`。
可通过PYTHON_BIN、PIANOKT_ROOT覆盖本机路径；参数直接传递给CLI。
真实主实验的数据入口问题已向用户询问，未绕过provenance要求。

## 用户指定数据接入

240个推荐特征Segment，26931个onset；两个学生历史分别6905、11218个onset。
文件自报synthetic与近似评分，因此不标记为真人历史。已接入local_feature_root/student_history。
指定examples/saved_model/pianokt_trained/qid_model.ckpt大小6553600字节，PyTorch无法读取ZIP central directory，且目录无config.json。等待完整模型或允许替代模型，未静默替换。
配置：configs/experiments/local_mu_probe.yaml。时长暂为首末onset跨度代理，不是完整MIDI时长。

## 已完成指定模型与数据运行

用户确认改用tse/qid_model.ckpt后，成功运行位于docs/tse/kt_mu_local_run_v3。
2个synthetic学生完整历史，3个预先按长度选定候选，完整314已知KC×5=1570 Probe。
预算4/16/32，扫描4/8各两次。所有方法共用候选和Probe。
Full MC平均独立Gain=.251169，One-step/Sweet Spot=.230435，Random=.236235；
但Full MC参考Regret=.034849，One-step/Sweet Spot=.001239，排序并不稳定。
两个学生all_wrong仍产生约.12正Gain，不能解释成真实学习增长。
40行五档CSV与JSON逐值一致；生成10张图并检查μ Gain图。
本轮新增/相关测试65+1(batch)+1(local input)通过。
批量FP32误差最大8.523e-6和1.1802e-5，使用2e-5门限；learned μ等价仍为1e-6。
v3复用了v2已完成的首学生结果，resume_lineage.json记录文件哈希与仅容差变化的原因。
正式32/256/512大规模运行及更深入的敏感性分析尚未完成。
