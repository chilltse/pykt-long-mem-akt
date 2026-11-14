# Yousician benchmark：开发与 HPC 使用

本框架在本地开发，HPC 独立运行，不需要 Codex、LLM 或网络 API。入口为
`examples/benchmark_yousician_fmkc.py`，支持 `prepare/train/evaluate/analyze/summarize/status/smoke`。

## 本地验收

```bash
conda activate pykt
bash scripts/yousician_benchmark/run_local.sh
```

执行依赖检查、回归测试、两规模 × 两模型的 CPU 合成训练、checkpoint 重载、独立 evaluate、
CPU analyze、事件身份完整性检查、Slurm 命令 dry-run、Shell 语法检查及归档。
仅合成数据参与训练；输出在临时目录，命令结束时打印实际路径。

## 正式数据 prepare（不训练）

```bash
python examples/benchmark_yousician_fmkc.py prepare \
  --raw_file data/yousician_full/yousician_ukulele.json \
  --run_dir saved_model/yousician_fmkc_benchmark/run_manual \
  --train_student_sizes 60 120 240 --include_full
```

目录必须不存在，避免覆盖结果。数据不上传 Git。默认固定 test=20%，剩余学生按现有
5-fold 规则取 fold 0 验证集；train pool 用 `random.Random(1024).shuffle` 排列后取嵌套前缀。
每规模独立 fit encoding，验证/测试完全固定。训练池不足请求人数直接报错。
canonical events 按学生存储一份，任务只编码自己 train/valid/test 学生。
Prepare 仍会在内存读取完整原始 JSON；应配置足够 CPU 内存。

## 提交前

检查变更，只提交代码和测试，不提交原始数据、模型或私人 cluster.env：

```bash
git switch -c feature/yousician-fmkc-benchmark
git add .gitignore pykt/benchmark pykt/datasets/student_block_dataset.py \
  pykt/models/train_model.py pykt/preprocess/yousician_field_preprocess.py \
  examples/benchmark_yousician_fmkc.py scripts/yousician_benchmark \
  tests/test_benchmark_prepare.py tests/test_benchmark_metrics.py tests/test_benchmark_export.py \
  docs/yousician_benchmark_hpc.md
git diff --cached --stat
git commit -m "Add staged Yousician AKT FMKC benchmark and Slurm workflow"
git push -u origin feature/yousician-fmkc-benchmark
```

这里仅提供命令，不自动 commit/push。不要把用户的原始 JSON fixture 加入提交。

## HPC 配置与提交

```bash
git fetch origin
git switch feature/yousician-fmkc-benchmark
git pull --ff-only
cp scripts/yousician_benchmark/cluster.env.example scripts/yousician_benchmark/cluster.env
```

编辑 PROJECT_ROOT、CONDA_SH、CONDA_ENV、YOUSICIAN_RAW_FILE、BENCHMARK_OUTPUT_ROOT、
account、partition、QoS、资源参数。示例资源值未在你的集群验证。
`GPU_RESOURCE` 是单一 sbatch 参数，例如 `--gres=gpu:1`、`--gres=gpu:a100:1` 或 `--gpus=1`；
不默认假设 GPU 型号。配置文件是可信 shell 文件，不提交私人路径。

```bash
bash scripts/yousician_benchmark/submit_all.sh \
  --train-student-sizes 60 120 240 --include-full --fold 0 --seed 3407 --dry-run
bash scripts/yousician_benchmark/submit_all.sh \
  --train-student-sizes 60 120 240 --include-full --fold 0 --seed 3407
```

支持 `--models akt`、`--folds 0 1`、`--seeds 42 3407`。
提交创建独立 detached worktree，任务启动校验代码 fingerprint；不要修改该目录。
Prepare CPU → afterok GPU array → afterany CPU analysis。
GPU 任务要求 CUDA 可用，绝不静默回退 CPU；不在 Job 内 git pull 或升级依赖。
日志和 submission.txt 位于相邻 `<run_id>_logs`，源码位于 `<run_id>_code`。

## 独立运行与恢复

```bash
python examples/benchmark_yousician_fmkc.py train --run_dir RUN --task_id 0 --require_cuda
python examples/benchmark_yousician_fmkc.py evaluate --run_dir RUN --task_id 0
python examples/benchmark_yousician_fmkc.py analyze --run_dir RUN --bootstrap_replicates 2000
python examples/benchmark_yousician_fmkc.py summarize --run_dir RUN
python examples/benchmark_yousician_fmkc.py status --run_dir RUN
bash scripts/yousician_benchmark/retry_failed.sh --run-dir RUN
```

任务以 task_manifest.json 查询参数，不靠模型/规模硬编码数组。完整任务通过产物 SHA-256 验证后跳过。
完成训练且 checkpoint 存在时仅重做 evaluate；中断训练从头开始，旧失败文件移入 attempts。
没有 optimizer resume。重试后重新执行 analyze。状态 running 但产物不完整的任务属于未完成，
需结合 squeue 确认已停止后再重试，避免并发重复执行同一 task。

## 归档

```bash
bash scripts/yousician_benchmark/collect_results.sh --run-dir RUN --mode summary
```

默认排除 canonical 数据、checkpoint、预测和日志；可增加 `--include-checkpoints`、
`--include-predictions`、`--include-logs` 或 `--mode full`。full 也不含 raw/canonical 数据。
输出唯一 tar.gz、SHA-256、文件清单。使用实际集群地址替换打印的 `<host>` 后下载。

## 科学解释与当前边界

保持原模型和训练行为：Adam weight_decay=0，PID 正则 l2=1e-5，梯度裁剪 5，
validation AUC 提升阈值 .001，patience=10。预测用最佳 checkpoint，真实历史 response。
普通 AKT 的 reset 按参数第一维等于 n_pid+1 清零；dense q embedding 也可能被清零。
本实验保留该行为并导出真实初始化，不将架构差异解释为单变量 embedding 消融。

事件输出目前使用流式 jsonl.gz；不强制安装 PyArrow。分析会将单个模型完整预测加载内存，
bootstrap 使用学生 multiplicity 权重，不复制重采样事件；精确 AUC 预排序并正确处理相同分数，千万级数据
的 2000 次 bootstrap 可能较慢且需要较大内存。可显式修改 replicate 数并在结果中记录，不能称作
已验证大规模性能。当前没有实现可选 dataset_prefix 模式。

报告不伪造缺失结果。固定模型 bootstrap CI 不代表训练随机性；不同规模 epoch 数相同并不等于
优化步骤/计算预算相同。宏学生 AUC 排除单类学生；概率指标和 reliability 数据包含于 metrics JSON。

已验证：本地 CPU 合成流程、模型回归、Shell 语法、Slurm dry-run。
未验证：真实 CUDA、真实 sbatch 提交、集群资源容量、千万级训练和分析性能。

## 本次实际验收记录（2026-11-14）

- `run_local.sh`：115 passed、1 skipped；四个 synthetic 任务全部完成；独立 evaluate/analyze、
  导出事件检查、Shell syntax、Slurm dry-run、summary archive 均成功。
- 合成产物目录：`/private/var/folders/kz/wk9mldfj64s3c0wh336bpm_w0000gn/T/yousician-local.5KTlrJ/synthetic`。
  这是本地临时目录，不是 HPC 配置；重跑会打印新目录。
- 真实子集 `data/yousician/yousician_ukulele.json`：10,387 条原始记录，26 名有效学生，
  307,659 个有效事件，1,106 个 KC；1,055 个排序键重复组，沿用原始输入顺序 tie-break。
- 真实子集 prepare 验证规模为 2、4、full（16）名训练学生；验证/测试均为 5 名学生，
  分别固定为 81,951 / 57,816 个事件。
- 三规模训练事件分别 11,844 / 24,170 / 167,892；question vocab 分别 127 / 423 / 861。
  此子集不够 60 人，不能把它当作正式 60/120/240 数据效率实验。
- 真实子集产物：`/private/tmp/yousician-subset-prepare-1794649469`，仅 prepare，没有训练。
- 570 MB 完整原始文件的本地 prepare 未完成并主动终止；完整数据验证需在 HPC 执行。
- 没有正式模型实验，没有 CUDA/Slurm 实测，没有 commit/push。
