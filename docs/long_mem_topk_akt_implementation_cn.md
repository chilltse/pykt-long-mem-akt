> 版本说明：本文原有内容描述 `legacy_all_layers`（保留默认值以兼容旧 checkpoint）。新增 V2 必须显式指定 `architecture_variant=v2_last_retriever_residual`，实现与测试见 [V2 验证报告](long_mem_topk_akt_v2_validation_cn.md)。V2 不使用下文的全层联合候选 attention。

# long_mem_topk_akt：原 AKT 的跨块缓存与远期检索

## 实现范围

模型名为 `long_mem_topk_akt`，类为 `LongMemTopKAKT`。继承仓库原始 AKT，不增加预测头、logit 残差或融合门控。原 AKT 文件未修改，其 SHA256 仍为：

```
f84d7bd21e6843749127f7440dd8bfc632a377f5aa724cd9a3d54954b6e7c1e0
```

原来的 `long_mem_akt` 是另一个模型，保持独立。设计依据是 [Coding Plan](AKT_Memory_Retrieval_Coding_Plan_CN.md)。该计划中的 E2/E3 指近期缓存和远期检索，不是旧 LongMemAKT 中已删除的选择性门控。

参数名称和形状与原 AKT 相同，可以 strict 加载 AKT state_dict。继承的 `forward(q_data, target, pid_data=None, qtest=False)` 始终走原 AKT 无状态路径，返回契约不变。需要记忆时，runner 显式调用 `forward_with_memory`。

| 模式 | 每层近期缓存 | Retriever 远期读取 |
|---|---|---|
| `off` | 无 | 无，直接调用原 AKT forward |
| `recent` | 有 | 无 |
| `retrieval` + `exact` | 有 | 当前参数投影、分块精确点积 Top-k |
| `retrieval` + `hnsw_ip` | 有 | 每学生/层/head 独立的 HNSW inner-product |
| `dense_oracle` | 有 | 同缓存特征下的全部远期记录，计划中的 oracle A |

支持 `qid`、`qid_pdiff`、`separate_qa`、`kq_same`；记忆路径明确拒绝 avgpool/linear 变体。Rasch embedding、原预测头、原残差/LayerNorm/FFN 数学行为未改动。

## 缓存与防泄漏机制

- 每个 QA、Question Attention 和 Retriever 分别捕获其投影前 K/V 输入；Retriever 的 Value 是当前 QA 编码器最终输出，Key 是进入该 Retriever 前的 x。
- MemoryBank 位于模型之外，按 student_id 隔离。模型只读取块起始快照，返回 detached pending records。完成整个块的预测和 optimizer.step 后才提交记录。
- QA/Question Attention 仅允许 `key_pos <= query_pos`；Retriever 仅允许 `key_pos < query_pos`。块首有历史时可以读取，只有没有合法候选的行返回零 attention。
- current、recent、remote 不重叠，按真实绝对位置排序。提交时 recent 超出 R 的记录才进入 Retriever 的 remote；remote 超出 M 按 FIFO 淘汰。其他层不保存远期记录。
- 历史 features 存在 CPU，无计算图；选中后由当前 W_K/W_V 重新投影，读取路径仍有梯度。记录 feature_version，索引另记投影版本。
- runner 每次 train/valid/test 调用创建空 bank，每个学生结束后释放其状态。epoch 内可以打乱学生，学生内部块始终顺序执行。重复学生轨迹、重复提交、位置倒退、错误 block_id 和未来快照记录会报错。
- padding 不参与 embedding、attention、写入或 loss。有效但 loss_mask=False 的前缀仍可在观察响应后成为后续历史。
- 默认单卡、batch=1 runner。模型接口可接同 batch 内不同学生，禁止同学生依赖块同时进入 batch；没有实现跨 rank 状态迁移。

训练数据仅 train/valid/test：指定 fold 为 valid，其余训练 folds 为 train，fold=-1 为 test。检查学生集合不相交。默认 legacy loss mask 固定来自原预处理 maxlen，不随计算块长改变，因此长输入 AKT 仍在相同位置计分。`student` 协议也可选，但同次实验必须统一。

上述防护和测试覆盖实现中的因果、状态和 split 路径；不能证明外部提供的预训练 checkpoint 没有被人为用测试数据训练。入口检查其模型名、数据集、fold、词表和输入类型，并记录文件 hash，使用者仍需提供来源正确的 AKT checkpoint。

## Attention 与检索边界

局部 K/V 是共享矩阵，远期候选按 query chunk gather，不构造全历史 `[B,H,L,M,Dh]` Value 广播张量。合并候选后统一计算原 AKT 的 `sqrt(tail_mass * gap)`、head-specific gamma、可选当前 Query pdiff，再以衰减因子乘有符号 logits，最后统一 softmax。距离 detach，概率/指数运算采用 FP32。

候选删减改变累计概率质量和 softmax 分母，真实位置跨度不能恢复被删掉的质量。原始点积 Top-k 也不等于 AKT 衰减后的最终权重 Top-k。保留这些近似，不宣称等价于全历史 AKT。

Exact 按 Query/Memory chunk 扫描，搜索使用当前 W_K；复杂度仍为 O(LMd)，不是次线性索引。

HNSW 使用 Faiss `IndexHNSWFlat(d, M, METRIC_INNER_PRODUCT)`，不做 L2 normalize。接口依据：[Faiss 官方构造函数](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSWFlat.html)、[官方 metric 说明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)。

HNSW 每 head 独立建图，稳定 ID 映射为 abs_pos；FIFO 删除采用失效 ID 过滤，超过 20% tombstone 重建；候选不足扩大搜索，仍不足则 exact fallback。每学生默认每 8 个已完成块刷新当前参数的投影索引。新迁入记录在提交时用更新后的参数建索引，但 feature_version 仍标为 forward 时版本。缺 Faiss 会明确 warning 并记录 `effective_backend=exact_fallback`，不能把此结果当作 ANN 性能。

## 文件与集成

| 文件 | 功能 |
|---|---|
| `pykt/models/long_mem_topk_akt.py` | AKT 子类、每层特征捕获与记忆 forward |
| `pykt/models/akt_candidate_attention.py` | 联合候选 attention 与可选 dense 诊断 |
| `pykt/models/akt_memory_bank.py` | 外部事务式缓存、迁移、容量与学生隔离 |
| `pykt/models/akt_memory_index.py` | Exact 和 HNSW 后端 |
| `pykt/models/long_mem_topk_akt_runner.py` | 顺序训练、重放评估、成本和分层指标 |
| `pykt/models/long_mem_topk_akt_workflow.py` | AKT 初始化检查、checkpoint 和验证选择 |
| `examples/wandb_long_mem_topk_akt_train.py` | 训练/测试入口 |
| `examples/evaluate_long_mem_topk_akt_ablation.py` | 同条件消融入口 |
| `tests/test_long_mem_topk_akt.py` | 数值、梯度、因果、隔离、ANN 和端到端测试 |

已注册到 `init_model/load_model/train_model/evaluate` 和学生数据初始化分支。专用入口完成原 AKT checkpoint 检查、配置和 hash 保存；`wandb_predict.py` 也支持该入口保存的模型。

## 运行

以下从仓库 `examples` 目录执行。`--akt_source` 必须指向已训练原 AKT 的目录，包含 `config.json` 和 `qid_model.ckpt`（或对应 emb_type 的 checkpoint）；模型维度从该配置继承。

```bash
# 可选 ANN 依赖，当前验证环境 Python 3.7 / Faiss CPU 1.7.3
python -m pip install -r ../requirements-memory.txt

# 一次联合微调：没有旧 LongMem 的多阶段 residual/selector 流程
PYTHONPATH=.. python wandb_long_mem_topk_akt_train.py \
  --akt_source saved_model/akt_baseline \
  --dataset_name bridge2algebra2006 --fold 0 \
  --block_len 200 --recent_capacity 200 --remote_capacity 8192 --remote_topk 32 \
  --retrieval_backend exact --learning_rate 1e-5 --epochs 20 \
  --output saved_model/long_mem_topk_akt

PYTHONPATH=.. python wandb_long_mem_topk_akt_train.py --stage evaluate \
  --source saved_model/long_mem_topk_akt --output saved_model/long_mem_topk_akt_test

# 默认 3 个种子、4 个同块长组；无早停，训练预算一致
PYTHONPATH=.. python evaluate_long_mem_topk_akt_ablation.py \
  --akt_source saved_model/akt_baseline \
  --dataset_name bridge2algebra2006 --fold 0 \
  --epochs 20 --seeds 42,43,44 --variants akt,recent,exact,hnsw_ip \
  --block_len 200 --recent_capacity 200 --remote_capacity 8192 --remote_topk 32 \
  --output saved_model/topk_comparison
```

默认 CPU，GPU 加 `--device cuda`。输出目录必须不存在。可以用 `--fixed_steps N` 固定总优化步数，必须给足 epochs；否则报错，不能把未完成预算的运行当成公平对照。

可选 `--variants akt,akt_long,recent,exact,hnsw_ip,dense_oracle` 跑六组。`akt_long` 的块长为基础块长的 2 倍，同 epoch 训练交互数相同但更新次数不同；脚本显式记录这一区别，不声称其更新粒度相同。dense_oracle 仅适合小规模诊断。

加 `--diagnostics` 另起评估进程，测原始 IP Recall@k、同一快照的 dense attention 保留质量、attention 输出 L2、remote mass、候选年龄均值、feature/index age 和重新打分差异；诊断耗时不混入普通测试的速度结果。另导出 dense A 完整 replay 的最终预测差异，明确该 replay 会改变后续缓存表征，不能与同快照的逐层误差混称。

每组从完全相同的 AKT state_dict 开始，使用相同 seed、学生 shuffle、Adam、学习率和训练预算，按 valid AUC 选择 checkpoint（单类别 valid 时用 NLL）。保存 best checkpoint 后才评估 test。统计比较逐条核对 uid/position/target，以及同块长组的训练标签数和优化步数。

## 输出与成本解释

训练目录包含 config、roles、memory_manifest、`<emb_type>_model.ckpt`、`training.json`、`last_epoch.pt`。last_epoch 保存 epoch 边界的 optimizer 和 RNG 状态，当前 CLI 没有自动 resume；禁止把它当作可直接在学生中途恢复的状态。

评估输出 `metrics.json`、`predictions.csv`，包括 AUC、ACC、NLL、按历史长度/KC 间隔分层指标、训练/测试耗时、块计算耗时、CPU 峰值 RSS、CUDA allocated/reserved 峰值、bank/index 字节数、索引更新/搜索/fallback 数据。CPU RSS 是隔离进程的峰值，包含 Python/PyTorch 开销；bank 字节数是记录 tensor payload，index 字节数包括 Faiss 序列化大小和索引侧数组。

块计算耗时覆盖 forward/backward/commit，不包含指标导出；总 seconds 包含 runner 的统计开销。正常 CUDA 模式逐块同步以测总耗时，细分组件默认是 CPU launch 时间；diagnostics 才逐组件同步。index_transfer_seconds 包括索引重投影和搬运，不能当作纯 PCIe 带宽测量。

消融汇总保存 `summary.csv`、`comparison.json`，其中保留完整命令、初始 checkpoint hash、实际训练步数和交互预算。不要用一次极短的 smoke 耗时做加速结论。

## 验证与未完成实验

数值测试采用 FP32 atol=1e-6、rtol=1e-5：旧 forward / qtest、空缓存输出与参数梯度、全前缀 recent 的多层切块等价、k 覆盖全部 remote 时 dense 等价、候选顺序和位置平移不变性。因果测试分别覆盖 Exact/HNSW 路径的当前/未来响应以及未来问题扰动；另验证混合学生 batch、读取梯度、detach、重复提交/FIFO/刷新、全 padding、零容量、BF16 和连续多块优化。

六组单种子合成数据的训练、验证、测试、诊断已实际运行，见 [验证报告](long_mem_topk_akt_validation_cn.md)。这些结果只验证流程，不构成性能改善证据。真实数据三种子主实验、GPU FP16 和 GPU 资源曲线尚未运行；当前执行环境无 CUDA。
