# Yousician 四字段 AKT：预处理与训练

与原版模型的详细对照见 [实现差异说明](yousician_akt_vs_original_cn.md)。当前工程已改为 pyKT 的“先预处理，再按 dataset_name/fold 训练”流程；训练不再接受或解析原始 Yousician JSON。

## 1. 工程入口

```text
examples/data_preprocess.py --dataset_name yousician
  → preprocess/yousician_field_preprocess.py：原始 schema 转换
  → preprocess/student_sequences.py：学生划分、各 fold 训练词表、保存产物
  → configs/data_config.json：登记数据集

examples/train_yousician_akt.py --dataset_name yousician --fold 0
  → examples/wandb_train.py
  → datasets/init_dataset.py → student_block_dataset.py
  → models/init_model.py → YousicianAKT
  → models/train_model.py / evaluate_model.py
```

四字段不能直接塞进原有一维 KC CSV 协议，因此使用新的 prepared artifact 格式和 DataLoader 类型，但注册到现有数据、模型及训练工厂，没有保留独立训练循环。

## 2. 预处理

在仓库根目录安装项目（`pip install -e .`）后进入 examples；未安装 editable 包时，可以在以下命令前加 `PYTHONPATH=..`。

```bash
cd examples
python data_preprocess.py --dataset_name yousician \
  --file_path ../data/yousician/yousician.json --kfold 5
```

默认写入 `../data/yousician/` 并更新 `../configs/data_config.json`。可通过 `--output_dir`、`--config_file` 覆盖位置。原始数据只在这一步读取。

产物包括：

- `student_sequences.json`：标准化的学生完整时序，每条为四字段、题目身份和响应；不含原始 events_data schema。
- `student_fold_0.json` 至 `student_fold_4.json`：各 fold 的训练、验证、测试学生 ID，以及仅由该 fold 训练学生拟合的字段/题目词表。
- data_config 中的数据集条目：产物路径、格式版本、字段名、fold 列表、划分 seed 等。

先固定约 20% 学生为 test，再将剩余学生分成 k 个验证 fold；选择 fold=i 时，其余 fold 为 train。默认划分 seed=1024，可用预处理的 `--seed` 修改。每名学生保持完整时序后才切 block。训练阶段只选择已保存的 fold，不重新划分或拟合编码。

`prepare_student_sequences` 接受通用的 `uid -> ordered events`，不依赖 Yousician schema。其他字段数据集可以提供自己的原始数据适配器，复用划分、词表和 block loader。

## 3. 字段与模型

四字段固定为 `prev_frets / prev_strings / frets / strings`；复用原有 pitch 累加、fret、弦集合、无效帧跳过和响应映射规则，每个有效帧一个预测。题目身份为规范化的 `prev_frets|prev_strings|frets|strings`。相同 KC vector 在所有歌曲、练习、位置和学生间共享题目 ID 与 μ，不使用曲目或位置身份。时间/session/part 只负责排序，不作为模型特征。未知组合统一映射为题目 0、μ=0；字段组合仍能参与预测。

此规则覆盖 PianoKT 参考手顺中的谱面位置 ID 要求。学生历史及已确认的局部窗口/缓存保留，因此相同 KC 不要求预测概率相同。

字段编码 PAD=0、UNK=1、MISSING=2；有效零值和 exercise 起点 inf 与缺失不同。未知题目 ID=0，其 μ 强制为 0。

四字段独立查表并拼接成 u，c=base_head(u)，g=response_embedding(r)。三个配置：

- none：question=c，interaction=c+g。
- shared：question=c+μ*d，interaction=question+g。
- response：question=c+μ*d，interaction=c+g+μ*((1-r)*f0+r*f1)。

d、f0、f1 为独立输出头，μ 零初始化；未启用的模块不创建。AKT 主干与输出 MLP 复用原实现。

## 4. 学生 block DataLoader 与窗口

`StudentBlockDataset` 每次迭代按学生组织 block，仅打乱训练学生顺序。每个学生/epoch/split 建立独立缓存。当前仅支持 num_workers=0，每次产生一个学生的一个 block（DataLoader batch_size=None），不是跨学生混合 batch。

默认 block_size=128、cache_blocks=1、seq_len=128。历史与当前最多合计 256 条输入；响应检索直接可见 [t-L,t-1]，包含当前位置的 attention 可见 [t-L+1,t]。各层最多 L 个直接可见键，允许多层间接传递更早信息。

缓存保存原始编码输入，不保存 KV。历史与当前在最新参数下重新前向；只对当前事件计算 BCE 和 μ 正则，梯度通过历史表示反传。缓存容量必须满足 block_size*cache_blocks >= seq_len。

密集 attention 仍有平方矩阵计算开销。有限缓存可能改变间接上下文，不保证与学生整段运行完全等价。

## 5. 按数据集训练

在 examples 目录执行：

```bash
python train_yousician_akt.py --dataset_name yousician --fold 0 --drift none
python train_yousician_akt.py --dataset_name yousician --fold 0 --drift shared
python train_yousician_akt.py --dataset_name yousician --fold 0 --drift response
```

可指定 `--data_config` 使用另一份配置，但不再有 `--data` 原始数据选项。入口为参数封装，实际调用共享 wandb_train、init_dataset4train、init_model、train_model、evaluate。默认 use_wandb=0，num_epochs=30；使用项目 train_config 中的优化器设置和指定学习率。

共享 train_model 按验证 AUC 选择 checkpoint，并在连续 10 epochs 未改善时早停。单类评估 AUC=-1，第一轮仍保存 checkpoint；这不表示有效的 AUC。新版保留 gradient clipping=5 以及当前 block 全部有效事件的监督（包含首条）。

## 6. 保存与加载

沿用 pyKT 产物约定：`qid_model.ckpt` 为纯 state_dict，`config.json` 保存 train_config/model_config/data_config/params。data_config 中记录当前 fold 的字段维度、题目数量、词表、学生划分和窗口参数。训练结束后重载最佳权重，在预处理指定的 test 学生上评估，写入 test_metrics.json。

可用通用 `load_model` 加载；也可用 `load_yousician_checkpoint` 同时读取邻近 config.json。后者只为兼容旧实验保留对旧 bundled checkpoint 的读取，新的训练不再生成旧格式。

题目身份策略记录为 question_identity=kc_vector_v1。此前基于曲目位置的预处理产物必须重新生成，其模型权重不可直接复用于新题目映射，应重新训练。

未实现断点续训，也不把保存权重描述为保存了 optimizer 状态。训练不依赖原始 JSON，但需要预处理产物；仅做新事件推理时可使用 config.json 中保存的词表。

## 7. 验证

在仓库根目录：

```bash
python -m pytest tests/test_yousician_akt.py tests/test_opera_integration_config.py -q
```

测试包含字段模型因果性、窗口、历史梯度、缓存隔离；新增预处理产物、各 fold 训练专属词表、标准数据/模型工厂、共享 train_model/evaluate、通用 checkpoint 加载与 config 元数据验证。此轮测试为 19 passed；另对小样本执行 CLI 预处理与训练联调，未执行完整训练。
