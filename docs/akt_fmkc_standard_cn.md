# 标准 AKT-FMKC 数据与训练入口

模型实现位于 `pykt/models/akt_fmkc.py`，训练适配位于 `akt_fmkc_runner.py`。旧 `yousician_akt` 模块仅保留兼容导入。模型从 `field_sizes` 获取字段数量，支持任意固定宽度的分类 KC 向量；同一个数据集的所有事件必须具有相同字段名称和顺序。模型的 backbone、drift、cache 和 loss 逻辑与此前相同。

每个字段分别拟合训练词表并嵌入，拼接后投影为 `d_model` 维。数字字段作为分类值处理，不作为连续数值回归输入，也不会自动分箱。0/1/2 分别保留给 PAD/UNK/MISSING。布尔值、数字和字符串使用可持久化的类型编码区分，null 和空字符串为 MISSING。

## 原始数据接口

`vector_kc_preprocess.py` 接收单个 attempt 对象或 attempt 列表，包括：

- `user_id`、`days_since_signup`；可选 `session_index`、`exercise_part_index`。
- `events_data.kc_feature_names`：有序、唯一的字段名。
- `events_data.kc_vectors`：每个事件一个等长标量向量。
- `events_data.responses`：0/1 标签；若未提供，则使用 `reject_reason.event`（或列表），0 为正确，正整数为错误。
- 可选 `onset_seconds`，用于 attempt 内排序；缺省保留原事件顺序。

适配器接受所有 play_mode（包括 PianoKT 的 challenge）。标签使用事件级结果，不根据音符级结果重新推导。KC identity 只依赖完整字段组合，独立于歌曲和事件位置。Yousician 的四维原始数据仍通过原适配器转换，两者输出相同的 `student_fields_v1` 标准。

## 预处理与训练

在仓库根目录，使用完整的多学生 PianoKT 文件：

```bash
conda activate pykt
python examples/data_preprocess.py --dataset_name pianokt --model_name akt_fmkc --input_format vector --file_path /path/to/pianokt_attempts.json --output_dir data/pianokt/prepared --config_file configs/data_config.json --kfold 5
python examples/train_akt_fmkc.py --dataset_name pianokt --fold 0
```

也可一键预处理并对比 dense AKT 与 vector AKT-FMKC：

```bash
python examples/compare_akt_fmkc.py --dataset_name pianokt --input_format vector --raw_file /path/to/pianokt_attempts.json --save_dir saved_model/pianokt_comparison
```

学生级划分、训练词表、cache block、窗口、只对新增事件计算 loss、验证最优 checkpoint 的流程保持一致。测试集默认占学生数 20%；剩余学生至少需要 kfold 个。仓库的 `pianokt_attempt_example_v1_1.json` 只有一个学生，支持解析验证，但不能生成互斥的三组训练划分，程序会明确拒绝，不会复制学生或退化为事件随机划分。
