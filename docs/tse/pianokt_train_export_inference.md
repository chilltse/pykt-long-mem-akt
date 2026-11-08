可以按下面流程跑。模型名用 akt_fmkc。当前示范文件只有 2 名学生，可以读取；正式训练请换成真实数据。如果有 5 名学生，预处理指定 --kfold 4。

1．预处理

在项目根目录执行：

```bash
conda activate pykt
cd examples
python data_preprocess.py --dataset_name pianokt
```

只需要指定数据集名称，程序会自动选择 PianoKT 读取器，读取 `data/pianokt/pianokt.json`，输出到 `data/pianokt/`，并更新 `configs/data_config.json`。`model_name`、`file_path`、`output_dir`、`config_file` 和 `seed` 均无需手动指定；这些可选参数只用于覆盖默认行为。

默认 `kfold=5`，按原划分逻辑至少需要 6 名学生。如果实际只有 5 名学生，显式使用 4 折：

```bash
python data_preprocess.py --dataset_name pianokt --kfold 4
```

以 4 折为例，生成：

```text
data/pianokt/
  student_sequences.json
  student_fold_0.json
  student_fold_1.json
  student_fold_2.json
  student_fold_3.json
```

同时更新 configs/data_config.json 中的 pianokt 配置。

原始数据按 learner_key 聚合，attempt 按时间排序，内部按 onset 排序；每个 onset 对应一个六维向量和一个标签。字段词表只在各折训练学生上拟合。

2．训练

仍在 `examples` 目录，最简命令为：

```bash
python train_akt_fmkc.py --dataset_name pianokt
```

脚本默认使用 `akt_fmkc`，自动读取项目的 `configs/data_config.json`，训练第 0 折。数据路径和划分文件由预处理配置决定，字段词表及模型输入大小在加载对应折时确定；训练不会重新划分学生或拟合词表。

数据配置与训练超参数分开：模型维度、层数、学习率、训练轮数等采用训练入口的默认值（例如 30 轮、d_model=256、4 层）。需要调整时才传相应参数。默认 block_size=128、cache_blocks=1、seq_len=128，满足缓存容量要求。

若预处理生成 4 折，要跑完整交叉验证，分别指定 `--fold 0`、`--fold 1`、`--fold 2`、`--fold 3`；单条最简命令只训练第 0 折。

训练复用 pyKT 的共享入口：`wandb_train.main → init_dataset4train → init_model → train_model`。字段向量数据通过新增的 `student_fields_v1` 加载器和 AKTFMKC 模型分支接入。

3．取得 checkpoint

训练会自动保存，无需另行执行导出命令。默认 save_dir 是相对运行目录的 saved_model，因此在 examples 中执行时，项目内路径为：
examples/saved_model/<本次训练参数目录>/
  qid_model.ckpt
  config.json
  test_metrics.json

- qid_model.ckpt：训练选择的最佳模型权重。
- config.json：模型参数、该折编码词表和数据配置。
- test_metrics.json：测试集 AUC、accuracy、BCE 等。
部署时必须一起携带 qid_model.ckpt 和同目录的 config.json。推理恢复依赖这份配置里的词表，不能重新拟合或改用其他折的词表。

查找权重路径：
find saved_model -name qid_model.ckpt
4．推理：先预测，再记录真实结果

回到项目根目录：
cd ..
下面代码展示完整的单学生增量推理过程。把 checkpoint 替换成实际路径，可保存为 Python 文件后运行：
import json

from pykt.preprocess.vector_kc_preprocess import category
from pykt.models.akt_fmkc_runner import load_fmkc_checkpoint

checkpoint = "examples/saved_model/<本次训练参数目录>/qid_model.ckpt"

model, metadata = load_fmkc_checkpoint(checkpoint, device="cpu")
encoding = metadata["data_config"]["encoding"]

def encode_kc(vector):
    if len(vector) != len(encoding["fields"]):
        raise ValueError("KC 向量维度与训练时不一致")

```
# 与预处理时完全相同的类型编码和题目身份规则。
tokens = [category(value) for value in vector]
identity = json.dumps(
    tokens, ensure_ascii=False, separators=(",", ":")
)
attrs = [
    2 if token is None else mapping.get(token, 1)
    for mapping, token in zip(encoding["fields"], tokens)
]
pid = encoding["questions"].get(identity, 0)
return attrs, pid
```



# 模型或词表更换后，需要更换版本标识并重建历史状态。

versions = {
    "model_version": "pianokt-model-v1",
    "feature_version": "pianokt-encoding-v1",
}

# 每名学生独立维护一个 cache。

cache = model.empty_cache(**versions)

# 当前目标事件的 KC，来自预测前已知的乐谱属性。

vector = [False, 1, False, 30, 1, 1]
attrs, pid = encode_kc(vector)

# 演奏前：预测当前事件正确的概率，不更新历史。

probability = model.predict_next(cache, attrs, pid, **versions)
print("正确概率：", float(probability))

# 演奏后：取得真实标签，再更新历史。

response = 0  # 实际 correct；只能为 0 或 1
cache, _ = model.observe(
    cache,
    attrs,
    pid,
    response,
    event_id="attempt-001:onset-000",
    **versions,
)

# 下一事件继续使用更新后的 cache。

next_attrs, next_pid = encode_kc([False, 2, True, 50, 1, 2])
next_probability = model.predict_next(
    cache, next_attrs, next_pid, **versions
)
print("下一事件正确概率：", float(next_probability))
这里有三个关键约定：

- 预测当前事件时不传当前真实标签；演奏完成后才调用 observe。
- 同一学生按时间顺序更新，不同学生使用独立 cache。已有学生需要先按顺序 observe 历史事件，再预测下一事件。
- event_id 应唯一，例如 attempt_id:onset_index。服务需要断点恢复、持久化和重复请求处理时，可以使用现有的 [StudentStateStore](/Users/tse/my_code/long_mem_akt/pykt-long-mem-akt/pykt/models/akt_state_store.py)。
目前题目身份仍按完整 KC 向量生成；未知字段值回退到 UNK，缺失值单独编码，未见完整组合的 pid 为 0。

