可以按下面流程跑。模型名用 akt_fmkc。当前示范文件只有 2 名学生，可以读取；正式训练请换成真实数据。如果有 5 名学生，预处理指定 --kfold 4。

1．预处理

在项目根目录执行：
conda activate pykt
cd examples

python data_preprocess.py   
  --dataset_name pianokt   
  --model_name akt_fmkc   
  --file_path ../data/pianokt/pianokt.json   
  --output_dir ../data/pianokt/prepared   
  --config_file ../configs/data_config.json   
  --kfold 4   
  --seed 1024
这一步会生成：
data/pianokt/prepared/
  student_sequences.json
  student_fold_0.json
  student_fold_1.json
  student_fold_2.json
  student_fold_3.json
同时更新 configs/data_config.json 中的 pianokt 配置。

原始数据按 learner_key 聚合，attempt 按时间排序，内部按 onset 排序；每个 onset 对应一个六维向量和一个标签。字段词表只在各折训练学生上拟合。

2．训练

仍在 examples 目录：
python train_akt_fmkc.py   
  --dataset_name pianokt   
  --data_config ../configs/data_config.json   
  --model_name akt_fmkc   
  --fold 0   
  --save_dir ../saved_model/pianokt   
  --num_epochs 30   
  --d_model 128   
  --d_ff 256   
  --field_dim 16   
  --n_blocks 2   
  --num_attn_heads 8   
  --dropout 0.2   
  --learning_rate 0.0001   
  --block_size 128   
  --cache_blocks 1   
  --seq_len 128   
  --use_wandb 0
这是一个可用的起始配置。fold 0 表示第 0 折验证；要跑完整交叉验证，分别执行 --fold 0/1/2/3。

block_size × cache_blocks 必须至少覆盖 seq_len，上面的参数满足要求。

3．取得 checkpoint

训练会自动保存，无需另行执行导出命令：
saved_model/pianokt/<本次训练参数目录>/
  qid_model.ckpt
  config.json
  test_metrics.json

- qid_model.ckpt：训练选择的最佳模型权重。
- config.json：模型参数、该折编码词表和数据配置。
- test_metrics.json：测试集 AUC、accuracy、BCE 等。
部署时必须一起携带 qid_model.ckpt 和同目录的 config.json。推理恢复依赖这份配置里的词表，不能重新拟合或改用其他折的词表。

查找权重路径：
find ../saved_model/pianokt -name qid_model.ckpt
4．推理：先预测，再记录真实结果

回到项目根目录：
cd ..
下面代码展示完整的单学生增量推理过程。把 checkpoint 替换成实际路径，可保存为 Python 文件后运行：
import json

from pykt.preprocess.vector_kc_preprocess import category
from pykt.models.akt_fmkc_runner import load_fmkc_checkpoint

checkpoint = "saved_model/pianokt/<本次训练参数目录>/qid_model.ckpt"

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

