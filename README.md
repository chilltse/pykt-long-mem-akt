# pyKT

[![Downloads](https://pepy.tech/badge/pykt-toolkit)](https://pepy.tech/project/pykt-toolkit)
[![GitHub Issues](https://img.shields.io/github/issues/pykt-team/pykt-toolkit.svg)](https://github.com/pykt-team/pykt-toolkit/issues)
[![Documentation](https://img.shields.io/website/http/pykt-team.github.io/index.html?down_color=red&down_message=offline&up_message=online)](https://pykt.org/)

pyKT is a python library build upon PyTorch to train deep learning based knowledge tracing models. The library consists of a standardized set of integrated data preprocessing procedures on more than 7 popular datasets across different domains, 5 detailed prediction scenarios, more than 10 frequently compared DLKT approaches for transparent and extensive experiments. More details about pyKT can see our [website](https://pykt.org/) and [docs](https://pykt-toolkit.readthedocs.io/en/latest/quick_start.html).




## Installation
Use the following command to install pyKT:

Create conda envirment.

```
conda create --name=pykt python=3.7.5
source activate pykt
```


```
pip install -U pykt-toolkit -i  https://pypi.python.org/simple 

```

## Hyper parameter tunning results
The hyper parameter tunning results of our experiments about all the DLKT models on various datasets can be found at https://drive.google.com/drive/folders/1MWYXj73Ke3zC6bm3enu1gxQQKAHb37hz?usp=drive_link.

## References
### Projects

1. https://github.com/hcnoh/knowledge-tracing-collection-pytorch 
2. https://github.com/arshadshk/SAKT-pytorch 
3. https://github.com/shalini1194/SAKT 
4. https://github.com/arshadshk/SAINT-pytorch 
5. https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing- 
6. https://github.com/arghosh/AKT 
7. https://github.com/JSLBen/Knowledge-Query-Network-for-Knowledge-Tracing 
8. https://github.com/xiaopengguo/ATKT 
9. https://github.com/jhljx/GKT 
10. https://github.com/THUwangcy/HawkesKT
11. https://github.com/ApexEDM/iekt
12. https://github.com/Badstu/CAKT_othermodels/blob/0c28d870c0d5cf52cc2da79225e372be47b5ea83/SKVMN/model.py
13. https://github.com/bigdata-ustc/EduKTM
14. https://github.com/shalini1194/RKT
15. https://github.com/shshen-closer/DIMKT
16. https://github.com/skewondr/FoLiBi
17. https://github.com/yxonic/DTransformer
18. https://github.com/lilstrawberry/ReKT

### Papers

1. DKT: Deep knowledge tracing 
2. DKT+: Addressing two problems in deep knowledge tracing via prediction-consistent regularization 
3. DKT-Forget: Augmenting knowledge tracing by considering forgetting behavior 
4. KQN: Knowledge query network for knowledge tracing: How knowledge interacts with skills 
5. DKVMN: Dynamic key-value memory networks for knowledge tracing 
6. ATKT: Enhancing Knowledge Tracing via Adversarial Training 
7. GKT: Graph-based knowledge tracing: modeling student proficiency using graph neural network 
8. SAKT: A self-attentive model for knowledge tracing 
9. SAINT: Towards an appropriate query, key, and value computation for knowledge tracing 
10. AKT: Context-aware attentive knowledge tracing 
11. HawkesKT: Temporal Cross-Effects in Knowledge Tracing
12. IEKT: Tracing Knowledge State with Individual Cognition and Acquisition Estimation
13. SKVMN: Knowledge Tracing with Sequential Key-Value Memory Networks
14. LPKT: Learning Process-consistent Knowledge Tracing
15. QIKT: Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations
16. RKT: Relation-aware Self-attention for Knowledge Tracing
17. DIMKT: Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect
18. ATDKT: Enhancing Deep Knowledge Tracing with Auxiliary Tasks
19. simpleKT: A Simple but Tough-to-beat Baseline for Knowledge Tracing
20. SparseKT: Towards Robust Knowledge Tracing Models via K-sparse Attention
21. FoLiBiKT: Forgetting-aware Linear Bias for Attentive Knowledge Tracing
22. DTransformer: Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer
23. stableKT: Enhancing Length Generalization for Attention Based Knowledge Tracing Models with Linear Biases
24. extraKT: Extending Context Window of Attention Based Knowledge Tracing Models via Length Extrapolation
25. csKT: Addressing Cold-start Problem in Knowledge Tracing via Kernel Bias and Cone Attention
26. LefoKT: Rethinking and Improving Student Learning and Forgetting Processes for Attention Based Knowledge Tracing Models
<!-- 27. FlucKT: Cognitive Fluctuations Enhanced Attention Network for Knowledge Tracing -->
27. UKT: Uncertainty-aware Knowledge Tracing
28. HCGKT: Hierarchical Contrastive Graph Knowledge Tracing with Multi-level Feature Learning
29. MTKT: Learning multi-granularity temporal characteristics for attention based knowledge tracing
30. RobustKT: Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data
31. MoCKT: : Mixture of Convolutions for Knowledge Tracing
32. DenoiseKT: Denoised Attention and Question-Augmented Representations for Knowledge Tracing

## LongMemAKT

现有 `long_mem_akt` 已完整替换为选择性长期修正模型。旧四模式、旧参数和旧 checkpoint 不再支持。
唯一训练入口为 `examples/wandb_long_mem_akt_train.py`。

```bash
cd examples

# 自动完成学生划分、基线训练、候选训练、收益头训练和校准
PYTHONPATH=.. python wandb_long_mem_akt_train.py --stage train \
  --dataset_name bridge2algebra2006 --fold 0 --block_size 200 \
  --epochs 20 --output saved_model/long_mem_akt_v2

# 评估最终模型，导出指标和每名学生的预测
PYTHONPATH=.. python wandb_long_mem_akt_train.py --stage evaluate \
  --source saved_model/long_mem_akt_v2/model --block_size 200 \
  --output saved_model/long_mem_akt_v2/test
```

默认 CPU，使用 GPU 时加 `--device cuda`。输出目录必须尚不存在。
模型保存于 `model/qid_model.ckpt`；评估结果为 `test/metrics.json` 和 `test/student_block_predictions.csv`。
`base/candidate/benefit` 是同一个模型的训练阶段，不是旧模型选项。
详细实现见 [实现说明](docs/long_mem_akt_implementation_cn.md)。

一键消融对比（在 `examples` 目录执行，使用已训练的同一份模型）：

```bash
PYTHONPATH=.. python evaluate_long_mem_akt_ablation.py \
  --source saved_model/long_mem_akt_v2/model \
  --output saved_model/long_mem_akt_v2/ablations
```

自动对比 E0（AKT 基线）、E1（完整长期残差）、E2（不确定性门控）、E3（双门控）。
分块、MC 设置和计分协议自动读取 checkpoint。默认复用各自校准结果；
加 `--recalibrate` 仅用 calibration 学生重新校准 E2/E3，不使用测试标签选参数。
结果包含 `summary.csv`、`summary.md`、`metrics.json`、`students.csv` 和四组逐条预测 CSV。
脚本校验目标/计分掩码与基线预测一致，不修改来源模型。GPU 使用 `--device cuda`。

## Citation

We now have a [paper](https://arxiv.org/abs/2206.11460?context=cs.CY) you can cite for the our pyKT library:

```bibtex
@inproceedings{liupykt2022,
  title={pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models},
  author={Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and Huang, Shuyan and Tang, Jiliang and Luo, Weiqi},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```
