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

## LongMemTopKAKT

新增 `long_mem_topk_akt`：保留原 AKT 参数和预测头，所有 attention 层加入近期缓存，仅 Knowledge Retriever 加入远期 Top-k 检索，支持 Exact 与 HNSW inner-product。与现有 `long_mem_akt` 独立。

训练入口：`examples/wandb_long_mem_topk_akt_train.py`（从原 AKT checkpoint 联合微调）；同条件消融入口：`examples/evaluate_long_mem_topk_akt_ablation.py`（默认比较 AKT / recent / exact / HNSW，统一 train/valid/test 和计分位置）。

运行命令、因果约束、近似边界和成本解释见 [实现说明](docs/long_mem_topk_akt_implementation_cn.md)。

## LongMemAKT

当前只保留 E1：有历史块时输出 `sigmoid(AKT_logit + residual_logit)`，不使用预测门控。首个有效块使用 AKT；非有限残差回退到 AKT。

## 数据与训练

采用标准 pykt fold 划分：指定 fold 为 valid，其余训练 fold 全部为 train，原始测试文件中 fold=-1 为 test。按学生隔离，无额外数据子集。

一个训练命令内部先训练 AKT（valid AUC 选择 checkpoint），再冻结 AKT，训练长期残差（valid logloss 选择 checkpoint）。两个阶段共用 train/valid，`epochs` 为每阶段最多轮数，连续 10 轮无改善早停。保留 E1 原训练语义，未改成联合训练。test 只在显式评估时读取用于预测。

## 记忆与残差

每名学生独立初始化 Value memory，共享 Key 槽；完整轨迹按连续非重叠块处理。块内 AKT 保持严格因果预测，整块读取进入该块时的记忆，观察完该块后才写入。

残差头输入为归一化问题表示、短期表示、长期读取表示及 AKT logit。最后一层零初始化。长期写入保留因果编码、时间衰减、erase/add 更新；这些写入机制不属于已删除的预测门控。

TBPTT 默认跨 2 块，残差训练要求至少 2 块；组间截断并重放边界写入，让后续块损失监督写入器。每批一名学生，避免轨迹或状态混用。

默认 `score_protocol=legacy` 保留 pykt 原分段计分掩码；`student` 按完整学生轨迹计分，跳过第一个交互。两种协议不要直接混比。

## 使用

在 `examples` 目录运行：

```bash
PYTHONPATH=.. python wandb_long_mem_akt_train.py --dataset_name bridge2algebra2006 --fold 0 --block_size 200 --epochs 20 --output saved_model/long_mem_e1
PYTHONPATH=.. python wandb_long_mem_akt_train.py --stage evaluate --source saved_model/long_mem_e1 --block_size 200 --output saved_model/long_mem_e1_test
PYTHONPATH=.. python evaluate_long_mem_akt_ablation.py --source saved_model/long_mem_e1 --output saved_model/long_mem_e1_comparison
```

默认 CPU，GPU 加 `--device cuda`。输出目录必须尚不存在。模型直接保存在输出目录的 `qid_model.ckpt`，配套 `config.json`、`roles.json`、`long_mem_manifest.json` 和训练日志，不再生成多阶段子目录。

也支持标准预测入口 `wandb_predict.py --save_dir saved_model/long_mem_e1 --use_wandb 0`。训练保留专用入口，以自动完成 E1 的两步优化和学生级记忆处理，内部复用 pykt `train_model`。

消融仅比较同一 checkpoint 的 E0（冻结 AKT）与 E1（完整残差），输出 `metrics.json` 和两份预测 CSV。旧选择性模型 checkpoint 不兼容，请重新训练。

详细说明见 [实现说明](docs/long_mem_akt_implementation_cn.md)。

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
