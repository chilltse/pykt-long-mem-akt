# 本地曲库与未训练学生历史实验

已完成小规模离线运行；真实冻结 checkpoint + 推荐曲库预提取特征 + 文件自报 synthetic 的两名学生历史。不是人类学习效果验证。

曲库721个distinct KC；已知314个（43.55%），固定1570个Probe。240段中39段完全已知，预先按onset数选择最短3段，未按Gain筛选。
选择MC=4，独立评价=16，参考=32；预算扫描4/8，各重复2次。参考只覆盖3个候选，仍有MC噪声；不是全曲库最优。

## 独立评价宏平均Gain

- sweet_spot: Gain=0.230435；参考Regret=0.001239。
- one_step_gain: Gain=0.230435；参考Regret=0.001239。
- full_segment_mc: Gain=0.251169；参考Regret=0.034849。
- random: Gain=0.236235；参考Regret=0.011468。

## 数值与状态诊断

- d80417e24d40737d181f: 历史6905步；No-op=0.0；批量与逐条最大误差=8.52e-06；μ非单调比例=0.00%。
- 588bf8f9eb998bec1451: 历史11218步；No-op=0.0；批量与逐条最大误差=1.18e-05；μ非单调比例=0.00%。

原始learned μ覆盖通过1e-6；批量路径因FP32计算次序差异采用2e-5。缓存窗口128，不等于完整历史无损保留。
五档中±0.25、±0.5超出learned μ范围；不是验证过的五级演奏难度。
时间归一化分母是首末onset跨度，不含最后音符持续时间。
仅两个合成学生，bootstrap区间只作描述；不能推断真实学生群体。未单独核验学生是否训练集之外，该属性来自用户说明。

## 复现

```bash
bash scripts/run_mu_benchmark.sh --mode offline --config configs/experiments/local_mu_probe.yaml --output /tmp/kt-mu-repeat --candidate-limit 3 --student-limit 2 --selection-mc 4 --evaluation-mc 16 --reference-mc 32
```

## 模型偏差提示

检查state_diagnostics.csv中的all_wrong：若全部答错仍产生正Gain，应优先排查历史状态更新和窗口移动，不能把正Gain直接解释为学习。

## 未完成

全预算32/256/512、全39候选、128预算扫描、自适应参考增采样、多历史窗口敏感性、真实future outcome校准未执行。现有runtime是评价耗时，不是各策略选曲成本。
下一步应先扩大候选和独立MC预算，检查优势是否超过误差，再判断是否值得做多Segment Planning。