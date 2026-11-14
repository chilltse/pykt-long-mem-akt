"""Supplement a finished run with diagnostic plots and a Chinese status report."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]); results=json.loads((root/'results.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
fig=root/'figures';fig.mkdir(exist_ok=True)
def save(name,title):
 plt.title(title+' (model simulation)');plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.close()
def export(name,rows):
 with (root/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
budgets=[dict(student_checkpoint=r['student_checkpoint'],**b) for r in results for b in r['budget_sweep']]
export('mc_budget_metrics.csv',budgets)
for metric in ['regret','top1_selection_stability']:
 for r in results:
  bs=sorted({b['mc_budget'] for b in r['budget_sweep']})
  plt.plot(bs,[np.mean([b[metric] for b in r['budget_sweep'] if b['mc_budget']==n]) for n in bs],marker='o',label=r['student_checkpoint'][:6])
 plt.xlabel('MC trajectories');plt.legend();save('budget_'+metric,metric)
plt.bar([r['student_checkpoint'][:6] for r in results],[r['diagnostics']['mu_nonmonotone_fraction'] for r in results]);save('mu_monotonicity','Fraction of nonmonotone KC curves')
for r in results:
 for method,s in r['selected'].items():
  plt.scatter(np.mean(s['evaluation']['p_before']),s['evaluation']['macro_gain'],label=method if r is results[0] else None)
plt.xlabel('Initial macro probability');plt.legend();save('initial_ability_gain','Initial prediction and gain (two histories)')
for r in results:
 for c in r['diagnostics']['candidates']:
  plt.scatter(c['probe_overlap_ratio'],c['variants']['original_order']['macro_gain'])
plt.xlabel('Exact KC overlap fraction');save('overlap_gain','Overlap and macro gain')
diagnostic_rows=[]
for r in results:
 for c in r['diagnostics']['candidates']:
  for name,v in c['variants'].items():
   diagnostic_rows.append(dict(student_checkpoint=r['student_checkpoint'],segment_id=c['segment_id'],variant=name,
    gain=v['macro_gain'],sem=v['macro_sem'],history_length=c['history_length'],cache_length=c['cache_length'],
    rollout_length=c['rollout_length'],probe_overlap_ratio=c['probe_overlap_ratio'],
    overlap_gain=c['gain_on_exact_overlapping_kcs'],nonoverlap_gain=c['gain_on_nonoverlapping_kcs']))
export('state_diagnostics.csv',diagnostic_rows)
export('student_checkpoints.csv',[dict(student_checkpoint=r['student_checkpoint'],student_cluster=r['student_cluster'],history_length=r['history_length']) for r in results])
metrics=list(csv.DictReader((root/'evaluation_metrics.csv').open()))
coverage=json.loads((root/'kc_coverage.json').read_text())
lines=['# 本地曲库与未训练学生历史实验','',
'已完成小规模离线运行；真实冻结 checkpoint + 推荐曲库预提取特征 + 文件自报 synthetic 的两名学生历史。不是人类学习效果验证。','',
f"曲库721个distinct KC；已知{coverage['known_kc']}个（{coverage['known_coverage']:.2%}），固定{coverage['probe_count']}个Probe。240段中39段完全已知，预先按onset数选择最短3段，未按Gain筛选。",
'选择MC=4，独立评价=16，参考=32；预算扫描4/8，各重复2次。参考只覆盖3个候选，仍有MC噪声；不是全曲库最优。','',
'## 独立评价宏平均Gain','']
for m in metrics:lines.append(f"- {m['method']}: Gain={float(m['mean_gain']):.6f}；参考Regret={float(m['mean_regret']):.6f}。")
lines+=['','## 数值与状态诊断','']
for r in results:
 lines.append(f"- {r['student_checkpoint']}: 历史{r['history_length']}步；No-op={r['noop_max_error']}；批量与逐条最大误差={r['probe_batch_max_abs_error']:.3g}；μ非单调比例={r['diagnostics']['mu_nonmonotone_fraction']:.2%}。")
lines+=['','原始learned μ覆盖通过1e-6；批量路径因FP32计算次序差异采用2e-5。缓存窗口128，不等于完整历史无损保留。',
'五档中±0.25、±0.5超出learned μ范围；不是验证过的五级演奏难度。',
'时间归一化分母是首末onset跨度，不含最后音符持续时间。',
'仅两个合成学生，bootstrap区间只作描述；不能推断真实学生群体。未单独核验学生是否训练集之外，该属性来自用户说明。',
'','## 复现','',
'```bash','bash scripts/run_mu_benchmark.sh --mode offline --config configs/experiments/local_mu_probe.yaml --output /tmp/kt-mu-repeat --candidate-limit 3 --student-limit 2 --selection-mc 4 --evaluation-mc 16 --reference-mc 32','```','',
'## 模型偏差提示','',
'检查state_diagnostics.csv中的all_wrong：若全部答错仍产生正Gain，应优先排查历史状态更新和窗口移动，不能把正Gain直接解释为学习。',
'','## 未完成','',
'全预算32/256/512、全39候选、128预算扫描、自适应参考增采样、多历史窗口敏感性、真实future outcome校准未执行。现有runtime是评价耗时，不是各策略选曲成本。',
'下一步应先扩大候选和独立MC预算，检查优势是否超过误差，再判断是否值得做多Segment Planning。']
(root/'report.md').write_text('\n'.join(lines))
print('\n'.join(lines[:14]))
