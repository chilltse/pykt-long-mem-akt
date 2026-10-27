"""Export plots and a Chinese report from the completed synthetic benchmark."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'B0_akt':'AKT pretrained','B1_akt_continue':'AKT continued','F_recent':'V2 recent',
        'F_exact':'V2 exact','F_additive':'V2 additive','G_exact':'V2 gate-only'}


def main(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    result=json.loads((source/'comparison.json').read_text())
    config=json.loads((source/'resolved_config.json').read_text())['arguments']
    if config['length'] != 1000 or config['block_len'] != 200:
        raise ValueError('This report layout expects length=1000, block_len=200')
    seeds=[int(v) for v in config['seeds'].split(',')]
    variants=list(LABELS)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    x=np.arange(len(variants))
    for j,(subset,title) in enumerate([('all','All scored positions'),('return_block','Final 200 interactions'),('first_return','First delayed revisit')]):
        for i,metric in enumerate(['auc','nll']):
            ax=axes[i,j]
            means=[result['summary'][v][subset][metric]['mean'] for v in variants]
            std=[result['summary'][v][subset][metric]['std'] for v in variants]
            ax.errorbar(x,means,yerr=std,fmt='none',capsize=4,ecolor='#475569')
            ax.scatter(x,means,s=50,c=['#a1a1aa','#475569','#22a699','#2563eb','#8b5cf6','#f59e0b'])
            ax.set_xlim(-.5,len(variants)-.5)
            ax.set_xticks(x);ax.set_xticklabels([LABELS[v] for v in variants],rotation=30,ha='right',fontsize=8)
            ax.set_title(title);ax.set_ylabel(metric.upper()+(' (higher better)' if metric=='auc' else ' (lower better)'))
            ax.set_ylim(max(0,min(means)-max(std)-.025),max(means)+max(std)+.025)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Synthetic 1000-step delayed revisit | mean +/- population SD, {} training seeds'.format(len(seeds)))
    fig.tight_layout();fig.savefig(destination/'synthetic_1000_ablation.png',dpi=160);plt.close(fig)

    with (source/'data/train.csv').open() as f: student=next(csv.DictReader(f))
    c=np.array(list(map(int,student['concepts'].split(','))))
    r=np.array(list(map(int,student['responses'].split(','))))
    fig,ax=plt.subplots(figsize=(12,3.2))
    ax.scatter(np.arange(len(c)),c,c=r,cmap='coolwarm',vmin=0,vmax=1,s=6)
    ax.axvspan(0,200,color='#2563eb',alpha=.08,label='Target concepts 0-31')
    ax.axvspan(200,800,color='#64748b',alpha=.08,label='Distractor concepts 32-47')
    ax.axvspan(800,1000,color='#2563eb',alpha=.08)
    for t in range(0,1001,200): ax.axvline(t,color='grey',alpha=.3,linewidth=.8)
    ax.set(xlabel='Interaction position',ylabel='Concept ID',title='One student: learn, 600-step absence, delayed revisit (red=correct, blue=incorrect)')
    ax.legend(loc='lower center',ncol=2);fig.tight_layout();fig.savefig(destination/'synthetic_1000_design.png',dpi=160);plt.close(fig)

    config=json.loads((source/'resolved_config.json').read_text())['arguments']
    lines=['# 1000 步合成学习序列：AKT 与 LongMemTopKAKT V2 消融',
           '', '## 数据与协议', '',
           '- 每学生 1000 次交互，48 个知识点；{}/{}/{} 名训练/验证/测试学生，共 {} 条交互。'.format(config['train_students'],config['valid_students'],config['test_students'],1000*(config['train_students']+config['valid_students']+config['test_students'])),
           '- 前 200 步练习知识点 0–31，中间 600 步练习 32–47，最后 200 步回访 0–31。首次回访间隔超过 600 步，recent=200 无法保存对应历史。',
           '- 每学生每知识点有独立初始掌握状态，未掌握状态每次练习以 0.015 概率转为掌握；回答包含能力、知识点难度和 Bernoulli 噪声。模型只能读取知识点与历史回答，不接收学生 ID、隐状态或阶段标签。',
           '- block_len=200；每块首题均不计分，其他有效位置一致。测试整体 {} 个位置，回访块 {} 个，首次回访 {} 个。'.format(config['test_students']*995,config['test_students']*199,config['test_students']*31),
           '- 数据种子固定 {}；模型种子 {}。报告均值 ± 总体标准差，反映训练随机性，不代表数据重采样置信区间。'.format(config['data_seed'],config['seeds']),
           '- AKT：qid、n_pid=0（无题目 ID 难度分支），d_model=32、n_blocks=2、4 heads、d_ff=64、dropout=0.1。remote_capacity=1000、Top-k=32、exact 检索。',
           '- B0 全参数预训练 {} epochs；所有继续训练消融均为 {} epochs，共用该种子的 B0 checkpoint。按验证 NLL 选 checkpoint，不根据测试集选配置。'.format(config['pretrain_epochs'],config['finetune_epochs']),
           '- B1 与 F 系列都只训练最后 Retriever 和预测头；G_exact 只训练 gate，因此它是冻结范围消融。学习率：预训练 1e-3、共享参数微调 1e-4、gate 1e-3。',
           '', '![数据设计](synthetic_1000_design.png)', '', '## 结果', '',
           '| 配置 | 整体 AUC | 整体 NLL | 整体 ACC | 回访块 AUC | 首次回访 AUC | 首次回访 NLL |',
           '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v,s,m):
        a=result['summary'][v][s][m]
        return '{:.4f} ± {:.4f}'.format(a['mean'],a['std'])
    for v in variants:
        lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(v,fmt(v,'all','auc'),fmt(v,'all','nll'),fmt(v,'all','acc'),fmt(v,'return_block','auc'),fmt(v,'first_return','auc'),fmt(v,'first_return','nll')))
    exact=result['summary']['F_exact'];base=result['summary']['B1_akt_continue'];recent=result['summary']['F_recent']
    lines+=['','本次结果：F_exact 相比匹配预算的 B1，整体 AUC 均值增加 {:.6f}，NLL 变化 {:+.6f}；但相比仅 recent，AUC 只增加 {:.6f}，NLL 变化 {:+.6f}。远期缓存的额外收益很小。'.format(exact['all']['auc']['mean']-base['all']['auc']['mean'],exact['all']['nll']['mean']-base['all']['nll']['mean'],exact['all']['auc']['mean']-recent['all']['auc']['mean'],exact['all']['nll']['mean']-recent['all']['nll']['mean']),
            '', '首次回访 AUC 仍约为 0.52，且 F_exact 的首次回访 NLL 均值未优于 B1。本轮没有显示出充分利用远期信息的效果。历史均值对照表明数据中确实有可预测的长期信号，而当前神经模型连块内统计对照也尚未超过；应把这视为当前训练/表示/记忆配置的小规模诊断，不据此否定长期历史的价值。',
            '', '没有进行显著性检验；三种子标准差仅描述训练随机性。上述测试结果没有用于重新选择学习率、候选数或训练轮数。']
    lines+=['','![消融结果](synthetic_1000_ablation.png)','','## 配对差异与记忆依赖','','下表为 F_exact − B1，同种子比较；AUC 正值/NLL 负值代表更好。','','| seed | 整体 ΔAUC | 整体 ΔNLL | 首次回访 ΔAUC | 首次回访 ΔNLL | 训练后关闭 gate：整体 ΔNLL |','|---|---:|---:|---:|---:|---:|']
    for seed in seeds:
        selected={r['variant']:r for r in result['results'] if r['seed']==seed}
        b=selected['B1_akt_continue']['report']['subsets'];f=selected['F_exact']['report']['subsets']
        inter=json.loads((source/('seed'+str(seed))/('F_exact_seed'+str(seed))/'gate_zero_metrics.json').read_text())
        lines.append('| {} | {:+.5f} | {:+.5f} | {:+.5f} | {:+.5f} | {:+.5f} |'.format(seed,f['all']['auc']-b['all']['auc'],f['all']['nll']-b['all']['nll'],f['first_return']['auc']-b['first_return']['auc'],f['first_return']['nll']-b['first_return']['nll'],inter['subsets']['all']['nll']-f['all']['nll']))
    sanity_path=destination/'synthetic_1000_history_signal.json'
    if sanity_path.exists():
        sanity=json.loads(sanity_path.read_text())['results']
        lines+=['','## 数据中长期信息的独立检查','','固定 Laplace 平滑的同知识点历史均值 `(过去正确次数+1)/(过去练习次数+2)`，严格先预测再更新当前回答；不读取隐状态、不调参。','','| 可用历史 | 整体 AUC | 首次回访 AUC | 首次回访 NLL |','|---|---:|---:|---:|']
        for mode in ['block_only','all_history']:
            r=sanity[mode]
            lines.append('| {} | {:.4f} | {:.4f} | {:.4f} |'.format(mode,r['all']['auc'],r['first_return']['auc'],r['first_return']['nll']))
        lines.append('\n这证明本合成任务存在可用的远期信息，但不保证神经模型的当前检索与融合方式能利用好它。')
    full_path=source/'full_context_diagnostic.json'
    if full_path.exists():
        full=json.loads(full_path.read_text())
        lines+=['','## AKT 完整 1000 步上下文诊断','','保持 B0 权重不变，仅把推理上下文从 200 扩为 1000，并保持完全相同的计分位置。这不是在 1000 步上下文上重新训练的模型。','','| 推理上下文 | 整体 AUC | 整体 NLL | 首次回访 AUC | 首次回访 NLL |','|---|---:|---:|---:|---:|']
        values=[]
        for subset,metric in [('all','auc'),('all','nll'),('first_return','auc'),('first_return','nll')]:
            items=[r['report']['subsets'][subset][metric] for r in full['results']]
            values.append('{:.4f} ± {:.4f}'.format(np.mean(items),np.std(items)))
        lines.append('| 1000 | '+' | '.join(values)+' |')
        lines.append('| 200（B0） | '+' | '.join(fmt('B0_akt',subset,metric) for subset,metric in [('all','auc'),('all','nll'),('first_return','auc'),('first_return','nll')])+' |')
    lines+=['','## 预算与资源','','| 配置 | 每种子 optimizer steps | 每种子计分训练标签 | 平均训练秒数 | 平均测试秒数 | 最大 bank 字节 |','|---|---:|---:|---:|---:|---:|']
    for v in variants:
        selected=[r for r in result['results'] if r['variant']==v]
        lines.append('| {} | {} | {} | {:.1f} | {:.1f} | {} |'.format(v,selected[0]['budget']['optimizer_steps'],selected[0]['budget']['scored_labels'],np.mean([r['budget']['train_seconds'] for r in selected]),np.mean([r['report']['seconds'] for r in selected]),max(r['report']['bank_bytes_peak'] for r in selected)))
    lines+=['','继续训练预算不含共同 B0 预训练；B0 本身不是匹配继续训练预算的对照。测试和训练均为 CPU；三个独立种子并发，每个进程一个 PyTorch 线程。运行秒数受并发和共享 CPU 影响，不作为严格吞吐基准。RSS 是各进程累计峰值，详细值保存在 comparison.json。',
            '', '## 验证与边界', '',
            '- 每种子的 B0/Z0 对齐相同 uid/position/target，并断言最大预测误差 ≤1e-6。各消融计分位置、optimizer steps、有效交互与计分训练标签一致。',
            '- 数据设计和可重复性测试：`tests/test_synthetic_longmem_benchmark.py`，2 项通过；另以 200 步小数据、2 种子跑通 12 个端到端训练/评估流程（仅作为流水线测试，不计入本表）。',
            '- 人造任务有意保留长期证据，结果仅反映该任务。不能据此推导真实数据集收益；样本量较小，只有一个数据种子，尚未做超参搜索。预训练只有 6 轮，验证损失仍在下降，不能视为充分收敛的最优模型对比。',
            '- memory additive 仅改变 memory 分支；current 仍为原 AKT。gate=0 是同一训练后 checkpoint 的依赖性干预，不是独立训练 baseline。',
            '', '## 复运行与产物', '', '```bash',
            'PYTHONPATH=. OMP_NUM_THREADS=1 /home/remote/u8016457/miniconda3/envs/pykt/bin/python \\',
            '  examples/experiment_long_mem_topk_akt_synthetic.py --output /tmp/synthetic1000-new-run',
            '```', '',
            '完整产物：`{}`，包括 `data/*.csv`、数据 SHA256、resolved configs、每种子的 best.pt、逐 epoch 日志、逐位置预测、gate 干预、`summary.csv` 和 `comparison.json`。输出目录须尚不存在。'.format(source),
            '', '中途停止的串行耗时试跑位于相邻的 `_pilot` 目录；其结果不进入任何汇总。']
    (destination/'synthetic_1000_ablation_cn.md').write_text('\n'.join(lines)+'\n')
    (destination/'synthetic_1000_ablation_summary.json').write_text(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--destination',default='docs/experiments')
    args=p.parse_args();main(args.source,args.destination)
