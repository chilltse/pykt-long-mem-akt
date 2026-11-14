"""CPU analysis of persisted predictions; never trains or selects models."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .runtime import read,save,rows,complete
from .metrics import metrics,frequency_group,cold_group,bootstrap


def summarize_rows(records):
    m=metrics([r['response'] for r in records],[r['predicted_probability'] for r in records])
    students=defaultdict(list)
    for r in records:students[r['student_id']].append(r)
    sm=[metrics([r['response'] for r in rs],[r['predicted_probability'] for r in rs]) for rs in students.values()]
    for name in ('auc','accuracy','logloss','brier'):
        values=[v[name] for v in sm if v[name] is not None]
        m['macro_student_'+name]=float(np.mean(values)) if values else None
    m.update(unique_student_count=len(students),unique_kc_count=len({r['kc_key'] for r in records}),
             macro_auc_valid_students=sum(v['auc'] is not None for v in sm),
             macro_auc_excluded_students=sum(v['auc'] is None for v in sm))
    return m


def csv_write(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in records for k,v in r.items() if not isinstance(v,(dict,list))))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)


def analyze(root,replicates=2000):
    root=Path(root); tasks=read(root/'task_manifest.json');overall=[];freq=[];cold=[];pairs=defaultdict(dict);missing=[]
    for task in tasks:
        if not complete(task):missing.append(task);continue
        d=Path(task['output_dir'])
        status=read(d/'task_status.json')
        identity=read(root/'code_identity.json')
        if any(status.get(k)!=v for k,v in identity.items()):raise ValueError('Mixed source versions cannot be compared')
        if status.get('dataset_sha256')!=read(root/'experiment_manifest.json')['dataset_sha256']:
            raise ValueError('Mixed dataset versions cannot be compared')
        records=rows(d/'predictions_test.jsonl.gz')
        base={k:task[k] for k in ('task_id','model_name','train_student_count','fold','seed','actual_train_student_count')}
        m=summarize_rows(records);save(d/'test_metrics.json',m)
        save(d/'validation_metrics.json',summarize_rows(rows(d/'predictions_valid.jsonl.gz')))
        scale=read(Path(read(task['prepared_config_path'])['yousician']['dpath'])/'scale_statistics.json')
        with (d/'epoch_metrics.csv').open() as stream:
            epochs=list(csv.DictReader(stream))
        overall.append(dict(base,**m,**read(d/'training_result.json'),**read(d/'parameter_counts.json'),**scale,
                            optimizer_steps=int(epochs[-1]['optimizer_steps']),supervised_events=int(epochs[-1]['supervised_events']),total_epochs=len(epochs)))
        for i in range(7):
            group=[r for r in records if frequency_group(r['train_frequency'])==i]
            freq.append(dict(base,group=f'G{i}',**summarize_rows(group),group_fraction_of_all_test_events=len(group)/len(records)))
        for name in ('seen','unseen_known_fields','unseen_unknown_fields'):
            group=[r for r in records if cold_group(r)==name]
            cold.append(dict(base,group=name,**summarize_rows(group)))
            for first in (True,False):
                cold.append(dict(base,group=f'{name}_first_{first}',**summarize_rows([r for r in group if r['is_first_kc_occurrence_for_student']==first])))
        unique=list({r['kc_key']:r for r in records}.values())
        save(d/'coverage_metrics.json',dict(question_vocabulary_size=scale['question_vocabulary_size'],field_vocabulary_sizes=scale['field_vocabulary_sizes'],
             fully_known_field_unseen_combination_rate=float(np.mean([r['is_unseen_combination'] and not r['has_unknown_field'] and not r['has_missing_field'] for r in records])),
             unique_kc_unseen_rate=float(np.mean([r['is_unseen_combination'] for r in unique])),
             unique_kc_unknown_field_rate=float(np.mean([r['has_unknown_field'] for r in unique])),unseen_combination_rate=np.mean([r['is_unseen_combination'] for r in records]).item(),
             unseen_field_rate=np.mean([r['has_unknown_field'] for r in records]).item(),
             missing_field_rate=np.mean([r['has_missing_field'] for r in records]).item()))
        pairs[(task['fold'],task['seed'],task['train_student_count'])][task['model_name']]=task
    comparisons=[]
    for key,pair in pairs.items():
        if set(pair)!= {'akt','akt_fmkc'}:continue
        a,b=(rows(Path(pair[m]['output_dir'])/'predictions_test.jsonl.gz') for m in ('akt','akt_fmkc'))
        result={'overall':bootstrap(a,b,replicates)}
        for i in range(7):result[f'G{i}']=bootstrap(a,b,replicates,mask=[frequency_group(r['train_frequency'])==i for r in a])
        for name in ('seen','unseen_known_fields','unseen_unknown_fields'):
            result[name]=bootstrap(a,b,replicates,mask=[cold_group(r)==name for r in a])
        save(Path(pair['akt']['output_dir']).parent/'bootstrap_results.json',result)
        for group,ms in result.items():
            for metric,value in ms.items():comparisons.append(dict(fold=key[0],seed=key[1],train_student_count=key[2],group=group,metric=metric,**value))
    for name,data in [('overall_metrics',overall),('scaling_metrics',overall),('frequency_group_metrics',freq),('longtail_metrics',freq),
                      ('coldstart_metrics',cold),('bootstrap_comparison',comparisons),('parameter_comparison',overall)]:
        csv_write(root/'summary'/f'{name}.csv',data)
    csv_write(root/'summary/frequency_group_comparison.csv',[r for r in comparisons if r['group'].startswith('G')])
    csv_write(root/'summary/data_efficiency_comparison.csv',[r for r in comparisons if r['group']=='overall'])
    save(root/'execution_status.json',dict(completed=len(overall),total=len(tasks),missing_or_failed=missing))
    save(root/'summary/bootstrap_results.json',comparisons)
    common_analysis(root,tasks)
    figures(root,overall)
    group_figures(root,freq)
    summarize(root)


def common_analysis(root,tasks):
    grouped=defaultdict(list);output=[]
    for t in tasks:grouped[(t['fold'],t['seed'])].append(t)
    for key,group in grouped.items():
        if not all(complete(t) for t in group):continue
        records=[rows(Path(t['output_dir'])/'predictions_test.jsonl.gz') for t in group]
        identities=[[(r['student_id'],r['event_index'],r['kc_key'],r['response']) for r in rs] for rs in records]
        if any(x!=identities[0] for x in identities):raise ValueError('Cross-scale test events differ')
        seen=set.intersection(*[{r['kc_key'] for r in rs if not r['is_unseen_combination']} for rs in records])
        unseen=set.intersection(*[{r['kc_key'] for r in rs if r['is_unseen_combination'] and not r['has_unknown_field'] and not r['has_missing_field']} for rs in records])
        for t,rs in zip(group,records):
            for label,kcs in [('common_seen',seen),('common_unseen',unseen)]:
                output.append(dict(fold=key[0],seed=key[1],model_name=t['model_name'],train_student_count=t['train_student_count'],group=label,
                                   **summarize_rows([r for r in rs if r['kc_key'] in kcs])))
    for name in ('common_seen','common_unseen'):csv_write(root/'summary'/f'{name}_metrics.csv',[r for r in output if r['group']==name])


def figures(root,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'figures';directory.mkdir(exist_ok=True)
    for metric in ('auc','logloss','brier','ece','total_parameters'):
        fig,ax=plt.subplots()
        groups=defaultdict(list)
        for row in records:groups[(row['model_name'],row['fold'],row['seed'])].append(row)
        for label,rs in groups.items():
            rs=sorted(rs,key=lambda r:r['actual_train_student_count'])
            ax.plot([r['actual_train_student_count'] for r in rs],[r.get(metric) for r in rs],marker='o',label=str(label))
        ax.set(xlabel='Training students',ylabel=metric)
        if groups:ax.legend()
        fig.tight_layout();fig.savefig(directory/f'{metric}.png');plt.close(fig)


def summarize(root):
    root=Path(root);state=read(root/'execution_status.json') if (root/'execution_status.json').exists() else dict(completed=0,total=len(read(root/'task_manifest.json')))
    report=f'''# Yousician AKT / AKT-FMKC 实验报告

完成任务：{state['completed']} / {state['total']}。缺失结果不视为零，不对未完成配对作统计推断。

研究问题：总体预测性能、自然冷启动与长尾泛化、固定训练协议下的数据可用量效应。

数据与划分详见 dataset_audit.json、experiment_manifest.json 和 prepared/*/*/split_manifest.json。
训练学生按固定 subset_seed 排列并取嵌套前缀；词表仅由各规模训练学生拟合。

协议详见 experiment_config.json；原共享训练器以 validation AUC 提升超过 0.001 保存 checkpoint，patience=10。
所有预测使用最佳验证 checkpoint、真实历史 response；replay 提供有限上下文，不代表完整历史。

结果见 summary/*.csv、各任务 test_metrics.json（含 15-bin reliability 数据）与 figures/。
PR-AUC 定义为 Average Precision；Log Loss clip epsilon=1e-7；Accuracy 阈值为 0.5。
空组和单类 AUC 为 null。Bootstrap 为学生级配对 percentile 95% CI；子组为 exploratory / unadjusted。

限制：架构差异同时涉及 encoder、drift、初始化和参数规模，不是单变量 embedding 消融。
原 AKT.reset 的按参数第一维零初始化规则保持不变，详见 initialization_audit.json。
训练词表不含 UNK 监督样本，必须结合这一点解释冷启动结果。
固定模型 CI 仅反映测试学生抽样不确定性；单一嵌套子集和 seed 不支持对所有训练子集的推广。
预测改善不表示因果知识增长。不同规模优化步数不同，不是等计算预算实验。

本地合成数据通过不代表 CUDA 或 Slurm 已验证。正式结果以任务产物为准。
'''
    (root/'report.md').write_text(report)


def group_figures(root,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('auc','logloss','brier','events','unique_kc_count'):
        groups=defaultdict(list)
        for r in records: groups[(r['fold'],r['seed'],r['train_student_count'])].append(r)
        for label,rs in groups.items():
            fig,ax=plt.subplots()
            for model in ('akt','akt_fmkc'):
                selected=[r for r in rs if r['model_name']==model]
                ax.plot([r['group'] for r in selected],[r[metric] for r in selected],marker='o',label=model)
            ax.set(ylabel=metric,xlabel='Training KC frequency group');ax.legend();fig.tight_layout()
            fig.savefig(root/'figures'/f'longtail_{label[0]}_{label[1]}_{label[2]}_{metric}.png');plt.close(fig)
