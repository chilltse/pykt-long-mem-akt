"""Reproducible 1000-interaction delayed-revisit benchmark for AKT and V2.

Synthetic diagnostic, not evidence of real-data superiority. No student ID,
latent mastery or phase labels are passed to the model.
"""
import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from pykt.models.long_mem_topk_akt import LongMemTopKAKT
from pykt.models.long_mem_topk_akt_runner import run_epoch
from pykt.models.long_mem_topk_akt_workflow import seed_all, write_json

VARIANTS = {
    'B1_akt_continue': ('off', 'last_layer', 'legacy_mul'),
    'F_recent': ('recent', 'last_layer', 'legacy_mul'),
    'F_exact': ('retrieval', 'last_layer', 'legacy_mul'),
    'F_additive': ('retrieval', 'last_layer', 'log_additive'),
    'G_exact': ('retrieval', 'gate_only', 'legacy_mul'),
}


def generate_dataset(output, seed=20261025, counts=(48,12,24), length=1000, block_len=200):
    """Student-independent learning states, disjoint role IDs, fixed schedule.

    Concepts 0..31 occur in the first and final fifth; 32..47 only in between.
    A latent unmastered concept becomes mastered with p=.015 per practice.
    Responses are Bernoulli(sigmoid(+/-1.8 + ability - difficulty)).
    """
    if length < 160 or length % 5 or length % block_len:
        raise ValueError('length must be >=160, divisible by 5 and block_len')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.RandomState(seed)
    difficulty = rng.normal(0,.25,48)
    roles, metadata, csv_rows = {}, {}, []
    fifth = length//5
    for role,count in zip(('train','valid','test'),counts):
        roles[role]=[]
        for i in range(count):
            uid='{}_{}'.format(role,i)
            def schedule(pool,n):
                return np.concatenate([rng.permutation(pool) for _ in range((n+len(pool)-1)//len(pool))])[:n]
            concepts=np.concatenate((schedule(np.arange(32),fifth),schedule(np.arange(32,48),3*fifth),schedule(np.arange(32),fifth)))
            mastery=rng.binomial(1,.5,48).astype(bool)
            ability=rng.normal(0,.2)
            response=[]; oracle=[]; first_return=[]; seen_return=set()
            for t,c in enumerate(concepts):
                probability=1/(1+np.exp(-((1.8 if mastery[c] else -1.8)+ability-difficulty[c])))
                oracle.append(float(probability));response.append(int(rng.rand()<probability))
                first_return.append(t>=4*fifth and int(c) not in seen_return)
                if t>=4*fifth: seen_return.add(int(c))
                if not mastery[c] and rng.rand()<.015: mastery[c]=True
            c=torch.tensor(concepts,dtype=torch.long)[None]
            r=torch.tensor(response,dtype=torch.float32)[None]
            mask=torch.ones_like(c,dtype=torch.bool);mask[:,::block_len]=False
            roles[role].append(dict(uid=[uid],cseqs=c,qseqs=c,rseqs=r,smasks=mask))
            metadata[uid]=dict(first_return=first_return,oracle_probability=oracle)
            csv_rows.append(dict(uid=uid,role=role,fold={'train':1,'valid':0,'test':-1}[role],
                                 concepts=','.join(map(str,concepts)),questions=','.join(map(str,concepts)),responses=','.join(map(str,response))))
    for role in roles:
        with (output/(role+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader()
            writer.writerows(row for row in csv_rows if row['role']==role)
    write_json(output/'diagnostic_labels.json',metadata)
    manifest=dict(seed=seed,length=length,block_len=block_len,counts=dict(zip(roles,counts)),n_concepts=48,
                  split='student-disjoint',return_start=4*fifth,score_mask='exclude every block first position',
                  labels='latent oracle and first-return labels are evaluation-only',
                  response='Bernoulli(sigmoid((mastered ? 1.8 : -1.8) + student ability - concept difficulty))',
                  learning_probability=.015,
                  sha256={role:hashlib.sha256((output/(role+'.csv')).read_bytes()).hexdigest() for role in roles})
    write_json(output/'manifest.json',manifest)
    return roles,metadata,manifest


def metrics(rows):
    y=np.array([float(r['target']) for r in rows]);p=np.array([float(r['probability']) for r in rows])
    return dict(count=len(y),auc=float(roc_auc_score(y,p)) if len(set(y))>1 else None,
                nll=float(log_loss(y,p,labels=[0,1])),acc=float(accuracy_score(y,p>=.5)))


def evaluate(net,loader,path,metadata,length):
    report=run_epoch(net,loader,save_path=str(path))[2]
    with path.open() as f: rows=list(csv.DictReader(f))
    report['subsets']={
        'all':metrics(rows),
        'return_block':metrics([r for r in rows if int(r['position'])>=4*length//5]),
        'first_return':metrics([r for r in rows if metadata[r['uid']]['first_return'][int(r['position'])]]),
    }
    return report,rows


def train_select(net,roles,epochs,lr,seed,directory):
    directory.mkdir(parents=True,exist_ok=False)
    optimizer=torch.optim.Adam(net.optimizer_groups(lr))
    best=float('inf');history=[]
    for epoch in range(epochs):
        order=np.random.RandomState(seed+epoch).permutation(len(roles['train']))
        train=[roles['train'][i] for i in order]
        loss,training=run_epoch(net,train,optimizer=optimizer)
        valid=run_epoch(net,roles['valid'])[2]
        history.append(dict(epoch=epoch+1,train_loss=loss,train=training,valid=valid))
        if valid['logloss']<best:
            best=valid['logloss'];best_epoch=epoch+1
            torch.save(net.state_dict(),directory/'best.pt')
        print('{} epoch={} loss={:.5f} valid_nll={:.5f} seconds={:.1f}'.format(directory.name,epoch+1,loss,valid['logloss'],training['seconds']),flush=True)
    net.load_state_dict(torch.load(directory/'best.pt',map_location='cpu'),strict=True)
    write_json(directory/'training.json',dict(history=history,best_epoch=best_epoch,selection='validation NLL only',
              trainable_parameters=[n for n,p in net.named_parameters() if p.requires_grad]))
    return dict(best_epoch=best_epoch,train_seconds=sum(x['train']['seconds'] for x in history),
                optimizer_steps=sum(x['train']['optimizer_steps'] for x in history),
                interactions=sum(x['train']['interactions'] for x in history),
                scored_labels=sum(x['train']['scored_labels'] for x in history))


def main(args):
    torch.set_num_threads(args.threads)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    roles,metadata,manifest=generate_dataset(output/'data',args.data_seed,(args.train_students,args.valid_students,args.test_students),args.length,args.block_len)
    config=dict(n_question=48,n_pid=0,d_model=32,n_blocks=2,dropout=.1,d_ff=64,final_fc_dim=64,
                num_attn_heads=4,block_len=args.block_len,recent_capacity=args.block_len,
                remote_capacity=args.length,remote_topk=32,retrieval_query_chunk=32,
                retrieval_memory_chunk=256,score_protocol='legacy',retrieval_backend='exact')
    write_json(output/'resolved_config.json',dict(arguments=vars(args),model=config,variants=VARIANTS,
               commit=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip(),
               torch_version=torch.__version__,device='cpu',note='working-tree implementation; fixed dataset across training seeds'))
    results=[];reference=None
    for seed in map(int,args.seeds.split(',')):
        seed_all(seed)
        baseline=LongMemTopKAKT(**config,memory_mode='off')
        base_dir=output/('B0_seed'+str(seed))
        base_budget=train_select(baseline,roles,args.pretrain_epochs,args.pretrain_lr,seed,base_dir)
        base_state=copy.deepcopy(baseline.state_dict())
        base_report,base_rows=evaluate(baseline,roles['test'],base_dir/'predictions.csv',metadata,args.length)
        signature=[(r['uid'],r['position'],r['target']) for r in base_rows]
        if reference is None: reference=signature
        assert signature==reference
        results.append(dict(variant='B0_akt',seed=seed,budget=base_budget,report=base_report))
        write_json(base_dir/'metrics.json',base_report)
        z=LongMemTopKAKT(**config,architecture_variant='v2_last_retriever_residual',training_stage='gate_only')
        z.load_baseline_state_dict(base_state)
        zero_report,zero_rows=evaluate(z,roles['test'],base_dir/'z0_predictions.csv',metadata,args.length)
        assert [(r['uid'],r['position'],r['target']) for r in zero_rows]==reference
        error=max(abs(float(a['probability'])-float(b['probability'])) for a,b in zip(base_rows,zero_rows))
        assert error<=1e-6
        write_json(base_dir/'zero_step.json',dict(max_error=error,report=zero_report))
        seed_results=[]
        for variant,(mode,stage,decay) in VARIANTS.items():
            seed_all(seed+10000)  # same dropout RNG and student order across ablations
            net=LongMemTopKAKT(**config,architecture_variant='v2_last_retriever_residual',
                    memory_mode=mode,training_stage=stage,memory_decay_mode=decay,gate_lr=args.gate_lr)
            net.load_baseline_state_dict(base_state)
            directory=output/(variant+'_seed'+str(seed))
            budget=train_select(net,roles,args.finetune_epochs,args.finetune_lr,seed+10000,directory)
            report,rows=evaluate(net,roles['test'],directory/'predictions.csv',metadata,args.length)
            assert [(r['uid'],r['position'],r['target']) for r in rows]==reference
            write_json(directory/'metrics.json',report)
            row=dict(variant=variant,seed=seed,budget=budget,report=report)
            seed_results.append(row);results.append(row)
            if variant=='F_exact':
                net.memory_alpha.data.zero_()
                intervention,_=evaluate(net,roles['test'],directory/'gate_zero_predictions.csv',metadata,args.length)
                write_json(directory/'gate_zero_metrics.json',intervention)
            write_json(output/'partial_results.json',results)
        for field in ('optimizer_steps','interactions','scored_labels'):
            assert len({r['budget'][field] for r in seed_results})==1,field
    summarize(output, results, manifest)


def summarize(output, results, manifest):
    summary={}
    for variant in ['B0_akt']+list(VARIANTS):
        selected=[r for r in results if r['variant']==variant]
        summary[variant]={subset:{metric:dict(mean=float(np.mean([r['report']['subsets'][subset][metric] for r in selected])),
                    std=float(np.std([r['report']['subsets'][subset][metric] for r in selected])))
                    for metric in ('auc','nll','acc')} for subset in ('all','return_block','first_return')}
    write_json(output/'comparison.json',dict(results=results,summary=summary,dataset=manifest,
               std_definition='population std over model seeds; one fixed dataset seed',
               interpretation='synthetic delayed-revisit diagnostic, not general real-data improvement'))
    with (output/'summary.csv').open('w',newline='') as f:
        fields=['variant','seed','subset','count','auc','nll','acc','train_seconds','optimizer_steps','scored_labels']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:
            for subset,value in r['report']['subsets'].items():
                writer.writerow(dict(variant=r['variant'],seed=r['seed'],subset=subset,**value,
                                     **{k:r['budget'][k] for k in ('train_seconds','optimizer_steps','scored_labels')}))
    print(json.dumps(summary,indent=2),flush=True)


def parallel_main(args):
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    seeds=args.seeds.split(',')
    if len(set(seeds))!=len(seeds):
        raise ValueError('Duplicate seeds')
    def worker(seed):
        child=output/('seed'+seed)
        command=[sys.executable,str(Path(__file__).resolve())]
        values=dict(vars(args),output=str(child),seeds=seed,workers=1)
        for name,value in values.items(): command.extend(['--'+name,str(value)])
        with (output/('seed'+seed+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed seed '+seed,flush=True)
        return json.loads((child/'comparison.json').read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        reports=list(executor.map(worker,seeds))
    assert all(r['dataset']['sha256']==reports[0]['dataset']['sha256'] for r in reports)
    (output/'data').symlink_to(Path('seed'+seeds[0])/'data',target_is_directory=True)
    write_json(output/'resolved_config.json',dict(arguments=vars(args),worker_configs=['seed'+s+'/resolved_config.json' for s in seeds]))
    summarize(output,[row for report in reports for row in report['results']],reports[0]['dataset'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--length',type=int,default=1000)
    p.add_argument('--block_len',type=int,default=200)
    p.add_argument('--data_seed',type=int,default=20261025)
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--train_students',type=int,default=48)
    p.add_argument('--valid_students',type=int,default=12)
    p.add_argument('--test_students',type=int,default=24)
    p.add_argument('--pretrain_epochs',type=int,default=6)
    p.add_argument('--finetune_epochs',type=int,default=4)
    p.add_argument('--pretrain_lr',type=float,default=.001)
    p.add_argument('--finetune_lr',type=float,default=.0001)
    p.add_argument('--gate_lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    (parallel_main if args.workers>1 else main)(args)
