"""Standalone, staged Yousician benchmark. No external service is required."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pykt.benchmark.prepare import prepare
from pykt.benchmark.runtime import DEFAULTS,save,read,sha,code_identity,environment,train,evaluate,complete


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','evaluate','analyze','summarize','smoke','status'])
    p.add_argument('--run_dir',required=True)
    p.add_argument('--raw_file')
    p.add_argument('--train_student_sizes',nargs='+',default=['60','120','240'])
    p.add_argument('--include_full',action='store_true')
    p.add_argument('--folds','--fold',nargs='+',type=int,default=[0])
    p.add_argument('--seeds','--seed',nargs='+',type=int,default=[3407])
    p.add_argument('--models',nargs='+',choices=['akt','akt_fmkc'],default=['akt','akt_fmkc'])
    p.add_argument('--split_seed',type=int,default=1024);p.add_argument('--subset_seed',type=int,default=1024)
    p.add_argument('--kfold',type=int,default=5);p.add_argument('--test_fraction',type=float,default=.2)
    p.add_argument('--task_id',type=int);p.add_argument('--require_cuda',action='store_true')
    p.add_argument('--bootstrap_replicates',type=int,default=2000)
    for k,v in DEFAULTS.items():p.add_argument('--'+k,type=type(v),default=v)
    a=p.parse_args(argv);root=Path(a.run_dir).resolve()
    if a.stage=='smoke':
        from unittest.mock import patch
        students={str(i):[(['inf' if j==0 else str(j%2),'0',str((i+j)%3),'1'],
                           '|'.join(['inf' if j==0 else str(j%2),'0',str((i+j)%3),'1']),j%2) for j in range(9)] for i in range(20)}
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            raw=Path(temp)/'synthetic.json';raw.write_text('[]')
            with patch('pykt.benchmark.prepare.read_students',return_value=(students,0)):
                prepare(raw,root,sizes=(2,4),kfold=5)
        a.num_epochs=1;a.d_model=16;a.d_ff=32;a.n_blocks=1;a.num_attn_heads=2;a.field_dim=4
        a.block_size=4;a.seq_len=4
        finalize_prepare(root,a,synthetic=True)
        for t in read(root/'task_manifest.json'):train(root,t['task_id'])
        from pykt.benchmark.analysis import analyze
        analyze(root,10)
    elif a.stage=='prepare':
        if not a.raw_file:p.error('--raw_file required')
        prepare(a.raw_file,root,a.train_student_sizes+(['full'] if a.include_full else []),a.folds,a.seeds,a.split_seed,a.subset_seed,a.kfold,a.test_fraction)
        finalize_prepare(root,a)
    elif a.stage in ('train','evaluate'):
        if a.task_id is None:p.error('--task_id required')
        (train if a.stage=='train' else evaluate)(root,a.task_id,a.require_cuda)
    elif a.stage=='analyze':
        from pykt.benchmark.analysis import analyze
        analyze(root,a.bootstrap_replicates)
    elif a.stage=='summarize':
        from pykt.benchmark.analysis import summarize
        summarize(root)
    else:
        print(json.dumps([dict(t,complete=complete(t)) for t in read(root/'task_manifest.json')],indent=2))


def finalize_prepare(root,a,synthetic=False):
    from pykt.benchmark.prepare import fingerprint
    tasks=[t for t in read(root/'task_manifest.json') if t['model_name'] in a.models]
    for i,t in enumerate(tasks):t['task_id']=i
    save(root/'task_manifest.json',tasks)
    manifest=read(root/'experiment_manifest.json');manifest['task_fingerprint']=fingerprint(tasks);manifest['synthetic']=synthetic
    save(root/'experiment_manifest.json',manifest)
    index=read(root/'prepared/student_index.json')
    save(root/'prepared/student_hashes.json',{u:sha(root/'prepared'/path) for u,path in index.items()})
    save(root/'code_identity.json',code_identity());save(root/'environment.json',environment())
    save(root/'experiment_config.json',dict(training={k:getattr(a,k) for k in DEFAULTS},arguments=vars(a),synthetic=synthetic))
    for task in tasks:save(Path(task['output_dir'])/'task_status.json',dict(task_id=task['task_id'],status='pending'))
    print(f'Prepared {len(tasks)} tasks: {root}')


if __name__=='__main__':main()
