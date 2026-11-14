"""Benchmark orchestration around the existing model factory and shared trainer."""
import contextlib
import csv
import gzip
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from collections import Counter
from pathlib import Path

from .prepare import fingerprint

DEFAULTS = dict(d_model=256,d_ff=512,n_blocks=4,num_attn_heads=8,field_dim=32,
                drift='response',dropout=.2,l2=1e-5,seq_len=128,block_size=128,cache_blocks=1,
                num_epochs=30,learning_rate=1e-4)


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def environment():
    import torch, numpy, sklearn
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,
                sklearn=sklearn.__version__,cuda=torch.version.cuda,hostname=socket.gethostname(),
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def code_identity():
    root=Path(__file__).resolve().parents[2]
    files=sorted(p for folder in ('pykt','examples','configs','scripts/yousician_benchmark')
                 for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.sh','.sbatch'))
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                code_fingerprint=fingerprint({str(p.relative_to(root)):sha(p) for p in files}))


def task_for(root,task_id):
    tasks=read(Path(root)/'task_manifest.json')
    task=next(t for t in tasks if t['task_id']==int(task_id))
    manifest=read(Path(root)/'experiment_manifest.json')
    if fingerprint(tasks)!=manifest['task_fingerprint']: raise ValueError('Task manifest changed')
    lock=Path(root)/'code_identity.json'
    if lock.exists() and read(lock)!=code_identity(): raise ValueError('Code version changed since prepare')
    config=read(task['prepared_config_path'])['yousician']
    d=Path(config['dpath'])
    artifact=read(d/config['fold_file'].format(fold=task['fold']))
    split=read(d/'split_manifest.json')
    if fingerprint(artifact['encoding'])!=split['encoding_fingerprint'] or artifact['splits']!=split['splits']:
        raise ValueError('Prepared encoding/splits changed')
    if fingerprint(read(d/'train_kc_frequency.json'))!=split['frequency_fingerprint']:
        raise ValueError('Prepared frequencies changed')
    index_path=Path(config['student_index_file']); index=read(index_path)
    hashes=read(index_path.parent/'student_hashes.json') if (index_path.parent/'student_hashes.json').exists() else None
    if hashes:
        for uid in sum(artifact['splits'].values(),[]):
            if sha(index_path.parent/index[uid])!=hashes[uid]: raise ValueError('Canonical student file changed')
    return task,config


def complete(task):
    d=Path(task['output_dir'])
    try:
        status=read(d/'task_status.json')
        return status['status']=='completed' and all(sha(d/p)==h for p,h in status['artifacts'].items())
    except (OSError,KeyError,ValueError): return False


def train(root,task_id,require_cuda=False):
    import torch
    from pykt.utils import set_seed
    from pykt.models.init_model import init_model
    from pykt.models.train_model import train_model
    from pykt.datasets.student_block_dataset import init_student_loaders
    task,config=task_for(root,task_id)
    if complete(task): return
    if require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA required for HPC training')
    d=Path(task['output_dir']); d.mkdir(parents=True,exist_ok=True)
    # Preserve failed attempts rather than overwrite their evidence.
    if (d/'qid_model.ckpt').exists() and (d/'training_result.json').exists():
        evaluate(root,task_id,require_cuda); return
    existing=[p for p in d.iterdir() if p.name!='attempts']
    if existing:
        archive=d/'attempts'/str(time.time_ns()); archive.mkdir(parents=True)
        for p in existing: p.rename(archive/p.name)
    params=read(Path(root)/'experiment_config.json')['training']
    status=dict(task_id=task_id,status='running',start_time=time.time(),end_time=None,exit_code=None,
        hostname=socket.gethostname(),slurm_job_id=os.getenv('SLURM_JOB_ID'),slurm_array_task_id=os.getenv('SLURM_ARRAY_TASK_ID'),
        **code_identity(),environment=environment(),environment_fingerprint=fingerprint(environment()),
        dataset_sha256=read(Path(root)/'experiment_manifest.json')['dataset_sha256'],
        prepared_manifest_fingerprint=sha(Path(config['dpath'])/'split_manifest.json'),
        gpu_name=environment()['gpu'],config_fingerprint=fingerprint(params))
    save(d/'task_status.json',status)
    try:
        set_seed(task['seed'])
        loaders=init_student_loaders(config,task['fold'],**{k:params[k] for k in ('block_size','cache_blocks','seq_len')},seed=task['seed'])
        model_config={k:v for k,v in params.items() if k not in ('block_size','cache_blocks','num_epochs','learning_rate')}
        model=init_model(task['model_name'],model_config,config,'qid','yousician')
        save(d/'config.json',dict(model_config=model_config,data_config=config,params=dict(task,emb_type='qid'),train_config=params))
        audit={n:dict(shape=list(p.shape),all_zero=bool((p==0).all()),first_row=p[0].detach().cpu().tolist() if p.ndim else p.item())
               for n,p in model.named_parameters() if 'embed' in n or 'difficult' in n}
        save(d/'initialization_audit.json',dict(parameters=audit,zero_parameters=[n for n,p in model.named_parameters() if bool((p==0).all())],
            unk_supervised_in_training=False,pid_zero='unseen complete combinations; inspect difficult_param first row',
            reserved_fields={'PAD':0,'UNK':1,'MISSING':2},note='Original reset behavior preserved; architecture comparison is not a single-variable ablation'))
        counts=dict(total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    by_parameter={n:p.numel() for n,p in model.named_parameters()})
        sizes=counts['by_parameter']
        counts['shared_backbone_parameters']=sum(v for n,v in sizes.items() if n.startswith('model.'))
        counts['drift_parameters']=sum(v for n,v in sizes.items() if any(k in n for k in ('drift','diff','scalar','difficult')))
        counts['question_encoder_parameters']=sum(v for n,v in sizes.items() if any(k in n for k in ('field_embeddings','q_embed','base_head')) and not any(k in n for k in ('diff','drift')))
        counts['parameter_category_rule']='name-based non-exhaustive categories; full by_parameter inventory retained'
        save(d/'parameter_counts.json',counts)
        opt=torch.optim.Adam(model.parameters(),lr=params['learning_rate'])
        start=time.monotonic()
        with (d/'train.log').open('w') as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            result=train_model(model,loaders[0],loaders[1],params['num_epochs'],opt,str(d),save_model=True,
                               epoch_callback=lambda row: append_epoch(d/'epoch_metrics.csv',row))
        save(d/'training_result.json',dict(best_epoch=result[-1],training_wall_time=time.monotonic()-start,
             optimizer='adam',weight_decay=0,gradient_clip=5,early_stopping_patience=10,selection_min_delta=.001))
        evaluate(root,task_id,require_cuda)
    except BaseException as exc:
        status.update(status='failed',end_time=time.time(),exit_code=1,error=repr(exc)); save(d/'task_status.json',status)
        raise


def append_epoch(path,row):
    exists=path.exists()
    with path.open('a') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if not exists:w.writeheader()
        w.writerow(row)


def evaluate(root,task_id,require_cuda=False):
    import torch
    from pykt.models.akt_fmkc_runner import load_fmkc_checkpoint,forward_field_batch
    from pykt.datasets.student_block_dataset import init_student_loaders
    task,config=task_for(root,task_id); d=Path(task['output_dir'])
    if require_cuda and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    params=read(Path(root)/'experiment_config.json')['training']
    model,metadata=load_fmkc_checkpoint(d/'qid_model.ckpt','cuda' if torch.cuda.is_available() else 'cpu')
    loaders=init_student_loaders(config,task['fold'],**{k:params[k] for k in ('block_size','cache_blocks','seq_len')},seed=task['seed'])
    encoding=config['encoding']; ef=fingerprint(encoding)
    index_path=Path(config['student_index_file']); index=read(index_path)
    freq=read(Path(config['dpath'])/'train_kc_frequency.json')
    for split,loader in zip(('valid','test'),loaders[1:]):
        path=d/f'predictions_{split}.jsonl.gz'; temp=path.with_name(path.name+'.tmp')
        count=0; current=None
        with gzip.open(temp,'wt',encoding='utf-8') as stream,torch.no_grad():
            model.eval()
            for batch in loader:
                uid=batch['uid']
                if uid!=current:
                    events=read(index_path.parent/index[uid]); position=0; seen=set(); current=uid
                pred,target,_=forward_field_batch(model,batch)
                offset=len(batch['response'])-len(pred)
                for j,(p,y) in enumerate(zip(pred.cpu().tolist(),target.cpu().tolist())):
                    attrs,kc,label=events[position]
                    if label!=y or not 0<=p<=1: raise ValueError('Invalid exported event')
                    encoded=batch['attrs'][offset+j].tolist(); pid=int(batch['pid'][offset+j])
                    unknown=[name for name,v in zip(encoding['field_names'],encoded) if v==1]
                    row=dict(student_id=str(uid),event_index=position,fold=task['fold'],seed=task['seed'],model_name=task['model_name'],
                        kc_vector=attrs,kc_key=kc,question_id=pid,encoded_pid=pid,train_frequency=freq.get(kc,0),
                        is_unseen_combination=kc not in encoding['questions'],has_unknown_field=bool(unknown),response=label,predicted_probability=p,
                        run_id=Path(root).name,scale_mode='fixed_holdout',train_student_count=task['train_student_count'],split=split,
                        encoded_attrs=encoded,unknown_fields=unknown,unknown_field_count=len(unknown),has_missing_field=2 in encoded,
                        history_length=offset+j,input_history_length=offset+j,history_events_seen=position,seq_len=params['seq_len'],
                        block_index=position//params['block_size'],is_first_student_event=position==0,
                        is_first_kc_occurrence_for_student=kc not in seen,encoding_fingerprint=ef)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n'); seen.add(kc); position+=1; count+=1
                if position>len(events):raise ValueError('Repeated events')
        expected=sum(len(loader.dataset.students[u]['response']) for u in loader.dataset.ids)
        if count!=expected:raise ValueError('Missing export events')
        temp.replace(path)
        save(d/f'predictions_{split}.metadata.json',dict(schema_version=1,events=count,encoding_fingerprint=ef,format='jsonl.gz'))
    from .analysis import summarize_rows
    for split in ('valid','test'):
        save(d/('validation_metrics.json' if split=='valid' else 'test_metrics.json'),summarize_rows(rows(d/f'predictions_{split}.jsonl.gz')))
    status=read(d/'task_status.json') if (d/'task_status.json').exists() else dict(task_id=task_id)
    files=['qid_model.ckpt','config.json','predictions_valid.jsonl.gz','predictions_test.jsonl.gz','training_result.json','validation_metrics.json','test_metrics.json']
    status.update(status='completed',exit_code=0,end_time=time.time(),checkpoint_path=str(d/'qid_model.ckpt'),
                  prediction_path=str(d/'predictions_test.jsonl.gz'),artifacts={p:sha(d/p) for p in files})
    save(d/'task_status.json',status)


def rows(path):
    with gzip.open(path,'rt') as f:
        return [json.loads(line) for line in f]
