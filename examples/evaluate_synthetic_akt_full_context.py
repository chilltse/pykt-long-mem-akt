"""Zero-shot context extension diagnostic; not a separately trained baseline."""
import argparse
import csv
import json
from pathlib import Path
import torch
from experiment_long_mem_topk_akt_synthetic import evaluate
from pykt.models.long_mem_topk_akt import LongMemTopKAKT
from pykt.models.long_mem_topk_akt_workflow import write_json


def main(source):
    source=Path(source).resolve();torch.set_num_threads(1)
    config_path=source/'resolved_config.json'
    children=sorted(p for p in source.glob('seed*') if p.is_dir())
    if config_path.exists():
        config=json.loads(config_path.read_text())['arguments']
    else:
        config=json.loads((children[0]/'resolved_config.json').read_text())['arguments']
        config['seeds']=','.join(p.name[4:] for p in children)
    data=source/'data' if (source/'data').exists() else children[0]/'data'
    metadata=json.loads((data/'diagnostic_labels.json').read_text())
    students=[]
    with (data/'test.csv').open() as f:
        for row in csv.DictReader(f):
            c=torch.tensor(list(map(int,row['concepts'].split(','))))[None]
            r=torch.tensor(list(map(int,row['responses'].split(','))),dtype=torch.float32)[None]
            mask=torch.ones_like(c,dtype=torch.bool);mask[:,::config['block_len']]=False
            students.append(dict(uid=[row['uid']],cseqs=c,qseqs=c,rseqs=r,smasks=mask))
    result=[]
    for seed in config['seeds'].split(','):
        child=source/('seed'+seed)
        if not (child/('B0_seed'+seed)/'metrics.json').exists():
            raise ValueError('Baseline training is not complete for seed '+seed)
        mc=json.loads((child/'resolved_config.json').read_text())['model']
        mc['block_len']=config['length']
        net=LongMemTopKAKT(**mc,memory_mode='off')
        net.load_state_dict(torch.load(child/('B0_seed'+seed)/'best.pt',map_location='cpu'),strict=True)
        report,rows=evaluate(net,students,child/('B0_seed'+seed)/'full_context_predictions.csv',metadata,config['length'])
        with (child/('B0_seed'+seed)/'predictions.csv').open() as f: original=list(csv.DictReader(f))
        assert [(r['uid'],r['position'],r['target']) for r in rows]==[(r['uid'],r['position'],r['target']) for r in original]
        result.append(dict(seed=int(seed),report=report))
    write_json(source/'full_context_diagnostic.json',dict(results=result,note='Same B0 checkpoint trained with block_len=200, evaluated with length=1000 and identical scoring mask. No additional training; not a full-context-trained baseline.'))
    print([(r['seed'],r['report']['subsets']) for r in result])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    main(p.parse_args().source)
