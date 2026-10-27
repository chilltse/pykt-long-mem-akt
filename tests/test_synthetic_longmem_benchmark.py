"""Dataset design and scoring invariants for the delayed-revisit benchmark."""
import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('benchmark',Path(__file__).resolve().parents[1]/'examples/experiment_long_mem_topk_akt_synthetic.py')
benchmark=importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_delayed_revisit_and_masks(tmp_path):
    roles,metadata,manifest=benchmark.generate_dataset(tmp_path/'a',counts=(2,1,1))
    seen=set()
    for role,students in roles.items():
        for student in students:
            uid=student['uid'][0]
            assert uid not in seen;seen.add(uid)
            c=student['cseqs'][0]
            assert len(c)==1000 and len(student['rseqs'][0])==1000
            assert set(c[:200].tolist())==set(range(32))
            assert set(c[200:800].tolist())==set(range(32,48))
            assert set(c[800:].tolist())==set(range(32))
            assert int(student['smasks'].sum())==995
            first=metadata[uid]['first_return']
            assert sum(first)==32
            last={}
            for t,k in enumerate(c.tolist()):
                if first[t]:
                    assert t-last[k]>600  # absent from current block and recent R=200
                last[k]=t
            assert set(student)=={'uid','cseqs','qseqs','rseqs','smasks'}
            assert torch.all((student['rseqs']==0)|(student['rseqs']==1))
    again,_,second=benchmark.generate_dataset(tmp_path/'b',counts=(2,1,1))
    assert manifest['sha256']==second['sha256']
    assert torch.equal(roles['train'][0]['rseqs'],again['train'][0]['rseqs'])


def test_metric_calculation():
    rows=[dict(target='0',probability='.1'),dict(target='1',probability='.9')]
    actual=benchmark.metrics(rows)
    assert actual['count']==2 and actual['auc']==1 and actual['acc']==1
    assert abs(actual['nll']-0.10536051565782628)<1e-7
