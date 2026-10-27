"""Small synthetic B0/Z0/G2 smoke, not evidence of real-data improvement.

Run from repository root with PYTHONPATH=. Uses disjoint synthetic students,
identical score masks and fixed three-step gate-only budgets; saves full reports.
"""
import argparse
from pathlib import Path
import torch
from pykt.models.long_mem_topk_akt import LongMemTopKAKT
from pykt.models.long_mem_topk_akt_runner import run_epoch
from pykt.models.long_mem_topk_akt_workflow import seed_all, write_json


def students(seed, prefix):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for i in range(3):
        c = torch.randint(0, 12, (1, 16), generator=generator)
        r = torch.randint(0, 2, (1, 16), generator=generator).float()
        mask = torch.ones_like(c, dtype=torch.bool)
        mask[:, ::4] = False  # same baseline block-first exclusion for every run
        result.append(dict(uid=[prefix+str(i)], cseqs=c, qseqs=c, rseqs=r, smasks=mask))
    return result


def main(output):
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(n_question=12,n_pid=12,d_model=16,n_blocks=2,dropout=.1,
                  d_ff=32,final_fc_dim=32,num_attn_heads=4,block_len=4,
                  recent_capacity=2,remote_capacity=16,remote_topk=4,
                  architecture_variant='v2_last_retriever_residual',training_stage='gate_only',
                  memory_decay_mode='legacy_mul',retrieval_backend='exact',gate_lr=.001,
                  collect_diagnostics=True)
    reports=[]
    for seed in (42,43,44):
        seed_all(seed)
        train,valid=students(seed,'train'),students(seed+100,'valid')
        base_config={k:v for k,v in config.items() if k not in ('architecture_variant','training_stage','gate_lr')}
        baseline=LongMemTopKAKT(**base_config,memory_mode='off')
        run_epoch(baseline,train,optimizer=torch.optim.Adam(baseline.parameters(),lr=.001),max_steps=3)
        net=LongMemTopKAKT(**config)
        net.load_baseline_state_dict(baseline.state_dict())
        bpath,zpath=output/f'b0_{seed}.csv',output/f'z0_{seed}.csv'
        b0=run_epoch(baseline,valid,save_path=str(bpath))[2]
        z0=run_epoch(net,valid,save_path=str(zpath))[2]
        import csv
        with bpath.open() as f: b=list(csv.DictReader(f))
        with zpath.open() as f: z=list(csv.DictReader(f))
        assert [(v['uid'],v['position']) for v in b]==[(v['uid'],v['position']) for v in z]
        error=max(abs(float(a['probability'])-float(c['probability'])) for a,c in zip(b,z))
        assert error <= 1e-6
        before={n:p.clone() for n,p in net.named_parameters() if n!='memory_alpha'}
        _,training=run_epoch(net,train,optimizer=torch.optim.Adam(net.optimizer_groups(.0001)),max_steps=3)
        assert all(torch.equal(before[n],p) for n,p in net.named_parameters() if n!='memory_alpha')
        g2=run_epoch(net,valid)[2]
        torch.save(net.state_dict(),output/f'v2_{seed}.pt')
        gate=net.memory_alpha.detach().clone()
        net.memory_alpha.data.zero_()
        gate_off=run_epoch(net,valid)[2]
        net.memory_alpha.data.copy_(gate)
        net.memory_mode='off'
        empty=run_epoch(net,valid)[2]
        reports.append(dict(seed=seed,zero_step_max_error=error,B0=b0,Z0=z0,G2=g2,training=training,
                            trained_gate_zero=gate_off,trained_memory_off=empty))
    write_json(output/'report.json',dict(kind='synthetic smoke only',config=config,results=reports))
    print([(r['seed'],r['zero_step_max_error'],r['B0']['logloss'],r['G2']['logloss']) for r in reports])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
