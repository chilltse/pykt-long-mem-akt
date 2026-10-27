"""V2 isolation, baseline compatibility, gradients and candidate mathematics."""
import pytest
import torch
from test_long_mem_topk_akt import model, sample, run
from pykt.models.akt_memory_bank import MemoryBank
from pykt.models.akt_candidate_attention import safe_masked_softmax


def v2(**kw):
    return model(architecture_variant='v2_last_retriever_residual', **kw)


@pytest.mark.parametrize('decay', ['legacy_mul', 'log_additive'])
@pytest.mark.parametrize('separate,kq,emb', [(a,b,c) for a in [False,True] for b in [0,1] for c in ['qid','qid_pdiff']])
def test_zero_gate_nonempty(decay, separate, kq, emb):
    m = v2(memory_decay_mode=decay, separate_qa=separate, kq_same=kq, emb_type=emb, recent_capacity=2).eval()
    with torch.no_grad():
        m.difficult_param.weight.normal_()
    c,r,p = sample()
    actual,bank = run(m,c,r,p)
    expected = torch.cat([m(c[:,i:i+3],r[:,i:i+3],p[:,i:i+3])[0] for i in range(0,12,3)],1)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    assert set(bank.students['a']['layers']) == {'decoder.3'}
    assert len(bank.students['a']['layers']['decoder.3']['remote']) > 0


@pytest.mark.parametrize('r,m,k', [(0,0,0),(0,5,0),(0,5,20),(2,0,3),(1,1,1)])
def test_edge_backward(r,m,k):
    net=v2(recent_capacity=r,remote_capacity=m,remote_topk=k)
    net.memory_alpha.data.fill_(.2)
    bank=MemoryBank(net)
    c,y,p=sample(6)
    for i in range(6):
        pos=torch.tensor([[i]])
        pred,reg,pending=net.forward_with_memory(c[:,i:i+1],y[:,i:i+1],p[:,i:i+1],pos,pos>=0,bank.snapshot(['a']))
        (pred.sum()+reg).backward()
        assert torch.isfinite(pred).all()
        assert all(torch.isfinite(t.grad).all() for t in net.parameters() if t.grad is not None)
        bank.commit(pending)
        net.zero_grad()
    pos=torch.tensor([[6]])
    out,reg,pending=net.forward_with_memory(c[:,:1],y[:,:1],p[:,:1],pos,pos<0,bank.snapshot(['a']))
    (out.sum()+reg).backward()
    bank.commit(pending)


def test_gate_only_update_and_modes():
    net=v2(training_stage='gate_only', dropout=.4,recent_capacity=2).train()
    assert not any(t.training for t in net.modules() if isinstance(t,torch.nn.Dropout))
    before={n:p.clone() for n,p in net.named_parameters()}
    opt=torch.optim.Adam(net.optimizer_groups(1e-4))
    c,r,p=sample()
    result,bank=run(net,c,r,p)
    torch.nn.functional.binary_cross_entropy(result,r.float()).backward()
    assert net.memory_alpha.grad.abs().sum()>0
    opt.step()
    assert not torch.equal(before['memory_alpha'],net.memory_alpha)
    assert all(torch.equal(before[n],p) for n,p in net.named_parameters() if n!='memory_alpha')
    assert all(rec.key.grad_fn is None and rec.value.grad_fn is None for layer in bank.students['a']['layers'].values() for rec in layer.values())
    net.configure_training('last_layer');net.train()
    assert not net.model.blocks_1[0].training and net.model.blocks_2[-1].training


def test_causality_and_shift_and_empty():
    net=v2().eval();net.memory_alpha.data.fill_(.4)
    c,r,p=sample()
    first,_=run(net,c,r,p)
    rr=r.clone();rr[:,7:]=1-rr[:,7:]
    second,_=run(net,c,rr,p)
    torch.testing.assert_close(first[:,:8],second[:,:8])
    cc=c.clone();cc[:,8:]=(cc[:,8:]+1)%12
    third,_=run(net,cc,r,p)
    torch.testing.assert_close(first[:,:8],third[:,:8])
    shifted,_=run(net,c,r,p,offset=2**40)
    torch.testing.assert_close(first,shifted)
    pos=torch.arange(12)[None]
    empty=net.forward_with_memory(c,r,p,pos,pos>=0,MemoryBank(net).snapshot(['a']))[0]
    torch.testing.assert_close(empty,net(c,r,p)[0],atol=1e-6,rtol=1e-5)


def test_checkpoint_whitelist_and_replay():
    base=model();net=v2()
    net.load_baseline_state_dict(base.state_dict())
    bad=dict(base.state_dict());bad.pop('out.0.bias')
    with pytest.raises(RuntimeError): net.load_baseline_state_dict(bad)
    net.memory_alpha.data.fill_(.3)
    clone=v2();clone.load_state_dict(net.state_dict(),strict=True)
    c,r,p=sample()
    torch.testing.assert_close(run(net.eval(),c,r,p)[0],run(clone.eval(),c,r,p)[0])


def test_safe_empty_rows_and_tail_counterexamples():
    scores=torch.randn(2,3,requires_grad=True)
    probs=safe_masked_softmax(scores,torch.zeros_like(scores,dtype=torch.bool))
    probs.sum().backward()
    assert torch.isfinite(scores.grad).all() and probs.sum()==0
    p=torch.tensor([.8,.1,.1])
    assert p[2]/p[1:].sum() > p[2]
    without_b = p[:2]/p[:2].sum()
    assert without_b[2:].sum() < p[2]
    assert safe_masked_softmax(torch.tensor([[99.]]),torch.tensor([[True]])).item()==1


def test_nonzero_gate_read_gradients_and_single_projection():
    net=v2(training_stage='last_layer').eval();net.memory_alpha.data.fill_(.2)
    c,r,p=sample(6)
    _,bank=run(net,c[:,:3],r[:,:3],p[:,:3])
    count=[]
    handle=net.model.blocks_2[-1].masked_attn_head.out_proj.register_forward_hook(lambda *args: count.append(1))
    out,_,_=net.forward_with_memory(c[:,3:],r[:,3:],p[:,3:],torch.arange(3,6)[None],torch.ones(1,3,dtype=torch.bool),bank.snapshot(['a']))
    out[:,0].sum().backward()  # current has no response history at block start
    handle.remove()
    assert len(count)==1
    assert net.model.blocks_2[-1].masked_attn_head.v_linear.weight.grad.abs().sum()>0


def test_v2_rejects_unadapted_paths():
    with pytest.raises(ValueError): v2(retrieval_backend='hnsw_ip')
    with pytest.raises(ValueError): v2(emb_type='qid_avgpool',memory_mode='off')


@pytest.mark.parametrize('decay',['legacy_mul','log_additive'])
def test_memory_matches_explicit_reference(decay):
    import math
    from pykt.models.akt_memory_bank import Records
    from pykt.models.akt_candidate_attention import residual_memory_attention
    from pykt.models.akt_memory_index import heads
    net=v2(memory_decay_mode=decay,remote_topk=10).eval()
    net.memory_alpha.data.fill_(.4)
    attn=net.model.blocks_2[-1].masked_attn_head
    x=torch.randn(1,16); y=torch.randn(1,16); pos=torch.tensor([100])
    features=torch.randn(3,16); values=torch.randn(3,16)
    rec=Records(torch.tensor([1,5,90]),features,values,torch.zeros(3,dtype=torch.long))
    mem={'recent':rec.take(slice(2,None)),'remote':rec.take(slice(0,2))}
    actual=residual_memory_attention(attn,x,x,y,pos,mem,None,net,{})
    q=heads(attn.k_linear,x,4,4); k=heads(attn.k_linear,features,4,4); v=heads(attn.v_linear,values,4,4)
    s=(q@k.transpose(-1,-2))/math.sqrt(4)
    p=s.softmax(-1)
    tail=torch.stack([p[...,i+1:].sum(-1) for i in range(3)],-1)
    distance=(tail*torch.tensor([99.,95.,10.])).sqrt().detach()
    penalty=torch.nn.functional.softplus(attn.gammas)*distance
    logits=s*(-penalty).exp().clamp(1e-5,1e5) if decay=='legacy_mul' else s-penalty
    output=logits.softmax(-1)@v
    expected=attn.out_proj((net.memory_alpha.tanh()[:,None,None]*output).transpose(0,1).reshape(1,16))
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_interleaved_students_and_transactions():
    net=v2().eval();net.memory_alpha.data.fill_(.2)
    c,r,p=sample(6)
    expected,_=run(net,c,r,p)
    bank=MemoryBank(net); collected={'a':[],'b':[]}
    for start in [0,3]:
        for uid in ['a','b']:
            pos=torch.arange(start,start+3)[None]
            pred,_,pending=net.forward_with_memory(c[:,start:start+3],r[:,start:start+3],p[:,start:start+3],pos,pos>=0,bank.snapshot([uid]))
            bank.commit(pending);collected[uid].append(pred)
            with pytest.raises(ValueError): bank.commit(pending)
    for outputs in collected.values(): torch.testing.assert_close(torch.cat(outputs,1),expected)


def test_workflow_save_restore(tmp_path):
    import csv
    from pykt.models.long_mem_topk_akt_workflow import fit, load_artifact, create_roles, role_loader
    from pykt.models.long_mem_topk_akt_runner import run_epoch
    path=tmp_path/'students.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['uid','fold','concepts','questions','responses'])
        writer.writeheader()
        for i,fold in enumerate([0,1,-1]):
            writer.writerow(dict(uid=str(i),fold=fold,concepts='1,2,3,4,5,6,7,8',questions='1,2,3,4,5,6,7,8',responses='0,1,0,1,0,1,0,1'))
    dc=dict(dpath=str(tmp_path),input_type=['questions','concepts'],folds=[0,1],maxlen=4,min_seq_len=3,num_c=12,num_q=12,train_valid_original_file=path.name,test_original_file=path.name,dataset_name='synthetic')
    roles=create_roles(dc,1,42)
    config=dict(d_model=16,n_blocks=2,dropout=.1,d_ff=32,final_fc_dim=32,num_attn_heads=4,block_len=4,recent_capacity=2,architecture_variant='v2_last_retriever_residual',training_stage='gate_only')
    from pykt.models.long_mem_topk_akt import LongMemTopKAKT
    net=LongMemTopKAKT(12,12,**config)
    fit(tmp_path/'artifact',net,config,roles,{},1,.0001,42)
    restored,saved,_=load_artifact(tmp_path/'artifact')
    assert saved['trainable_parameters']==['memory_alpha']
    assert saved['model_version']=='v2_last_retriever_residual'
    assert run_epoch(net,role_loader(roles,'valid'))[2]['logloss']==run_epoch(restored,role_loader(roles,'valid'))[2]['logloss']
