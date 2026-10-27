"""Original-AKT equivalence, causality, transactions, and retrieval acceptance."""
import copy
import pytest
import torch
from pykt.models.akt import AKT
from pykt.models.long_mem_topk_akt import LongMemTopKAKT
from pykt.models.akt_memory_bank import MemoryBank

torch.set_num_threads(1)


def model(**kwargs):
    torch.manual_seed(71)
    config = dict(n_question=12, n_pid=12, d_model=16, n_blocks=2, dropout=0.,
                  d_ff=32, final_fc_dim=32, num_attn_heads=4, block_len=32,
                  recent_capacity=32, remote_capacity=32, remote_topk=32,
                  retrieval_query_chunk=2, retrieval_memory_chunk=3)
    config.update(kwargs)
    return LongMemTopKAKT(**config)


def sample(n=12):
    torch.manual_seed(87)
    return torch.randint(0, 12, (1, n)), torch.randint(0, 2, (1, n)), torch.randint(0, 12, (1, n))


def run(m, c, r, pid, cuts=3, offset=0, bank=None, uid='a'):
    bank = bank or MemoryBank(m)
    result = []
    for start in range(0, c.size(1), cuts):
        end = min(start+cuts, c.size(1))
        pos = torch.arange(start+offset, end+offset)[None]
        p, _, pending = m.forward_with_memory(c[:, start:end], r[:, start:end], pid[:, start:end],
                                               pos, torch.ones_like(pos, dtype=torch.bool), bank.snapshot([uid]))
        result.append(p)
        bank.commit(pending)
    return torch.cat(result, 1), bank


@pytest.mark.parametrize('separate', [False, True])
@pytest.mark.parametrize('kq', [0, 1])
@pytest.mark.parametrize('emb', ['qid', 'qid_pdiff'])
def test_original_and_empty_memory_outputs_and_gradients(separate, kq, emb):
    m = model(separate_qa=separate, kq_same=kq, emb_type=emb).eval()
    baseline = AKT(12, 12, 16, 2, 0., d_ff=32, final_fc_dim=32, num_attn_heads=4,
                   separate_qa=separate, kq_same=kq, emb_type=emb).eval()
    with torch.no_grad():
        m.difficult_param.weight.normal_(std=.2)
    baseline.load_state_dict(m.state_dict(), strict=True)
    c, r, pid = sample(7)
    original, reg = baseline(c, r, pid)
    inherited, _ = m(c, r, pid)
    torch.testing.assert_close(inherited, original, atol=1e-6, rtol=1e-5)
    pos = torch.arange(7)[None]
    actual, reg2, _ = m.forward_with_memory(c, r, pid, pos, torch.ones_like(pos, dtype=torch.bool), MemoryBank(m).snapshot(['a']))
    torch.testing.assert_close(actual, original, atol=1e-6, rtol=1e-5)
    (original.sum()+reg).backward()
    (actual.sum()+reg2).backward()
    for name, parameter in m.named_parameters():
        other = dict(baseline.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('emb', ['qid', 'qid_pdiff'])
def test_recent_full_prefix_matches_full_sequence(emb):
    m = model(memory_mode='recent', emb_type=emb).eval()
    with torch.no_grad():
        m.difficult_param.weight.normal_(std=.2)
    c, r, pid = sample()
    full = m(c, r, pid)[0]
    blocked, bank = run(m, c, r, pid)
    torch.testing.assert_close(blocked, full, atol=1e-6, rtol=1e-5)
    shifted, _ = run(m, c, r, pid, offset=1000000)
    torch.testing.assert_close(blocked, shifted, atol=1e-6, rtol=1e-5)
    assert not bank.students['a']['layers']['qa.0']['recent'].key.requires_grad


def test_exact_all_remote_matches_dense_and_order_invariant():
    m = model(recent_capacity=2).eval()
    c, r, pid = sample()
    exact, bank = run(m, c, r, pid)
    m.memory_mode = 'dense_oracle'
    dense, _ = run(m, c, r, pid)
    torch.testing.assert_close(exact, dense, atol=1e-6, rtol=1e-5)
    snapshot = bank.snapshot(['a'])
    pos = torch.arange(12, 15)[None]
    inputs = [x[:, :3] for x in (c, r, pid)]
    first = m.forward_with_memory(*inputs, pos, torch.ones_like(pos, dtype=torch.bool), snapshot)[0]
    for layer in snapshot[0]['layers'].values():
        for name in ('recent', 'remote'):
            if name in layer:
                layer[name] = layer[name].take(torch.randperm(len(layer[name])))
    second = m.forward_with_memory(*inputs, pos, torch.ones_like(pos, dtype=torch.bool), snapshot)[0]
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('backend', ['exact', 'hnsw_ip'])
def test_multilayer_no_current_or_future_response_leakage(backend):
    m = model(recent_capacity=2, remote_topk=2, retrieval_backend=backend).eval()
    c, r, pid = sample()
    reference, _ = run(m, c, r, pid)
    for t in range(c.size(1)):
        changed = r.clone()
        changed[:, t:] = 1-changed[:, t:]
        changed_c, changed_pid = c.clone(), pid.clone()
        changed_c[:, t+1:] = (changed_c[:, t+1:]+1) % 12
        changed_pid[:, t+1:] = (changed_pid[:, t+1:]+1) % 12
        other, _ = run(m, changed_c, changed, changed_pid)
        torch.testing.assert_close(reference[:, :t+1], other[:, :t+1], atol=1e-6, rtol=1e-5)


def test_transactions_isolation_fifo_and_padding():
    m = model(recent_capacity=2, remote_capacity=3).eval()
    c, r, pid = sample()
    reference, bank = run(m, c, r, pid)
    for name, layer in bank.students['a']['layers'].items():
        assert len(layer['recent']) == 2
        if 'remote' in layer:
            assert layer['remote'].pos.tolist() == [7, 8, 9]
            assert set(layer['recent'].pos.tolist()).isdisjoint(layer['remote'].pos.tolist())
    independent, _ = run(m, c, r, pid, bank=bank, uid='b')
    torch.testing.assert_close(reference, independent)
    with pytest.raises(ValueError):
        bank.snapshot(['a', 'a'])
    pos = torch.arange(12, 15)[None]
    p, _, pending = m.forward_with_memory(c[:, :3], r[:, :3], pid[:, :3], pos,
                                         torch.zeros_like(pos, dtype=torch.bool), bank.snapshot(['a']))
    assert torch.isfinite(p).all()
    bank.commit(pending)
    with pytest.raises(ValueError, match='duplicate'):
        bank.commit(pending)
    with pytest.raises(ValueError, match='increase'):
        m.forward_with_memory(c[:, :3], r[:, :3], pid[:, :3], pos-12,
                              torch.ones_like(pos, dtype=torch.bool), bank.snapshot(['a']))


@pytest.mark.parametrize('r,m,k', [(0, 0, 0), (0, 3, 5), (2, 0, 3), (2, 3, 0)])
def test_capacity_edges_and_multiple_optimizer_steps(r, m, k):
    net = model(recent_capacity=r, remote_capacity=m, remote_topk=k).train()
    c, response, pid = sample(15)
    bank = MemoryBank(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    for start in range(0, 15, 3):
        pos = torch.arange(start, start+3)[None]
        prediction, reg, pending = net.forward_with_memory(c[:, start:start+3], response[:, start:start+3],
            pid[:, start:start+3], pos, torch.ones_like(pos, dtype=torch.bool), bank.snapshot(['a']))
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(prediction, response[:, start:start+3].float()) + reg
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step()
        bank.commit(pending, parameter_version=start//3+1)


def test_mixed_student_batch_and_history_read_gradients():
    m = model(recent_capacity=2, remote_topk=2, kq_same=0).eval()
    c, r, pid = sample(9)
    _, bank = run(m, c[:, :6], r[:, :6], pid[:, :6])
    _, bank = run(m, c[:, :6], 1-r[:, :6], pid[:, :6], bank=bank, uid='b')
    pos = torch.arange(6, 9)[None]
    args = (c[:, 6:], r[:, 6:], pid[:, 6:])
    a = m.forward_with_memory(*args, pos, torch.ones_like(pos, dtype=torch.bool), bank.snapshot(['a']))[0]
    b = m.forward_with_memory(*args, pos, torch.ones_like(pos, dtype=torch.bool), bank.snapshot(['b']))[0]
    combined = m.forward_with_memory(*(x.repeat(2, 1) for x in args), pos.repeat(2, 1),
                                    torch.ones(2, 3, dtype=torch.bool), bank.snapshot(['a', 'b']))[0]
    torch.testing.assert_close(combined, torch.cat([a, b]))
    combined.sum().backward()
    attention = m.attention_layer('decoder.1')
    for parameter in (attention.q_linear.weight, attention.k_linear.weight, attention.v_linear.weight, attention.gammas):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    for layer in bank.students['a']['layers'].values():
        for name in ('recent', 'remote'):
            if name in layer:
                assert layer[name].key.grad_fn is None and not layer[name].value.requires_grad


def test_hnsw_fifo_refresh_bounded_size_and_dense_diagnostics():
    pytest.importorskip('faiss')
    m = model(retrieval_backend='hnsw_ip', recent_capacity=1, remote_capacity=5,
              remote_topk=2, index_refresh_blocks=3, collect_diagnostics=True, diagnose_dense=True).eval()
    c, r, pid = sample(30)
    _, bank = run(m, c, r, pid, cuts=2)
    layer = bank.students['a']['layers']['decoder.1']
    index, remote = layer['index'], layer['remote']
    assert len(index.ids) <= 7 and index.refreshes > 1
    attention = m.attention_layer('decoder.1')
    q = torch.randn(attention.h, 3, attention.d_k)
    ids = index.search(q, remote, attention, 20)
    assert ids.min() >= 0 and ids.max() < len(remote)
    assert m.last_memory_diagnostics['recall_rows'] > 0
    assert m.last_memory_diagnostics['dense_rows'] > 0


def test_padding_nonfinite_safe_softmax_and_bfloat16():
    from pykt.models.akt_candidate_attention import safe_masked_softmax
    scores = torch.randn(2, 3, dtype=torch.bfloat16)
    valid = torch.tensor([[False, False, False], [True, False, True]])
    result = safe_masked_softmax(scores, valid)
    assert torch.isfinite(result).all() and result[0].sum() == 0
    m = model(recent_capacity=1, remote_topk=2).eval()
    c, r, pid = sample(6)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result, _ = run(m, c, r, pid)
    assert torch.isfinite(result).all()
    pos = torch.tensor([[0, 1, 2, 3, 4, 5]])
    valid = torch.tensor([[False, True, True, False, True, False]])
    m = model().eval()
    padded = m.forward_with_memory(c, r, pid, pos, valid, MemoryBank(m).snapshot(['a']))[0]
    compact = m.forward_with_memory(c[:, valid[0]], r[:, valid[0]], pid[:, valid[0]], pos[:, valid[0]],
        torch.ones(1, 3, dtype=torch.bool), MemoryBank(m).snapshot(['a']))[0]
    torch.testing.assert_close(padded[:, valid[0]], compact)


def test_cli_matched_ablation(tmp_path):
    import csv
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    path = tmp_path/'students.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['uid', 'fold', 'concepts', 'questions', 'responses'])
        writer.writeheader()
        for i in range(6):
            writer.writerow(dict(uid=str(i), fold=0 if i < 2 else (1 if i < 4 else -1),
                concepts=','.join(str(j%10) for j in range(13)),
                questions=','.join(str(j%10) for j in range(13)),
                responses=','.join(str((j+i)%2) for j in range(13))))
    dc = dict(dpath=str(tmp_path), input_type=['questions', 'concepts'], folds=[0, 1],
              maxlen=4, min_seq_len=3, num_c=12, num_q=12,
              train_valid_original_file=path.name, test_original_file=path.name, dataset_name='synthetic')
    data_config = tmp_path/'data_config.json'
    data_config.write_text(json.dumps({'synthetic': dc}))
    mc = dict(d_model=16, n_blocks=2, dropout=0., d_ff=32, final_fc_dim=32, num_attn_heads=4)
    torch.manual_seed(1234)
    baseline = AKT(12, 12, **mc)
    # A genuine tiny baseline optimization, only training students/fold 0.
    from pykt.models.long_mem_akt_workflow import create_roles, role_loader
    from pykt.models.long_mem_topk_akt_runner import run_epoch
    initial = model(memory_mode='off', block_len=4)
    initial.load_state_dict(baseline.state_dict())
    roles = create_roles(dc, 1, 42)
    optimizer = torch.optim.Adam(initial.parameters(), lr=.001)
    run_epoch(initial, role_loader(roles, 'train'), optimizer=optimizer)
    source = tmp_path/'baseline'
    source.mkdir()
    torch.save(initial.state_dict(), source/'qid_model.ckpt')
    (source/'config.json').write_text(json.dumps(dict(model_config=mc, data_config=dc,
        params=dict(model_name='akt', emb_type='qid', fold=1, dataset_name='synthetic'))))
    env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(root/'examples/evaluate_long_mem_topk_akt_ablation.py'),
        '--akt_source', str(source), '--output', str(tmp_path/'comparison'), '--data_config', str(data_config),
        '--dataset_name', 'synthetic', '--fold', '1', '--epochs', '1', '--seeds', '42',
        '--variants', 'akt,recent,exact,hnsw_ip,akt_long,dense_oracle', '--threads', '1',
        '--block_len', '4', '--recent_capacity', '2', '--remote_capacity', '5', '--remote_topk', '2',
        '--diagnostics']
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path/'comparison/comparison.json').read_text())
    assert len(report['results']) == 6
    assert {r['count'] for r in report['results']} == {18}
    assert len({r['scored_training_labels'] for r in report['results']}) == 1
    for row in report['results']:
        assert row['optimizer_steps'] > 0 and row['train_seconds'] > 0
    work = tmp_path/'examples'
    work.mkdir()
    configs = tmp_path/'configs'
    configs.mkdir()
    (configs/'wandb.json').write_text('{}')
    artifact = tmp_path/'comparison/exact_seed42'
    prediction = subprocess.run([sys.executable, str(root/'examples/wandb_predict.py'),
        '--save_dir', str(artifact), '--use_wandb', '0'], cwd=work, env=env,
        capture_output=True, text=True, timeout=60)
    assert prediction.returncode == 0, prediction.stdout + prediction.stderr
    standard = json.loads((artifact/'test_metrics.json').read_text())
    original = json.loads((artifact/'test/metrics.json').read_text())
    assert standard['logloss'] == pytest.approx(original['logloss'])
    from pykt.models import load_model
    saved = json.loads((artifact/'config.json').read_text())
    restored = load_model('long_mem_topk_akt', saved['model_config'], saved['data_config'], 'qid', str(artifact))
    assert restored.model_name == 'long_mem_topk_akt'


def test_no_pid_old_qtest_and_rejected_modes():
    m = model(n_pid=0, memory_mode='off').eval()
    baseline = AKT(12, 0, 16, 2, 0., d_ff=32, final_fc_dim=32, num_attn_heads=4).eval()
    baseline.load_state_dict(m.state_dict())
    c, r, _ = sample()
    expected = baseline(c, r, qtest=True)
    actual = m(c, r, qtest=True)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[2], expected[2])
    for option in ('qid_avgpool', 'qid_linear'):
        with pytest.raises(ValueError, match='supports only'):
            model(emb_type=option)
    with pytest.raises(ValueError, match='detached'):
        model(memory_detach=False)


def test_wrong_block_and_future_snapshot_are_rejected():
    m = model().eval()
    c, r, pid = sample(3)
    bank = MemoryBank(m)
    pos = torch.arange(3)[None]
    valid = torch.ones_like(pos, dtype=torch.bool)
    with pytest.raises(ValueError, match='Out-of-order'):
        m.forward_with_memory(c, r, pid, pos, valid, bank.snapshot(['a']), block_ids=[1])
    _, _, pending = m.forward_with_memory(c, r, pid, pos, valid, bank.snapshot(['a']))
    bank.commit(pending)
    snapshot = bank.snapshot(['a'])
    snapshot[0]['last_pos'] = 0
    with pytest.raises(ValueError, match='future'):
        m.forward_with_memory(c, r, pid, pos+3, valid, snapshot)
    pos = torch.tensor([[4, 3, 5]])
    with pytest.raises(ValueError, match='increase'):
        m.forward_with_memory(c, r, pid, pos, valid, bank.snapshot(['a']))


def test_registry_and_split_replay_are_stateless(tmp_path):
    import csv
    from torch.utils.data import DataLoader
    from pykt.datasets.long_mem_akt_dataset import LongMemAKTDataset
    from pykt.models.long_mem_topk_akt_runner import run_epoch
    from pykt.models import init_model, evaluate
    mc = dict(d_model=16, n_blocks=2, dropout=0., num_attn_heads=4, block_len=4,
              recent_capacity=2, remote_capacity=3, remote_topk=2)
    m = init_model('long_mem_topk_akt', mc, dict(num_c=12, num_q=12), 'qid').cpu()
    path = tmp_path/'students.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['uid', 'fold', 'concepts', 'questions', 'responses'])
        writer.writeheader()
        writer.writerow(dict(uid='a', fold=0, concepts='1,2,3,4,5,6,7,8,9',
                             questions='1,2,3,4,5,6,7,8,9', responses='0,1,0,1,0,1,0,1,0'))
    loader = DataLoader(LongMemAKTDataset(path, ['concepts', 'questions'], {0}, 4, 3), batch_size=1)
    first = run_epoch(m, loader)
    second = run_epoch(m, loader)
    assert first[2]['logloss'] == second[2]['logloss']
    assert evaluate(m, loader, 'long_mem_topk_akt') == first[:2]
    assert not any('memory' in key for key in m.state_dict())
