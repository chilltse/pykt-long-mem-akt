import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pykt.models.akt import AKT
from pykt.models.akt_dual_window import AKTDualWindow, window_masks, select_history
from pykt.models.akt_dual_window_runner import run_epoch

torch.set_num_threads(1)


def config(**kw):
    result = dict(n_question=20, n_pid=20, d_model=16, n_blocks=2, dropout=0.,
                  d_ff=32, final_fc_dim=32, num_attn_heads=4)
    result.update(kw)
    return result


def model(**kw):
    torch.manual_seed(42)
    return AKTDualWindow(**config(), block_size=4, short_window=4, **kw)


def inputs(length=12, batch=1):
    torch.manual_seed(7)
    return torch.randint(0, 20, (batch, length)), torch.randint(0, 2, (batch, length))


def test_required_windows_and_topk():
    pos = torch.tensor([[384, 399, 449, 511]])
    keys = torch.arange(512)[None]
    short, long = window_masks(pos, keys, torch.ones_like(keys, dtype=torch.bool), 128, 256)
    for row, (lo, hi, slo, shi) in enumerate([(1, 256, 257, 384), (16, 271, 272, 399),
                                             (66, 321, 322, 449), (128, 383, 384, 511)]):
        assert (long[0, row].nonzero().flatten() + 1).tolist() == list(range(lo, hi+1))
        assert (short[0, row].nonzero().flatten() + 1).tolist() == list(range(slo, shi+1))
    assert not (short & long).any()
    scores = torch.arange(512).float()[None, None, None].expand(1, 4, 4, 512)
    allowed, selected = select_history(scores, short, long, 32)
    assert (selected.sum(-1) == 32).all()
    assert not (selected & ~long[:, None]).any()
    assert (allowed.sum(-1) == 160).all()
    for row in range(4):
        expected = long[0, row].nonzero().flatten()[-32:]
        assert torch.equal(selected[0, 0, row].nonzero().flatten(), expected)


@pytest.mark.parametrize('separate', [False, True])
@pytest.mark.parametrize('emb_type', ['qid', 'qid_pdiff'])
@pytest.mark.parametrize('kq_same', [0, 1])
def test_first_block_original_equivalence(separate, emb_type, kq_same):
    cfg = config(separate_qa=separate, emb_type=emb_type, kq_same=kq_same)
    original = AKT(**cfg).eval()
    dual = AKTDualWindow(**cfg, block_size=4, short_window=4, long_window=0).eval()
    with torch.no_grad():
        original.difficult_param.weight.normal_()
    dual.load_state_dict(original.state_dict())
    q, r = inputs(4, 2)
    p, reg = original(q, r, q)
    actual, actual_reg = dual(q, r, q)
    torch.testing.assert_close(p, actual, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reg, actual_reg)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('long_window', [0, 8])
def test_future_response_and_question_causality(long_window):
    m = model(long_window=long_window, topk=2).eval()
    q, r = inputs()
    expected, _ = m(q, r, q)
    for t in range(q.size(1)):
        changed_q, changed_r = q.clone(), r.clone()
        changed_q[:, t+1:] = (changed_q[:, t+1:] + 1) % 20
        changed_r[:, t:] = 1 - changed_r[:, t:]
        actual, _ = m(changed_q, changed_r, changed_q)
        torch.testing.assert_close(expected[:, :t+1], actual[:, :t+1])


def test_projection_cache_capacity_detach_and_gradients():
    m = model(long_window=4, topk=2)
    q, r = inputs(16)
    seen = []
    attn = m.model.blocks_1[0].masked_attn_head
    handle = attn.v_linear.register_forward_hook(lambda module, args, output: seen.append(output))
    _, _, memory, _ = m.predict_block(q[:, :4], r[:, :4], q[:, :4], student_ids=['a'])
    expected = seen[0].view(1, 4, 4, 4).transpose(1, 2)
    torch.testing.assert_close(memory['layers']['encoder.0']['v'], expected)
    handle.remove()
    seen[0].retain_grad()
    history = memory['layers']['encoder.0']['v'].clone()
    p, _, next_memory, _ = m.predict_block(q[:, 4:8], r[:, 4:8], q[:, 4:8], student_ids=['a'], memory=memory)
    p.sum().backward()
    for name in ('k_linear', 'v_linear'):
        assert getattr(attn, name).weight.grad.abs().sum() > 0
    assert attn.gammas.grad.abs().sum() > 0
    assert m.q_embed.weight.grad.abs().sum() > 0
    assert seen[0].grad_fn is not None
    assert seen[0].grad is None  # Later blocks cannot backpropagate into history.
    torch.testing.assert_close(memory['layers']['encoder.0']['v'], history)
    for start in (8, 12):
        _, _, next_memory, _ = m.predict_block(q[:, start:start+4], r[:, start:start+4], q[:, start:start+4],
                                               student_ids=['a'], memory=next_memory)
    for cache in next_memory['layers'].values():
        assert cache['positions'].tolist() == [list(range(8, 16))]
        for name in ('k', 'v'):
            assert cache[name].grad_fn is None and not cache[name].requires_grad


def test_student_isolation_and_per_row_padding():
    m = model(long_window=4, topk=2).eval()
    q, r = inputs(8, 2)
    valid = torch.tensor([[True]*4, [True, True, False, False]])
    _, _, memory, _ = m.predict_block(q[:, :4], r[:, :4], q[:, :4], student_ids=['a', 'b'], valid=valid)
    assert memory['next_position'].tolist() == [4, 2]
    actual, _, _, _ = m.predict_block(q[:, 4:], r[:, 4:], q[:, 4:], student_ids=['a', 'b'], memory=memory)
    for row, length in [(0, 4), (1, 2)]:
        _, _, single, _ = m.predict_block(q[row:row+1, :length], r[row:row+1, :length], q[row:row+1, :length], student_ids=['solo'])
        expected, _, _, _ = m.predict_block(q[row:row+1, 4:], r[row:row+1, 4:], q[row:row+1, 4:], student_ids=['solo'], memory=single)
        torch.testing.assert_close(actual[row:row+1], expected)
    with pytest.raises(ValueError, match='Student batch changed'):
        m.predict_block(q[:, :4], r[:, :4], q[:, :4], student_ids=['b', 'a'], memory=memory)
    first, _ = m(q, r, q)
    m(19-q, 1-r, 19-q)
    torch.testing.assert_close(m(q, r, q)[0], first)


def test_empty_histories_and_insufficient_topk():
    m = model(long_window=0, topk=32)
    q, r = inputs(4, 2)
    valid = torch.tensor([[True, False, False, False], [False]*4])
    pred, _, memory, stats = m.predict_block(q, r, q, student_ids=['a', 'b'], valid=valid)
    assert torch.isfinite(pred).all()
    assert stats['long_selected'] == 0
    assert memory['next_position'].tolist() == [1, 0]
    s, l = window_masks(torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([[True]]), 4, 8)
    allowed, _ = select_history(torch.zeros(1, 4, 1, 1), s, l, 32)
    assert not allowed.any()


def test_runner_resets_per_student_and_epoch():
    m = model(long_window=4, topk=2)
    q, r = inputs()
    records = [dict(uid=uid, cseqs=q[0], qseqs=q[0], rseqs=r[0].float(),
                    allowed=torch.ones(12, dtype=torch.bool)) for uid in ('a', 'b')]
    original = m.predict_block
    starts = []
    def wrapped(*args, **kwargs):
        starts.append((kwargs['student_ids'][0], kwargs['memory'] is None))
        return original(*args, **kwargs)
    m.predict_block = wrapped
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(2):
        report, _ = run_epoch(m, records, 4, opt)
        assert report['optimizer_steps'] == 2
        assert report['scored_labels'] == 22
    assert [reset for _, reset in starts] == [True, False, False] * 4
    report, _ = run_epoch(m, records, 4)
    assert report['boundary']['count'] == 4
    assert report['long_usage']['long_selected'] > 0


def test_ab_cli(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    for filename, folds in [('train.csv', [0, 1, 1]), ('test.csv', [-1, -1])]:
        with (data / filename).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=['uid', 'fold', 'concepts', 'questions', 'responses'])
            writer.writeheader()
            for i, fold in enumerate(folds):
                writer.writerow(dict(uid=filename+str(i), fold=fold, concepts='1,2,3,4,1,2,3,4,1,2,3,4',
                    questions='1,2,3,4,1,2,3,4,1,2,3,4', responses='0,1,0,1,1,0,1,0,0,1,0,1'))
    cfg = tmp_path / 'data.json'
    cfg.write_text(json.dumps(dict(tiny=dict(num_c=20, num_q=20, input_type=['concepts', 'questions'],
        folds=[0, 1], train_valid_original_file='train.csv', test_original_file='test.csv'))))
    output = tmp_path / 'ab'
    script = Path(__file__).resolve().parents[1] / 'examples/experiment_akt_dual_window.py'
    subprocess.run([sys.executable, str(script), '--data-config', str(cfg), '--dataset-name', 'tiny',
        '--data-dir', str(data), '--output', str(output), '--epochs', '1', '--seeds', '42',
        '--block-size', '4', '--short-window', '4', '--long-window', '8', '--topk', '2',
        '--d-model', '16', '--d-ff', '32', '--n-blocks', '2', '--num-attn-heads', '4',
        '--final-fc-dim', '32', '--dropout', '0'], check=True, capture_output=True, text=True)
    reports = json.loads((output / 'comparison.json').read_text())
    assert len(reports) == 3
    assert {r['overall']['count'] for r in reports} == {22}
    assert {r['boundary']['count'] for r in reports} == {4}
    assert reports[2]['long_usage']['long_selected'] > 0
    assert reports[1]['long_usage']['long_selected'] == 0


def test_cached_layers_equal_full_akt_without_truncation():
    original = AKT(**config()).eval()
    dual = AKTDualWindow(**config(), block_size=4, short_window=16, long_window=0).eval()
    dual.load_state_dict(original.state_dict())
    q, r = inputs(12, 2)
    torch.testing.assert_close(dual(q, r, q)[0], original(q, r, q)[0], atol=1e-6, rtol=1e-5)
