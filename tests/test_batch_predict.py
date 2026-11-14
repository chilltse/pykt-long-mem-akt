"""Tensor candidate parity, isolation, encoding deduplication and opt-in timings."""
from dataclasses import replace
import os
import time

import pytest
import torch

from test_akt_fmkc_incremental import model_for, VERSIONS, single_thread
from test_fmkc_serving import engine, event, CANDIDATES, database


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('limit,history', [(1, 0), (1, 3), (3, 2), (3, 3), (3, 4), (None, 64)])
@pytest.mark.parametrize('drift,kq', [('none', 0), ('shared', 1), ('response', 0), ('response', 1)])
def test_batch_parity_and_immutable(device, limit, history, drift, kq):
    model = model_for(limit=limit, drift=drift, kq=kq).to(device)
    cache = model.empty_cache(**VERSIONS)
    for i in range(history):
        cache, _ = model.observe(cache, [1+i%4, 2], i%4, i%2, str(i), **VERSIONS)
    before = cache.state_dict()
    attrs = torch.randint(1, 5, (37, 2), device=device)
    pid = torch.arange(37, device=device) % 4
    expected = torch.stack([model.predict_next(cache, a, p, **VERSIONS) for a, p in zip(attrs, pid)])
    for size in [1, 7, 64]:
        actual = model.predict_candidates(cache, attrs, pid, batch_size=size, **VERSIONS)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    order = torch.randperm(len(pid), device=device)
    shuffled = model.predict_candidates(cache, attrs[order], pid[order], **VERSIONS)
    torch.testing.assert_close(shuffled, expected[order], atol=1e-5, rtol=0)
    repeats = model.predict_candidates(cache, attrs[:1].expand(100, -1), pid[:1].expand(100), **VERSIONS)
    torch.testing.assert_close(repeats, expected[:1].expand(100), atol=1e-5, rtol=0)
    after = cache.state_dict()
    for name in before:
        if name.endswith('_layers'):
            for a, b in zip(before[name], after[name]):
                for key in a:
                    assert torch.equal(a[key], b[key])
        else:
            assert before[name] == after[name]


def test_service_dedup_and_limits(engine, monkeypatch, caplog):
    engine.observe('a', event())
    before = database(engine)
    candidates = [dict(candidate_id=str(i), kc_vector=['oov-'+str(i), 999]) for i in range(100)]
    original = engine.model.predict_candidates
    calls = []
    def capture(cache, attrs, pid, **kw):
        calls.append((attrs, pid))
        return original(cache, attrs, pid, **kw)
    monkeypatch.setattr(engine.model, 'predict_candidates', capture)
    with caplog.at_level('INFO'):
        out = engine.predict_candidates('a', candidates)
    assert len(calls) == 1 and calls[0] == ([[1, 1]], [0])
    assert len({p['probability'] for p in out['predictions']}) == 1
    assert [p['candidate_id'] for p in out['predictions']] == [str(i) for i in range(100)]
    assert out['predictions'][0]['probability'] == engine.predict_next('a', ['unknown', 999])['predictions'][0]['probability']
    assert 'count=100 unique_kc=1' in caplog.text
    assert database(engine) == before
    for candidates in [[], candidates * 3, [CANDIDATES[0]] * 2]:
        with pytest.raises(ValueError):
            engine.predict_candidates('a', candidates)


def test_validation():
    model = model_for()
    cache = model.empty_cache(**VERSIONS)
    for attrs, pid in [([[0, 1]], [0]), ([[5, 1]], [0]), ([[1, 1]], [4]),
                       ([[1.5, 1]], [0]), ([[1, 1]], [0.5]), ([1, 2], [0])]:
        with pytest.raises(ValueError):
            model.predict_candidates(cache, attrs, pid, **VERSIONS)
    for bad in [replace(cache, model_version='bad'), replace(cache, feature_version='bad'),
                replace(cache, cache_schema_version=999), replace(cache, history_revision=1)]:
        with pytest.raises(ValueError):
            model.predict_candidates(bad, [[1, 2]], [0], **VERSIONS)
    for size in [0, -1, True]:
        with pytest.raises(ValueError):
            model.predict_candidates(cache, [[1, 2]], [0], batch_size=size, **VERSIONS)


@pytest.mark.skipif(not os.getenv('RUN_BATCH_BENCHMARK'), reason='opt-in benchmark')
def test_benchmark():
    model = model_for(limit=64, blocks=2)
    cache = model.empty_cache(**VERSIONS)
    for i in range(64):
        cache, _ = model.observe(cache, [1, 2], 0, i%2, str(i), **VERSIONS)
    # 64 distinct encoded KCs; repeated workload separates dedup from batching.
    unique = [([a, b], p) for a in range(1, 5) for b in range(1, 5) for p in range(4)]
    def scalar(items):
        return [float(model.predict_next(cache, a, p, **VERSIONS)) for a, p in items]
    def batch(items):
        return model.predict_candidates(cache, [a for a, _ in items], [p for _, p in items], **VERSIONS).tolist()
    scalar(unique[:2]); batch(unique)
    for n in [100, 1000, 5000]:
        items = [unique[i % len(unique)] for i in range(n)]
        timings, outputs = {}, {}
        for name, fn, data in [('scalar', scalar, items), ('dedup_scalar', scalar, unique),
                               ('tensor_only', batch, items), ('dedup_tensor', batch, unique)]:
            start = time.perf_counter()
            values = fn(data)
            outputs[name] = values if len(data) == n else [values[i%len(unique)] for i in range(n)]
            timings[name] = (time.perf_counter()-start)*1000
        for values in outputs.values():
            torch.testing.assert_close(torch.tensor(values), torch.tensor(outputs['scalar']), atol=1e-5, rtol=0)
        print(dict(count=n, unique_kc=len(unique), device='cpu', threads=torch.get_num_threads(), total_ms=timings))
