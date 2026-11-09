"""Service contract tests with a tiny synthetic model; no checkpoint required."""
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient

from pykt.models.akt_fmkc import AKTFMKC
from pykt.models.akt_state_store import StateConflict
from pykt.preprocess.student_sequences import fit_encoding
from pykt.preprocess.vector_kc_preprocess import category
from pykt.serving import AKTFMKCInferenceEngine, Settings
from pykt.serving.api import create_app
from pykt.serving.encoder import KCEncoder
from pykt.serving.errors import ServiceError

NAMES = ['key', 'count']
VECTORS = [[False, 1], [True, 2], [None, 'inf'], [0, 1.0]]
CANDIDATES = [dict(candidate_id=str(i), kc_vector=v) for i, v in enumerate(VECTORS)]


def event(i='e1', vector=None, response=1):
    return dict(event_id=i, kc_vector=VECTORS[0] if vector is None else vector, response=response)


def artifacts():
    rows = []
    for v in VECTORS:
        tokens = [category(x) for x in v]
        rows.append((tokens, json.dumps(tokens, ensure_ascii=False, separators=(',', ':')), 1))
    encoding = fit_encoding({'train': rows}, ['train'], NAMES)
    config = dict(field_sizes=[len(m)+3 for m in encoding['fields']], n_pid=len(encoding['questions']),
                  d_model=8, n_blocks=2, dropout=.2, field_dim=3, num_attn_heads=2,
                  d_ff=12, final_fc_dim=16, seq_len=3)
    torch.manual_seed(42)
    return AKTFMKC(**config), dict(data_config=dict(encoding=encoding), model_config=config)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def engine(tmp_path):
    model, metadata = artifacts()
    e = AKTFMKCInferenceEngine(model, metadata, Settings(state_db_path=str(tmp_path / 'state.sqlite3')))
    yield e
    e.close()


def database(e):
    return {table: e.store.db.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()
            for table in ('events', 'snapshots', 'event_requests', 'attempts')}


def test_encoding_training_parity(engine):
    for i, vector in enumerate(VECTORS):
        q = engine.encode_kc(vector)
        tokens = [category(x) for x in vector]
        assert q.attrs == [2 if t is None else m.get(t, 1) for t, m in zip(tokens, engine.encoder.encoding['fields'])]
        assert q.pid == i + 1
        assert not q.unknown_question and not q.unknown_fields
    q = engine.encode_kc(['unseen', 999])
    assert q.attrs == [1, 1] and q.pid == 0 and q.unknown_fields == NAMES
    assert engine.encode_kc(['', 'inf']).attrs[0] == 2
    # Same known categories in a previously unseen combination.
    assert engine.encode_kc([False, 2]).unknown_question
    assert engine.encode_kc([False, 1]).attrs != engine.encode_kc([0, 1.0]).attrs


@pytest.mark.parametrize('vector', [[1], [1, 2, 3], [{}, 1], [[], 1], [float('nan'), 1], [1, float('inf')]])
def test_invalid_vectors(engine, vector):
    with pytest.raises(ValueError):
        engine.encode_kc(vector)


def test_observe_recover_isolation_and_readonly(engine):
    initial = database(engine)
    p = engine.predict_candidates('alice', CANDIDATES)
    assert p['state']['history_revision'] == 0
    assert database(engine) == initial
    first = engine.observe('alice', event(), expected_revision=0)
    assert first['state']['history_revision'] == 1
    assert first['observations'][0]['probability'] == p['predictions'][0]['probability']
    assert engine.get_student_state('bob')['next_position'] == 0
    assert engine.predict_candidates('bob', CANDIDATES)['predictions'] == p['predictions']
    second = engine.observe('alice', event(), expected_revision=0)
    assert second['state']['next_position'] == 1 and second['observations'][0]['duplicate']
    assert second['observations'][0]['probability'] == first['observations'][0]['probability']
    with pytest.raises(StateConflict):
        engine.observe('alice', event(response=0), expected_revision=1)
    snapshot = database(engine)
    after = engine.predict_candidates('alice', CANDIDATES)
    assert database(engine) == snapshot
    engine.store.close()
    from pykt.models.akt_state_store import StudentStateStore
    engine.store = StudentStateStore(engine.settings.state_db_path, engine.model, **engine.versions)
    engine.store.bind_versions()
    assert engine.predict_candidates('alice', CANDIDATES) == after
    assert engine.rebuild_student('alice')['history_revision'] == 1
    assert engine.predict_candidates('alice', CANDIDATES) == after


def test_oov_raw_payload_dedup(engine):
    engine.observe('a', event(vector=['unknown-a', 999]), 0)
    with pytest.raises(StateConflict):
        engine.observe('a', event(vector=['unknown-b', 999]), 1)


def test_batch_atomicity_prevalidation_and_inference_failure(engine, monkeypatch):
    events = [event(str(i), response=i % 2) for i in range(4)]
    original = database(engine)
    with pytest.raises(ValueError):
        engine.observe_batch('a', events + [event('bad', response=2)], 0, 'attempt')
    assert database(engine) == original
    real_observe = engine.model.observe
    def fail(cache, *args, **kwargs):
        if cache.next_position == 2:
            raise RuntimeError('injected inference failure')
        return real_observe(cache, *args, **kwargs)
    monkeypatch.setattr(engine.model, 'observe', fail)
    with pytest.raises(RuntimeError, match='injected'):
        engine.observe_batch('a', events, 0, 'attempt')
    assert database(engine) == original
    monkeypatch.setattr(engine.model, 'observe', real_observe)
    out = engine.observe_batch('a', events, 0, 'attempt')
    assert out['atomic'] and out['state']['history_revision'] == 4
    repeated = engine.observe_batch('a', events, 0, 'attempt')
    assert all(o['duplicate'] for o in repeated['observations'])
    with pytest.raises(StateConflict):
        engine.observe_batch('a', list(reversed(events)), 4, 'attempt')
    with pytest.raises(StateConflict):
        engine.observe_batch('a', [events[0], event('new')], 4)
    with pytest.raises(StateConflict):
        engine.observe_batch('a', [event('new')], 0)
    assert engine.get_student_state('a')['history_revision'] == 4


def test_expected_revision_serializes_workers(engine):
    def write(i):
        try:
            engine.observe('a', event(str(i)), expected_revision=0)
            return 'ok'
        except StateConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(write, range(2))) == ['conflict', 'ok']
    assert engine.get_student_state('a')['history_revision'] == 1


def test_branch_lifecycle_isolation_ownership_and_ttl(engine):
    engine.observe('a', event(), 0)
    before = database(engine)
    a = engine.fork_student('a', principal='alice')
    b = engine.fork_student('a', principal='alice')
    assert database(engine) == before
    original_b = engine.predict_branch(b['branch_id'], CANDIDATES, principal='alice')
    out = engine.simulate_step(a['branch_id'], VECTORS[0], 0, principal='alice')
    assert out['branch']['next_position'] == 2
    assert out['branch']['simulated_steps'] == 1
    assert engine.predict_branch(b['branch_id'], CANDIDATES, principal='alice')['predictions'] == original_b['predictions']
    assert engine.branch_info(b['branch_id'], principal='alice')['next_position'] == 1
    assert database(engine) == before
    with pytest.raises(ServiceError):
        engine.predict_branch(a['branch_id'], CANDIDATES, principal='mallory')
    engine.discard_branch(a['branch_id'], principal='alice')
    with pytest.raises(ServiceError):
        engine.simulate_step(a['branch_id'], VECTORS[0], 1, principal='alice')
    engine.branches.clock = lambda: b['expires_at'] + 1
    with pytest.raises(ServiceError):
        engine.predict_branch(b['branch_id'], CANDIDATES, principal='alice')


def test_forced_rollout_manual_oracle_and_no_writes(engine):
    engine.observe('a', event(), 0)
    before = database(engine)
    weights = {k: v.clone() for k, v in engine.model.state_dict().items()}
    cache = engine.store.recover('a')
    sequence = [dict(kc_vector=v, response=i % 2) for i, v in enumerate(VECTORS)]
    manual = []
    for i, step in enumerate(sequence):
        q = engine.encode_kc(step['kc_vector'])
        cache, p = engine.model.observe(cache, q.attrs, q.pid, step['response'], f'manual-{i}', **engine.versions)
        manual.append(float(p))
    result = engine.run_rollout('a', sequence, 'forced', seed=1, forecast_candidates=CANDIDATES, retain_branch=True)
    assert [s['probability'] for s in result['trajectories'][0]['steps']] == manual
    assert result['trajectories'][0]['final_position'] == 5
    assert result['branch']['base_revision'] == 1
    assert result['branch']['simulated_steps'] == 4
    predicted = engine.predict_branch(result['branch']['branch_id'], CANDIDATES)['predictions']
    assert predicted == engine._predict(cache, engine._candidates(CANDIDATES))
    assert database(engine) == before
    for key, weight in engine.model.state_dict().items():
        assert torch.equal(weights[key], weight) and not weight.requires_grad
    q = engine.encode_kc(VECTORS[0])
    real = engine.store.recover('a')
    _, expected = engine.model.observe(real, q.attrs, q.pid, 1, 'real2', **engine.versions)
    observed = engine.observe('a', event('real2'), 1)
    assert observed['state']['next_position'] == 2
    assert observed['observations'][0]['probability'] == float(expected)


def test_sample_repeatable_independent_trajectories(engine):
    engine.observe('a', event(), 0)
    before = database(engine)
    args = dict(student_id='a', sequence=[dict(kc_vector=v) for v in VECTORS],
                mode='sample', seed=123, num_trajectories=5, forecast_candidates=CANDIDATES)
    a, b = engine.run_rollout(**args), engine.run_rollout(**args)
    assert a == b
    assert database(engine) == before
    assert all(t['final_position'] == 5 for t in a['trajectories'])
    first_p = [t['steps'][0]['probability'] for t in a['trajectories']]
    assert len(set(first_p)) == 1
    one = engine.run_rollout(**dict(args, num_trajectories=1))
    assert one['trajectories'][0] == a['trajectories'][0]
    assert all(0 <= s['mean'] <= 1 for s in a['forecast_statistics'].values())
    assert not engine.branches._branches


def test_versions_and_encoding_validation(engine):
    model, metadata = artifacts()
    with torch.no_grad():
        model.base_head.bias.add_(1)
    with pytest.raises(ValueError, match='version mismatch'):
        AKTFMKCInferenceEngine(model, metadata, engine.settings)
    with pytest.raises(ValueError, match='order'):
        engine.encoder.check_names(list(reversed(NAMES)))
    bad = copy.deepcopy(metadata['data_config']['encoding'])
    bad['fields'][0]['false'] = 999
    with pytest.raises(ValueError):
        KCEncoder(bad, [e.num_embeddings for e in model.field_embeddings], model.n_pid)


def test_limits(engine):
    engine.branches.settings = replace(engine.settings, max_branches=1)
    branch = engine.fork_student('a')
    with pytest.raises(ServiceError, match='count'):
        engine.fork_student('a')
    engine.discard_branch(branch['branch_id'])
    engine.settings = replace(engine.settings, max_rollout_steps=1, max_output_steps=1)
    with pytest.raises(ValueError, match='steps'):
        engine.run_rollout('a', [dict(kc_vector=v) for v in VECTORS])
    with pytest.raises(ValueError, match='output'):
        engine.run_rollout('a', [dict(kc_vector=VECTORS[0])], num_trajectories=2)
    engine.settings = replace(engine.settings, max_branch_bytes=1)
    with pytest.raises(ServiceError, match='memory'):
        engine.run_rollout('a', [dict(kc_vector=VECTORS[0])])


def test_api_all_routes_auth_and_errors(engine):
    token, admin = 'worker-' + 'x' * 40, 'admin-' + 'y' * 40
    settings = replace(engine.settings, api_token=token, admin_token=admin, allowed_students='a')
    app = create_app(engine, settings)
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/kt/model/info').status_code == 401
        assert client.get('/docs').status_code == 200
        client.headers['Authorization'] = 'Bearer ' + token
        assert client.get('/kt/model/info').json()['kc_feature_names'] == NAMES
        assert client.get('/kt/students/b/state').status_code == 403
        assert client.get('/kt/students/a/state').json()['history_revision'] == 0
        predict = dict(kc_feature_names=NAMES, candidates=CANDIDATES)
        assert client.post('/kt/students/a/predict', json=predict).status_code == 200
        body = dict(kc_feature_names=NAMES, expected_revision=0, event=event())
        assert client.post('/kt/students/a/observe', json=body).status_code == 200
        assert client.post('/kt/students/a/observe', json=body).json()['observations'][0]['duplicate']
        body['event'] = event('new')
        assert client.post('/kt/students/a/observe', json=body).status_code == 409
        body['expected_revision'] = 1
        body['event']['response'] = True
        assert client.post('/kt/students/a/observe', json=body).status_code == 422
        wrong = dict(predict, kc_feature_names=list(reversed(NAMES)))
        assert client.post('/kt/students/a/predict', json=wrong).status_code == 422
        batch = dict(kc_feature_names=NAMES, expected_revision=1, attempt_id='attempt2', events=[event('second')])
        assert client.post('/kt/students/a/observe-batch', json=batch).json()['state']['history_revision'] == 2
        branch = client.post('/kt/branches', json=dict(student_id='a')).json()['branch_id']
        assert client.post(f'/kt/branches/{branch}/step', json=dict(kc_feature_names=NAMES, kc_vector=VECTORS[0], response=0)).status_code == 200
        assert client.post(f'/kt/branches/{branch}/predict', json=predict).status_code == 200
        rollout = dict(student_id='a', kc_feature_names=NAMES, sequence=[dict(kc_vector=v) for v in VECTORS],
                       mode='sample', num_trajectories=2, seed=23, forecast_candidates=CANDIDATES)
        response = client.post('/kt/rollouts/evaluate', json=rollout)
        assert response.status_code == 200, response.text
        assert response.json()['base_revision'] == 2
        assert client.delete(f'/kt/branches/{branch}').status_code == 200
        assert client.post(f'/kt/branches/{branch}/predict', json=predict).status_code == 404
        assert client.post('/kt/students/a/rebuild').status_code == 403
        assert client.post('/kt/students/a/rebuild', headers={'Authorization': 'Bearer ' + admin}).status_code == 200


def test_auth_fail_closed(engine):
    with pytest.raises(ValueError, match='KT_API_TOKEN'):
        create_app(engine)
    with TestClient(create_app(engine, replace(engine.settings, dev_auth=True))) as client:
        assert client.get('/kt/students/a/state').status_code == 200
        assert client.post('/kt/students/a/rebuild').status_code == 403


def test_checkpoint_loader_and_fingerprints(tmp_path):
    model, metadata = artifacts()
    checkpoint = tmp_path / 'qid_model.ckpt'
    # Standard weights + sibling config export.
    metadata['data_config'].update(field_sizes=metadata['model_config']['field_sizes'],
                                   num_q=metadata['model_config']['n_pid'])
    torch.save(model.state_dict(), checkpoint)
    (tmp_path / 'config.json').write_text(json.dumps(metadata))
    settings = Settings(checkpoint_path=str(checkpoint), state_db_path=str(tmp_path / 'state.sqlite'))
    e = AKTFMKCInferenceEngine.from_env(settings)
    try:
        assert e.predict_next('a', VECTORS[0])['state']['history_revision'] == 0
    finally:
        e.close()


@pytest.mark.skipif(not os.environ.get('FMKC_CHECKPOINT_PATH'), reason='FMKC_CHECKPOINT_PATH not set')
def test_real_checkpoint_smoke(tmp_path):
    settings = replace(Settings.from_env(), state_db_path=str(tmp_path / 'smoke.sqlite3'))
    e = AKTFMKCInferenceEngine.from_env(settings)
    try:
        tokens = json.loads(next(iter(e.encoder.encoding['questions'])))
        vector = [None if t is None else json.loads(t) for t in tokens]
        assert 0 <= e.predict_next('smoke', vector)['predictions'][0]['probability'] <= 1
        assert e.observe('smoke', event(vector=vector), 0)['state']['history_revision'] == 1
        before = database(e)
        rollout = e.run_rollout('smoke', [dict(kc_vector=vector)], seed=42, num_trajectories=2)
        assert rollout['base_revision'] == 1 and database(e) == before
    finally:
        e.close()


def test_empty_student_rollout_is_readonly_and_rng_is_local(engine):
    import random
    before = database(engine)
    python_rng, torch_rng = random.getstate(), torch.random.get_rng_state()
    result = engine.run_rollout('new', [dict(kc_vector=VECTORS[0])], seed=99, num_trajectories=3)
    assert result['base_revision'] == 0
    assert database(engine) == before
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


def test_branch_tensor_storage_is_independent(engine):
    engine.observe('a', event(), 0)
    a, b = engine.fork_student('a'), engine.fork_student('a')
    left = engine.branches.get(a['branch_id'], 'python').cache
    right = engine.branches.get(b['branch_id'], 'python').cache
    for name in ('response_layers', 'question_layers', 'retrieval_layers'):
        for x, y in zip(getattr(left, name), getattr(right, name)):
            assert x.key.data_ptr() != y.key.data_ptr()
            assert x.value.data_ptr() != y.value.data_ptr()
            assert x.positions.data_ptr() != y.positions.data_ptr()
    left.response_layers[0].key.add_(100)
    real = engine.store.recover('a')
    torch.testing.assert_close(right.response_layers[0].key, real.response_layers[0].key)


def test_failed_simulation_and_batch_validation_do_not_advance(engine):
    branch = engine.fork_student('a')
    with pytest.raises(ValueError):
        engine.simulate_step(branch['branch_id'], VECTORS[0], True)
    assert engine.branch_info(branch['branch_id'])['simulated_steps'] == 0
    before = database(engine)
    with pytest.raises(ValueError):
        engine.run_rollout('a', [dict(kc_vector=VECTORS[0], response=1),
                                dict(kc_vector=[{}], response=0)], mode='forced')
    assert database(engine) == before
    with pytest.raises(ValueError):
        engine.run_rollout('a', [dict(kc_vector=VECTORS[0])], mode='forced')
    with pytest.raises(ValueError):
        engine.run_rollout('a', [dict(kc_vector=VECTORS[0], response=1)], mode='sample')
    with pytest.raises(ValueError):
        engine.run_rollout('a', [dict(kc_vector=VECTORS[0])], retain_branch=True, num_trajectories=2)


def test_version_guard_without_snapshot(engine):
    engine.observe('a', event(), 0)
    with engine.store.db:
        engine.store.db.execute('DELETE FROM snapshots')
    model, metadata = artifacts()
    metadata['data_config']['encoding']['field_names'] = ['different', 'count']
    with pytest.raises(ValueError, match='version mismatch'):
        AKTFMKCInferenceEngine(model, metadata, engine.settings)
    assert engine.get_student_state('a')['history_revision'] == 1


def test_same_weight_different_window_fingerprint(tmp_path):
    model, metadata = artifacts()
    e = AKTFMKCInferenceEngine(model, metadata, Settings(state_db_path=str(tmp_path / 'state.sqlite')))
    settings = e.settings
    e.close()
    other = AKTFMKC(**dict(metadata['model_config'], seq_len=5))
    other.load_state_dict(model.state_dict())
    with pytest.raises(ValueError, match='version mismatch'):
        AKTFMKCInferenceEngine(other, metadata, settings)


def test_unbounded_window_simulation_limit(tmp_path):
    model, metadata = artifacts()
    model.seq_len = None
    settings = Settings(state_db_path=str(tmp_path / 'state.sqlite'), max_rollout_steps=8)
    e = AKTFMKCInferenceEngine(model, metadata, settings)
    try:
        result = e.run_rollout('a', [dict(kc_vector=VECTORS[0])] * 8, seed=4, retain_branch=True)
        branch = e.branches.get(result['branch']['branch_id'], 'python')
        assert branch.cache.response_layers[0].key.shape[2] == 8
        with pytest.raises(ServiceError, match='steps'):
            e.simulate_step(branch.branch_id, VECTORS[0], 1)
    finally:
        e.close()
