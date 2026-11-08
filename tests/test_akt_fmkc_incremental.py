import io
import pytest
import torch
from pykt.models.akt_fmkc import AKTFMKC
from pykt.models.akt_cache import StudentCache
from pykt.models.akt_state_store import StudentStateStore

VERSIONS = dict(model_version='weights-1', feature_version='vocab-1')


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model_for(limit=2, blocks=4, drift='response', kq=1):
    torch.manual_seed(42)
    model = AKTFMKC([5, 6], 3, d_model=8, n_blocks=blocks, dropout=.2,
                    field_dim=3, num_attn_heads=2, d_ff=12, final_fc_dim=16,
                    drift=drift, seq_len=limit, kq_same=kq).eval()
    with torch.no_grad():
        if drift != 'none':
            model.question_scalar.weight.normal_()  # Includes nonzero pid=0 weight.
        for block in list(model.model.blocks_1) + list(model.model.blocks_2):
            block.masked_attn_head.out_proj.bias.normal_()
    return model


@pytest.mark.parametrize('limit', [1, 2, 3, None])
@pytest.mark.parametrize('blocks', [1, 4])
@pytest.mark.parametrize('drift', ['none', 'shared', 'response'])
@pytest.mark.parametrize('kq', [0, 1])
@torch.no_grad()
def test_full_prefix_equivalence(limit, blocks, drift, kq):
    model = model_for(limit, blocks, drift, kq)
    attrs = torch.randint(1, 5, (1, 9, 2))
    responses = torch.randint(0, 2, (1, 9))
    pid = torch.arange(9).remainder(4).view(1, -1)
    reference, _, features = model(attrs, responses, pid, qtest=True)
    cache = model.empty_cache(**VERSIONS)
    for t in range(9):
        before = cache.state_dict()
        p = model.predict_next(cache, attrs[0, t], pid[0, t], **VERSIONS)
        torch.testing.assert_close(p, reference[0, t], atol=1e-5, rtol=1e-4)
        # Candidate order and repetitions must not modify historical tensors.
        model.predict_next(cache, [2, 3], 0, **VERSIONS)
        torch.testing.assert_close(p, model.predict_next(cache, attrs[0, t], pid[0, t], **VERSIONS))
        for name in ('response_layers', 'question_layers', 'retrieval_layers'):
            for a, b in zip(before[name], cache.state_dict()[name]):
                for key in a:
                    assert torch.equal(a[key], b[key])
        other, opposite = model.observe(cache, attrs[0, t], pid[0, t], 1-responses[0,t], str(t), **VERSIONS)
        cache, observed = model.observe(cache, attrs[0, t], pid[0, t], responses[0, t], str(t), **VERSIONS)
        torch.testing.assert_close(observed, p)
        torch.testing.assert_close(opposite, p)
        prefix = model(attrs[:, :t+1], responses[:, :t+1], pid[:, :t+1])[0][0, -1]
        torch.testing.assert_close(prefix, p, atol=1e-5, rtol=1e-4)
        assert cache.response_layers[0].key.size(2) == (t+1 if limit is None else min(limit, t+1))
        if t == 3:
            buffer = io.BytesIO()
            torch.save(cache.state_dict(), buffer)
            buffer.seek(0)
            cache = StudentCache.from_state_dict(torch.load(buffer, weights_only=True))
    # Existing checkpoint keys/shapes and padded training oracle remain supported.
    clone = model_for(limit, blocks, drift, kq)
    clone.load_state_dict(model.state_dict(), strict=True)
    padded_attrs = torch.cat((attrs, torch.full((1, 2, 2), 999)), 1)
    padded_responses = torch.cat((responses, torch.full((1, 2), -1)), 1)
    padded_pid = torch.cat((pid, torch.full((1, 2), 999)), 1)
    torch.testing.assert_close(model(padded_attrs, padded_responses, padded_pid)[0][:, :9], reference)


def test_store_recovery_dedup_and_isolation(tmp_path):
    model = model_for()
    path = tmp_path / 'students.sqlite'
    store = StudentStateStore(path, model, **VERSIONS)
    a, p = store.observe('alice', [1, 2], 0, 1, 'a')
    store.observe('bob', [3, 2], 1, 0, 'a')
    store.observe('alice', [2, 3], 1, 0, 'b')
    duplicate, same = store.observe('alice', [1, 2], 0, 1, 'a')
    assert duplicate.next_position == 2 and same == p
    assert store.recover('bob').next_position == 1
    expected = store.predict_next('alice', [2, 4], 2)
    with pytest.raises(ValueError, match='different payload'):
        store.observe('alice', [1, 2], 0, 0, 'a')
    store.close()
    store = StudentStateStore(path, model, **VERSIONS)
    torch.testing.assert_close(store.predict_next('alice', [2, 4], 2), expected)
    store.rebuild('alice')
    torch.testing.assert_close(store.predict_next('alice', [2, 4], 2), expected)
    # Simulate a crash after durable event commit but before snapshot update.
    with store.db:
        store.db.execute("INSERT INTO events VALUES ('alice',2,'c','[[1, 2], 0, 1]',NULL)")
    assert store.recover('alice').next_position == 3
    store.model_version = 'weights-2'
    with pytest.raises(ValueError, match='version'):
        store.recover('alice')
    assert store.rebuild('alice').model_version == 'weights-2'
    store.close()


def test_invalid_inputs_and_versions():
    model = model_for()
    cache = model.empty_cache(**VERSIONS)
    for attrs, pid in [([0, 1], 0), ([5, 2], 0), ([1, 2], 4), ([1.5, 2], 0)]:
        with pytest.raises(ValueError):
            model.predict_next(cache, attrs, pid, **VERSIONS)
    with pytest.raises(ValueError):
        model.observe(cache, [1, 2], 0, -1, 'x', **VERSIONS)
    with pytest.raises(ValueError):
        model.predict_next(cache, [1, 2], 0, model_version='new', feature_version='vocab-1')
    model.train()
    with pytest.raises(ValueError, match='eval'):
        model.predict_next(cache, [1, 2], 0, **VERSIONS)


@pytest.mark.parametrize('drift', ['none', 'shared', 'response'])
@torch.no_grad()
def test_original_encoding_formula(drift):
    model = model_for(drift=drift)
    attrs = torch.tensor([[[1, 2], [3, 4]]])
    pid = torch.tensor([[0, 2]])
    response = torch.tensor([[0, 1]])
    u = torch.cat([emb(attrs[..., i]) for i, emb in enumerate(model.field_embeddings)], -1)
    c = model.base_head(u)
    g = model.response_embedding(response)
    q, interaction = c, c + g
    if drift != 'none':
        mu = model.question_scalar(pid) * (pid != 0).unsqueeze(-1)
        q = c + mu * model.question_drift_head(u)
        if drift == 'shared':
            interaction = q + g
        else:
            r = response.unsqueeze(-1)
            interaction = c + g + mu * ((1-r)*model.wrong_drift_head(u) + r*model.correct_drift_head(u))
    question, context = model.encode_question(attrs, pid)
    torch.testing.assert_close(question, q, atol=0, rtol=0)
    torch.testing.assert_close(model.encode_interaction(context, response), interaction, atol=0, rtol=0)


@torch.no_grad()
def test_layer_projections_and_chunk_boundaries():
    model = model_for(limit=2)
    attrs = torch.randint(1, 5, (1, 13, 2))
    pid = torch.arange(13).remainder(4).view(1, -1)
    response = torch.randint(0, 2, (1, 13))
    projections = {}
    handles = []
    def capture(name, head):
        def hook(module, args):
            q, k, v = args[:3]
            projections[name] = (head.k_linear(k).view(1, -1, head.h, head.d_k).transpose(1, 2),
                                 head.v_linear(v).view(1, -1, head.h, head.d_k).transpose(1, 2))
        return hook
    for group, blocks in [('response_layers', model.model.blocks_1),
                          ('question_layers', model.model.blocks_2[::2]),
                          ('retrieval_layers', model.model.blocks_2[1::2])]:
        for i, block in enumerate(blocks):
            head = block.masked_attn_head
            handles.append(head.register_forward_pre_hook(capture((group, i), head)))
    reference = model(attrs, response, pid)[0]
    for handle in handles:
        handle.remove()
    for chunk_size in [1, 3, 7]:
        cache = model.empty_cache(**VERSIONS)
        for start in range(0, 13, chunk_size):
            for t in range(start, min(13, start+chunk_size)):
                cache, p = model.observe(cache, attrs[0,t], pid[0,t], response[0,t], str(t), **VERSIONS)
                torch.testing.assert_close(p, reference[0,t], atol=1e-5, rtol=1e-4)
                for (group, i), (k, v) in projections.items():
                    layer = getattr(cache, group)[i]
                    torch.testing.assert_close(layer.key, k[:, :, max(0,t-1):t+1], atol=1e-5, rtol=1e-4)
                    torch.testing.assert_close(layer.value, v[:, :, max(0,t-1):t+1], atol=1e-5, rtol=1e-4)
