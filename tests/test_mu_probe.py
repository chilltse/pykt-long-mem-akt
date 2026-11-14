import io
import pytest
import torch
from test_akt_fmkc_incremental import model_for, VERSIONS, single_thread


@pytest.mark.parametrize('drift', ['none', 'shared', 'response'])
@pytest.mark.parametrize('history', [0, 1, 5])
def test_override_equivalence_and_isolation(drift, history):
    model = model_for(drift=drift)
    cache = model.empty_cache(**VERSIONS)
    for i in range(history):
        cache, _ = model.observe(cache, [2, 3], 1, i % 2, str(i), **VERSIONS)
    def snapshot():
        b = io.BytesIO()
        torch.save((model.state_dict(), cache.state_dict()), b)
        return b.getvalue()
    before = snapshot()
    original = model.predict_next(cache, [2, 3], 1, **VERSIONS)
    mu = model.question_scalar.weight[1].item() if drift != 'none' else 0.
    actual = model.predict_next_with_mu_override(cache, [2, 3], 1, mu, **VERSIONS)
    torch.testing.assert_close(actual, original, atol=1e-6, rtol=0)
    for level in [-.5, -.25, 0., .25, .5]:
        result = model.predict_next_with_mu_override(cache, [2, 3], 1, level, **VERSIONS)
        if drift == 'none':
            torch.testing.assert_close(result, original, atol=0, rtol=0)
    attrs, pid = torch.tensor([[[2, 3]]]), torch.tensor([[1]])
    question, context = model.encode_question(attrs, pid, mu_override=0.)
    torch.testing.assert_close(question, context[1], atol=0, rtol=0)
    assert snapshot() == before
