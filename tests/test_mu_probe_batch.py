import pytest
import torch
from test_akt_fmkc_incremental import model_for, VERSIONS, single_thread


def test_experiment_batch_parity():
    from pianokt_backend.experiments.benchmark import Benchmark, MU
    model=model_for(drift='response')
    cache=model.empty_cache(**VERSIONS)
    for i in range(5):
        cache,_=model.observe(cache,[2,3],1,i%2,str(i),**VERSIONS)
    bank=[dict(attrs=[2,3],pid=pid,mu_override=mu) for pid in [1,2,3]*10 for mu in MU]
    runner=Benchmark(model,bank,VERSIONS)
    actual=runner.probe(cache)
    expected=torch.tensor([float(model.predict_next_with_mu_override(cache,p['attrs'],p['pid'],p['mu_override'],**VERSIONS)) for p in bank]).reshape(-1,5)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
    assert cache.next_position==5
