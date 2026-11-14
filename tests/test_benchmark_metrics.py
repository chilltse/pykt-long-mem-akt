import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from pykt.benchmark.metrics import metrics,frequency_group,validate_pair,bootstrap


def test_metrics_endpoints_and_weighting():
    y=np.array([0,1,1,0]);p=np.array([0,1,.5,.5]);w=np.array([2,1,3,2])
    m=metrics(y,p,w)
    assert m['auc']==pytest.approx(roc_auc_score(np.repeat(y,w),np.repeat(p,w)))
    assert sum(b['count'] for b in m['reliability'])==8
    assert m['brier']==pytest.approx(np.average((p-y)**2,weights=w))
    assert metrics([1,1],[.2,.4])['auc'] is None
    assert metrics([],[])['auc'] is None


def test_frequency_boundaries():
    assert [frequency_group(n) for n in (0,1,10,11,50,51,100,101,500,501,1000,1001)]==[0,1,1,2,2,3,3,4,4,5,5,6]


def records():
    return [dict(student_id=str(i//2),event_index=i%2,response=i%2,predicted_probability=.2+.6*(i%2),
        kc_vector=['a'],kc_key='a',train_frequency=1,encoded_attrs=[3],unknown_fields=[],has_missing_field=False,
        fold=0,seed=1,train_student_count='2',encoding_fingerprint='x',split='test') for i in range(6)]


def test_pair_and_bootstrap():
    a=records();b=records()[::-1]
    assert validate_pair(a,b)==a
    result=bootstrap(a,b,20)
    assert all(v['ci_lower']==0 and v['ci_upper']==0 for v in result.values())
    with pytest.raises(ValueError):validate_pair(a,b[:-1])
    with pytest.raises(ValueError):validate_pair(a,b+[b[0]])
    b=records();b[0]['response']=1
    with pytest.raises(ValueError):validate_pair(a,b)


def test_bootstrap_weighted_auc_ties_reference():
    a=records();b=records()
    scores_a=[.5,.5,.1,.8,.8,.4];scores_b=[.1,.3,.2,.3,.8,.8]
    for r,p in zip(a,scores_a):r['predicted_probability']=p
    for r,p in zip(b,scores_b):r['predicted_probability']=p
    result=bootstrap(a,b,30,seed=7)['auc']
    rng=np.random.default_rng(7);reference=[]
    for _ in range(30):
        multiplicity=rng.multinomial(3,np.full(3,1/3))
        weights=np.repeat(multiplicity,2)
        y=[r['response'] for r in a]
        reference.append(roc_auc_score(y,scores_b,sample_weight=weights)-roc_auc_score(y,scores_a,sample_weight=weights))
    assert result['bootstrap_mean_difference']==pytest.approx(np.mean(reference))
    assert result['ci_lower']==pytest.approx(np.quantile(reference,.025))
