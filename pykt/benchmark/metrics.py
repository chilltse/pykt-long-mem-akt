"""Probability metrics and paired student bootstrap (fixed fitted models)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def metrics(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid predictions')
    if not w.sum():
        return dict(auc=None, accuracy=None, logloss=None, brier=None, ece=None,
                    average_precision=None, reason='empty group', events=0, reliability=[])
    mean = lambda x: float(np.average(x, weights=w))
    q = np.clip(p, 1e-7, 1-1e-7)
    both = len(np.unique(y[w > 0])) == 2
    bins = np.minimum((p * 15).astype(int), 14)
    reliability = []
    for i in range(15):
        mask = bins == i
        n = float(w[mask].sum())
        reliability.append(dict(bin=i, lower=i/15, upper=(i+1)/15, count=n,
            accuracy=float(np.average(y[mask], weights=w[mask])) if n else None,
            probability=float(np.average(p[mask], weights=w[mask])) if n else None))
    return dict(events=int((w > 0).sum()), auc=float(roc_auc_score(y,p,sample_weight=w)) if both else None,
        auc_reason=None if both else 'single class',
        average_precision=float(average_precision_score(y,p,sample_weight=w)) if both else None,
        accuracy=mean((p >= .5) == y), logloss=mean(-y*np.log(q)-(1-y)*np.log1p(-q)),
        brier=mean((p-y)**2), positive_response_rate=mean(y), mean_predicted_probability=mean(p),
        ece=sum(b['count']/w.sum()*abs(b['accuracy']-b['probability']) for b in reliability if b['count']),
        reliability=reliability)


def frequency_group(n):
    return next((i for i, upper in enumerate((0,10,50,100,500,1000)) if n <= upper), 6)


def cold_group(row):
    if not row['is_unseen_combination']:
        return 'seen'
    return 'unseen_unknown_fields' if row['has_unknown_field'] else 'unseen_known_fields'


def validate_pair(a, b):
    keys = lambda rows: [(r['student_id'],r['event_index']) for r in rows]
    ka, kb = keys(a), keys(b)
    if len(set(ka)) != len(ka) or len(set(kb)) != len(kb) or set(ka) != set(kb):
        raise ValueError('Duplicate or unmatched event identities')
    lookup = dict(zip(kb,b))
    b = [lookup[k] for k in ka]
    fields = ('response','kc_vector','kc_key','train_frequency','encoded_attrs','unknown_fields',
              'has_missing_field','fold','seed','train_student_count','encoding_fingerprint','split')
    for x,y in zip(a,b):
        if any(x[k] != y[k] for k in fields):
            raise ValueError('Paired metadata mismatch')
    return b


def bootstrap(a,b,replicates=2000,seed=2026,mask=None):
    b = validate_pair(a,b)
    ids, inverse = np.unique([r['student_id'] for r in a], return_inverse=True)
    y = np.array([r['response'] for r in a])
    pa,pb = (np.array([r['predicted_probability'] for r in rows]) for rows in (a,b))
    selected = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask)
    names = ('auc','logloss','brier','accuracy')
    draws = {k: [] for k in names}
    rng = np.random.default_rng(seed)
    def prepared(prob):
        order=np.argsort(prob,kind='stable')
        starts=np.r_[0,np.flatnonzero(np.diff(prob[order]))+1]
        q=np.clip(prob,1e-7,1-1e-7)
        additive=np.stack((-y*np.log(q)-(1-y)*np.log1p(-q),(prob-y)**2,(prob>=.5)==y))
        # Aggregate additive metrics by student once, preserving multiplicities.
        sums=np.stack([np.bincount(inverse,weights=v*selected,minlength=len(ids)) for v in additive])
        return order,starts,sums
    plans=[prepared(prob) for prob in (pa,pb)]
    counts=np.bincount(inverse,weights=selected,minlength=len(ids))
    for _ in range(replicates):
        multiplicity=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
        total=counts@multiplicity
        if not total:continue
        w=multiplicity[inverse]*selected
        results=[]
        for order,starts,sums in plans:
            positive=np.add.reduceat((w*y)[order],starts)
            negative=np.add.reduceat((w*(1-y))[order],starts)
            denominator=positive.sum()*negative.sum()
            auc=float(np.dot(positive,np.cumsum(negative)-.5*negative)/denominator) if denominator else None
            results.append(dict(zip(('logloss','brier','accuracy'),(sums@multiplicity/total).tolist()),auc=auc))
        ma,mb=results
        for k in names:
            if ma[k] is not None and mb[k] is not None:
                draws[k].append(mb[k]-ma[k])
    ma,mb = metrics(y,pa,selected),metrics(y,pb,selected)
    return {k: dict(observed_difference=mb[k]-ma[k] if ma[k] is not None and mb[k] is not None else None,
        bootstrap_mean_difference=float(np.mean(v)) if v else None,
        ci_lower=float(np.quantile(v,.025)) if v else None, ci_upper=float(np.quantile(v,.975)) if v else None,
        valid_bootstrap_replicates=len(v),invalid_bootstrap_replicates=replicates-len(v),
        student_count=len(ids),event_count=int(selected.sum()), interpretation='fixed-model student sampling; exploratory subgroup CIs unadjusted')
        for k,v in draws.items()}
