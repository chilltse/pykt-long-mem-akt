"""Matched student-ordered AKT / short-only / dual-window experiment loop."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

from .akt_dual_window import AKTDualWindow


def metrics(rows):
    if not rows:
        return dict(count=0, auc=None, accuracy=None, logloss=None)
    y, p = np.array([(r['target'], r['prediction']) for r in rows]).T
    return dict(count=len(y), auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                accuracy=float(accuracy_score(y, p >= .5)),
                logloss=float(log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1])))


def run_epoch(model, records, block_size, optimizer=None, seed=0):
    """One optimizer step per complete student; backprop each block then detach.

    K/V never survives an optimizer update, student boundary or epoch boundary.
    All arms score the same positions (all allowed labels except student token 0).
    Baseline AKT resets its context at each block, including at evaluation.
    """
    training = optimizer is not None
    model.train(training)
    order = np.random.default_rng(seed).permutation(len(records)) if training else range(len(records))
    device = next(model.parameters()).device
    rows, usage = [], dict(long_selected=0, long_candidates=0, query_heads=0, long_query_heads=0)
    total_loss, total_count, steps = 0., 0, 0
    with torch.set_grad_enabled(training):
        for index in order:
            record = records[int(index)]
            memory = None
            labels = record['allowed'].clone().bool()
            labels[0] = False
            nlabels = int(labels.sum())
            if training:
                optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(labels), block_size):
                sl = slice(start, start + block_size)
                c, q, r = [record[key][sl].to(device)[None].long() for key in ('cseqs', 'qseqs', 'rseqs')]
                selected = labels[sl].to(device)[None]
                if isinstance(model, AKTDualWindow):
                    pred, reg, memory, stats = model.predict_block(c, r, q,
                        student_ids=[record['uid']], memory=memory)
                    for key in usage:
                        usage[key] += stats[key]
                else:
                    pred, reg = model(c, r, q)
                if selected.any():
                    loss = F.binary_cross_entropy(pred[selected], r[selected].float(), reduction='sum')
                    total_loss += float(loss.detach())
                    total_count += int(selected.sum())
                    if training:
                        (loss / nlabels + reg / nlabels).backward()
                    else:
                        for offset in selected[0].nonzero().flatten().tolist():
                            position = start + offset
                            rows.append(dict(uid=record['uid'], position=position,
                                target=int(r[0, offset]), prediction=float(pred[0, offset]),
                                boundary=position > 0 and position % block_size == 0))
            if training and nlabels:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                steps += 1
            memory = None
    report = dict(loss=total_loss / max(total_count, 1), scored_labels=total_count, optimizer_steps=steps,
                  overall=metrics(rows), boundary=metrics([r for r in rows if r['boundary']]), long_usage=usage)
    usage['mean_selected_per_query_head'] = usage['long_selected'] / max(usage['query_heads'], 1)
    usage['fraction_query_heads_using_long'] = usage['long_query_heads'] / max(usage['query_heads'], 1)
    return report, rows
