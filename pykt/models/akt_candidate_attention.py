"""AKT monotonic attention over a time-sorted joint candidate set.

Local K/V are shared matrices. Only selected remote features are gathered per
query chunk. Exact scanning never broadcasts all historical Values per query.
"""
import math
import time
import torch
import torch.nn.functional as F
from .akt_memory_index import ExactIPIndex, heads


def safe_masked_softmax(scores, valid):
    if scores.size(-1) == 0:
        return scores.float() * 0
    if not bool(torch.isfinite(scores.masked_select(valid)).all()):
        raise ValueError('Nonfinite valid attention scores')
    scores = scores.float().masked_fill(~valid, float('-inf'))
    any_valid = valid.any(-1, keepdim=True)
    scores = torch.where(any_valid, scores, torch.zeros_like(scores))
    return scores.softmax(-1).masked_fill(~valid, 0.)


def monotonic_weights(scores, valid, positions, query_pos, gammas, pdiff=None):
    with torch.no_grad():
        p = safe_masked_softmax(scores, valid)
        tail = (p.sum(-1, keepdim=True) - p.cumsum(-1)).clamp_min(0)
        gap = (query_pos[None, :, None] - positions).clamp_min(0).float()
        distance = (tail * gap).clamp_min(0).sqrt()
    exponent = -F.softplus(gammas.float()) * distance
    if pdiff is not None:
        exponent = exponent * pdiff.float().sigmoid().exp()[None]
    return safe_masked_softmax(scores.float() * exponent.exp().clamp(1e-5, 1e5), valid)


def _remote_projection(linear, features, h, d):
    weight = linear.weight.reshape(h, d, -1)
    output = torch.einsum('hqcf,hdf->hqcd', features, weight)
    if linear.bias is not None:
        output = output + linear.bias.reshape(h, 1, 1, d)
    return output


def _clock(model, device):
    if model.collect_diagnostics and device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def candidate_attention(attention, query, key, value, pos, layer_memory,
                        strict, pdiff, model, diagnostics):
    # Inputs belong to one student, with padding removed and increasing positions.
    h, d = attention.h, attention.d_k
    projection_start = _clock(model, query.device)
    q = heads(attention.k_linear if attention.kq_same else attention.q_linear, query, h, d)
    recent = layer_memory.get('recent')
    if recent is not None and len(recent):
        local_key = torch.cat((recent.key.to(key), key), 0)
        local_value = torch.cat((recent.value.to(value), value), 0)
        local_pos = torch.cat((recent.pos.to(pos.device), pos))
    else:
        local_key, local_value, local_pos = key, value, pos
    order = local_pos.argsort()
    local_pos = local_pos[order]
    local_k = heads(attention.k_linear, local_key[order], h, d)
    local_v = heads(attention.v_linear, local_value[order], h, d)
    diagnostics['projection_seconds'] += _clock(model, query.device)-projection_start
    remote = layer_memory.get('remote') if strict else None
    has_remote = remote is not None and len(remote) > 0
    k = len(remote) if has_remote and model.memory_mode == 'dense_oracle' else model.remote_topk
    k = min(k, len(remote)) if has_remote else 0
    outputs = []
    for start in range(0, len(pos), model.retrieval_query_chunk):
        end = min(start+model.retrieval_query_chunk, len(pos))
        qc, qp = q[:, start:end], pos[start:end]
        retrieval_start = _clock(model, query.device)
        if k:
            if model.memory_mode == 'dense_oracle':
                indices = torch.arange(len(remote), device=q.device)[None, None].expand(h, end-start, -1)
            else:
                index = layer_memory.get('index') or ExactIPIndex()
                indices = index.search(qc.detach(), remote, attention, k, model.retrieval_memory_chunk)
            if indices.min() < 0 or indices.max() >= len(remote):
                raise ValueError('Retrieved record ID outside this student/layer snapshot')
            remote_pos = remote.pos[indices.cpu()].to(pos.device)
            if not bool((remote_pos < qp[None, :, None]).all()):
                raise ValueError('Retrieved a future/current record')
        diagnostics['retrieval_seconds'] += _clock(model, query.device)-retrieval_start
        projection_start = _clock(model, query.device)
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores_local = qc.float() @ local_k.float().transpose(-1, -2) / math.sqrt(d)
        positions = local_pos[None, None].expand(h, end-start, -1)
        valid = positions < qp[None, :, None] if strict else positions <= qp[None, :, None]
        scores = scores_local
        if k:
            feature_k = remote.key[indices.cpu()].to(q)
            feature_v = remote.value[indices.cpu()].to(q)
            rk = _remote_projection(attention.k_linear, feature_k, h, d)
            rv = _remote_projection(attention.v_linear, feature_v, h, d)
            scores_remote = (qc[:, :, None].float() * rk.float()).sum(-1) / math.sqrt(d)
            scores = torch.cat((scores_local, scores_remote), -1)
            positions = torch.cat((positions, remote_pos), -1)
            valid = torch.cat((valid, torch.ones_like(remote_pos, dtype=torch.bool)), -1)
        diagnostics['projection_seconds'] += _clock(model, query.device)-projection_start
        attention_start = _clock(model, query.device)
        order = positions.argsort(-1)
        positions = positions.gather(-1, order)
        valid = valid.gather(-1, order)
        # Snapshot invariant: record_id == abs_pos, unique across recent/remote/current.
        if bool((positions[..., 1:] == positions[..., :-1]).any()):
            raise ValueError('Duplicate record ID in joint candidates')
        weights = monotonic_weights(scores.gather(-1, order), valid, positions, qp,
                                    attention.gammas, pdiff[start:end] if pdiff is not None else None)
        original_weights = torch.zeros_like(weights).scatter(-1, order, weights)
        if model.collect_diagnostics and strict:
            diagnostics['remote_mass_sum'] += float(original_weights[..., local_pos.numel():].sum().detach())
            diagnostics['query_heads'] += h*(end-start)
            if k:
                diagnostics['remote_hits'] += h*(end-start)
                diagnostics['candidate_age_sum'] += float((qp[None, :, None]-remote_pos).sum())
                diagnostics['remote_candidates'] += remote_pos.numel()
                diagnostics['feature_age_sum'] += float((model.feature_version-remote.version[indices.cpu()]).sum())
            if model.diagnose_dense and has_remote:
                _diagnose(attention, qc, qp, pdiff[start:end] if pdiff is not None else None,
                          local_k, local_v, local_pos, remote, indices if k else None,
                          original_weights, rv if k else None, model, diagnostics, layer_memory.get('index'))
        dropped = attention.dropout(original_weights).to(local_v.dtype)
        result = dropped[..., :local_pos.numel()] @ local_v
        if k:
            result = result + (dropped[..., local_pos.numel():, None] * rv).sum(-2)
        outputs.append(result)
        diagnostics['attention_seconds'] += _clock(model, query.device)-attention_start
    combined = torch.cat(outputs, 1).transpose(0, 1).reshape(len(pos), -1)
    return attention.out_proj(combined)


@torch.no_grad()
def _diagnose(attention, q, pos, pdiff, lk, lv, lp, remote, indices, candidate_weights,
              candidate_values, model, diagnostics, index):
    # Optional expensive oracle A: same cached features, all remote records.
    # Never enabled by the normal training/evaluation or speed benchmark path.
    h, d = attention.h, attention.d_k
    rk = heads(attention.k_linear, remote.key.to(q), h, d)
    rv = heads(attention.v_linear, remote.value.to(q), h, d)
    allk, allv = torch.cat((lk, rk), 1), torch.cat((lv, rv), 1)
    positions = torch.cat((lp, remote.pos.to(pos.device)))
    order = positions.argsort()
    positions, allk, allv = positions[order], allk[:, order], allv[:, order]
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = q.float() @ allk.float().transpose(-1, -2) / math.sqrt(d)
    positions = positions[None, None].expand(h, len(pos), -1)
    weights = monotonic_weights(scores, positions < pos[None, :, None], positions, pos,
                                attention.gammas, pdiff)
    dense_output = weights.to(allv.dtype) @ allv
    sparse_output = candidate_weights[..., :len(lp)].to(lv.dtype) @ lv
    kept = torch.zeros(h, len(pos), len(lp)+len(remote), device=q.device, dtype=torch.bool)
    kept[..., :len(lp)] = True
    if indices is not None:
        kept.scatter_(-1, indices+len(lp), True)
        sparse_output += (candidate_weights[..., len(lp):, None].to(candidate_values.dtype)*candidate_values).sum(-2)
        exact = ExactIPIndex.search(q, remote, attention, indices.size(-1), model.retrieval_memory_chunk)
        recall = (indices[..., :, None] == exact[..., None, :]).any(-1).float().mean(-1)
        diagnostics['recall_sum'] += float(recall.sum())
        diagnostics['recall_rows'] += recall.numel()
        if index is not None and len(index.ids):
            mapping = {int(v): i for i, v in enumerate(index.ids)}
            slots = torch.tensor([mapping[int(v)] for v in remote.pos[indices.cpu()].flatten()]).reshape(indices.shape)
            indexed = torch.from_numpy(index.keys)[torch.arange(h)[:, None, None], slots].to(q)
            current = rk[torch.arange(h, device=q.device)[:, None, None], indices]
            diagnostics['rescore_abs_sum'] += float(((indexed-current)*q[:, :, None]).sum(-1).abs().sum())
            diagnostics['index_age_sum'] += float((model.feature_version-torch.from_numpy(index.versions)[slots]).sum())
            diagnostics['indexed_candidates'] += slots.numel()
    diagnostics['retained_mass_sum'] += float((weights*kept[..., order]).sum())
    diagnostics['dense_output_l2_sum'] += float((dense_output-sparse_output).norm(dim=-1).sum())
    diagnostics['dense_rows'] += h*len(pos)


def residual_memory_attention(attn, query, key, value, pos, memory, pdiff, model, diagnostics):
    """V2: original block attention plus independent candidate-context memory.

    Latest memory candidate has zero tail distance, even if very old. This is
    not full-history AKT and does not impose absolute-age decay.
    """
    from .akt import attention as original_attention
    h, d = attn.h, attn.d_k
    projection_started = _clock(model, query.device)
    q = heads(attn.k_linear if attn.kq_same else attn.q_linear, query, h, d)
    k = heads(attn.k_linear, key, h, d)
    v = heads(attn.v_linear, value, h, d)
    mask = torch.ones(len(pos), len(pos), device=pos.device, dtype=torch.bool).tril(-1)[None, None]
    current = original_attention(q[None], k[None], v[None], d, mask, attn.dropout,
                                 True, attn.gammas, pdiff[None] if pdiff is not None else None)[0]
    recent = memory.get('recent')
    remote = memory.get('remote') if model.memory_mode in ('retrieval', 'dense_oracle') else None
    nr = len(recent) if recent is not None else 0
    nk = min(len(remote), model.remote_topk) if remote is not None else 0
    if remote is not None and model.memory_mode == 'dense_oracle':
        nk = len(remote)
    # All records must be committed, unique and strictly before this block.
    records = [r for r in (recent, remote) if r is not None and len(r)]
    if records:
        rp = torch.cat([r.pos for r in records])
        if len(rp.unique()) != len(rp) or bool((rp >= pos[0].cpu()).any()):
            raise ValueError('Duplicate or future memory record')
        if any(not bool(torch.isfinite(t).all()) for r in records for t in (r.key, r.value)):
            raise ValueError('Nonfinite memory features')
    rk = heads(attn.k_linear, recent.key.to(query), h, d) if nr else k[:, :0]
    rv = heads(attn.v_linear, recent.value.to(query), h, d) if nr else v[:, :0]
    diagnostics["projection_seconds"] = diagnostics.get("projection_seconds", 0.) + _clock(model, query.device)-projection_started
    outputs = []
    for start in range(0, len(pos), model.retrieval_query_chunk):
        stop = min(start + model.retrieval_query_chunk, len(pos))
        qc, qp = q[:, start:stop], pos[start:stop]
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = qc.float() @ rk.float().transpose(-1, -2) / math.sqrt(d)
        positions = (recent.pos.to(pos.device) if nr else pos[:0])[None, None].expand(h, stop-start, -1)
        if nk:
            retrieval_started = _clock(model, query.device)
            ids = ExactIPIndex.search(qc.detach(), remote, attn, nk, model.retrieval_memory_chunk)
            diagnostics["retrieval_seconds"] = diagnostics.get("retrieval_seconds", 0.) + _clock(model, query.device)-retrieval_started
            remote_k = _remote_projection(attn.k_linear, remote.key[ids.cpu()].to(query), h, d)
            remote_v = _remote_projection(attn.v_linear, remote.value[ids.cpu()].to(query), h, d)
            scores = torch.cat((scores, (qc[:, :, None].float()*remote_k.float()).sum(-1)/math.sqrt(d)), -1)
            positions = torch.cat((positions, remote.pos[ids.cpu()].to(pos.device)), -1)
        attention_started = _clock(model, query.device)
        order = positions.argsort(-1)
        ordered_pos = positions.gather(-1, order)
        scores = scores.gather(-1, order)
        valid = ordered_pos < qp[None, :, None]
        with torch.no_grad():
            initial = safe_masked_softmax(scores, valid)
            tail = (initial.sum(-1, keepdim=True)-initial.cumsum(-1)).clamp_min(0)
            distance = (tail*(qp[None, :, None]-ordered_pos).clamp_min(0).float()).sqrt()
        penalty = F.softplus(attn.gammas.float())*distance
        if pdiff is not None:
            penalty = penalty*pdiff[start:stop].float().sigmoid().exp()[None]
        logits = scores*torch.exp(-penalty).clamp(1e-5, 1e5) if model.memory_decay_mode == 'legacy_mul' else scores-penalty
        weights = safe_masked_softmax(logits, valid)
        weights = torch.zeros_like(weights).scatter(-1, order, weights)
        if model.collect_diagnostics:
            diagnostics['remote_mass_sum'] += float(weights[..., nr:].sum().detach())
            diagnostics['query_heads'] += h*(stop-start)
            if nk:
                diagnostics['remote_hits'] += h*(stop-start)
                ages = qp[None, :, None]-remote.pos[ids.cpu()].to(pos.device)
                diagnostics['candidate_age_sum'] += float(ages.sum())
                diagnostics['remote_candidates'] += ages.numel()
                diagnostics['feature_age_sum'] += float((model.feature_version-remote.version[ids.cpu()]).sum())
        dropped = attn.dropout(weights).to(v.dtype)
        result = dropped[..., :nr] @ rv
        if nk:
            result = result + (dropped[..., nr:, None]*remote_v).sum(-2)
        outputs.append(result)
        diagnostics["attention_seconds"] = diagnostics.get("attention_seconds", 0.) + _clock(model, query.device)-attention_started
    mem = torch.cat(outputs, 1)
    injected = model.memory_alpha.tanh()[:, None, None]*mem
    if model.collect_diagnostics:
        for name, tensor in (('current_norm', current), ('memory_norm', mem), ('injected_norm', injected)):
            diagnostics[name] = diagnostics.get(name, 0.) + float(tensor.detach().norm())
    combined = (current+injected).transpose(0, 1).reshape(len(pos), -1)
    return attn.out_proj(combined)
