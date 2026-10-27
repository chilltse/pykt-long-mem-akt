"""Original AKT with per-layer, detached, student-isolated sliding K/V caches."""
import math

import torch
import torch.nn.functional as F

from .akt import AKT


def window_masks(query_positions, key_positions, valid, short_window, long_window, peek=False):
    age = query_positions[:, :, None] - key_positions[:, None, :]
    eligible = valid[:, None, :]
    short = eligible & (age >= (0 if peek else 1)) & (age <= short_window)
    long = eligible & (age > short_window) & (age <= short_window + long_window)
    return short, long


def select_history(scores, short, long, topk):
    selected = torch.zeros_like(scores, dtype=torch.bool)
    if topk:
        indices = scores.detach().masked_fill(~long[:, None], -torch.inf).topk(
            min(topk, scores.size(-1)), dim=-1).indices
        selected.scatter_(-1, indices, True)
        selected &= long[:, None]
    return short[:, None] | selected, selected


def masked_softmax(scores, mask):
    """FP32 softmax; fully masked rows are exactly zero."""
    scores = scores.float()
    return scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1) * mask


def attention_branches(attn, q, k, v, short, long, topk, positions, key_positions, pdiff=None):
    """Shared normalized Q/K, independent Short AKT and Long similarity attention."""
    # Explicitly disable autocast: matmul/softmax/distance must stay FP32.
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (q.float() @ k.float().transpose(-2, -1)) * math.sqrt(attn.d_k)
        _, selected = select_history(scores, short, long, topk)
        short_mask = short[:, None]
        with torch.no_grad():
            p_s = masked_softmax(scores, short_mask)
            age = (positions[:, :, None] - key_positions[:, None, :]).abs().float()[:, None]
            distance = ((p_s.sum(-1, keepdim=True) - p_s.cumsum(-1)) * age).clamp(min=0).sqrt()
        exponent = distance * (-F.softplus(attn.gammas.float())[None])
        if pdiff is not None:
            exponent = exponent * pdiff.float()[:, None].sigmoid().exp()
        a_s = masked_softmax(scores * exponent.exp().clamp(1e-5, 1e5), short_mask)
        a_l = masked_softmax(scores, selected)
        o_s = attn.dropout(a_s).to(v.dtype) @ v
        o_l = attn.dropout(a_l).to(v.dtype) @ v
    return o_s, o_l, selected


def fuse_heads(short_output, long_output, selected, gate_bias):
    gate = gate_bias.sigmoid()[None, :, None, None].to(short_output.dtype)
    effective = gate * selected.any(-1, keepdim=True).to(gate.dtype)
    return (1 - effective) * short_output + effective * long_output


class AKTDualWindow(AKT):
    """AKT with independent Short/Long attention and static per-head gates.

    ``predict_block`` returns (predictions, regularizer, memory, usage).
    Keep the returned memory only for the next block of the same student batch.
    Start every sequence/epoch with memory=None. Valid tokens must be a prefix
    in each row; padding does not advance that student's absolute position.
    """
    def __init__(self, *args, block_size=128, short_window=128, long_window=256,
                 topk=32, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= block_size <= short_window or long_window < 0 or topk < 0:
            raise ValueError("Require 1 <= B <= W, M >= 0 and k >= 0")
        if not self.emb_type.startswith('qid') or self.emb_type.endswith(('avgpool', 'linear')):
            raise ValueError("akt-dual-window requires AKT Q/K/V attention")
        self.model_name = 'akt-dual-window'
        self.block_size, self.short_window = block_size, short_window
        self.long_window, self.topk = long_window, topk
        for layer in list(self.model.blocks_1) + list(self.model.blocks_2):
            attn = layer.masked_attn_head
            attn.register_parameter('gate_bias', torch.nn.Parameter(
                torch.full((attn.h,), math.log(0.05 / 0.95))))

    def load_compatible_state_dict(self, state_dict):
        """Accept legacy AKT weights only when the sole missing keys are gates."""
        expected = set(self.state_dict())
        supplied = set(state_dict)
        gates = {key for key in expected if key.endswith('.gate_bias')}
        missing, extra = expected - supplied, supplied - expected
        if missing - gates or extra:
            raise RuntimeError(f'Incompatible weights: missing={sorted(missing - gates)}, unexpected={sorted(extra)}')
        return self.load_state_dict(state_dict, strict=False)

    def gate_values(self):
        return {name: module.gate_bias.detach().sigmoid().cpu().tolist()
                for name, module in self.named_modules() if hasattr(module, 'gate_bias')}

    def _layer(self, layer, query, key, values, peek, apply_pos, pdiff,
               positions, valid, history):
        attn = layer.masked_attn_head
        bs, length, _ = query.shape
        def project(linear, value):
            return linear(value).view(bs, length, attn.h, attn.d_k).transpose(1, 2)
        q = project(attn.k_linear if attn.kq_same else attn.q_linear, query)
        k, v = project(attn.k_linear, key), project(attn.v_linear, values)
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
        kp, kv_valid = positions, valid
        if history is not None:
            k = torch.cat((history['k'].detach(), k), dim=2)
            v = torch.cat((history['v'].detach(), v), dim=2)
            kp = torch.cat((history['positions'], kp), dim=1)
            kv_valid = torch.cat((history['valid'], valid), dim=1)
        short, long = window_masks(positions, kp, kv_valid, self.short_window,
                                  self.long_window, peek)
        short &= valid[:, :, None]
        long &= valid[:, :, None]
        o_s, o_l, selected = attention_branches(attn, q, k, v, short, long, self.topk,
            positions, kp, pdiff if 'pdiff' in self.emb_type else None)
        output = fuse_heads(o_s, o_l, selected, attn.gate_bias)
        output = output.transpose(1, 2).contiguous().view(bs, length, -1)
        output = attn.out_proj(output)
        query = layer.layer_norm1(query + layer.dropout1(output))
        if apply_pos:
            residual = layer.linear2(layer.dropout(layer.activation(layer.linear1(query))))
            query = layer.layer_norm2(query + layer.dropout2(residual))
        # Compact each student's valid history independently, retaining W+M tokens.
        capacity = self.short_window + self.long_window
        counts = kv_valid.sum(-1).clamp(max=capacity)
        width = int(counts.max())
        indices = torch.zeros(bs, width, dtype=torch.long, device=k.device)
        keep_valid = torch.arange(width, device=k.device)[None] < counts[:, None]
        for row in range(bs):
            ix = kv_valid[row].nonzero().flatten()[-capacity:]
            indices[row, :ix.numel()] = ix
        gather = indices[:, None, :, None].expand(-1, attn.h, -1, attn.d_k)
        cache = dict(k=k.gather(2, gather).detach(), v=v.gather(2, gather).detach(),
                     positions=kp.gather(1, indices).detach(), valid=keep_valid)
        usage = dict(long_selected=int(selected.sum()),
                     long_candidates=int(long.sum()) * attn.h,
                     query_heads=int(valid.sum()) * attn.h,
                     long_query_heads=int(selected.any(-1).sum()),
                     queries=int(valid.sum()), long_queries=int(selected.any(1).any(-1).sum()),
                     _has_long=selected.any(1).any(-1).detach())
        return query, cache, usage

    def predict_block(self, q_data, target, pid_data=None, *, student_ids, memory=None, valid=None):
        if q_data.ndim != 2 or not 0 < q_data.size(1) <= self.block_size:
            raise ValueError("Expected [batch, length] with 0 < length <= B")
        ids = tuple(str(uid) for uid in student_ids)
        if len(ids) != q_data.size(0) or len(set(ids)) != len(ids):
            raise ValueError("Each batch row must have a distinct student ID")
        if valid is None:
            valid = torch.ones_like(q_data, dtype=torch.bool)
        if valid.shape != q_data.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean mask matching the input")
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Valid tokens must precede padding")
        if memory is not None and memory['student_ids'] != ids:
            raise ValueError("Student batch changed: start with memory=None")
        if memory is not None and memory.get('format') != 'normalized-gated-v1':
            raise ValueError('Discard legacy caches and restart with memory=None')
        start = torch.zeros(q_data.size(0), dtype=torch.long, device=q_data.device) if memory is None else memory['next_position']
        positions = start[:, None] + valid.long().cumsum(1) - 1
        q_data, target = q_data.masked_fill(~valid, 0), target.long().masked_fill(~valid, 0)
        if pid_data is not None:
            pid_data = pid_data.masked_fill(~valid, 0)
        qe, qa, diff, reg = self.encode_embeddings(q_data, target, pid_data)
        if diff is not None:
            reg = diff.squeeze(-1).square().masked_select(valid).sum() * self.l2
        caches, usage = {}, dict(long_selected=0, long_candidates=0, query_heads=0, long_query_heads=0)
        layer_usage = {}
        has_long = torch.zeros_like(valid)
        def run(name, layer, query, key, value, peek, apply_pos):
            out, cache, stats = self._layer(layer, query, key, value, peek, apply_pos,
                diff, positions, valid, None if memory is None else memory['layers'][name])
            caches[name] = cache
            has_long.logical_or_(stats.pop('_has_long'))
            layer_usage[name] = stats
            if not peek:  # Report Knowledge Retriever usage only.
                for metric in usage:
                    usage[metric] += stats[metric]
            return out
        y, x = qa, qe
        for i, layer in enumerate(self.model.blocks_1):
            y = run(f'encoder.{i}', layer, y, y, y, True, True)
        for i, layer in enumerate(self.model.blocks_2):
            peek = i % 2 == 0
            x = run(f'retriever.{i}', layer, x, x, x if peek else y, peek, not peek)
        features = torch.cat((x, qe), dim=-1)
        predictions = self.out(features).squeeze(-1).sigmoid()
        usage.update(layers=layer_usage, _has_long=has_long)
        new_memory = dict(format='normalized-gated-v1', student_ids=ids, next_position=(start + valid.sum(1)).detach(), layers=caches)
        return predictions, reg, new_memory, usage

    def forward(self, q_data, target, pid_data=None, qtest=False):
        """Independent complete student sequences; no memory persists between calls."""
        if qtest:
            raise ValueError("Use predict_block for streaming; qtest features are not exposed")
        memory, predictions, regularizer = None, [], 0.
        for start in range(0, q_data.size(1), self.block_size):
            sl = slice(start, start + self.block_size)
            pred, reg, memory, _ = self.predict_block(q_data[:, sl], target[:, sl],
                None if pid_data is None else pid_data[:, sl],
                student_ids=range(q_data.size(0)), memory=memory)
            predictions.append(pred)
            regularizer = regularizer + reg
        return torch.cat(predictions, dim=1), regularizer
