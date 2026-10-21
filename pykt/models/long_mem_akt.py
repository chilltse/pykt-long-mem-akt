"""Original AKT within blocks; DKVMN-style memory with block-summary writes.

Question and interaction embeddings are shared with AKT, not additional PID
vectors. Only the writer has a separate causal encoder. Runtime memory is an
explicit argument, never persistent module state.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .akt import AKT


class LongMemAKT(nn.Module):
    model_name = "long_mem_akt"

    def __init__(self, n_question, n_pid, d_model=128, n_blocks=2, dropout=.1,
                 num_slots=32, value_dim=None, block_size=200, tbptt_blocks=2,
                 time_weight=True, event_distance_fallback=False, time_unit=1000.,
                 vector_gate=False, ablation="gate", num_attn_heads=8, d_ff=256,
                 kq_same=1, separate_qa=False, l2=1e-5, final_fc_dim=512,
                 emb_type="qid", write_encoding="causal", score_protocol="legacy"):
        super().__init__()
        if d_model % num_attn_heads or min(block_size, tbptt_blocks, num_slots, n_blocks) < 1:
            raise ValueError("Positive sizes and d_model divisible by num_attn_heads required")
        if time_unit <= 0 or emb_type != "qid":
            raise ValueError("time_unit must be positive; currently supports emb_type='qid'")
        if ablation not in {"gate", "concat", "short", "long"}:
            raise ValueError("Unknown branch ablation")
        if write_encoding not in {"causal", "independent"} or score_protocol not in {"legacy", "student"}:
            raise ValueError("Unknown write encoding or scoring protocol")
        value_dim = value_dim or d_model
        self.emb_type, self.block_size, self.tbptt_blocks = emb_type, block_size, tbptt_blocks
        self.time_weight, self.time_unit = bool(time_weight), time_unit
        self.event_distance_fallback = bool(event_distance_fallback)
        self.ablation, self.write_encoding, self.score_protocol = ablation, write_encoding, score_protocol
        self.akt = AKT(n_question, n_pid, d_model, n_blocks, dropout, d_ff=d_ff,
                       kq_same=kq_same, final_fc_dim=final_fc_dim,
                       num_attn_heads=num_attn_heads, separate_qa=separate_qa,
                       l2=l2, emb_type=emb_type)
        self.pos_embed = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model, num_attn_heads, d_ff, dropout,
                                          batch_first=True, activation="relu")
        self.write_encoder = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
        self.keys = nn.Parameter(torch.empty(num_slots, d_model))
        self.memory_init = nn.Parameter(torch.empty(num_slots, value_dim))
        # Same learned key / initial value initialization as repository DKVMN.
        nn.init.kaiming_normal_(self.keys)
        nn.init.kaiming_normal_(self.memory_init)
        self.query_proj = nn.Linear(d_model, d_model)
        self.long_proj, self.short_proj = nn.Linear(value_dim, d_model), nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(3*d_model + 1, d_model), nn.ReLU(),
                                  nn.Linear(d_model, d_model if vector_gate else 1))
        self.concat_head = nn.Sequential(nn.Linear(3*d_model, final_fc_dim), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(final_fc_dim, 1))
        self.write_z = nn.Linear(d_model, value_dim)
        self.erase = nn.Sequential(nn.Linear(2*value_dim + 1, value_dim), nn.Sigmoid())
        self.add = nn.Sequential(nn.Linear(2*value_dim + 1, value_dim), nn.Tanh())
        self.raw_lambda = nn.Parameter(torch.tensor(-5.))
        self.raw_kappa = nn.Parameter(torch.tensor(0.))

    def initial_memory(self, batch, device):
        # clone gives each runtime state separate storage while retaining M_init gradients.
        return self.memory_init.unsqueeze(0).expand(batch, -1, -1).clone().to(device)

    def address(self, query):
        # Scaled/projected DKVMN dot-product addressing, as specified in the guide.
        return (self.query_proj(query) @ self.keys.T / math.sqrt(self.keys.size(1))).softmax(-1)

    def predict_block(self, c, q, r, valid, memory):
        if c.shape != valid.shape or c.size(1) > self.block_size or c.size(1) == 0:
            raise ValueError("Expected a nonempty block of length <= block_size")
        c = c.masked_fill(~valid, 0).long()
        q = q.masked_fill(~valid, 0).long()
        r = r.masked_fill(~valid, 0).long()
        qe, interaction, difficulty, _ = self.akt.encode_embeddings(c, r, q)
        qe = qe.masked_fill(~valid[..., None], 0)
        interaction = interaction.masked_fill(~valid[..., None], 0)
        # Run the ORIGINAL AKT architecture without shifts. Its interaction
        # encoder sees <=t; its retriever sees strictly <t and has zero first-row
        # attention. The residual still carries the known current question.
        # Compact valid tokens to exclude padding also from AKT's distance metric.
        outputs = []
        for i in range(c.size(0)):
            ids = valid[i].nonzero(as_tuple=True)[0]
            result = torch.zeros_like(qe[i])
            if ids.numel():
                pdiff = None if difficulty is None else difficulty[i:i+1, ids]
                encoded = self.akt.model(qe[i:i+1, ids], interaction[i:i+1, ids], pdiff)
                result = result.index_copy(0, ids, encoded[0])
            outputs.append(result)
        short = torch.stack(outputs)
        weights = self.address(qe)
        long = torch.einsum("bls,bsv->blv", weights, memory)
        l = F.layer_norm(self.long_proj(long), (qe.size(-1),))
        s = F.layer_norm(self.short_proj(short), (qe.size(-1),))
        count = valid.long().cumsum(1) - valid.long()
        gate = self.gate(torch.cat([qe, l, s, count[..., None].float().log1p()], -1)).sigmoid()
        if self.ablation == "short":
            # Exactly original AKT head/hidden state, not a transformed surrogate.
            logits = self.akt.out(torch.cat([short, qe], -1)).squeeze(-1)
        elif self.ablation == "concat":
            logits = self.concat_head(torch.cat([qe, l, s], -1)).squeeze(-1)
        else:
            fused = l if self.ablation == "long" else gate*l + (1-gate)*s
            logits = self.akt.out(torch.cat([fused, qe], -1)).squeeze(-1)
        reg_loss = qe.sum()*0 if difficulty is None else self.akt.l2 * difficulty[valid].square().sum()
        return logits.sigmoid(), {"query": qe, "interaction": interaction,
                                  "weights": weights, "gate": gate, "short": short,
                                  "logits": logits, "reg_loss": reg_loss}

    def encode_write(self, valid, cache):
        """Observed interaction encoder; <=t attention, independent of predictor states."""
        results = []
        for i in range(valid.size(0)):
            ids = valid[i].nonzero(as_tuple=True)[0]
            z = cache["interaction"].new_zeros(valid.size(1), self.write_z.out_features)
            if ids.numel():
                interaction = cache["interaction"][i:i+1, ids]
                if self.write_encoding == "causal":
                    pos = self.pos_embed(torch.arange(ids.numel(), device=valid.device))[None]
                    mask = torch.ones(ids.numel(), ids.numel(), dtype=torch.bool, device=valid.device).triu(1)
                    interaction = self.write_encoder(interaction + pos, mask=mask)
                z = z.index_copy(0, ids, self.write_z(interaction)[0])
            results.append(z)
        return torch.stack(results)

    def summarize(self, r, valid, timestamps, cache):
        z, weights = self.encode_write(valid, cache), cache["weights"]
        # NaN means unavailable timestamp. Keep double precision until differences
        # are computed: float32 epoch milliseconds erase short time intervals.
        event_pos = valid.long().cumsum(1) - 1
        event_delta = (valid.sum(1, keepdim=True) - 1 - event_pos).clamp_min(0)
        reliable = torch.zeros(valid.size(0), dtype=torch.bool, device=valid.device)
        delta = event_delta.to(torch.float64)
        if timestamps is not None and not self.event_distance_fallback:
            timestamps = timestamps.double()
            reliable = ((torch.isfinite(timestamps) & (timestamps >= 0)) | ~valid).all(1) & valid.any(1)
            for i in range(valid.size(0)):
                times = timestamps[i, valid[i]]
                if times.numel() and reliable[i] and (times[1:] >= times[:-1]).all():
                    delta[i] = ((times[-1] - timestamps[i]) / self.time_unit).clamp_min(0)
                else:
                    reliable[i] = False
        delta = delta.masked_fill(~valid, 0).to(z.dtype)
        alpha = (-F.softplus(self.raw_lambda) * delta).exp() if self.time_weight else torch.ones_like(delta)
        alpha = alpha * valid
        raw = torch.einsum("bls,bl->bs", weights, valid.to(weights.dtype))
        weighted_count = torch.einsum("bls,bl->bs", weights, alpha)
        summary = torch.einsum("bls,bl,blv->bsv", weights, alpha, z) / weighted_count.clamp_min(1e-8)[..., None]
        return {"summary": summary, "raw_count": raw, "has_valid": valid.any(1),
                "weighted_count": weighted_count, "uses_timestamps": reliable, "alpha": alpha}

    def write(self, memory, summary):
        raw = summary["raw_count"]
        inputs = torch.cat([summary["summary"], memory, raw.log1p()[..., None]], -1)
        rho = -torch.expm1(-F.softplus(self.raw_kappa) * raw)[..., None]
        updated = memory * (1 - rho*self.erase(inputs)) + rho*self.add(inputs)
        return torch.where(summary["has_valid"][:, None, None], updated, memory)
