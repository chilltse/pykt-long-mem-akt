"""Original AKT within blocks; DKVMN-style memory with block-summary writes.

Question and interaction embeddings are shared with AKT, not additional PID
vectors. Only the writer has a separate causal encoder. Runtime memory is an
explicit argument, never persistent module state.
"""
import math
import hashlib
import time
import torch
from torch import nn
import torch.nn.functional as F
from .akt import AKT


def smoothstep(value, threshold, width):
    if width <= 0:
        raise ValueError("width must be positive")
    x = ((value - threshold) / width).clamp(0, 1)
    return x.square() * (3 - 2*x)


class LongMemAKT(nn.Module):
    model_name = "long_mem_akt"

    def __init__(self, n_question, n_pid, d_model=128, n_blocks=2, dropout=.1,
                 num_slots=32, value_dim=None, block_size=200, tbptt_blocks=2,
                 time_weight=True, event_distance_fallback=False, time_unit=1000.,
                 phase="inference", num_attn_heads=8, d_ff=256,
                 kq_same=1, separate_qa=False, l2=1e-5, final_fc_dim=512,
                 emb_type="qid", write_encoding="causal", score_protocol="legacy",
                 correction_hidden=64, benefit_hidden=32, mc_samples=20, mc_seed=42):
        super().__init__()
        if d_model % num_attn_heads or min(block_size, tbptt_blocks, num_slots, n_blocks) < 1:
            raise ValueError("Positive sizes and d_model divisible by num_attn_heads required")
        if time_unit <= 0 or emb_type != "qid":
            raise ValueError("time_unit must be positive; currently supports emb_type='qid'")
        if phase not in {"base", "candidate", "inference"}:
            raise ValueError("Unknown training/inference phase")
        if write_encoding not in {"causal", "independent"} or score_protocol not in {"legacy", "student"}:
            raise ValueError("Unknown write encoding or scoring protocol")
        value_dim = value_dim or d_model
        self.emb_type, self.block_size, self.tbptt_blocks = emb_type, block_size, tbptt_blocks
        self.time_weight, self.time_unit = bool(time_weight), time_unit
        self.event_distance_fallback = bool(event_distance_fallback)
        self.phase, self.write_encoding, self.score_protocol = phase, write_encoding, score_protocol
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
        self.write_z = nn.Linear(d_model, value_dim)
        self.erase = nn.Sequential(nn.Linear(2*value_dim + 1, value_dim), nn.Sigmoid())
        self.add = nn.Sequential(nn.Linear(2*value_dim + 1, value_dim), nn.Tanh())
        self.raw_lambda = nn.Parameter(torch.tensor(-5.))
        self.raw_kappa = nn.Parameter(torch.tensor(0.))
        self.mc_samples, self.mc_seed = mc_samples, mc_seed
        if mc_samples < 2:
            raise ValueError("mc_samples must be >= 2")
        self.correction_head = nn.Sequential(nn.Linear(3*d_model+1, correction_hidden),
                                             nn.GELU(), nn.Linear(correction_hidden, 1))
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        self.benefit_head = nn.Sequential(nn.Linear(3*d_model+2, benefit_hidden),
                                          nn.GELU(), nn.Linear(benefit_hidden, 1))
        self.register_buffer("calibration", torch.full((4,), float("nan"), dtype=torch.float64))
        self.register_buffer("selector_ready", torch.tensor(False))
        self.register_buffer("uncertainty_only", torch.tensor(False))
        self.set_stage(phase)

    def set_stage(self, stage):
        if stage not in {"base", "candidate", "benefit", "inference"}:
            raise ValueError("Unknown residual stage")
        if stage == "candidate" and self.tbptt_blocks < 2:
            raise ValueError("Candidate training requires tbptt_blocks >= 2")
        for name, parameter in self.named_parameters():
            enabled = (stage == "base" and name.startswith("akt.")) or (stage == "benefit" and name.startswith("benefit_head.")) or (
                stage == "candidate" and not name.startswith(
                    ("akt.", "benefit_head.")))
            parameter.requires_grad_(enabled)
            parameter.grad = None
        self.stage = stage
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        stage = getattr(self, "stage", "inference")
        if stage != "base":
            self.akt.eval()
        if stage not in {"base", "candidate"}:
            for module in self.children():
                module.eval()
            self.benefit_head.train(mode and stage == "benefit")
        return self

    def set_calibration(self, tau_u, width_u, tau_g, width_g, uncertainty_only=False):
        values = [tau_u, width_u, tau_g, width_g]
        if not all(math.isfinite(v) for v in values) or min(width_u, width_g) <= 0 or min(tau_u, tau_g) < 0:
            raise ValueError("Finite nonnegative thresholds and positive widths required")
        self.calibration.copy_(self.calibration.new_tensor(values))
        self.uncertainty_only.fill_(uncertainty_only)

    @torch.no_grad()
    def mc_uncertainty(self, c, q, r, valid, uid, block_index):
        """Sample AKT only, preserving individual modes, gradients and caller RNG."""
        probability_dtype = self.akt.q_embed.weight.dtype
        if not valid.any():
            return torch.zeros_like(r, dtype=probability_dtype)
        key = f"{self.mc_seed}:{uid}:{block_index}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63-1)
        modes = [(module, module.training) for module in self.akt.modules()]
        devices = [c.device.index] if c.is_cuda else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if c.is_cuda:
                    with torch.cuda.device(c.device):
                        torch.cuda.manual_seed(seed)
                self.akt.eval()
                for module in self.akt.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                mean = torch.zeros_like(r, dtype=torch.float64)
                m2 = torch.zeros_like(mean)
                for i in range(self.mc_samples):
                    p = self._forward_short(c, q, r, valid)["logits"].sigmoid().double()
                    delta = p - mean
                    mean += delta / (i+1)
                    m2 += delta * (p-mean)
                return (m2/self.mc_samples).clamp_min(0).to(probability_dtype).masked_fill(~valid, 0)
        finally:
            for module, mode in modes:
                module.training = mode

    def initial_memory(self, batch, device):
        # clone gives each runtime state separate storage while retaining M_init gradients.
        return self.memory_init.unsqueeze(0).expand(batch, -1, -1).clone().to(device)

    def address(self, query):
        # Scaled/projected DKVMN dot-product addressing, as specified in the guide.
        return (self.query_proj(query) @ self.keys.T / math.sqrt(self.keys.size(1))).softmax(-1)

    def _forward_short(self, c, q, r, valid):
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
        logits = self.akt.out(torch.cat([short, qe], -1)).squeeze(-1)
        reg_loss = qe.sum()*0 if difficulty is None else self.akt.l2 * difficulty[valid].square().sum()
        return {"query": qe, "interaction": interaction, "short": short,
                "logits": logits, "reg_loss": reg_loss}

    def predict_block(self, c, q, r, valid, memory, has_history=False, uid="", block_index=0):
        if self.phase == "base":
            base = self._forward_short(c, q, r, valid)
        else:
            self.akt.eval()
            with torch.no_grad():
                base = self._forward_short(c, q, r, valid)
        qe, short = base["query"], base["short"]
        weights = self.address(qe)
        if self.phase == "base":
            return base["logits"].sigmoid(), dict(base, weights=weights)
        long = torch.einsum("bls,bsv->blv", weights, memory)
        l = F.layer_norm(self.long_proj(long), (qe.size(-1),))
        s = F.layer_norm(self.short_proj(short), (qe.size(-1),))
        return self._residual_prediction(c, q, r, valid, base, weights, l, s,
                                         has_history, uid, block_index)

    def _residual_prediction(self, c, q, r, valid, base, weights, l, s,
                             has_history, uid, block_index):
        selective = self.phase == "inference"
        if selective and (not bool(self.selector_ready) or not torch.isfinite(self.calibration).all()):
            raise ValueError("Selective inference requires trained benefit and calibration")
        z = base["logits"]
        history = torch.as_tensor(has_history, device=z.device, dtype=torch.bool)
        if history.ndim == 1:
            history = history[:, None]
        history = history.expand_as(valid) & valid
        features = torch.cat([F.layer_norm(base["query"], (s.size(-1),)), s, l, z[..., None]], -1)
        uncertainty, mc_seconds = torch.zeros_like(z), 0.
        active = history.clone()
        if selective and history.any():
            if c.is_cuda:
                torch.cuda.synchronize(c.device)
            started = time.perf_counter()
            uncertainty = self.mc_uncertainty(c, q, r, valid, uid, block_index)
            if c.is_cuda:
                torch.cuda.synchronize(c.device)
            mc_seconds = time.perf_counter()-started
            active &= uncertainty > self.calibration[0]
        delta = torch.zeros_like(z)
        gain = torch.full_like(z, float("nan"))
        if active.any():
            delta = delta.masked_scatter(active, self.correction_head(features[active]).squeeze(-1))
        benefit_input = torch.cat([features, delta[..., None]], -1)
        if selective and not bool(self.uncertainty_only) and active.any():
            gain[active] = self.benefit_head(benefit_input[active]).squeeze(-1)
        finite = torch.isfinite(delta) & torch.isfinite(features).all(-1)
        if selective:
            finite &= torch.isfinite(uncertainty)
            if not bool(self.uncertainty_only):
                finite &= ~active | torch.isfinite(gain)
        alpha = torch.zeros_like(z)
        accepted = active & finite
        if selective:
            tu, wu, tg, wg = self.calibration.tolist()
            if accepted.any():
                a = smoothstep(uncertainty[accepted], tu, wu)
                if not bool(self.uncertainty_only):
                    a = a * smoothstep(gain[accepted], tg, wg)
                alpha[accepted] = a
        else:
            alpha[accepted] = 1
        final = z.clone()
        intervention = accepted & (alpha > 0)
        final[intervention] = z[intervention] + alpha[intervention]*delta[intervention]
        bad = history & ~finite
        reason = torch.zeros_like(r, dtype=torch.long)
        reason[~history] = 1
        if selective:
            reason[history & ~active] = 2
            reason[accepted & ~intervention] = 3
        reason[bad] = 4
        candidate = torch.full_like(z, float("nan"))
        candidate[active & finite] = (z + delta)[active & finite].sigmoid()
        cache = dict(base, weights=weights, logits=final, base_logits=z,
                     base_probability=z.sigmoid(), candidate_probability=candidate,
                     short_uncertainty=uncertainty, predicted_gain=gain, alpha=alpha,
                     residual_logit=delta, has_history=history, candidate_evaluated=active,
                     intervention=intervention, fallback_reason=reason,
                     benefit_input=benefit_input, mc_seconds=mc_seconds,
                     numerical_fallback_count=int(bad.sum()))
        return final.sigmoid(), cache

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
