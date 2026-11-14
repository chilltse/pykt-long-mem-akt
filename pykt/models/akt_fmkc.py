"""Variable-width categorical KC vector embeddings with the original AKT backbone and prediction head."""
import torch
from torch import nn

from .akt import AKT


class AKTFMKC(AKT):
    def __init__(self, field_sizes, n_pid, d_model=256, n_blocks=4, dropout=0.2,
                 field_dim=32, drift="response", l2=1e-5, seq_len=None, **kwargs):
        if kwargs.get("emb_type", "qid") != "qid" or kwargs.get("separate_qa", False):
            raise ValueError("Only the original qid backbone with field-based interactions is supported")
        if not field_sizes or any(size < 3 for size in field_sizes):
            raise ValueError("Nonempty vocabularies including PAD/UNK/MISSING are required")
        if drift not in {"none", "shared", "response"}:
            raise ValueError("drift must be none, shared or response")
        if d_model % kwargs.get("num_attn_heads", 8):
            raise ValueError("d_model must be divisible by num_attn_heads")
        # Reuse AKT's exact Architecture and output head, without discrete KC paths.
        super().__init__(1, 0, d_model, n_blocks, dropout, l2=l2, **kwargs)
        del self.q_embed, self.qa_embed
        self.model_name = "akt_fmkc"
        self.student_block_input = True
        self.n_pid = n_pid
        self.num_fields = len(field_sizes)
        self.drift = drift
        if seq_len is not None and seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len
        self.field_embeddings = nn.ModuleList([
            nn.Embedding(size, field_dim, padding_idx=0) for size in field_sizes
        ])
        self.base_head = nn.Linear(self.num_fields * field_dim, d_model)
        self.response_embedding = nn.Embedding(2, d_model)
        if drift != "none":
            self.question_scalar = nn.Embedding(n_pid + 1, 1, padding_idx=0)
            nn.init.zeros_(self.question_scalar.weight)
            self.question_drift_head = nn.Linear(self.num_fields * field_dim, d_model)
        if drift == "response":
            self.wrong_drift_head = nn.Linear(self.num_fields * field_dim, d_model)
            self.correct_drift_head = nn.Linear(self.num_fields * field_dim, d_model)

    def forward(self, attrs, target, pid_data, qtest=False, loss_mask=None):
        """One row per event; only contiguous right padding is supported.

        loss_mask selects new events for regularization when replaying cached blocks.
        BCE is computed by the runner on the same selected events.
        """
        if attrs.ndim != 3 or attrs.shape[-1] != self.num_fields:
            raise ValueError(f"attrs must have shape [batch, time, {self.num_fields}]")
        if target.shape != attrs.shape[:2] or pid_data.shape != target.shape:
            raise ValueError("Event tensors must have matching batch/time dimensions")
        valid = target != -1
        if ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("Only right padding is supported")
        if ((target[valid] != 0) & (target[valid] != 1)).any():
            raise ValueError("Responses must be 0, 1, or padding -1")
        safe_attrs = attrs.masked_fill(~valid.unsqueeze(-1), 0)
        response = target.masked_fill(~valid, 0).long()
        pid = pid_data.masked_fill(~valid, 0)
        question, context = self.encode_question(safe_attrs, pid)
        interaction = self.encode_interaction(context, response)
        reg = question.new_zeros(())
        if self.drift != "none":
            selected = valid if loss_mask is None else valid & loss_mask.bool()
            reg = context[2].squeeze(-1)[selected].square().sum() * self.l2
        output = self.model(question, interaction, None, seq_len=self.seq_len)
        features = torch.cat([output, question], dim=-1)
        preds = torch.sigmoid(self.out(features).squeeze(-1))
        return (preds, reg, features) if qtest else (preds, reg)

    def encode_question(self, attrs, pid, *, mu_override=None):
        if attrs.dtype not in (torch.int32, torch.int64) or pid.dtype not in (torch.int32, torch.int64):
            raise ValueError("Field IDs and pid must be integers")
        if attrs.shape[-1] != self.num_fields or pid.shape != attrs.shape[:-1]:
            raise ValueError("Field and pid shapes do not match")
        for i, emb in enumerate(self.field_embeddings):
            if ((attrs[..., i] < 0) | (attrs[..., i] >= emb.num_embeddings)).any():
                raise ValueError("Field ID out of range")
        if ((pid < 0) | (pid > self.n_pid)).any():
            raise ValueError("pid out of range")
        u = torch.cat([emb(attrs[..., i]) for i, emb in enumerate(self.field_embeddings)], -1)
        c = self.base_head(u)
        mu = None
        question = c
        if self.drift != "none":
            mu = self.question_scalar(pid) * (pid != 0).unsqueeze(-1)
            if mu_override is not None:
                mu = torch.as_tensor(mu_override, dtype=c.dtype, device=c.device)
                if mu.numel() != 1 or not torch.isfinite(mu).all():
                    raise ValueError("mu_override must be a finite scalar")
                mu = mu.reshape(1).expand_as(self.question_scalar(pid))
            question = c + mu * self.question_drift_head(u)
        return question, (u, c, mu, question)

    def encode_interaction(self, context, response):
        if ((response != 0) & (response != 1)).any():
            raise ValueError("Response must be 0 or 1")
        u, c, mu, question = context
        g = self.response_embedding(response.long())
        if self.drift == "none":
            return c + g
        if self.drift == "shared":
            return question + g
        r = response.unsqueeze(-1).to(c.dtype)
        return c + g + mu * ((1-r)*self.wrong_drift_head(u) + r*self.correct_drift_head(u))

    def empty_cache(self, model_version, feature_version):
        from .akt_cache import LayerCache, StudentCache
        if not model_version or not feature_version:
            raise ValueError("Explicit model and feature versions are required")
        def empty(block):
            head = block.masked_attn_head
            key = head.k_linear.weight.new_empty(1, head.h, 0, head.d_k)
            return LayerCache(key, key.clone(), torch.empty(0, dtype=torch.long, device=key.device))
        return StudentCache(model_version, feature_version, self.seq_len,
                            tuple(empty(b) for b in self.model.blocks_1),
                            tuple(empty(b) for b in self.model.blocks_2[::2]),
                            tuple(empty(b) for b in self.model.blocks_2[1::2]))

    def validate_cache(self, cache, model_version, feature_version):
        from .akt_cache import CACHE_SCHEMA_VERSION
        if self.training:
            raise ValueError("Incremental inference requires eval mode")
        if (cache.model_version != model_version or cache.feature_version != feature_version
                or cache.seq_len != self.seq_len or cache.cache_schema_version != CACHE_SCHEMA_VERSION):
            raise ValueError("Incompatible cache version or seq_len; replay history")
        if cache.next_position < 0 or cache.history_revision != cache.next_position:
            raise ValueError("Invalid cache revision")
        count = cache.next_position if self.seq_len is None else min(cache.next_position, self.seq_len)
        param = next(self.parameters())
        if param.dtype != torch.float32:
            raise ValueError("Incremental inference currently requires FP32")
        for name in ('response_layers', 'question_layers', 'retrieval_layers'):
            layers = getattr(cache, name)
            if len(layers) != len(self.model.blocks_1):
                raise ValueError("Invalid cache layer count")
            for c in layers:
                head = self.model.blocks_1[0].masked_attn_head
                expected = (1, head.h, count, head.d_k)
                positions = torch.arange(cache.next_position-count, cache.next_position, device=param.device)
                if (c.key.shape != expected or c.value.shape != expected
                        or c.key.device != param.device or c.value.device != param.device
                        or c.key.dtype != param.dtype or c.value.dtype != param.dtype
                        or c.positions.dtype != torch.long or not torch.equal(c.positions, positions)):
                    raise ValueError("Invalid cache tensor shape, placement or positions")

    def _event_question(self, attrs, pid):
        device = next(self.parameters()).device
        attrs = torch.as_tensor(attrs, device=device)
        pid = torch.as_tensor(pid, device=device)
        if attrs.shape != (self.num_fields,) or pid.ndim != 0:
            raise ValueError("Expected one event: attrs [fields], scalar pid")
        if (attrs == 0).any():
            raise ValueError("Incremental events cannot contain PAD fields")
        return self.encode_question(attrs.view(1, 1, -1), pid.view(1, 1))

    def _question_step(self, cache, question, y=None):
        x = question
        questions, retrievals = [], []
        for i, (self_layer, retrieval) in enumerate(zip(self.model.blocks_2[::2], self.model.blocks_2[1::2])):
            x, key, value = self_layer.forward_step(x, cache.question_layers[i], cache.next_position,
                                                   self.seq_len, True, values=x, apply_pos=False)
            if y is not None:
                questions.append(cache.question_layers[i].append(key, value, cache.next_position, self.seq_len))
            x, key, value = retrieval.forward_step(x, cache.retrieval_layers[i], cache.next_position,
                                                   self.seq_len, False, values=y)
            if y is not None:
                retrievals.append(cache.retrieval_layers[i].append(key, value, cache.next_position, self.seq_len))
        probability = torch.sigmoid(self.out(torch.cat((x, question), -1))).reshape(-1)
        return probability.squeeze(0), tuple(questions), tuple(retrievals)

    @torch.no_grad()
    def predict_next(self, cache, attrs, pid, *, model_version, feature_version):
        """Read-only candidate prediction at the next real event position."""
        self.validate_cache(cache, model_version, feature_version)
        question, _ = self._event_question(attrs, pid)
        return self._question_step(cache, question)[0]

    @torch.inference_mode()
    def predict_next_with_mu_override(self, cache, attrs, pid, mu_override, *,
                                      model_version, feature_version):
        """Read-only counterfactual query; existing interactions retain learned drift."""
        self.validate_cache(cache, model_version, feature_version)
        # Reuse the original incremental input validation (including PAD checks).
        self._event_question(attrs, pid)
        device = next(self.parameters()).device
        attrs = torch.as_tensor(attrs, device=device).reshape(1, 1, -1)
        pid = torch.as_tensor(pid, device=device).reshape(1, 1)
        mu = torch.as_tensor(mu_override, device=device)
        if mu.numel() != 1 or not torch.isfinite(mu).all():
            raise ValueError("mu_override must be a finite scalar")
        question, _ = self.encode_question(attrs, pid, mu_override=mu)
        return self._question_step(cache, question)[0]

    @torch.no_grad()
    def predict_candidates(self, cache, attrs, pid, *, model_version, feature_version,
                           batch_size=256):
        """Tensor batches of independent candidates, sharing one immutable cache.

        Inputs are encoded IDs [candidates, fields] and [candidates]. Service-level
        deduplication happens before this method. Chunking bounds temporary K/V
        and attention memory; each chunk is a real tensor batch, not a scalar loop.
        """
        self.validate_cache(cache, model_version, feature_version)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        device = next(self.parameters()).device
        attrs, pid = torch.as_tensor(attrs, device=device), torch.as_tensor(pid, device=device)
        if attrs.ndim != 2 or attrs.shape[1] != self.num_fields or pid.shape != attrs.shape[:1]:
            raise ValueError("Expected attrs [candidates, fields] and pid [candidates]")
        if (attrs == 0).any():
            raise ValueError("Incremental events cannot contain PAD fields")
        # Validate all IDs before any attention work, including empty tensors.
        question, _ = self.encode_question(attrs.unsqueeze(1), pid.unsqueeze(1))
        if not len(attrs):
            return question.new_empty(0)
        return torch.cat([self._question_step(cache, q)[0].reshape(-1)
                          for q in question.split(batch_size)])

    @torch.no_grad()
    def observe(self, cache, attrs, pid, response, event_id, *, model_version, feature_version):
        """Return a new snapshot and pre-response probability. Store handles deduplication."""
        from dataclasses import replace
        self.validate_cache(cache, model_version, feature_version)
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a nonempty string")
        if event_id == cache.last_event_cursor:
            raise ValueError("Duplicate event; use the state store for idempotent delivery")
        question, context = self._event_question(attrs, pid)
        response = torch.as_tensor(response, device=question.device)
        if response.ndim != 0 or response.item() not in (0, 1):
            raise ValueError("Response must be scalar 0 or 1")
        y = self.encode_interaction(context, response.view(1, 1))
        responses = []
        for block, old in zip(self.model.blocks_1, cache.response_layers):
            y, key, value = block.forward_step(y, old, cache.next_position, self.seq_len, True, values=y)
            responses.append(old.append(key, value, cache.next_position, self.seq_len))
        probability, questions, retrievals = self._question_step(cache, question, y)
        return replace(cache, response_layers=tuple(responses), question_layers=questions,
                       retrieval_layers=retrievals, next_position=cache.next_position+1,
                       history_revision=cache.history_revision+1, last_event_cursor=event_id), probability
