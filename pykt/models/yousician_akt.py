"""Four-field embeddings with the original AKT backbone and prediction head."""
import torch
from torch import nn

from .akt import AKT


class AKTFMKC(AKT):
    def __init__(self, field_sizes, n_pid, d_model=256, n_blocks=4, dropout=0.2,
                 field_dim=32, drift="response", l2=1e-5, seq_len=None, **kwargs):
        if kwargs.get("emb_type", "qid") != "qid" or kwargs.get("separate_qa", False):
            raise ValueError("Only the original qid backbone with field-based interactions is supported")
        if len(field_sizes) != 4 or any(size < 3 for size in field_sizes):
            raise ValueError("Four vocabularies including PAD/UNK/MISSING are required")
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
        self.drift = drift
        if seq_len is not None and seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len
        self.field_embeddings = nn.ModuleList([
            nn.Embedding(size, field_dim, padding_idx=0) for size in field_sizes
        ])
        self.base_head = nn.Linear(4 * field_dim, d_model)
        self.response_embedding = nn.Embedding(2, d_model)
        if drift != "none":
            self.question_scalar = nn.Embedding(n_pid + 1, 1, padding_idx=0)
            nn.init.zeros_(self.question_scalar.weight)
            self.question_drift_head = nn.Linear(4 * field_dim, d_model)
        if drift == "response":
            self.wrong_drift_head = nn.Linear(4 * field_dim, d_model)
            self.correct_drift_head = nn.Linear(4 * field_dim, d_model)

    def forward(self, attrs, target, pid_data, qtest=False, loss_mask=None):
        """One row per event; only contiguous right padding is supported.

        loss_mask selects new events for regularization when replaying cached blocks.
        BCE is computed by the runner on the same selected events.
        """
        if attrs.ndim != 3 or attrs.shape[-1] != 4:
            raise ValueError("attrs must have shape [batch, time, 4]")
        if target.shape != attrs.shape[:2] or pid_data.shape != target.shape:
            raise ValueError("Event tensors must have matching batch/time dimensions")
        valid = target != -1
        if ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("Only right padding is supported")
        if ((target[valid] != 0) & (target[valid] != 1)).any():
            raise ValueError("Responses must be 0, 1, or padding -1")
        safe_attrs = attrs.masked_fill(~valid.unsqueeze(-1), 0)
        response = target.masked_fill(~valid, 0).long()
        u = torch.cat([emb(safe_attrs[..., i])
                       for i, emb in enumerate(self.field_embeddings)], dim=-1)
        c = self.base_head(u)
        g = self.response_embedding(response)
        question, interaction = c, c + g
        reg = c.new_zeros(())
        if self.drift != "none":
            pid = pid_data.masked_fill(~valid, 0)
            # Force unknown question scalar to zero, including after checkpoint load.
            mu = self.question_scalar(pid) * (pid != 0).unsqueeze(-1)
            question = c + mu * self.question_drift_head(u)
            if self.drift == "shared":
                interaction = question + g
            else:
                r = response.unsqueeze(-1).to(c.dtype)
                interaction = c + g + mu * (
                    (1 - r) * self.wrong_drift_head(u) + r * self.correct_drift_head(u))
            selected = valid if loss_mask is None else valid & loss_mask.bool()
            reg = mu.squeeze(-1)[selected].square().sum() * self.l2
        output = self.model(question, interaction, None, seq_len=self.seq_len)
        features = torch.cat([output, question], dim=-1)
        preds = torch.sigmoid(self.out(features).squeeze(-1))
        return (preds, reg, features) if qtest else (preds, reg)


# Compatibility with previously saved experiments and imports.
YousicianAKT = AKTFMKC
