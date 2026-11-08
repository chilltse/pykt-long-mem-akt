"""Detached, per-student AKT inference state (never model parameters)."""
from dataclasses import dataclass, replace
import torch

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LayerCache:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor

    def append(self, key, value, position, limit):
        positions = torch.tensor([position], device=key.device, dtype=torch.long)
        key = torch.cat((self.key, key.detach()), dim=2)
        value = torch.cat((self.value, value.detach()), dim=2)
        positions = torch.cat((self.positions, positions))
        start = 0 if limit is None else max(0, len(positions) - limit)
        return LayerCache(key[:, :, start:].clone(), value[:, :, start:].clone(),
                          positions[start:].clone())

    def to(self, device, dtype):
        return LayerCache(self.key.detach().to(device=device, dtype=dtype).clone(),
                          self.value.detach().to(device=device, dtype=dtype).clone(),
                          self.positions.to(device=device).clone())


@dataclass(frozen=True)
class StudentCache:
    model_version: str
    feature_version: str
    seq_len: object
    response_layers: tuple
    question_layers: tuple
    retrieval_layers: tuple
    next_position: int = 0
    history_revision: int = 0
    last_event_cursor: object = None
    cache_schema_version: int = CACHE_SCHEMA_VERSION

    def to(self, device='cpu', dtype=torch.float32):
        return replace(self, **{name: tuple(c.to(device, dtype) for c in getattr(self, name))
                                for name in ('response_layers', 'question_layers', 'retrieval_layers')})

    def state_dict(self):
        result = dict(vars(self))
        for name in ('response_layers', 'question_layers', 'retrieval_layers'):
            result[name] = [dict(key=c.key.detach().cpu().clone(),
                                 value=c.value.detach().cpu().clone(),
                                 positions=c.positions.cpu().clone()) for c in getattr(self, name)]
        return result

    @classmethod
    def from_state_dict(cls, state, device='cpu', dtype=torch.float32):
        state = dict(state)
        if state.get('cache_schema_version') != CACHE_SCHEMA_VERSION:
            raise ValueError('Unsupported cache schema version; replay history')
        for name in ('response_layers', 'question_layers', 'retrieval_layers'):
            state[name] = tuple(LayerCache(**c) for c in state[name])
        return cls(**state).to(device, dtype)
