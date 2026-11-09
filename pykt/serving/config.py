"""Single-process deployment settings; checkpoints must be trusted local files."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    checkpoint_path: str = ''
    device: str = 'cpu'
    state_db_path: str = 'kt_state.sqlite3'
    model_version: str = ''
    feature_version: str = ''
    branch_ttl_seconds: int = 900
    max_branches: int = 64
    max_branch_bytes: int = 256 * 1024 * 1024
    max_rollout_steps: int = 512
    max_trajectories: int = 32
    max_rollout_work: int = 8192
    max_candidates: int = 256
    max_batch_events: int = 512
    max_output_steps: int = 2048
    # Explicit opt-in: loopback development, never a production auth substitute.
    dev_auth: bool = False
    api_token: str = ''
    principal: str = 'pianokt-worker'
    allowed_students: str = ''
    admin_token: str = ''

    def __post_init__(self):
        for name in ('branch_ttl_seconds', 'max_branches', 'max_branch_bytes',
                     'max_rollout_steps', 'max_trajectories', 'max_rollout_work',
                     'max_candidates', 'max_batch_events', 'max_output_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')

    @classmethod
    def from_env(cls):
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            key = 'FMKC_CHECKPOINT_PATH' if name == 'checkpoint_path' else 'KT_' + name.upper()
            if key in os.environ:
                raw = os.environ[key]
                if field.type is bool:
                    if raw.lower() not in ('true', 'false', '1', '0'):
                        raise ValueError(f'{key} must be true or false')
                    values[name] = raw.lower() in ('true', '1')
                else:
                    values[name] = field.type(raw)
        return cls(**values)
