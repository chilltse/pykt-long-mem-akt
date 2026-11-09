"""Private in-memory branches. All access is serialized by the engine lock."""
import secrets
import time
from dataclasses import dataclass

from .errors import ServiceError


def cache_bytes(cache):
    return sum(t.numel() * t.element_size()
               for name in ('response_layers', 'question_layers', 'retrieval_layers')
               for layer in getattr(cache, name) for t in (layer.key, layer.value, layer.positions))


@dataclass
class Branch:
    branch_id: str
    student_id: str
    owner: str
    base_revision: int
    cache: object
    simulated_steps: int
    created_at: float
    last_access_at: float
    expires_at: float

    def info(self):
        return dict(branch_id=self.branch_id, student_id=self.student_id,
                    base_revision=self.base_revision, simulated_steps=self.simulated_steps,
                    next_position=self.cache.next_position, model_version=self.cache.model_version,
                    feature_version=self.cache.feature_version, created_at=self.created_at,
                    last_access_at=self.last_access_at, expires_at=self.expires_at)


class RolloutBranchManager:
    def __init__(self, settings, clock=time.time):
        self.settings = settings
        self.clock = clock
        self._branches = {}

    def expire(self):
        now = self.clock()
        for key in list(self._branches):
            if self._branches[key].expires_at <= now:
                del self._branches[key]

    def check_memory(self, cache, replacing=None):
        self.expire()
        used = sum(cache_bytes(b.cache) for k, b in self._branches.items() if k != replacing)
        if used + cache_bytes(cache) > self.settings.max_branch_bytes:
            raise ServiceError('branch_memory_limit', 'Branch memory budget exceeded', 429)

    def create(self, student, cache, owner, base_revision=None, simulated_steps=0):
        self.expire()
        if len(self._branches) >= self.settings.max_branches:
            raise ServiceError('branch_limit', 'Maximum branch count reached', 429)
        self.check_memory(cache)
        now = self.clock()
        branch = Branch(secrets.token_urlsafe(32), student, owner,
                        cache.history_revision if base_revision is None else base_revision,
                        cache.to(next(iter(cache.response_layers)).key.device), simulated_steps,
                        now, now, now + self.settings.branch_ttl_seconds)
        self._branches[branch.branch_id] = branch
        return branch

    def get(self, branch_id, owner):
        self.expire()
        branch = self._branches.get(branch_id)
        if branch is None or branch.owner != owner:
            raise ServiceError('branch_not_found', 'Branch missing, expired, or inaccessible', 404)
        branch.last_access_at = self.clock()
        return branch

    def update(self, branch, cache):
        self.check_memory(cache, replacing=branch.branch_id)
        if branch.branch_id not in self._branches:
            raise ServiceError('branch_not_found', 'Branch expired during operation', 404)
        branch.cache = cache
        branch.simulated_steps += 1

    def discard(self, branch_id, owner):
        self.get(branch_id, owner)
        del self._branches[branch_id]
