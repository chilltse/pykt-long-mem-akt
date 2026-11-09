"""Storage contract for future adapters; the engine does not depend on SQL details."""
from typing import Protocol
from pykt.models.akt_cache import StudentCache


class StateRepository(Protocol):
    def bind_versions(self) -> None:
        """Reject incompatible model/feature versions, even if snapshots are absent."""
        ...

    def recover(self, student: str) -> StudentCache:
        ...

    def observe_batch(self, student: str, events: list, expected_revision=None,
                      attempt_id=None) -> tuple:
        """Validate ordered retries and atomically persist log plus final state."""
        ...

    def rebuild(self, student: str) -> StudentCache:
        ...

    def close(self) -> None:
        ...
