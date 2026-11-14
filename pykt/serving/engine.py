"""Python inference API. The model is shared; real and hypothetical histories are not."""
import hashlib
import logging
import time
import random
import secrets
import threading
from pathlib import Path

import torch

from pykt.models.akt_fmkc import AKTFMKC
from pykt.models.akt_fmkc_runner import load_fmkc_checkpoint
from pykt.models.akt_state_store import StudentStateStore
from .branch_manager import RolloutBranchManager, cache_bytes
from .config import Settings
from .encoder import KCEncoder, canonical
from .errors import ServiceError
from .rollout import statistics
from .schemas import Candidate, Event, RolloutRequest, Step
from .storage import StateRepository


def identifier(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError('Identifiers must be nonempty strings of at most 256 characters')
    return value


def weight_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    # seq_len is not a tensor, but changes incremental inference semantics.
    heads = [b.masked_attn_head for b in list(model.model.blocks_1) + list(model.model.blocks_2)]
    digest.update(canonical({'seq_len': model.seq_len, 'drift': model.drift,
                             'attention': [(h.h, h.d_k, h.kq_same) for h in heads]}).encode())
    return digest.hexdigest()


class AKTFMKCInferenceEngine:
    def __init__(self, model, metadata, settings=None, store_factory=StudentStateStore):
        self.settings = settings or Settings()
        if not isinstance(model, AKTFMKC) or model.model_name != 'akt_fmkc':
            raise ValueError('Service requires an akt_fmkc checkpoint')
        if metadata.get('params', {}).get('model_name', 'akt_fmkc') != 'akt_fmkc':
            raise ValueError('Checkpoint metadata model_name must be akt_fmkc')
        self.model = model.float().eval()
        self.model.requires_grad_(False)
        if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
            raise ValueError('Checkpoint contains nonfinite model weights')
        try:
            encoding = metadata['data_config']['encoding']
        except (KeyError, TypeError) as exc:
            raise ValueError('Checkpoint must include metadata.data_config.encoding') from exc
        self.encoder = KCEncoder(encoding, [e.num_embeddings for e in model.field_embeddings], model.n_pid)
        self.model_version = (self.settings.model_version or 'sha256') + ':' + weight_fingerprint(model)
        self.feature_version = (self.settings.feature_version or 'sha256') + ':' + self.encoder.version
        self.versions = dict(model_version=self.model_version, feature_version=self.feature_version)
        self.lock = threading.RLock()
        self.store: StateRepository = store_factory(self.settings.state_db_path, model, **self.versions)
        try:
            self.store.bind_versions()  # DB-wide binding also protects logs without snapshots.
            cache = model.empty_cache(**self.versions)
            model.validate_cache(cache, **self.versions)
            model.predict_next(cache, [2] * model.num_fields, 0, **self.versions)
        except Exception:
            self.store.close()
            raise
        self.branches = RolloutBranchManager(self.settings)

    @classmethod
    def from_env(cls, settings=None):
        settings = settings or Settings.from_env()
        if not settings.checkpoint_path or not Path(settings.checkpoint_path).is_file():
            raise ValueError('FMKC_CHECKPOINT_PATH must identify a trusted checkpoint file')
        device = torch.device(settings.device)
        if device.type not in ('cpu', 'cuda', 'mps'):
            raise ValueError('KT_DEVICE must be cpu, cuda, or mps')
        # Fail during startup for unsupported devices or invalid checkpoint/config.
        torch.empty(1, device=device)
        model, metadata = load_fmkc_checkpoint(settings.checkpoint_path, device=device)
        return cls(model, metadata, settings)

    def close(self):
        with self.lock:
            self.branches._branches.clear()
            self.store.close()

    def model_info(self):
        return dict(model_type='akt_fmkc', **self.versions, kc_feature_names=self.encoder.field_names,
                    seq_len=self.model.seq_len, device=str(next(self.model.parameters()).device),
                    dtype='float32', single_process=True,
                    limits={k: v for k, v in vars(self.settings).items() if k.startswith('max_')})

    def encode_kc(self, kc_vector):
        return self.encoder.encode(kc_vector)

    def _state(self, student, cache):
        return dict(student_id=student, history_revision=cache.history_revision,
                    next_position=cache.next_position,
                    cache_length=cache.response_layers[0].key.shape[2], **self.versions)

    def get_student_state(self, student_id):
        with self.lock:
            return self._state(identifier(student_id), self.store.recover(student_id))

    def _candidates(self, candidates, allow_empty=False):
        if not isinstance(candidates, list) or len(candidates) > self.settings.max_candidates or (not candidates and not allow_empty):
            raise ValueError('Invalid number of candidates')
        items = [Candidate.model_validate(c) for c in candidates]
        if len({c.candidate_id for c in items}) != len(items):
            raise ValueError('candidate_id must be unique within a request')
        return [(c.candidate_id, self.encode_kc(c.kc_vector)) for c in items]

    def _predict(self, cache, encoded):
        if not encoded:
            return []
        unique, indices, inverse = [], {}, []
        for _, q in encoded:
            key = (tuple(q.attrs), q.pid)
            if key not in indices:
                indices[key] = len(unique)
                unique.append(q)
            inverse.append(indices[key])
        probabilities = self.model.predict_candidates(
            cache, [q.attrs for q in unique], [q.pid for q in unique], **self.versions).tolist()
        return [dict(candidate_id=cid, probability=probabilities[index], **q.as_dict())
                for (cid, q), index in zip(encoded, inverse)]

    def predict_candidates(self, student_id, candidates):
        started = time.perf_counter()
        identifier(student_id)
        encoded = self._candidates(candidates)
        with self.lock:
            cache = self.store.recover(student_id)
            result = dict(state=self._state(student_id, cache), predictions=self._predict(cache, encoded))
            logging.getLogger(__name__).info(
                "candidate_prediction count=%d unique_kc=%d total_ms=%.3f device=%s mode=tensor_batch",
                len(encoded), len({(tuple(q.attrs), q.pid) for _, q in encoded}),
                (time.perf_counter() - started) * 1000, next(self.model.parameters()).device)
            return result

    def predict_next(self, student_id, kc_vector):
        return self.predict_candidates(student_id, [dict(candidate_id='next', kc_vector=kc_vector)])

    def observe(self, student_id, event, expected_revision=None):
        return self.observe_batch(student_id, [event], expected_revision=expected_revision)

    def observe_batch(self, student_id, events, expected_revision=None, attempt_id=None):
        identifier(student_id)
        if attempt_id is not None:
            identifier(attempt_id)
        if not isinstance(events, list) or not 1 <= len(events) <= self.settings.max_batch_events:
            raise ValueError('Invalid batch size')
        encoded = []
        for value in events:
            event = Event.model_validate(value)
            q = self.encode_kc(event.kc_vector)
            encoded.append((q.attrs, q.pid, event.response, event.event_id, canonical(event.model_dump())))
        with self.lock:
            cache, results = self.store.observe_batch(student_id, encoded, expected_revision, attempt_id)
            return dict(state=self._state(student_id, cache), observations=results, atomic=True)

    def fork_student(self, student_id, *, principal='python'):
        identifier(student_id)
        identifier(principal)
        with self.lock:
            return self.branches.create(student_id, self.store.recover(student_id), principal).info()

    def branch_info(self, branch_id, *, principal='python'):
        with self.lock:
            return self.branches.get(branch_id, principal).info()

    def simulate_step(self, branch_id, kc_vector, response, *, principal='python'):
        step = Step(kc_vector=kc_vector, response=response)
        q = self.encode_kc(step.kc_vector)
        with self.lock:
            branch = self.branches.get(branch_id, principal)
            if branch.simulated_steps >= self.settings.max_rollout_steps:
                raise ServiceError('step_limit', 'Maximum simulated steps reached', 429)
            self._check_projected_memory(branch.cache, 1, replacing=branch_id)
            cache, p = self.model.observe(branch.cache, q.attrs, q.pid, response,
                                          'simulation:' + secrets.token_hex(16), **self.versions)
            self.branches.update(branch, cache)
            return dict(branch=branch.info(), probability=float(p), hypothetical_response=response)

    def predict_branch(self, branch_id, candidates, *, principal='python'):
        encoded = self._candidates(candidates)
        with self.lock:
            branch = self.branches.get(branch_id, principal)
            return dict(branch=branch.info(), predictions=self._predict(branch.cache, encoded))

    def discard_branch(self, branch_id, *, principal='python'):
        with self.lock:
            self.branches.discard(branch_id, principal)

    def _check_projected_memory(self, cache, steps, replacing=None):
        length = cache.next_position + steps
        if cache.seq_len is not None:
            length = min(length, cache.seq_len)
        per_position = sum(2 * layer.key.shape[1] * layer.key.shape[3] * layer.key.element_size() + 8
                           for name in ('response_layers', 'question_layers', 'retrieval_layers')
                           for layer in getattr(cache, name))
        self.branches.expire()
        used = sum(cache_bytes(b.cache) for k, b in self.branches._branches.items() if k != replacing)
        # Reserve old/new cache plus a base clone during a step (conservative bound).
        if used + 3 * length * per_position > self.settings.max_branch_bytes:
            raise ServiceError('branch_memory_limit', 'Projected simulation memory exceeds budget', 429)

    def run_rollout(self, student_id, sequence, mode='sample', seed=None, num_trajectories=1,
                    forecast_candidates=None, retain_branch=False, include_steps=True, *, principal='python'):
        identifier(student_id)
        identifier(principal)
        request = RolloutRequest(student_id=student_id, kc_feature_names=self.encoder.field_names,
                                 sequence=sequence, mode=mode, seed=seed, num_trajectories=num_trajectories,
                                 forecast_candidates=forecast_candidates or [], retain_branch=retain_branch,
                                 include_steps=include_steps)
        count = len(request.sequence)
        if count > self.settings.max_rollout_steps or num_trajectories > self.settings.max_trajectories:
            raise ValueError('Rollout steps or trajectories exceed configured limits')
        if count * num_trajectories > self.settings.max_rollout_work:
            raise ValueError('Total rollout work exceeds configured limit')
        if include_steps and count * num_trajectories > self.settings.max_output_steps:
            raise ValueError('Step output exceeds limit; set include_steps=false')
        if retain_branch and num_trajectories != 1:
            raise ValueError('retain_branch requires exactly one trajectory')
        if mode == 'forced' and any(s.response is None for s in request.sequence):
            raise ValueError('forced mode requires a response for every event')
        if mode == 'sample' and any(s.response is not None for s in request.sequence):
            raise ValueError('sample mode does not accept supplied responses')
        encoded = [(self.encode_kc(s.kc_vector), s.response) for s in request.sequence]
        forecast = self._candidates(request.forecast_candidates, allow_empty=True)
        if len(forecast) * num_trajectories > self.settings.max_output_steps:
            raise ValueError('Forecast output exceeds configured limit')
        actual_seed = secrets.randbits(63) if seed is None else seed
        with self.lock:
            base = self.store.recover(student_id)
            self._check_projected_memory(base, count)
            self.branches.expire()
            if retain_branch and len(self.branches._branches) >= self.settings.max_branches:
                raise ServiceError('branch_limit', 'Maximum branch count reached', 429)
            trajectories = []
            simulation_id = secrets.token_hex(16)
            for index in range(num_trajectories):
                # A private RNG per trajectory; never touch global Python/torch RNG.
                rng = random.Random(f'{actual_seed}:{index}')
                cache = base.to(next(self.model.parameters()).device)
                steps, correct = [], 0
                for position, (q, supplied_response) in enumerate(encoded):
                    p = float(self.model.predict_next(cache, q.attrs, q.pid, **self.versions))
                    response = supplied_response if mode == 'forced' else int(rng.random() < p)
                    cache, _ = self.model.observe(cache, q.attrs, q.pid, response,
                        f'rollout:{simulation_id}:{index}:{position}', **self.versions)
                    correct += response
                    if include_steps:
                        steps.append(dict(probability=p, hypothetical_response=response))
                trajectories.append(dict(trajectory_index=index, correct_count=correct,
                    correct_rate=correct/count, final_position=cache.next_position,
                    steps=steps if include_steps else None, forecast_predictions=self._predict(cache, forecast)))
            branch = self.branches.create(student_id, cache, principal, base.history_revision, count).info() if retain_branch else None
            return dict(state=self._state(student_id, base), base_revision=base.history_revision,
                        seed=actual_seed, mode=mode, trajectories=trajectories, branch=branch,
                        correct_count=statistics([t['correct_count'] for t in trajectories]),
                        correct_rate=statistics([t['correct_rate'] for t in trajectories]),
                        forecast_statistics={cid: statistics([t['forecast_predictions'][i]['probability']
                                             for t in trajectories]) for i, (cid, _) in enumerate(forecast)})

    def rebuild_student(self, student_id):
        """Trusted internal/admin operation; same model and encoding only."""
        with self.lock:
            return self._state(identifier(student_id), self.store.rebuild(student_id))
