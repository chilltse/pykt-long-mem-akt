"""SQLite event log + disposable snapshots for single-process inference services.

Use one store per database within a process. Legacy observe commits before inference;
failed cache updates are recovered by replay. observe_batch atomically commits the
log and computed final snapshot. SQLite transactions protect snapshots
with revision CAS across connections. Historical corrections require rebuilding the
log from an authoritative source into a new store.
"""
import io
import json
import sqlite3
import threading
import torch
from .akt_cache import StudentCache


class StateConflict(ValueError):
    """A revision, event payload, or ordering conflict (safe to report as HTTP 409)."""


class StudentStateStore:
    def __init__(self, path, model, model_version, feature_version):
        self.model = model
        self.model_version = model_version
        self.feature_version = feature_version
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events (
                student TEXT NOT NULL, cursor INTEGER NOT NULL, event_id TEXT NOT NULL,
                payload TEXT NOT NULL, probability REAL,
                PRIMARY KEY(student, cursor), UNIQUE(student, event_id));
            CREATE TABLE IF NOT EXISTS snapshots (
                student TEXT PRIMARY KEY, revision INTEGER NOT NULL, state BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS event_requests (
                student TEXT NOT NULL, event_id TEXT NOT NULL, identity TEXT NOT NULL,
                PRIMARY KEY(student, event_id));
            CREATE TABLE IF NOT EXISTS attempts (
                student TEXT NOT NULL, attempt_id TEXT NOT NULL, identity TEXT NOT NULL,
                PRIMARY KEY(student, attempt_id));
        ''')

    def close(self):
        self.db.close()

    def bind_versions(self):
        """Opt-in serving guard: never reuse a database with a different encoding/model.

        Legacy stores with data need an explicit migration rather than assuming the
        encoded event log belongs to this checkpoint. A new model uses a new DB and
        a replay of authoritative raw events.
        """
        with self.lock, self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS serving_versions (id INTEGER PRIMARY KEY, model TEXT, feature TEXT)')
            row = self.db.execute('SELECT model,feature FROM serving_versions WHERE id=1').fetchone()
            versions = (self.model_version, self.feature_version)
            if row is not None and row != versions:
                raise ValueError('Database version mismatch; use a new database and replay raw events')
            if row is None:
                if self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]:
                    raise ValueError('Unversioned legacy database; explicitly migrate raw events to a new database')
                self.db.execute('INSERT INTO serving_versions VALUES (1,?,?)', versions)

    def _load(self, student):
        row = self.db.execute('SELECT revision,state FROM snapshots WHERE student=?', (student,)).fetchone()
        if row is None:
            cache = self.model.empty_cache(self.model_version, self.feature_version)
            self.model.validate_cache(cache, self.model_version, self.feature_version)
            return cache
        state = torch.load(io.BytesIO(row[1]), map_location='cpu', weights_only=True)
        cache = StudentCache.from_state_dict(state, next(self.model.parameters()).device)
        self.model.validate_cache(cache, self.model_version, self.feature_version)
        if row[0] != cache.history_revision:
            raise ValueError('Snapshot revision mismatch')
        return cache

    def _recover(self, student):
        cache = self._load(student)
        rows = self.db.execute('SELECT cursor,event_id,payload FROM events WHERE student=? AND cursor>=? ORDER BY cursor',
                               (student, cache.next_position)).fetchall()
        for cursor, event_id, payload in rows:
            if cursor != cache.next_position:
                raise ValueError('Noncontiguous event history')
            attrs, pid, response = json.loads(payload)
            updated, probability = self.model.observe(cache, attrs, pid, response, event_id,
                model_version=self.model_version, feature_version=self.feature_version)
            buffer = io.BytesIO()
            torch.save(updated.state_dict(), buffer)
            with self.db:
                if cache.history_revision == 0:
                    changed = self.db.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?,?)',
                        (student, updated.history_revision, buffer.getvalue())).rowcount
                else:
                    changed = self.db.execute('UPDATE snapshots SET revision=?,state=? WHERE student=? AND revision=?',
                        (updated.history_revision, buffer.getvalue(), student, cache.history_revision)).rowcount
                if changed != 1:
                    raise RuntimeError('Concurrent student update; retry from latest snapshot')
                self.db.execute('UPDATE events SET probability=? WHERE student=? AND cursor=?',
                                (float(probability), student, cursor))
            cache = updated
        return cache

    def recover(self, student):
        with self.lock:
            return self._recover(str(student))

    def predict_next(self, student, attrs, pid):
        with self.lock:
            cache = self._recover(str(student))
            return self.model.predict_next(cache, attrs, pid, model_version=self.model_version,
                                           feature_version=self.feature_version)

    def observe(self, student, attrs, pid, response, event_id):
        student = str(student)
        with self.lock:
            # Validate before committing malformed events to the durable log.
            self.model._event_question(attrs, pid)
            if torch.as_tensor(response).ndim != 0 or response not in (0, 1):
                raise ValueError('Response must be scalar 0 or 1')
            if not isinstance(event_id, str) or not event_id:
                raise ValueError('event_id must be a nonempty string')
            payload = json.dumps([torch.as_tensor(attrs).tolist(), int(pid), int(response)])
            self._recover(student)  # Reject stale model/feature versions before logging.
            with self.db:
                self.db.execute('BEGIN IMMEDIATE')
                row = self.db.execute('SELECT payload FROM events WHERE student=? AND event_id=?',
                                      (student, event_id)).fetchone()
                if row is not None:
                    if row[0] != payload:
                        raise ValueError('event_id already exists with a different payload')
                else:
                    cursor = self.db.execute('SELECT COUNT(*) FROM events WHERE student=?', (student,)).fetchone()[0]
                    self.db.execute('INSERT INTO events VALUES (?,?,?,?,NULL)', (student, cursor, event_id, payload))
            cache = self._recover(student)
            probability = self.db.execute('SELECT probability FROM events WHERE student=? AND event_id=?',
                                          (student, event_id)).fetchone()[0]
            return cache, probability

    def rebuild(self, student):
        """Explicitly discard a snapshot and replay the entire durable event log.

        Also used after changing model/feature versions (IDs must remain compatible;
        vocabulary remapping requires a new authoritative log).
        """
        with self.lock:
            with self.db:
                self.db.execute('DELETE FROM snapshots WHERE student=?', (str(student),))
                self.db.execute('UPDATE events SET probability=NULL WHERE student=?', (str(student),))
            return self._recover(str(student))

    @torch.no_grad()
    def observe_batch(self, student, events, expected_revision=None, attempt_id=None):
        """Atomically commit ordered events, raw request identities and the final snapshot.

        Each event is (attrs, pid, response, event_id, raw_identity). Inference runs
        inside the transaction, so a failure rolls back the whole new batch. Recovery
        of an older interrupted single-event write can happen before this transaction.
        Full retries return stored probabilities, even with the original revision.
        """
        student = str(student)
        if not events or len({e[3] for e in events}) != len(events):
            raise ValueError('Batch must be nonempty with unique event IDs')
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
            raise ValueError('expected_revision must be a nonnegative integer')
        if attempt_id is not None and (not isinstance(attempt_id, str) or not attempt_id):
            raise ValueError('attempt_id must be nonempty')
        prepared = []
        for attrs, pid, response, event_id, identity in events:
            self.model._event_question(attrs, pid)
            if type(response) is not int or response not in (0, 1):
                raise ValueError('Response must be integer 0 or 1')
            if not isinstance(event_id, str) or not event_id:
                raise ValueError('event_id must be nonempty')
            payload = json.dumps([list(attrs), int(pid), response])
            prepared.append((event_id, payload, identity))
        attempt_identity = json.dumps(prepared, ensure_ascii=False)
        with self.lock:
            self._recover(student)
            with self.db:
                self.db.execute('BEGIN IMMEDIATE')
                cache = self._load(student)
                if attempt_id is not None:
                    row = self.db.execute('SELECT identity FROM attempts WHERE student=? AND attempt_id=?',
                                          (student, attempt_id)).fetchone()
                    if row and row[0] != attempt_identity:
                        raise StateConflict('attempt_id already exists with a different payload or order')
                existing = []
                for event_id, payload, identity in prepared:
                    row = self.db.execute('SELECT cursor,payload,probability FROM events WHERE student=? AND event_id=?',
                                          (student, event_id)).fetchone()
                    raw = self.db.execute('SELECT identity FROM event_requests WHERE student=? AND event_id=?',
                                          (student, event_id)).fetchone()
                    if row and (row[1] != payload or (raw and raw[0] != identity)):
                        raise StateConflict('event_id already exists with a different payload')
                    existing.append(row)
                cursors = [row[0] for row in existing if row]
                if cursors and cursors != list(range(cursors[0], cursors[0] + len(cursors))):
                    raise StateConflict('Retry events must retain their original contiguous order')
                new_count = sum(row is None for row in existing)
                if new_count:
                    if expected_revision is not None and expected_revision != cache.history_revision:
                        raise StateConflict('expected_revision does not match current history_revision')
                    prefix = len(existing) - new_count
                    if any(row is not None for row in existing[prefix:]):
                        raise StateConflict('Existing events must form a prefix of the batch')
                    if cursors and cursors[-1] != cache.next_position - 1:
                        raise StateConflict('Retry prefix must end at the current history position')
                results = []
                for event, prepared_event, row in zip(events, prepared, existing):
                    attrs, pid, response, event_id, identity = event
                    if row:
                        probability = row[2]
                    else:
                        cursor = cache.next_position
                        cache, probability = self.model.observe(cache, attrs, pid, response, event_id,
                            model_version=self.model_version, feature_version=self.feature_version)
                        probability = float(probability)
                        self.db.execute('INSERT INTO events VALUES (?,?,?,?,?)',
                                        (student, cursor, event_id, prepared_event[1], probability))
                        self.db.execute('INSERT INTO event_requests VALUES (?,?,?)',
                                        (student, event_id, identity))
                    results.append(dict(event_id=event_id, probability=probability, duplicate=row is not None))
                if new_count:
                    buffer = io.BytesIO()
                    torch.save(cache.state_dict(), buffer)
                    self.db.execute('INSERT OR REPLACE INTO snapshots VALUES (?,?,?)',
                                    (student, cache.history_revision, buffer.getvalue()))
                if attempt_id is not None:
                    self.db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,?)',
                                    (student, attempt_id, attempt_identity))
            return cache, results
