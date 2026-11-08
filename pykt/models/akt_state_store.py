"""SQLite event log + disposable snapshots for single-process inference services.

Use one store per database within a process. Events are committed before inference;
failed cache updates are recovered by replay. SQLite transactions protect snapshots
with revision CAS across connections. Historical corrections require rebuilding the
log from an authoritative source into a new store.
"""
import io
import json
import sqlite3
import threading
import torch
from .akt_cache import StudentCache


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
        ''')

    def close(self):
        self.db.close()

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
