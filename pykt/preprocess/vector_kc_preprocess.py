"""Categorical-vector adapter for flat attempts and grouped PianoKT exports."""
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .student_sequences import prepare_student_sequences


def category(value):
    """Stable typed tokens survive JSON round trips; null stays MISSING."""
    if value is None or value == '':
        return None
    if not isinstance(value, (str, bool, int, float)) or (
            isinstance(value, float) and not math.isfinite(value)):
        raise ValueError('KC fields must be finite JSON scalars or null')
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _attempts(records):
    if isinstance(records, dict) and 'students' in records:
        flattened = []
        for student in records['students']:
            uid = student['learner_key']
            for visit in student['visits']:
                for attempt in visit['attempts']:
                    # Parse offsets rather than comparing timestamp strings.
                    timestamp = attempt.get('ordering_timestamp') or attempt.get('started_at')
                    if not isinstance(timestamp, str) or not timestamp:
                        raise ValueError('Each attempt requires an ordering_timestamp')
                    try:
                        time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except (TypeError, ValueError) as exc:
                        raise ValueError('Invalid attempt ordering_timestamp') from exc
                    if time.tzinfo is None:
                        raise ValueError('Attempt ordering_timestamp must include timezone')
                    flattened.append((str(uid), time, attempt))
        return [(uid, record) for uid, _, record in sorted(
            flattened, key=lambda item: (item[0], item[1]))]
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty attempt object or list of attempts')
    records.sort(key=lambda r: (str(r['user_id']), r['days_since_signup'],
                                r.get('session_index', 0), r.get('exercise_part_index', 0)))
    return [(str(record['user_id']), record) for record in records]


def read_students(path):
    records = _attempts(json.loads(Path(path).read_text()))
    students = defaultdict(list)
    names = None
    for uid, record in records:
        events = record['events_data']
        if isinstance(events, str):
            events = json.loads(events)
        vectors = events['kc_vectors']
        if not isinstance(vectors, list):
            raise ValueError('kc_vectors must be a list')
        if not vectors:
            continue
        if not isinstance(vectors[0], list) or not vectors[0]:
            raise ValueError('KC vector width must be positive')
        # Positions define fields; descriptive names are optional metadata.
        fields = events.get('kc_feature_names', [f'kc_{i}' for i in range(len(vectors[0]))])
        if not isinstance(fields, list) or not fields or any(
                not isinstance(f, str) or not f for f in fields) or len(set(fields)) != len(fields):
            raise ValueError('kc_feature_names must be unique nonempty strings')
        if names is None:
            names = fields
        if fields != names:
            raise ValueError('KC field names and order must match across attempts')
        if 'correct' in events or 'responses' in events:
            responses = events.get('correct', events.get('responses'))
            if any(type(r) is not int or r not in (0, 1) for r in responses):
                raise ValueError('responses must contain binary integers')
            if 'correct' in events and 'responses' in events and events['responses'] != responses:
                raise ValueError('correct and responses labels disagree')
        if 'reject_reason' in events:
            reasons = events['reject_reason']
            reasons = reasons['event'] if isinstance(reasons, dict) else reasons
            if any(type(r) is not int or r < 0 for r in reasons):
                raise ValueError('reject_reason must contain nonnegative integer codes')
            derived = [int(r == 0) for r in reasons]
            if ('correct' in events or 'responses' in events) and responses != derived:
                raise ValueError('Response labels disagree with reject_reason.event')
            responses = derived
        elif 'correct' not in events and 'responses' not in events:
            raise ValueError('Expected correct, responses, or reject_reason labels')
        if len(vectors) != len(responses):
            raise ValueError('KC vectors and responses must have equal lengths')
        onsets = events.get('onset_seconds', list(range(len(vectors))))
        if len(onsets) != len(vectors) or any(
                not isinstance(t, (int, float)) or not math.isfinite(t) for t in onsets):
            raise ValueError('onset_seconds must match events and contain finite numbers')
        for i in sorted(range(len(vectors)), key=lambda i: (onsets[i], i)):
            vector = vectors[i]
            if not isinstance(vector, list) or len(vector) != len(names):
                raise ValueError('KC vector width must match across events and attempts')
            attrs = [category(value) for value in vector]
            # Identity depends only on the KC tuple, never score/song/event location.
            identity = json.dumps(attrs, ensure_ascii=False, separators=(',', ':'))
            students[uid].append((attrs, identity, responses[i]))
    if not students:
        raise ValueError('No vector KC events found')
    return dict(students), names


def prepare_dataset(read_file, output_dir, config_file, dataset_name='pianokt',
                    kfold=5, seed=1024):
    students, names = read_students(read_file)
    entry = prepare_student_sequences(students, names, output_dir, kfold, seed)
    examples = Path(__file__).resolve().parents[2] / 'examples'
    entry.update(dpath=os.path.relpath(Path(output_dir).resolve(), examples),
                 dataset_name=dataset_name, question_identity='kc_vector_json_v1',
                 category_encoding='typed_json_scalar_v1')
    path = Path(config_file)
    config = json.loads(path.read_text()) if path.exists() else {}
    config[dataset_name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=4) + '\n')
    return entry
