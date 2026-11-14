"""Immutable, training-only benchmark preparation; no model imports."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from pykt.preprocess.student_sequences import fit_encoding
from pykt.preprocess.yousician_field_preprocess import FIELD_NAMES, read_students


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare(raw_file, run_dir, sizes=(60, 120, 240, 'full'), folds=(0,), seeds=(3407,),
            split_seed=1024, subset_seed=1024, kfold=5, test_fraction=.2):
    if kfold < 2 or not 0 < test_fraction < 1:
        raise ValueError('Invalid split configuration')
    if not folds or any(f < 0 or f >= kfold for f in folds):
        raise ValueError('Invalid folds')
    if len(set(map(str, sizes))) != len(sizes):
        raise ValueError('Duplicate sizes')
    raw_file, root = Path(raw_file).resolve(), Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite {root}')
    raw_audit = {}
    students, skipped = read_students(raw_file, audit=raw_audit)
    ids = sorted(students)
    random.Random(split_seed).shuffle(ids)
    ntest = max(1, int(len(ids) * test_fraction))
    test, remaining = ids[:ntest], ids[ntest:]
    if len(remaining) < kfold:
        raise ValueError('Insufficient students for nonempty folds')
    plans = []
    for fold in folds:
        valid = remaining[fold::kfold]
        pool = [uid for uid in remaining if uid not in set(valid)]
        random.Random(subset_seed).shuffle(pool)
        for size in sizes:
            count = len(pool) if str(size) == 'full' else int(size)
            if not 0 < count <= len(pool):
                raise ValueError(f'Training size {size} unavailable: pool={len(pool)}')
            plans.append((fold, str(size), pool[:count], valid))
    root.mkdir(parents=True)
    # One canonical copy, individually addressable without parsing other students.
    index = {}
    for number, (uid, events) in enumerate(students.items()):
        filename = f'students/{number}.json'
        write_json(root / 'prepared' / filename, events)
        index[uid] = filename
    write_json(root / 'prepared/student_index.json', index)
    lengths = sorted(map(len, students.values()))
    frequencies = Counter(kc for events in students.values() for _, kc, _ in events)
    correct = sum(y for events in students.values() for _, _, y in events)
    digest = hashlib.sha256()
    with raw_file.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    import numpy as np
    field_frequency = {name: Counter(str(attrs[i]) for events in students.values() for attrs, _, _ in events)
                       for i, name in enumerate(FIELD_NAMES)}
    write_json(root / 'field_frequency.json', field_frequency)
    write_json(root / 'dataset_audit.json', dict(**raw_audit,
        sequence_length_median=float(np.median(lengths)),
        sequence_length_quantiles={str(q): float(np.quantile(lengths,q)) for q in (0,.25,.5,.75,.9,.95,.99,1)},
        unique_values_per_field={k:len(v) for k,v in field_frequency.items()},
        student_correct_rates={u:sum(y for _,_,y in e)/len(e) for u,e in students.items()},raw_file=str(raw_file), raw_file_sha256=digest.hexdigest(),
        valid_student_count=len(students), valid_event_count=sum(lengths), skipped_record_count=skipped,
        correct_count=correct, incorrect_count=sum(lengths)-correct, correct_rate=correct/sum(lengths),
        unique_kc_count=len(frequencies), field_names=FIELD_NAMES,
        sequence_length_min=min(lengths), sequence_length_max=max(lengths),
        sequence_length_mean=sum(lengths)/len(lengths),
        student_event_counts={u: len(e) for u,e in students.items()},
        ordering='stable sort: str(user_id), days_since_signup, session_index, exercise_part_index; ties retain raw order'))
    write_json(root / 'kc_frequency.json', frequencies)
    tasks = []
    for fold, size, train, valid in plans:
        encoding = fit_encoding(students, train, FIELD_NAMES)
        splits = dict(train=train, valid=valid, test=test)
        groups = list(map(set, splits.values()))
        assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
        directory = root / 'prepared' / f'fold_{fold}' / f'train_{size}'
        freq = Counter(kc for uid in train for _, kc, _ in students[uid])
        artifact = dict(format_version=1, fold=fold, seed=split_seed, encoding=encoding, splits=splits)
        write_json(directory / f'student_fold_{fold}.json', artifact)
        write_json(directory / 'train_kc_frequency.json', freq)
        write_json(directory / 'encoding.json', encoding)
        write_json(directory / 'train_field_frequency.json', {name:Counter(str(attrs[i]) for uid in train for attrs,_,_ in students[uid]) for i,name in enumerate(FIELD_NAMES)})
        write_json(directory / 'scale_statistics.json', dict(train_students=len(train),valid_students=len(valid),test_students=len(test),
            train_events=sum(len(students[u]) for u in train),valid_events=sum(len(students[u]) for u in valid),test_events=sum(len(students[u]) for u in test),
            question_vocabulary_size=len(encoding['questions']),field_vocabulary_sizes=[len(v)+3 for v in encoding['fields']]))
        write_json(directory / 'split_manifest.json', dict(splits=splits,
            subset_rule='random.Random(subset_seed).shuffle(train_pool)', subset_seed=subset_seed,
            encoding_fingerprint=fingerprint(encoding), frequency_fingerprint=fingerprint(freq),
            event_fingerprints={s: fingerprint([(u, students[u]) for u in us]) for s, us in splits.items()}))
        for split, uids in splits.items():
            write_json(directory / f'{split}_student_ids.json', uids)
        config = dict(dataset_format='student_fields_v1', dpath=str(directory),
            student_index_file=str(root / 'prepared/student_index.json'),
            fold_file='student_fold_{fold}.json', folds=[fold], field_names=FIELD_NAMES,
            input_type=['questions', 'attributes'], emb_path='')
        write_json(directory / 'data_config.json', {'yousician': config})
        for seed in seeds:
            for model in ('akt', 'akt_fmkc'):
                tasks.append(dict(task_id=len(tasks), model_name=model, train_student_count=size,
                    actual_train_student_count=len(train), fold=fold, seed=seed,
                    prepared_config_path=str(directory / 'data_config.json'),
                    output_dir=str(root / f'fold_{fold}/seed_{seed}/train_{size}/{model}')))
    write_json(root / 'task_manifest.json', tasks)
    write_json(root / 'experiment_manifest.json', dict(schema_version=1, dataset_sha256=digest.hexdigest(),
        task_fingerprint=fingerprint(tasks), split_seed=split_seed, subset_seed=subset_seed,
        kfold=kfold, test_fraction=test_fraction, scale_mode='fixed_holdout'))
    return tasks
