"""Reusable student-disjoint fold preparation for categorical field sequences."""
import json
import random
from pathlib import Path


def fit_encoding(students, train_ids, field_names):
    # 0 PAD, 1 UNK, 2 MISSING. "inf" is a known exercise-start token.
    fields = [dict() for _ in field_names]
    questions = {}
    for uid in train_ids:
        for attrs, pid, _ in students[uid]:
            for mapping, value in zip(fields, attrs):
                if value is not None and value != "" and value not in mapping:
                    mapping[value] = len(mapping) + 3
            if pid not in questions:
                questions[pid] = len(questions) + 1
    return {"field_names": list(field_names), "fields": fields, "questions": questions}



def prepare_student_sequences(students, field_names, output_dir, kfold=5, seed=1024,
                              test_fraction=0.2):
    """Input: uid -> ordered (field_values, question_identity, response) events.

    Persist canonical events and a vocabulary fitted on each fold's training
    students. Validation/test identities never contribute to that vocabulary.
    """
    ids = sorted(students)
    lengths = [len(events) for events in students.values()]
    event_count = sum(lengths)
    correct = sum(response for events in students.values() for _, _, response in events)
    print(f'Loaded students: {len(ids)}, interactions: {event_count}, KC fields: {len(field_names)}',
          flush=True)
    if event_count:
        print(f'Sequence length min/mean/max: {min(lengths)}/{event_count / len(ids):.2f}/{max(lengths)}; '
              f'correct: {correct}, incorrect: {event_count - correct}, '
              f'correct rate: {correct / event_count:.2%}', flush=True)
    if not 0 < test_fraction < 1 or kfold < 2:
        raise ValueError("Require 0 < test_fraction < 1 and kfold >= 2")
    random.Random(seed).shuffle(ids)
    n_test = max(1, int(len(ids) * test_fraction))
    remaining, test = ids[n_test:], ids[:n_test]
    print(f'Student-disjoint split: seed={seed}, kfold={kfold}, '
          f'test_fraction={test_fraction}, test students={len(test)}, '
          f'train/validation pool={len(remaining)}', flush=True)
    if len(remaining) < kfold:
        raise ValueError(f"Not enough students for nonempty validation folds: "
                         f"{len(ids)} total, {len(test)} test, {len(remaining)} remaining, "
                         f"but kfold={kfold}. Add students or explicitly reduce kfold (>=2).")
    for uid, events in students.items():
        if not events or any(len(attrs) != len(field_names) or response not in (0, 1)
                             for attrs, _, response in events):
            raise ValueError(f"Invalid events for student {uid}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "student_sequences.json").write_text(json.dumps(students, ensure_ascii=False))
    print(f'Saved sequences: {(output / "student_sequences.json").resolve()}', flush=True)
    folds = [remaining[i::kfold] for i in range(kfold)]
    for fold in range(kfold):
        valid = folds[fold]
        train = [uid for uid in remaining if uid not in set(valid)]
        encoding = fit_encoding(students, train, field_names)
        artifact = {"format_version": 1, "fold": fold, "seed": seed,
                    "encoding": encoding,
                    "splits": {"train": train, "valid": valid, "test": test}}
        (output / f"student_fold_{fold}.json").write_text(json.dumps(artifact, ensure_ascii=False))
        split_counts = ', '.join(
            f'{key}={len(group)} students/{sum(len(students[uid]) for uid in group)} events'
            for key, group in artifact['splits'].items())
        print(f'Fold {fold}: {split_counts}', flush=True)
        print(f'Fold {fold} training vocab: field sizes (PAD/UNK/MISSING included)='
              f'{[len(mapping) + 3 for mapping in encoding["fields"]]}, '
              f'question identities={len(encoding["questions"])}', flush=True)
        print(f'Saved fold: {(output / f"student_fold_{fold}.json").resolve()}', flush=True)
    return {"dataset_format": "student_fields_v1", "student_file": "student_sequences.json",
            "fold_file": "student_fold_{fold}.json", "folds": list(range(kfold)),
            "field_names": list(field_names), "split_seed": seed,
            "input_type": ["questions", "attributes"], "emb_path": ""}
