"""Raw Yousician adapter; used only by data preprocessing."""
import json
from collections import defaultdict

from .yousician_preprocess_kc_fmkc_standard import _events_to_event_rows

FIELD_NAMES = ("prev_frets", "prev_strings", "frets", "strings")


def read_students(path, audit=None):
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    if audit is not None:
        audit.update(raw_record_count=len(records), raw_student_count=len({str(r.get("user_id")) for r in records}),
                     non_play_record_count=sum(r.get("play_mode") != "play" for r in records))
    records = [r for r in records if r.get("play_mode") == "play"]
    if audit is not None:
        from collections import Counter
        keys = Counter((str(r["user_id"]), r["days_since_signup"], r.get("session_index", 0),
                        r.get("exercise_part_index", 0)) for r in records)
        audit.update(filtered_play_record_count=len(records), sorting_tie_groups=sum(v > 1 for v in keys.values()),
                     sorting_tie_extra_records=sum(v-1 for v in keys.values()))
    records.sort(key=lambda r: (str(r["user_id"]), r["days_since_signup"],
                                r.get("session_index", 0), r.get("exercise_part_index", 0)))
    students = defaultdict(list)
    skipped = 0
    for record in records:
        try:
            events = _events_to_event_rows(record["events_data"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        for _, kc, response in events:
            # Canonical local fields define identity, independently of score location.
            students[str(record["user_id"])].append((kc.split("|"), kc, response))
    if not students:
        raise ValueError("No valid play events")
    if audit is not None:
        audit["skipped_reasons"] = {"invalid_events_key_type_or_value": skipped}
    return dict(students), skipped



def prepare_dataset(read_file, output_dir, config_file, dataset_name="yousician",
                    kfold=5, seed=1024):
    from pathlib import Path
    import os
    from .student_sequences import prepare_student_sequences

    students, skipped = read_students(read_file)
    entry = prepare_student_sequences(students, FIELD_NAMES, output_dir, kfold, seed)
    # Match pyKT dpath convention: relative to examples/, not the raw input path.
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    entry.update(dpath=os.path.relpath(Path(output_dir).resolve(), examples_dir),
                 dataset_name=dataset_name, skipped_records=skipped,
                 question_identity="kc_vector_v1")
    path = Path(config_file)
    config = json.loads(path.read_text()) if path.exists() else {}
    config[dataset_name] = dict(config.get(dataset_name, {}), **entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=4) + "\n")
    return entry
