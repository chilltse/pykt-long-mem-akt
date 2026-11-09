#!/usr/bin/env python3
"""Replay PianoKT onset-level practice records through the real AKT-FMKC HTTP API.

Supports both PianoKT groupings:
  v1: students[].attempts[].events[] or events_data{}
  v2: students[].visits[].attempts[].events_data{}
The input may be JSON or JSON.gz.

Usage:
  pip install httpx
  python test_two_students_api.py --data data/pianokt/two_students.json \
      --base-url http://127.0.0.1:8000 --output-dir results/two_students

For local loopback development, configure KT_DEV_AUTH=true on the SERVER.
Otherwise set KT_API_TOKEN in the CLIENT shell. ALWAYS use a separate test
KT_STATE_DB_PATH; this program intentionally writes actual observable events.
"""
import argparse
import csv
import gzip
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import httpx

V11_NAMES = [
    "has_black_key", "simultaneous_note_count", "both_hands",
    "onset_interval_bin_cs", "change_count", "mean_move_bin",
]


def read_data(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("students"), list):
        raise ValueError("Expected root object with students[]")
    if len(data["students"]) != 2:
        raise ValueError(f"This test expects exactly TWO students; got {len(data['students'])}")
    if len(set(str(s.get("learner_key")) for s in data["students"])) != 2:
        raise ValueError("Students must have distinct learner_key values")
    return data


def parse_time(value):
    if not value:
        return datetime.max.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def extract_events(attempt):
    """Support both grouped format and packed events_data format."""
    if isinstance(attempt.get("events"), list):
        events = attempt["events"]
    elif isinstance(attempt.get("events_data"), dict):
        ed = attempt["events_data"]
        vectors = ed.get("kc_vectors")
        correct = ed.get("correct")
        if not isinstance(vectors, list) or not isinstance(correct, list):
            raise ValueError("Packed events_data requires kc_vectors[] and correct[]")
        if len(vectors) != len(correct):
            raise ValueError("events_data.kc_vectors and correct length differ")
        indexes = ed.get("event_index")
        if indexes is None:
            indexes = list(range(len(vectors)))
        if not isinstance(indexes, list) or len(indexes) != len(vectors):
            raise ValueError("events_data.event_index has wrong length")
        events = [
            {"event_index": indexes[i], "kc_vector": vector, "correct": correct[i]}
            for i, vector in enumerate(vectors)
        ]
        if isinstance(ed.get("onset_seconds"), list) and len(ed["onset_seconds"]) != len(vectors):
            raise ValueError("events_data.onset_seconds length differs from kc_vectors")
    else:
        raise ValueError("Each attempt must contain events[] or events_data{}")
    events = sorted(events, key=lambda e: int(e["event_index"]))
    ids = [int(e["event_index"]) for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate event_index in attempt {attempt.get('attempt_id')}")
    for e in events:
        if not isinstance(e.get("kc_vector"), list):
            raise ValueError("Every event needs kc_vector[]")
        response = e.get("correct")
        if type(response) not in (int, bool) or response not in (0, 1):
            raise ValueError("Every event needs binary correct (0/1). No label inference is performed")
    return events


def collect_attempts(raw):
    """Normalize v1 students[].attempts and v2 students[].visits[].attempts."""
    direct = raw.get("attempts")
    if direct is not None and not isinstance(direct, list):
        raise ValueError("student.attempts must be an array")
    visits = raw.get("visits")
    if visits is not None and not isinstance(visits, list):
        raise ValueError("student.visits must be an array")

    collected = list(direct or [])
    for visit in visits or []:
        if not isinstance(visit, dict):
            raise ValueError("Each visit must be an object")
        grouped = visit.get("attempts", [])
        if not isinstance(grouped, list):
            raise ValueError("visit.attempts must be an array")
        for attempt in grouped:
            if not isinstance(attempt, dict):
                raise ValueError("Each attempt must be an object")
            copy = dict(attempt)
            # Fallback to the visit timestamp only if missing on the attempt.
            if not copy.get("ordering_timestamp"):
                copy["ordering_timestamp"] = visit.get("ordering_timestamp") or visit.get("started_at")
            collected.append(copy)
    return collected


def prepare_students(data, feature_names, prefix):
    # v1.1 and the currently used v1.2.0 labeler both export six KC features.
    # A matching width alone is not enough: compare names and ordering.
    if (data.get("feature_version") == "v1.1" or
            data.get("kc_labeler_version") == "1.2.0") and feature_names not in (
                V11_NAMES,
                [f"kc_{i}" for i in range(6)],
            ):
        raise ValueError(
            "Training checkpoint field_names differ from PianoKT v1.1. "
            f"Checkpoint: {feature_names}; PianoKT v1.1: {V11_NAMES}. "
            "Do not force vectors into the wrong order!"
        )
    students = []
    for raw in data["students"]:
        key = raw.get("learner_key")
        if not isinstance(key, str) or not key:
            raise ValueError("Each student requires learner_key")
        attempts = sorted(collect_attempts(raw), key=lambda a: (
            parse_time(a.get("ordering_timestamp")), a.get("attempt_id", "")
        ))
        prepared = []
        for attempt in attempts:
            attempt_id = attempt.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                raise ValueError("Missing attempt_id")
            ed = attempt.get("events_data", {})
            if isinstance(ed, dict) and ed.get("kc_feature_names") is not None:
                if ed["kc_feature_names"] != feature_names:
                    raise ValueError(f"attempt {attempt_id}: KC feature names/order mismatch")
            events = extract_events(attempt)
            for e in events:
                if len(e["kc_vector"]) != len(feature_names):
                    raise ValueError(f"attempt {attempt_id}: wrong KC vector width")
            if events:
                prepared.append((attempt_id, events))
        students.append({"student_id": prefix + key, "attempts": prepared})
    if not all(s["attempts"] for s in students):
        for s in students:
            print(f"Parsed student {s['student_id']}: nonempty attempts={len(s['attempts'])}")
        raise ValueError(
            "Both students must have at least one nonempty attempt. "
            "Expected students[].attempts[] (v1) or students[].visits[].attempts[] (v2)."
        )
    return students


def api_call(client, method, endpoint, **kwargs):
    response = client.request(method, endpoint, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(
            f"{method} {endpoint}: HTTP {response.status_code}: "
            f"{response.text[:1400]}"
        )
    return response.json()


def state(client, student):
    return api_call(client, "GET", f"/kt/students/{quote(student, safe='')}/state")


def predict(client, student, names, candidates):
    return api_call(client, "POST", f"/kt/students/{quote(student, safe='')}/predict",
                    json={"kc_feature_names": names, "candidates": candidates})


def summarize(rows):
    if not rows:
        return {"events": 0}
    ys = [int(r["actual_correct"]) for r in rows]
    ps = [float(r["predicted_probability_before_event"]) for r in rows]
    eps = 1e-12
    return {
        "events": len(rows),
        "actual_correct_rate": round(sum(ys) / len(ys), 6),
        "mean_predicted_probability": round(sum(ps) / len(ps), 6),
        "accuracy_at_threshold_0_5": round(sum((p >= .5) == y for p, y in zip(ps, ys)) / len(ys), 6),
        "brier_score": round(sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ys), 6),
        "log_loss": round(-sum(y * math.log(max(eps, min(1 - eps, p))) +
                                   (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
                                   for p, y in zip(ps, ys)) / len(ys), 6),
        "duplicates_on_this_run": sum(str(r["duplicate"]).lower() == "true" for r in rows),
    }


def save_json(path, obj):
    """Write a readable progress report atomically (avoid half-written JSON)."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def show_parsed(students, data):
    print(f"Input schema={data.get('grouping_schema_version', 'v1')}, "
          f"source={data.get('source')}, data_origin={data.get('data_origin')}, "
          f"kc_labeler={data.get('kc_labeler_version', data.get('feature_version'))}", flush=True)
    for s in students:
        total = sum(len(evts) for _, evts in s['attempts'])
        print(f"  {s['student_id']}: {len(s['attempts'])} attempts / {total} events", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Grouped PianoKT JSON or JSON.gz")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="results/two_students")
    parser.add_argument("--student-prefix", default="localtest-")
    parser.add_argument("--verify-retry", action="store_true")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Validate and count input without starting or contacting the API")
    parser.add_argument("--max-attempts-per-student", type=int, default=None,
                        help="Replay only first N complete attempts per student (safe quick smoke test)")
    args = parser.parse_args()
    if args.max_attempts_per_student is not None and args.max_attempts_per_student < 1:
        parser.error("--max-attempts-per-student must be >= 1")

    data = read_data(args.data)
    if args.inspect_only:
        students = prepare_students(data, V11_NAMES, args.student_prefix)
        show_parsed(students, data)
        print("Input parsed OK. No API calls; no model predictions were computed.", flush=True)
        return

    token = os.environ.get("KT_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {results_dir.resolve()}", flush=True)
    print("Connecting to inference API...", flush=True)

    with httpx.Client(base_url=args.base_url, timeout=httpx.Timeout(120, connect=10),
                      headers=headers) as client:
        health = api_call(client, "GET", "/health")
        info = api_call(client, "GET", "/kt/model/info")
        names = info["kc_feature_names"]
        students = prepare_students(data, names, args.student_prefix)
        show_parsed(students, data)
        if args.max_attempts_per_student is not None:
            for s in students:
                s["attempts"] = s["attempts"][:args.max_attempts_per_student]
            print(f"QUICK TEST: first {args.max_attempts_per_student} complete attempts/student only", flush=True)

        max_batch = int(info.get("limits", {}).get("max_batch_events", 512))
        max_candidates = int(info.get("limits", {}).get("max_candidates", 256))
        unique = {}
        for s in students:
            for _, events in s["attempts"]:
                for e in events:
                    signature = json.dumps(e["kc_vector"], ensure_ascii=False, separators=(",", ":"))
                    unique.setdefault(signature, e["kc_vector"])
        candidates = [{"candidate_id": f"probe_{i+1}", "kc_vector": v}
                      for i, v in enumerate(list(unique.values())[:min(5, max_candidates)])]
        before = {}
        for s in students:
            sid = s["student_id"]
            initial = state(client, sid)
            pred = predict(client, sid, names, candidates)
            before[sid] = {"state": initial, "predictions": pred["predictions"]}
            print(f"Before {sid}: revision={initial['history_revision']} "
                  f"probes={[round(p['probability'], 4) for p in pred['predictions']]}", flush=True)
            if initial['history_revision']:
                print("  WARNING: student has existing server-side history. "
                      "A rerun may return duplicate=true; use an isolated SQLite DB for fresh tests.", flush=True)

        report = {"health": health, "model_info": info, "students": {}, "completed": False}
        save_json(results_dir / "progress.json", {
            "status": "started", "input_students": len(students),
            "planned_events": {s["student_id"]: sum(len(events) for _, events in s["attempts"])
                               for s in students}, "students": {}})

        for ix, student in enumerate(students):
            sid = student["student_id"]
            rows = []
            revision = int(before[sid]["state"]["history_revision"])
            first_request = None
            outfile = results_dir / f"student_{ix+1}_event_predictions.csv"
            fields = ["student_id", "attempt_id", "event_index", "event_id", "kc_vector",
                      "actual_correct", "predicted_probability_before_event", "duplicate"]
            # Create the CSV now and stream each successful attempt immediately.
            with outfile.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                f.flush()
                n_attempts = len(student["attempts"])
                for k, (attempt_id, events) in enumerate(student["attempts"], 1):
                    for j in range(0, len(events), max_batch):
                        chunk = events[j:j + max_batch]
                        chunk_num = j // max_batch
                        request_id = attempt_id if len(events) <= max_batch else f"{attempt_id}:part:{chunk_num}"
                        payload = {
                            "kc_feature_names": names, "attempt_id": request_id,
                            "expected_revision": revision,
                            "events": [
                                {"event_id": f"{attempt_id}:{int(e['event_index'])}",
                                 "kc_vector": e["kc_vector"], "response": int(e["correct"])}
                                for e in chunk
                            ],
                        }
                        endpoint = f"/kt/students/{quote(sid, safe='')}/observe-batch"
                        print(f"  [{ix+1}/2] attempt {k}/{n_attempts}: "
                              f"POST {len(chunk)} events (revision={revision})...", flush=True)
                        try:
                            result = api_call(client, "POST", endpoint, json=payload)
                        except Exception as ex:
                            print(f"  REQUEST FAILED: {ex}", flush=True)
                            print(f"  Partial CSV kept at {outfile} "
                                  "(last failed batch not included).", flush=True)
                            raise
                        if first_request is None:
                            first_request = (endpoint, payload)
                        observations = result["observations"]
                        if len(observations) != len(chunk):
                            raise RuntimeError(f"Observation count mismatch for {attempt_id}")
                        batch_rows = []
                        for e, ob in zip(chunk, observations):
                            expected_id = f"{attempt_id}:{int(e['event_index'])}"
                            if ob["event_id"] != expected_id:
                                raise RuntimeError("Observation order/IDs do not match request")
                            batch_rows.append({
                                "student_id": sid, "attempt_id": attempt_id,
                                "event_index": int(e["event_index"]), "event_id": expected_id,
                                "kc_vector": json.dumps(e["kc_vector"], ensure_ascii=False),
                                "actual_correct": int(e["correct"]),
                                "predicted_probability_before_event": ob["probability"],
                                "duplicate": bool(ob["duplicate"]),
                            })
                        writer.writerows(batch_rows)
                        f.flush()
                        rows.extend(batch_rows)
                        revision = int(result["state"]["history_revision"])
                    print(f"  [{ix+1}/2] attempt {k}/{n_attempts} OK; "
                          f"written_rows={len(rows)}, revision={revision}", flush=True)
                    save_json(results_dir / "progress.json", {
                        "status": "running", "student_index": ix + 1,
                        "student_id": sid, "completed_attempts_this_student": k,
                        "total_attempts_this_student": n_attempts,
                        "rows_written_this_student": len(rows),
                        "history_revision": revision,
                        "csv_file": str(outfile),
                    })

            if args.verify_retry and first_request:
                endpoint, payload = first_request
                old_rev = revision
                again = api_call(client, "POST", endpoint, json=payload)
                assert all(x["duplicate"] for x in again["observations"]), "Retry was not a duplicate"
                assert again["state"]["history_revision"] == old_rev, "Retry unexpectedly updated state"
                print(f"  {sid}: idempotent retry verified", flush=True)

            after_state = state(client, sid)
            after_pred = predict(client, sid, names, candidates)["predictions"]
            assert after_state["history_revision"] == revision
            report["students"][sid] = {
                "initial_state": before[sid]["state"],
                "final_state": after_state,
                "metrics": summarize(rows),
                "probes": [
                    {"candidate_id": a["candidate_id"], "kc_vector": candidates[i]["kc_vector"],
                     "before": a["probability"], "after": b["probability"],
                     "delta": b["probability"] - a["probability"],
                     "unknown_question": b["unknown_question"],
                     "unknown_fields": b["unknown_fields"]}
                    for i, (a, b) in enumerate(zip(before[sid]["predictions"], after_pred))
                ],
            }
            # Persist report after student A, before processing student B.
            save_json(results_dir / "summary.json", report)
            if ix == 0:
                other_id = students[1]["student_id"]
                other_current = state(client, other_id)
                assert other_current["history_revision"] == before[other_id]["state"]["history_revision"], (
                    "Student 2 state changed while replaying only student 1!"
                )
                print("Student isolation check: student 2 untouched [OK]", flush=True)
            print(f"After {sid}: {json.dumps(report['students'][sid]['metrics'])}", flush=True)

        report["completed"] = True
        save_json(results_dir / "summary.json", report)
        save_json(results_dir / "progress.json", {"status": "completed", "students": len(students),
                                                  "events": sum(v["metrics"]["events"]
                                                                for v in report["students"].values())})
        print(f"\nAll done. Reports: {results_dir.resolve()}", flush=True)
        print("NOTE: scores are prequential predictions from the loaded checkpoint. "
              "The provided dataset has synthetic/approximate labels.", flush=True)


if __name__ == "__main__":
    main()
