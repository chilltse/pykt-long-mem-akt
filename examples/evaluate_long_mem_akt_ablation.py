"""Compare E0–E3 using one trained LongMemAKT and identical test targets."""
import argparse
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import torch

from pykt.models.long_mem_akt_runner import evaluate_long_mem_akt
from pykt.models.long_mem_akt_workflow import (
    load_artifact, role_loader, collect_predictions, calibrate,
    probability_metrics, write_json, json_digest)


EXPERIMENTS = [
    ("E0", "AKT baseline", "base"),
    ("E1", "Full historical residual", "candidate"),
    ("E2", "Uncertainty gate", "inference"),
    ("E3", "Uncertainty + benefit gates", "inference"),
]
SUMMARY_COLUMNS = ["experiment", "description", "count", "auc", "accuracy", "brier", "logloss",
                   "brier_change", "logloss_change", "intervention_rate", "mean_alpha",
                   "mean_probability_change", "wrong_to_right", "right_to_wrong", "seconds", "mc_seconds"]


def compare_students(baseline_path, prediction_path):
    """Check all targets/masks and baseline probabilities before paired reporting."""
    students, ids = {}, []
    identity = ("uid", "block", "position", "target", "scored", "protocol")
    with baseline_path.open() as base_handle, prediction_path.open() as handle:
        pairs = itertools.zip_longest(csv.DictReader(base_handle), csv.DictReader(handle))
        for base, row in pairs:
            if base is None or row is None or any(base[key] != row[key] for key in identity):
                raise ValueError("Experiment targets or scoring masks differ from E0")
            baseline = float(base["probability"])
            if "base_probability" in row and not np.isclose(
                    baseline, float(row["base_probability"]), rtol=0, atol=1e-7):
                raise ValueError("AKT baseline predictions changed between experiments")
            if row["scored"] != "True":
                continue
            ids.append([row["uid"], int(row["position"])])
            values = students.setdefault(row["uid"], {"y": [], "p": [], "base": [], "alpha": []})
            for key, value in (("y", float(row["target"])), ("p", float(row["probability"])),
                               ("base", baseline), ("alpha", float(row.get("alpha", 0)))):
                values[key].append(value)
    results = []
    for uid, values in students.items():
        y, p, base, alpha = [np.asarray(values[key]) for key in ("y", "p", "base", "alpha")]
        metrics = probability_metrics(y, p)
        baseline = probability_metrics(y, base)
        results.append(dict(uid=uid, count=len(y), **metrics,
                            brier_change=metrics["brier"]-baseline["brier"],
                            logloss_change=metrics["logloss"]-baseline["logloss"],
                            intervention_rate=float((alpha > 0).mean()),
                            wrong_to_right=int((((base >= .5) != y) & ((p >= .5) == y)).sum()),
                            right_to_wrong=int((((base >= .5) == y) & ((p >= .5) != y)).sum())))
    return results, ids


def write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output exists; choose a new directory")
    model, config, manifest, roles = load_artifact(args.source, args.device)
    if manifest["stage"] != "calibrate" or model.phase != "inference":
        raise ValueError("--source must be the final model/ directory produced by --stage train")
    model.set_stage("inference")  # Freeze all weights for every experiment.
    policies = manifest.get("calibration_report") or {}
    recalibrate = args.recalibrate or not all(key in policies for key in ("E2", "E3"))
    if recalibrate:
        print("Calibrating E2/E3 on calibration students only...", flush=True)
        tolerance = policies.get("E3", {}).get("logloss_tolerance", 0.)
        model.phase = "candidate"
        data = collect_predictions(model, role_loader(roles, "calibration"))
        policies = {"E2": calibrate(model, data, tolerance, uncertainty_only=True),
                    "E3": calibrate(model, data, tolerance)}
        del data
    if not policies["E2"]["uncertainty_only"] or policies["E3"]["uncertainty_only"]:
        raise ValueError("Invalid E2/E3 calibration policies")

    # No test labels are consumed until both policies have been fixed.
    test_loader = role_loader(roles, "test")
    output.mkdir(parents=True)
    write_json(output / "calibration.json", policies)
    report, summaries, student_rows, target_ids = {}, [], [], None
    for name, description, phase in EXPERIMENTS:
        print("Evaluating " + name + ": " + description, flush=True)
        model.phase = phase
        if name in ("E2", "E3"):
            model.set_calibration(*policies[name]["selected"]["thresholds"], uncertainty_only=name == "E2")
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize(next(model.parameters()).device)
        started = time.perf_counter()
        path = output / (name + "_predictions.csv")
        auc, accuracy, diagnostics = evaluate_long_mem_akt(model, test_loader, str(path))
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize(next(model.parameters()).device)
        seconds = time.perf_counter()-started
        students, ids = compare_students(output / "E0_predictions.csv", path)
        if target_ids is None:
            target_ids = ids
        elif target_ids != ids:
            raise ValueError("Scored target IDs differ between experiments")
        if name == "E0":
            diagnostics.update(brier_change=0., logloss_change=0., intervention_rate=0.,
                               mean_alpha=0., mean_probability_change=0., wrong_to_right=0,
                               right_to_wrong=0, mc_seconds=0.)
        result = dict(experiment=name, description=description, auc=auc, accuracy=accuracy,
                      seconds=seconds, **diagnostics)
        result["student_brier_change_quantiles"] = np.quantile(
            [student["brier_change"] for student in students], [0, .25, .5, .75, 1]).tolist()
        report[name] = result
        summaries.append({key: result[key] for key in SUMMARY_COLUMNS})
        student_rows.extend(dict(experiment=name, **student) for student in students)

    write_json(output / "metrics.json", report)
    write_json(output / "target_indices.json", target_ids)
    write_json(output / "run.json", dict(source=str(Path(args.source).resolve()),
               weights_sha256=manifest["weights_sha256"], roles_sha256=manifest["roles_sha256"],
               model_config=config, device=args.device, threads=args.threads,
               calibration_source="recomputed on calibration role" if recalibrate else "saved artifact",
               target_ids_sha256=json_digest(target_ids),
               timing_note="Shared student/block evaluator, including memory maintenance and CSV export"))
    write_csv(output / "summary.csv", summaries, SUMMARY_COLUMNS)
    write_csv(output / "students.csv", student_rows, list(student_rows[0]))
    lines = ["| Group | AUC | Brier | Logloss | Intervention | Seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append("| {experiment} | {auc:.6f} | {brier:.6f} | {logloss:.6f} | {intervention_rate:.2%} | {seconds:.2f} |".format(**row))
    table = "\n".join(lines) + "\n"
    with (output / "summary.md").open("w") as handle:
        handle.write(table)
    print(table)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Final model/ artifact directory")
    parser.add_argument("--output", required=True, help="New directory for comparison results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--recalibrate", action="store_true", help="Recalibrate E2/E3 using calibration students only")
    main(parser.parse_args())
