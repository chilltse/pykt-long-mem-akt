"""Compare the frozen AKT baseline (E0) and full long-memory residual (E1)."""
import argparse
from pathlib import Path
import torch
from pykt.models.long_mem_akt_workflow import load_artifact, role_loader, write_json
from pykt.models.long_mem_akt_runner import evaluate_long_mem_akt


def main(args):
    torch.set_num_threads(args.threads)
    model, _, _, roles = load_artifact(args.source, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    loader = role_loader(roles, "test")
    reports = {}
    for name, stage in (("E0", "base"), ("E1", "inference")):
        model.set_stage(stage)
        auc, accuracy, diagnostics = evaluate_long_mem_akt(model, loader, str(output / (name + "_predictions.csv")))
        reports[name] = dict(auc=auc, accuracy=accuracy, **diagnostics)
    write_json(output / "metrics.json", reports)
    print(reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    main(parser.parse_args())
