"""Four-stage LongMemAKT workflow. Run from examples; see README commands."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from pykt.models.long_mem_akt import LongMemAKT
from pykt.models.long_mem_akt_runner import evaluate_long_mem_akt
from pykt.models.long_mem_akt_workflow import (
    create_roles, role_loader, save_artifact, load_artifact, collect_predictions,
    fit_benefit, calibrate, write_json, json_digest)
from pykt.models.train_model import train_model


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["train", "split", "base", "candidate", "benefit", "calibrate", "evaluate"])
    p.add_argument("--output", required=True)
    p.add_argument("--roles")
    p.add_argument("--source", help="Previous stage artifact directory")
    p.add_argument("--dataset_name", default="assist2009")
    p.add_argument("--data_config", default="../configs/data_config.json")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--block_size", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--benefit_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=.001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--num_attn_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--final_fc_dim", type=int, default=512)
    p.add_argument("--num_slots", type=int, default=32)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--tbptt_blocks", type=int, default=2)
    p.add_argument("--score_protocol", choices=["legacy", "student"], default="legacy")
    p.add_argument("--mc_samples", type=int, default=None, help="Candidate default 20; later stages inherit and check")
    p.add_argument("--mc_seed", type=int, default=None, help="Candidate default 42; later stages inherit and check")
    p.add_argument("--correction_hidden", type=int, default=64)
    p.add_argument("--benefit_hidden", type=int, default=32)
    p.add_argument("--logloss_tolerance", type=float, default=0.)
    return p


def main(args):
    if args.epochs < 1 or args.block_size < 1:
        raise ValueError("epochs and block_size must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    # New experiment artifacts must not overwrite an existing run or parent.
    if output.exists():
        raise ValueError("Output already exists; choose a new experiment directory/file")
    if args.stage == "split":
        dc = json.loads(Path(args.data_config).read_text())[args.dataset_name]
        dc["dataset_name"] = args.dataset_name
        roles = create_roles(dc, args.fold, args.seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, roles)
        print({role: len(ids) for role, ids in roles["roles"].items()})
        return
    if args.stage == "train":
        output.mkdir(parents=True)
        roles_path = output / "roles.json"
        source = None
        for stage, name in [("split", "roles.json"), ("base", "base"),
                            ("candidate", "candidate"), ("benefit", "benefit"),
                            ("calibrate", "model")]:
            step = copy.copy(args)
            step.stage, step.output = stage, str(output / name)
            step.roles, step.source = str(roles_path), str(source) if source else None
            if stage == "benefit":
                step.epochs = args.benefit_epochs
            main(step)
            if stage != "split":
                source = output / name
        print("Trained long_mem_akt: " + str(output / "model"))
        return
    provenance = {}
    if args.stage == "base":
        if not args.roles:
            raise ValueError("Base stage requires --roles")
        roles = json.loads(Path(args.roles).read_text())
        mc = {key: getattr(args, key) for key in ("d_model", "n_blocks", "num_attn_heads", "d_ff",
              "final_fc_dim", "num_slots", "dropout", "tbptt_blocks", "score_protocol", "block_size")}
        mc.update(phase="base", correction_hidden=args.correction_hidden, benefit_hidden=args.benefit_hidden,
                  mc_samples=args.mc_samples if args.mc_samples is not None else 20,
                  mc_seed=args.mc_seed if args.mc_seed is not None else 42)
        dc = roles["data_config"]
        model = LongMemAKT(dc["num_c"], dc["num_q"], **mc).to(args.device)
    else:
        if not args.source:
            raise ValueError("This stage requires --source")
        model, mc, previous, roles = load_artifact(args.source, args.device)
        mc = copy.deepcopy(mc)
        if args.roles and json_digest(json.loads(Path(args.roles).read_text())) != json_digest(roles):
            raise ValueError("Role manifest differs from source artifact")
        if args.block_size != model.block_size:
            raise ValueError("block_size differs from source: retrain and recalibrate")
        if args.stage != "candidate":
            for key in ("mc_samples", "mc_seed"):
                value = getattr(args, key)
                if value is not None and value != getattr(model, key):
                    raise ValueError(key + " differs from source: regenerate benefit data and recalibrate")
        required = dict(candidate="base", benefit="candidate", calibrate="benefit", evaluate="calibrate")
        if previous["stage"] != required[args.stage]:
            raise ValueError("Unexpected source stage: " + previous["stage"])
        provenance = dict(previous["provenance"], **{previous["stage"]+"_sha256": previous["weights_sha256"]})
    if args.stage == "candidate":
        if mc["phase"] != "base":
            raise ValueError("A protected baseline must come from the base training stage")
        old = model
        mc.update(phase="candidate", mc_samples=args.mc_samples if args.mc_samples is not None else model.mc_samples,
                  mc_seed=args.mc_seed if args.mc_seed is not None else model.mc_seed,
                  tbptt_blocks=args.tbptt_blocks)
        dc = roles["data_config"]
        model = LongMemAKT(dc["num_c"], dc["num_q"], **mc).to(args.device)
        model.load_state_dict(old.state_dict())
        # Only a checked base artifact can initialize the protected AKT.
        for key, value in old.akt.state_dict().items():
            if not torch.equal(value, model.akt.state_dict()[key]):
                raise ValueError("Baseline adapter weight mismatch")
    output.mkdir(parents=True)
    if args.stage in {"base", "candidate"}:
        provenance["training"] = dict(seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
                                      selection="dev logloss" if args.stage == "candidate" else "dev AUC")
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        train_model(model, role_loader(roles, "fit"), role_loader(roles, "dev"),
                    args.epochs, opt, str(output), save_model=True)
        model.load_state_dict(torch.load(output / "qid_model.ckpt", map_location=args.device))
        if args.stage == "candidate":
            for key, value in old.akt.state_dict().items():
                if not torch.equal(value, model.akt.state_dict()[key]):
                    raise ValueError("Frozen AKT changed during candidate training")
        evaluate_long_mem_akt(model, role_loader(roles, "dev"), str(output / "dev_predictions.csv"))
        save_artifact(output, model, mc, roles, args.stage, provenance)
    elif args.stage == "benefit":
        model.set_stage("inference")
        data = collect_predictions(model, role_loader(roles, "selector_fit"))
        report = fit_benefit(model, data, args.epochs, args.learning_rate, args.seed)
        write_json(output / "benefit_report.json", report)
        write_json(output / "target_indices.json", data["ids"])
        save_artifact(output, model, mc, roles, "benefit", provenance)
    elif args.stage == "calibrate":
        model.set_stage("inference")
        data = collect_predictions(model, role_loader(roles, "calibration"))
        e2 = calibrate(model, data, args.logloss_tolerance, uncertainty_only=True)
        e3 = calibrate(model, data, args.logloss_tolerance)
        model.phase = mc["phase"] = "inference"
        report = dict(E2=e2, E3=e3)
        write_json(output / "calibration.json", report)
        write_json(output / "target_indices.json", data["ids"])
        save_artifact(output, model, mc, roles, "calibrate", provenance, report)
    else:
        model.set_stage("inference")
        auc, accuracy, diagnostics = evaluate_long_mem_akt(
            model, role_loader(roles, "test"), str(output / "student_block_predictions.csv"))
        report = dict(auc=auc, accuracy=accuracy, **diagnostics)
        write_json(output / "metrics.json", report)
        print({key: report[key] for key in ("auc", "accuracy", "brier", "logloss", "intervention_rate")})


if __name__ == "__main__":
    main(parser().parse_args())
