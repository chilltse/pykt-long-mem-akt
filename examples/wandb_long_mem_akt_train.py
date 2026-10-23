"""E1 full residual training with standard train/valid/test folds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pykt.models.long_mem_akt import LongMemAKT
from pykt.models.long_mem_akt_runner import evaluate_long_mem_akt
from pykt.models.long_mem_akt_workflow import create_roles, role_loader, save_artifact, load_artifact, write_json
from pykt.models.train_model import train_model

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="train", choices=["train", "evaluate"])
    p.add_argument("--output", required=True)
    p.add_argument("--source", help="Trained model directory")
    p.add_argument("--dataset_name", default="assist2009")
    p.add_argument("--data_config", default="../configs/data_config.json")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--block_size", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
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
    p.add_argument("--correction_hidden", type=int, default=64)
    return p


def main(args):
    if args.epochs < 1 or args.block_size < 1:
        raise ValueError("epochs and block_size must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists; choose a new experiment directory")
    if args.stage == "evaluate":
        if not args.source:
            raise ValueError("Evaluation requires --source")
        model, _, _, roles = load_artifact(args.source, args.device)
        if args.block_size != model.block_size:
            raise ValueError("block_size differs from trained model")
        output.mkdir(parents=True)
        auc, accuracy, diagnostics = evaluate_long_mem_akt(
            model, role_loader(roles, "test"), str(output / "student_block_predictions.csv"))
        report = dict(auc=auc, accuracy=accuracy, **diagnostics)
        write_json(output / "metrics.json", report)
        print(report)
        return
    dc = json.loads(Path(args.data_config).read_text())[args.dataset_name]
    dc["dataset_name"] = args.dataset_name
    roles = create_roles(dc, args.fold, args.seed)
    mc = {key: getattr(args, key) for key in (
        "d_model", "n_blocks", "num_attn_heads", "d_ff", "final_fc_dim", "num_slots",
        "dropout", "tbptt_blocks", "score_protocol", "block_size", "correction_hidden")}
    model = LongMemAKT(dc["num_c"], dc["num_q"], phase="base", **mc).to(args.device)
    train, valid = role_loader(roles, "train"), role_loader(roles, "valid")
    output.mkdir(parents=True)
    # Preserve E1: pretrain AKT, then freeze it and fit the full residual.
    # Both optimization phases use exactly the same train/valid students.
    for stage in ("base", "candidate"):
        model.set_stage(stage)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        train_model(model, train, valid, args.epochs, opt, str(output), save_model=True)
        model.load_state_dict(torch.load(output / "qid_model.ckpt", map_location=args.device))
    mc["phase"] = "candidate"
    save_artifact(output, model, mc, roles, "train",
                  dict(seed=args.seed, epochs_per_phase=args.epochs, learning_rate=args.learning_rate))
    print("Trained E1 long_mem_akt: " + str(output))


if __name__ == "__main__":
    main(parser().parse_args())
