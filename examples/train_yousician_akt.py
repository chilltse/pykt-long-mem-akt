"""pyKT entry point: preprocess first, then select dataset_name and fold."""
import argparse

try:
    from .wandb_train import main
except ImportError:
    from wandb_train import main


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="yousician")
    parser.add_argument("--data_config", default="../configs/data_config.json")
    parser.add_argument("--model_name", choices=["akt", "akt_fmkc", "yousician_akt"], default="akt_fmkc")
    parser.add_argument("--emb_type", choices=["qid"], default="qid")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--save_dir", default="saved_model")
    parser.add_argument("--drift", choices=["none", "shared", "response"], default="response")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--cache_blocks", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--field_dim", type=int, default=32)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.num_epochs < 1:
        parser.error("num_epochs must be positive")
    main(vars(args))
