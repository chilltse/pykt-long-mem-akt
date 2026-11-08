"""Compatibility entry point; prefer train_akt_fmkc.py."""
try:
    from .train_akt_fmkc import build_parser, main
except ImportError:
    from train_akt_fmkc import build_parser, main

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.num_epochs < 1:
        parser.error("num_epochs must be positive")
    main(vars(args))
