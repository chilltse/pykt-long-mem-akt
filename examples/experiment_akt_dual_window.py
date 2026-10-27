"""Student-isolated A/B/C comparison starting from identical original AKT weights."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pykt.datasets.long_mem_akt_dataset import LongMemAKTDataset
from pykt.models.akt import AKT
from pykt.models.akt_dual_window import AKTDualWindow
from pykt.models.akt_dual_window_runner import run_epoch


def main(args):
    if args.epochs < 1 or args.block_size < 1 or args.short_window < args.block_size:
        raise ValueError('Require epochs >= 1 and W >= B >= 1')
    variants = args.variants.split(',')
    if len(set(variants)) != len(variants) or set(variants) - {'akt', 'short-only', 'akt-dual-window'}:
        raise ValueError('Supported variants: akt,short-only,akt-dual-window')
    torch.set_num_threads(args.threads)
    config = json.loads(Path(args.data_config).read_text())[args.dataset_name]
    # Repository data_config paths are relative to examples/, like existing entry points.
    root = Path(args.data_dir).resolve() if args.data_dir else (Path(__file__).parent / config['dpath']).resolve()
    def dataset(filename, folds):
        return LongMemAKTDataset(root / filename, config['input_type'], folds).records
    folds = set(config['folds'])
    if args.fold not in folds:
        raise ValueError('Validation fold is not in data_config')
    train = dataset(config['train_valid_original_file'], folds - {args.fold})
    valid = dataset(config['train_valid_original_file'], {args.fold})
    test = dataset(config['test_original_file'], {-1})
    groups = [{r['uid'] for r in records} for records in (train, valid, test)]
    if not all(groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Require nonempty, student-disjoint train/validation/test splits')
    if any(not any(bool(r['allowed'][1:].any()) for r in records) for records in (train, valid, test)):
        raise ValueError('Each split needs scored labels')
    if config['num_q'] and 'questions' not in config['input_type']:
        raise ValueError('PID embeddings require question IDs')
    model_config = dict(n_question=config['num_c'], n_pid=config['num_q'], d_model=args.d_model,
        n_blocks=args.n_blocks, dropout=args.dropout, d_ff=args.d_ff,
        num_attn_heads=args.num_attn_heads, final_fc_dim=args.final_fc_dim)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(dict(args=vars(args), model_config=model_config,
        data_config=config, attention='normalized QK sqrt(d_k); independent short/long; static head gates 0.05', scoring='all allowed positions except first student token; boundary = first token of blocks after block 0',
        optimizer='one step per student; block backward; no cache across steps'), indent=2))
    summary = []
    reference = None
    for seed in map(int, args.seeds.split(',')):
        torch.manual_seed(seed)
        base = AKT(**model_config)
        initial = base.state_dict()
        if args.init_checkpoint:
            initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
            validator = AKTDualWindow(**model_config)
            validator.load_compatible_state_dict(initial)
        for variant in variants:
            torch.manual_seed(seed)
            if variant == 'akt':
                model = AKT(**model_config)
            else:
                model = AKTDualWindow(**model_config, block_size=args.block_size,
                    short_window=args.short_window, long_window=args.long_window,
                    topk=0 if variant == 'short-only' else args.topk)
            if isinstance(model, AKTDualWindow):
                model.load_compatible_state_dict(initial)
            else:
                # Only discard validated per-layer gates when loading into original AKT.
                model.load_state_dict({key: initial[key] for key in model.state_dict()}, strict=True)
            model.to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            run = output / f'{variant}_seed{seed}'
            run.mkdir()
            best, history = float('inf'), []
            for epoch in range(args.epochs):
                training, _ = run_epoch(model, train, args.block_size, optimizer, seed + epoch)
                validation, _ = run_epoch(model, valid, args.block_size)
                history.append(dict(epoch=epoch + 1, train=training, validation=validation))
                value = validation['overall']['logloss']
                if value < best:
                    best = value
                    torch.save(model.state_dict(), run / 'best.pt')
                    best_epoch = epoch + 1
                print(f'{variant} seed={seed} epoch={epoch+1} val_logloss={value:.6f}', flush=True)
            model.load_state_dict(torch.load(run / 'best.pt', map_location=args.device, weights_only=True))
            report, rows = run_epoch(model, test, args.block_size)
            targets = [(r['uid'], r['position'], r['target']) for r in rows]
            if reference is not None and reference != targets:
                raise RuntimeError('A/B scoring targets differ')
            reference = targets
            with (run / 'predictions.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (run / 'training.json').write_text(json.dumps(history, indent=2))
            (run / 'metrics.json').write_text(json.dumps(report, indent=2))
            summary.append(dict(variant=variant, seed=seed, best_epoch=best_epoch, **report))
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-config', default=str(Path(__file__).resolve().parents[1] / 'configs/data_config.json'))
    p.add_argument('--dataset-name', default='assist2009')
    p.add_argument('--data-dir', help='Override original CSV directory')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--output', required=True)
    p.add_argument('--variants', default='akt,short-only,akt-dual-window')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--init-checkpoint', help='Original AKT or short-only/dual-window state dict (best.pt)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=1)
    for name, default in [('block-size', 128), ('short-window', 128), ('long-window', 256), ('topk', 32),
                          ('d-model', 256), ('d-ff', 512), ('num-attn-heads', 8), ('n-blocks', 4), ('final-fc-dim', 512)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--dropout', type=float, default=.2)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
