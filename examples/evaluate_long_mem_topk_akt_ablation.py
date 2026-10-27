"""Matched AKT/memory fine-tuning and evaluation in isolated processes."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

VARIANTS = {
    'akt': ('off', 'exact', 1), 'akt_long': ('off', 'exact', 2),
    'recent': ('recent', 'exact', 1), 'exact': ('retrieval', 'exact', 1),
    'hnsw_ip': ('retrieval', 'hnsw_ip', 1), 'dense_oracle': ('dense_oracle', 'exact', 1),
}


def main(args):
    variants = args.variants.split(',')
    seeds = [int(seed) for seed in args.seeds.split(',')]
    if not args.akt_source or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('Require --akt_source and distinct supported variants')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('wandb_long_mem_topk_akt_train.py').resolve()
    env = dict(os.environ, PYTHONPATH=str(script.parent.parent)+os.pathsep+os.environ.get('PYTHONPATH', ''),
               OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    results, commands = [], []
    reference_targets = source_hash = None
    for seed in seeds:
        for variant in variants:
            mode, backend, multiplier = VARIANTS[variant]
            run = output/(variant+'_seed'+str(seed))
            command = [sys.executable, str(script), '--akt_source', str(Path(args.akt_source).resolve()),
                       '--output', str(run), '--data_config', str(Path(args.data_config).resolve()),
                       '--dataset_name', args.dataset_name, '--fold', str(args.fold), '--seed', str(seed),
                       '--epochs', str(args.epochs), '--patience', '0', '--learning_rate', str(args.learning_rate),
                       '--memory_mode', mode, '--retrieval_backend', backend, '--device', args.device,
                       '--threads', str(args.threads), '--block_len', str(args.block_len*multiplier)]
            for name in ('architecture_variant', 'training_stage', 'memory_decay_mode', 'gate_lr',
                         'recent_capacity', 'remote_capacity', 'remote_topk', 'retrieval_query_chunk',
                         'retrieval_memory_chunk', 'index_refresh_blocks', 'hnsw_connections',
                         'hnsw_ef_construction', 'hnsw_ef_search', 'score_protocol'):
                command += ['--'+name, str(getattr(args, name))]
            if args.fixed_steps is not None:
                command += ['--fixed_steps', str(args.fixed_steps)]
            commands.append(command)
            subprocess.run(command, env=env, check=True)
            evaluation = [sys.executable, str(script), '--stage', 'evaluate', '--source', str(run),
                          '--output', str(run/'test'), '--device', args.device, '--threads', str(args.threads)]
            commands.append(evaluation)
            subprocess.run(evaluation, env=env, check=True)
            train = json.loads((run/'training.json').read_text())
            report = json.loads((run/'test/metrics.json').read_text())
            checkpoint_hash = train['provenance']['checkpoint_sha256']
            if source_hash is None:
                source_hash = checkpoint_hash
            if source_hash != checkpoint_hash:
                raise ValueError('Initial checkpoint changed during comparison')
            with (run/'test/predictions.csv').open() as handle:
                targets = [(r['uid'], r['position'], r['target']) for r in csv.DictReader(handle)]
            if reference_targets is None:
                reference_targets = targets
            if reference_targets != targets:
                raise ValueError('Unfair comparison: evaluation targets differ')
            row = dict(variant=variant, effective_backend=report['effective_backend'], seed=seed, auc=report['auc'], accuracy=report['accuracy'],
                       logloss=report['logloss'], count=report['count'], test_seconds=report['seconds'],
                       train_seconds=train['train_seconds'], optimizer_steps=train['optimizer_steps'],
                       scored_training_labels=train['scored_training_labels'],
                       bank_bytes_peak=report['bank_bytes_peak'], index_bytes_peak=report['index_bytes_peak'],
                       cpu_peak_rss_bytes=report['cpu_peak_rss_bytes'],
                       cuda_peak_allocated=report['cuda_peak_allocated'], cuda_peak_reserved=report['cuda_peak_reserved'],
                       fallback_fraction=report['fallback_fraction'], best_epoch=train['best_epoch'])
            results.append(row)
            if args.diagnostics and mode != 'off':
                diagnostic = evaluation.copy()
                diagnostic[diagnostic.index('--output')+1] = str(run/'diagnostics')
                diagnostic.append('--diagnostics')
                commands.append(diagnostic)
                subprocess.run(diagnostic, env=env, check=True)
    for seed in seeds:
        comparable = [r for r in results if r['seed'] == seed and r['variant'] != 'akt_long']
        for key in ('optimizer_steps', 'scored_training_labels'):
            if len({r[key] for r in comparable}) > 1:
                raise ValueError('Unmatched '+key)
    with (output/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {}
    for variant in variants:
        selected = [r for r in results if r['variant'] == variant]
        summary[variant] = {key: dict(mean=float(np.mean([r[key] for r in selected])),
                                    std=float(np.std([r[key] for r in selected])))
                            for key in ('auc', 'logloss', 'accuracy', 'train_seconds', 'test_seconds')
                                if all(r[key] is not None for r in selected)}
    metadata = dict(results=results, summary=summary, commands=commands, config=vars(args),
                    initial_checkpoint_sha256=source_hash,
                    note='akt_long changes update granularity: report interaction and step budgets separately. '
                         'All modes score identical uid/position/target triples. Dense oracle means cached features (A).')
    (output/'comparison.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    from wandb_long_mem_topk_akt_train import parser
    p = parser()
    p.description = __doc__
    p.add_argument('--variants', default='akt,recent,exact,hnsw_ip', help='Also accepts akt_long,dense_oracle')
    p.add_argument('--seeds', default='42,43,44')
    main(p.parse_args())
