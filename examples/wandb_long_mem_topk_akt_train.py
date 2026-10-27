"""Fine-tune original AKT with recent/Top-k memory; train/valid/test folds only."""
import argparse
import json
from pathlib import Path
import torch
from pykt.models.long_mem_topk_akt import LongMemTopKAKT
from pykt.models.long_mem_topk_akt_workflow import (
    baseline_config, fit, evaluate_artifact, seed_all, create_roles)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train', 'evaluate'], default='train')
    p.add_argument('--output', required=True)
    p.add_argument('--akt_source', help='Original AKT directory: config.json + emb_type_model.ckpt')
    p.add_argument('--source', help='Trained long_mem_topk_akt directory for evaluation')
    p.add_argument('--data_config', default='../configs/data_config.json')
    p.add_argument('--dataset_name', default='assist2009')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--learning_rate', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=10, help='0 disables early stopping for fixed-budget comparisons')
    p.add_argument('--fixed_steps', type=int, default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--architecture_variant', choices=['legacy_all_layers', 'v2_last_retriever_residual'], default='legacy_all_layers')
    p.add_argument('--memory_decay_mode', choices=['legacy_mul', 'log_additive'], default='legacy_mul')
    p.add_argument('--training_stage', choices=['gate_only', 'last_layer', 'full'], default='full')
    p.add_argument('--gate_lr', type=float, default=1e-3)
    p.add_argument('--memory_mode', choices=['off', 'recent', 'retrieval', 'dense_oracle'], default='retrieval')
    p.add_argument('--retrieval_backend', choices=['exact', 'hnsw_ip'], default='exact')
    for name, default in [('block_len', 200), ('recent_capacity', 200), ('remote_capacity', 8192),
                          ('remote_topk', 32), ('retrieval_query_chunk', 32), ('retrieval_memory_chunk', 1024),
                          ('index_refresh_blocks', 8), ('hnsw_connections', 16),
                          ('hnsw_ef_construction', 100), ('hnsw_ef_search', 64)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--score_protocol', choices=['legacy', 'student'], default='legacy')
    p.add_argument('--diagnostics', action='store_true', help='Expensive dense/recall diagnostics, not a speed benchmark')
    return p


def main(args):
    torch.set_num_threads(args.threads)
    try:
        import faiss
        faiss.omp_set_num_threads(args.threads)
    except ImportError:
        pass
    seed_all(args.seed)
    if args.stage == 'evaluate':
        if not args.source:
            raise ValueError('evaluate requires --source')
        print(evaluate_artifact(args.source, args.output, args.device, args.diagnostics))
        return
    if not args.akt_source or args.epochs < 1 or (args.fixed_steps is not None and args.fixed_steps < 1):
        raise ValueError('Training requires --akt_source and positive epochs/fixed_steps')
    dc = json.loads(Path(args.data_config).read_text())[args.dataset_name]
    dc['dataset_name'] = args.dataset_name
    roles = create_roles(dc, args.fold, args.seed)
    mc, state, provenance = baseline_config(args.akt_source, dc, args.fold)
    for name in ('architecture_variant', 'memory_decay_mode', 'training_stage', 'gate_lr', 'memory_mode', 'retrieval_backend', 'block_len', 'recent_capacity', 'remote_capacity',
                 'remote_topk', 'retrieval_query_chunk', 'retrieval_memory_chunk', 'index_refresh_blocks',
                 'hnsw_connections', 'hnsw_ef_construction', 'hnsw_ef_search', 'score_protocol'):
        mc[name] = getattr(args, name)
    model = LongMemTopKAKT(dc['num_c'], dc['num_q'], **mc).to(args.device)
    model.load_baseline_state_dict(state)
    fit(args.output, model, mc, roles, provenance, args.epochs, args.learning_rate,
        args.seed, args.patience, args.fixed_steps)


if __name__ == '__main__':
    main(parser().parse_args())
