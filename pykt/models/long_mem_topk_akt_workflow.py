"""Reproducible fine-tuning from a checked original-AKT checkpoint."""
import inspect
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .akt import AKT
from .long_mem_topk_akt import LongMemTopKAKT
from .long_mem_akt_workflow import create_roles, role_loader, write_json, file_digest
from .long_mem_topk_akt_runner import run_epoch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def baseline_config(source, data_config, fold):
    source = Path(source)
    config = json.loads((source/'config.json').read_text())
    params = config['params']
    if params['model_name'] != 'akt':
        raise ValueError('--akt_source must be an original AKT artifact')
    if params.get('fold') != fold or params.get('dataset_name') != data_config['dataset_name']:
        raise ValueError('Baseline fold/dataset differs; this could leak validation students')
    dc = config['data_config']
    if any(dc.get(key) != data_config.get(key) for key in ('num_c', 'num_q', 'input_type', 'folds')):
        raise ValueError('Baseline vocabulary/input configuration differs')
    supported = set(inspect.signature(AKT.__init__).parameters)-{'self', 'n_question', 'n_pid'}
    mc = {key: value for key, value in config['model_config'].items() if key in supported}
    mc['emb_type'] = params.get('emb_type', 'qid')
    path = source/(mc['emb_type']+'_model.ckpt')
    state = torch.load(path, map_location='cpu')
    baseline = AKT(dc['num_c'], dc['num_q'], **mc)
    baseline.load_state_dict(state, strict=True)
    return mc, state, dict(checkpoint=str(path.resolve()), checkpoint_sha256=file_digest(path),
                           config_sha256=file_digest(source/'config.json'),
                           akt_source_sha256=file_digest(Path(__file__).with_name('akt.py')))


def fit(output, model, model_config, roles, initial_provenance, epochs, learning_rate, seed,
        patience=10, fixed_steps=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable parameters; use evaluate for freeze_backbone')
    optimizer = torch.optim.Adam(model.optimizer_groups(learning_rate), lr=learning_rate)
    train_dataset = role_loader(roles, 'train').dataset
    valid = role_loader(roles, 'valid')
    best, best_epoch, total_steps = -float('inf'), 0, 0
    history = []
    for epoch in range(1, epochs+1):
        # Shuffle only students; never blocks. Identical order across variants.
        generator = torch.Generator().manual_seed(seed+epoch)
        train = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator)
        remaining = None if fixed_steps is None else fixed_steps-total_steps
        loss, train_report = run_epoch(model, train, optimizer=optimizer, max_steps=remaining)
        total_steps += train_report['optimizer_steps']
        auc, acc, valid_report = run_epoch(model, valid)
        selection = auc if np.isfinite(auc) else -valid_report['logloss']
        row = dict(epoch=epoch, train_loss=loss, train=train_report, valid=valid_report)
        history.append(row)
        if best_epoch == 0 or selection > best+1e-6:
            best, best_epoch = selection, epoch
            torch.save(model.state_dict(), output/(model.emb_type+'_model.ckpt'))
        # Epoch-boundary state only. No claim of mid-student/bank resume support.
        torch.save(dict(epoch=epoch, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        feature_version=model.feature_version, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                   output/'last_epoch.pt')
        print(f'epoch={epoch} train_loss={loss:.6f} valid_auc={auc:.6f} valid_nll={valid_report["logloss"]:.6f}', flush=True)
        if (fixed_steps is not None and total_steps >= fixed_steps) or (patience and epoch-best_epoch >= patience):
            break
    if fixed_steps is not None and total_steps < fixed_steps:
        raise ValueError('epochs exhausted before fixed_steps; increase --epochs for a matched step budget')
    model_config = dict(model_config)
    for name in ('architecture_variant', 'memory_decay_mode', 'training_stage', 'gate_lr'):
        model_config[name] = getattr(model, name)
    saved_config = dict(model_config={k: v for k, v in model_config.items() if k != 'emb_type'},
                        data_config=roles['data_config'],
                        params=dict(model_name='long_mem_topk_akt', emb_type=model.emb_type,
                                    dataset_name=roles['data_config']['dataset_name'], fold=roles['fold']),
                        train_config=dict(epochs=epochs, learning_rate=learning_rate, seed=seed,
                                          patience=patience, fixed_steps=fixed_steps))
    saved_config['trainable_parameters'] = [n for n, p in model.named_parameters() if p.requires_grad]
    saved_config['model_version'] = model.architecture_variant
    write_json(output/'config.json', saved_config)
    write_json(output/'roles.json', roles)
    write_json(output/'training.json', dict(history=history, best_epoch=best_epoch,
               optimizer_steps=total_steps, train_seconds=sum(row['train']['seconds'] for row in history),
               scored_training_labels=sum(row['train']['scored_labels'] for row in history),
               provenance=initial_provenance, resume_scope='epoch-boundary state saved; mid-block replay not supported'))
    write_json(output/'memory_manifest.json', dict(
        config_sha256=file_digest(output/'config.json'), roles_sha256=file_digest(output/'roles.json'),
        weights_sha256=file_digest(output/(model.emb_type+'_model.ckpt'))))
    return history


def load_artifact(source, device='cpu'):
    source = Path(source)
    config = json.loads((source/'config.json').read_text())
    manifest = json.loads((source/'memory_manifest.json').read_text())
    emb_type = config['params']['emb_type']
    for name, key in (('config.json', 'config_sha256'), ('roles.json', 'roles_sha256'),
                      (emb_type+'_model.ckpt', 'weights_sha256')):
        if file_digest(source/name) != manifest[key]:
            raise ValueError('Artifact checksum mismatch: '+name)
    dc = config['data_config']
    model = LongMemTopKAKT(dc['num_c'], dc['num_q'], emb_type=emb_type, **config['model_config']).to(device)
    model.load_state_dict(torch.load(source/(emb_type+'_model.ckpt'), map_location=device), strict=True)
    roles = json.loads((source/'roles.json').read_text())
    return model, config, roles


def evaluate_artifact(source, output, device='cpu', diagnostics=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, config, roles = load_artifact(source, device)
    model.collect_diagnostics = diagnostics
    model.diagnose_dense = diagnostics
    auc, acc, report = run_epoch(model, role_loader(roles, 'test'), save_path=str(output/'predictions.csv'))
    if diagnostics and model.memory_mode != 'off':
        import csv
        model.memory_mode = 'dense_oracle'
        model.diagnose_dense = False
        run_epoch(model, role_loader(roles, 'test'), save_path=str(output/'dense_A_predictions.csv'))
        with (output/'predictions.csv').open() as handle:
            candidate = list(csv.DictReader(handle))
        with (output/'dense_A_predictions.csv').open() as handle:
            dense = list(csv.DictReader(handle))
        if [(r['uid'], r['position']) for r in candidate] != [(r['uid'], r['position']) for r in dense]:
            raise ValueError('Dense diagnostic target mismatch')
        differences = np.array([float(a['probability'])-float(b['probability']) for a, b in zip(candidate, dense)])
        report['dense_A_replay_prediction_mean_absolute_difference'] = float(np.abs(differences).mean())
        report['dense_A_replay_note'] = 'Full replay can change later cached features; attention-level diagnostic uses identical snapshots.'
    write_json(output/'metrics.json', report)
    write_json(output/'config.json', config)
    return report
