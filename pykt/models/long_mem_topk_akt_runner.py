"""Single-rank, one-student-at-a-time ordered replay; fresh banks each call."""
import csv
import resource
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from .akt_memory_bank import MemoryBank
from .long_mem_akt_runner import _uid, _student, _blocks, _select


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_epoch(model, loader, optimizer=None, save_path='', max_steps=None, grad_clip=5.):
    training = optimizer is not None
    model.train(training and not model.freeze_backbone)
    device = next(model.parameters()).device
    bank = MemoryBank(model)
    ys, ps, rows, losses, block_times = [], [], [], [], []
    diagnostics = {}
    stats = dict(bank_bytes_peak=0, index_bytes_peak=0, refreshes=0, fallbacks=0, searches=0,
                 index_update_seconds=0., index_transfer_seconds=0., index_search_seconds=0.)
    steps = blocks = interactions = scored = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    seen_students = set()
    for raw in loader:
        uid = _uid(raw)
        if uid in seen_students:
            raise ValueError('Runner expects one complete trajectory per student')
        seen_students.add(uid)
        student = _student(raw, device)
        last_kc = {}
        for start, block in _blocks(student, model.block_len):
            _sync(device)
            block_started = time.perf_counter()
            pos = torch.arange(start, start+block['rseqs'].size(1), device=device)[None]
            selected = _select(model, block, start)
            with torch.set_grad_enabled(training and not model.freeze_backbone):
                prediction, reg, pending = model.forward_with_memory(
                    block['cseqs'], block['rseqs'], block['qseqs'], pos, block['valid'], bank.snapshot([uid]),
                    block_ids=[start//model.block_len])
                if selected.any():
                    loss = F.binary_cross_entropy(prediction[selected], block['rseqs'][selected]) + reg
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        if loss.requires_grad:
                            loss.backward()
                            if hasattr(model, "memory_alpha") and model.memory_alpha.grad is not None:
                                diagnostics["gate_gradient_norm_sum"] = diagnostics.get("gate_gradient_norm_sum", 0.) + float(model.memory_alpha.grad.norm())
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            optimizer.step()
                            model.feature_version += 1
                            steps += 1
                    losses.append((float(loss.detach()), int(selected.sum())))
            # Pending features retain the pre-update version; search keys use current parameters.
            bank.commit(pending, parameter_version=model.feature_version)
            _sync(device)
            block_times.append(time.perf_counter()-block_started)
            for key, value in model.last_memory_diagnostics.items():
                diagnostics[key] = diagnostics.get(key, 0.) + value
            current = bank.stats()
            stats['bank_bytes_peak'] = max(stats['bank_bytes_peak'], current['bank_bytes'])
            stats['index_bytes_peak'] = max(stats['index_bytes_peak'], current['index_bytes'])
            blocks += 1
            interactions += int(block['valid'].sum())
            scored += int(selected.sum())
            if not training:
                target = block['rseqs'][0].detach().cpu().tolist()
                probabilities = prediction[0].detach().cpu().tolist()
                concepts = block['cseqs'][0].cpu().tolist()
                valid = block['valid'][0].cpu().tolist()
                selected_cpu = selected[0].cpu().tolist()
                for offset, (y, p, kc, valid_event, use) in enumerate(zip(target, probabilities, concepts, valid, selected_cpu)):
                    absolute = start+offset
                    gap = absolute-last_kc[kc] if kc in last_kc else None
                    if use:
                        ys.append(y)
                        ps.append(p)
                        rows.append(dict(uid=uid, position=absolute, target=y, probability=p,
                                         history_length=absolute, kc_gap=gap))
                    if valid_event:
                        last_kc[kc] = absolute
            if training and max_steps is not None and steps >= max_steps:
                break
        final_stats = bank.stats()
        for key in stats:
            if key in final_stats:
                stats[key] += final_stats[key]
        bank.clear(uid)  # Finished students do not accumulate resident features or indexes.
        if training and max_steps is not None and steps >= max_steps:
            break
    _sync(device)
    stats.update(seconds=time.perf_counter()-started, blocks=blocks, optimizer_steps=steps,
                 interactions=interactions, scored_labels=scored,
                 block_seconds_mean=float(np.mean(block_times)) if block_times else 0.,
                 cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 cuda_peak_allocated=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                 cuda_peak_reserved=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                 feature_version=model.feature_version, diagnostics=diagnostics,
                 backend=model.retrieval_backend, memory_mode=model.memory_mode,
                 diagnostic_oracle_enabled=model.diagnose_dense)
    try:
        import faiss
        stats['faiss_version'] = faiss.__version__
    except ImportError:
        stats['faiss_version'] = None
    stats['effective_backend'] = ('exact_fallback' if model.retrieval_backend == 'hnsw_ip' and stats['faiss_version'] is None
                                  else model.retrieval_backend)
    if hasattr(model, 'memory_alpha'):
        stats['gate'] = model.memory_alpha.detach().tanh().cpu().tolist()
    if model.architecture_variant == 'v2_last_retriever_residual':
        stats['dense_oracle_semantics'] = 'All remote cached features in the last retriever memory branch; not full-history AKT'
        stats['dense_attention_diagnostics_available'] = False
    stats['architecture_variant'] = model.architecture_variant
    stats['training_stage'] = model.training_stage
    stats['memory_decay_mode'] = model.memory_decay_mode
    stats['torch_version'] = torch.__version__
    stats['component_timing'] = ('synchronized' if model.collect_diagnostics or device.type == 'cpu'
                                 else 'CPU launch timing; use diagnostics for synchronized CUDA components')
    normalized = {}
    for result, numerator, denominator in (
        ('remote_attention_mass_mean', 'remote_mass_sum', 'query_heads'),
        ('remote_read_hit_fraction', 'remote_hits', 'query_heads'),
        ('candidate_age_mean', 'candidate_age_sum', 'remote_candidates'),
        ('feature_age_mean', 'feature_age_sum', 'remote_candidates'),
        ('raw_ip_recall_at_k', 'recall_sum', 'recall_rows'),
        ('retained_dense_attention_mass', 'retained_mass_sum', 'dense_rows'),
        ('dense_output_l2_mean', 'dense_output_l2_sum', 'dense_rows'),
        ('index_age_mean', 'index_age_sum', 'indexed_candidates'),
        ('index_rescore_absolute_difference_mean', 'rescore_abs_sum', 'indexed_candidates')):
        count = diagnostics.get(denominator, 0)
        normalized[result] = diagnostics.get(numerator, 0)/count if count else None
    stats['retrieval_quality'] = normalized
    stats['fallback_fraction'] = stats['fallbacks']/max(1, stats['searches'])
    if training:
        mean_loss = sum(loss*n for loss, n in losses)/max(1, sum(n for _, n in losses))
        model.last_train_diagnostics = stats
        return mean_loss, stats
    if not ys:
        raise ValueError('No selected evaluation labels')
    if save_path:
        with open(save_path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    ys, ps = np.asarray(ys), np.asarray(ps)
    auc = metrics.roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float('nan')
    stats.update(count=len(ys), logloss=metrics.log_loss(ys, ps, labels=[0, 1]),
                 brier=float(np.mean((ys-ps)**2)), accuracy=metrics.accuracy_score(ys, ps >= .5), auc=auc)
    strata = {}
    for kind in ('history_length', 'kc_gap'):
        for low, high in ((0, 200), (200, 400), (400, 800), (800, float('inf'))):
            mask = np.array([row[kind] is not None and low <= row[kind] < high for row in rows])
            if mask.any():
                strata[kind+'_'+str(low)+'_'+str(high)] = dict(
                    count=int(mask.sum()), logloss=metrics.log_loss(ys[mask], ps[mask], labels=[0, 1]),
                    accuracy=metrics.accuracy_score(ys[mask], ps[mask] >= .5),
                    auc=metrics.roc_auc_score(ys[mask], ps[mask]) if len(np.unique(ys[mask])) > 1 else None)
    stats['strata'] = strata
    model.last_eval_diagnostics = stats
    return auc, stats['accuracy'], stats


def train_long_mem_topk_akt(model, loader, optimizer, **kwargs):
    return run_epoch(model, loader, optimizer=optimizer, **kwargs)[0]


def evaluate_long_mem_topk_akt(model, loader, save_path=''):
    return run_epoch(model, loader, save_path=save_path)
