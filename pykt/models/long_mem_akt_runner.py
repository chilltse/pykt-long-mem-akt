"""Ordered student-level training and evaluation for :mod:`long_mem_akt`."""
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


RESIDUAL_COLUMNS = ["base_probability", "residual_logit", "has_history",
                    "residual_applied", "fallback_reason"]
FALLBACK_REASONS = ["none", "no_history", "nonfinite"]


def _uid(raw):
    return str(raw["uid"][0]) if isinstance(raw["uid"], (list, tuple)) else str(raw["uid"])


def _student(data, device):
    if data["rseqs"].ndim != 2 or data["rseqs"].size(0) != 1:
        raise ValueError("Student runner requires batch_size=1; never flatten different students together")
    return {k: v.to(device).reshape(1, -1) for k, v in data.items() if k != "uid"}


def _blocks(student, size):
    length = student["rseqs"].size(1)
    for start in range(0, length, size):
        end = min(start + size, length)
        block = {k: v[:, start:end] for k, v in student.items()}
        block["valid"] = block["rseqs"] >= 0
        yield start, block


def _select(model, block, absolute_start):
    if model.score_protocol == "legacy":
        if "smasks" not in block:
            raise ValueError("Legacy scoring requires dataset smasks, preserving original segment selection")
        return block["valid"] & block["smasks"].bool()
    selected = block["valid"].clone() & block.get("allowed", block["valid"]).bool()
    if absolute_start == 0:
        selected[:, 0] = False
    return selected


def _grad_norm(modules):
    return sum(float(p.grad.detach().square().sum())
               for module in modules for p in module.parameters() if p.grad is not None) ** .5


def train_long_mem_akt(model, loader, optimizer, grad_clip=5.0):
    """One epoch with bounded TBPTT and delayed boundary writes.

    The retained pending summary is detached only *after* its group has been
    optimized.  The next group calls ``write`` again, so that next group's
    prediction loss supervises writer parameters without reusing an old graph.
    """
    if model.phase == "inference":
        raise ValueError("Set stage to base or candidate before training")
    if model.phase == "candidate":
        model.set_stage("candidate")
    model.train()
    losses = []
    writer_norms, encoder_norms = [], []
    device = next(model.parameters()).device
    for raw in loader:
        student = _student(raw, device)
        memory = model.initial_memory(1, device)
        pending = None
        completed_valid_blocks = 0
        blocks = list(_blocks(student, model.block_size))
        for group_start in range(0, len(blocks), model.tbptt_blocks):
            group = blocks[group_start:group_start + model.tbptt_blocks]
            if pending is not None:
                memory = model.write(pending["memory"], pending["summary"])
                pending = None
            total, nlabels = 0., 0
            next_pending = None
            for pos, (absolute_start, block) in enumerate(group):
                pred, cache = model.predict_block(block["cseqs"].long(), block["qseqs"].long(),
                                                  block["rseqs"], block["valid"], memory,
                                                  has_history=completed_valid_blocks > 0,
                                                  uid=_uid(raw), block_index=absolute_start//model.block_size)
                select = _select(model, block, absolute_start)
                count = select.sum()
                block_loss = F.binary_cross_entropy_with_logits(cache["logits"][select], block["rseqs"][select], reduction="sum")
                total = total + block_loss + cache["reg_loss"]
                nlabels = nlabels + count
                summary = model.summarize(block["rseqs"], block["valid"], block["tseqs"], cache)
                completed_valid_blocks += int(block["valid"].any())
                if pos == len(group) - 1:
                    next_pending = {"memory": memory, "summary": summary}
                else:
                    memory = model.write(memory, summary)
            optimizer.zero_grad(set_to_none=True)
            if nlabels:
                loss = total / nlabels
                if loss.requires_grad:
                    loss.backward()
                    writer_norms.append(_grad_norm([model.erase, model.add]))
                    encoder_norms.append(_grad_norm([model.write_encoder, model.write_z]))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                losses.append(float(loss.detach().cpu()))
            # Truncate only between groups; next group replays its boundary write.
            pending = {"memory": next_pending["memory"].detach().clone(),
                       "summary": {k: (v.detach() if torch.is_tensor(v) else v)
                                   for k, v in next_pending["summary"].items()}}
            memory = pending["memory"]
    model.last_train_diagnostics = {
        "writer_grad_norm": float(np.mean(writer_norms)) if writer_norms else 0.,
        "write_encoder_grad_norm": float(np.mean(encoder_norms)) if encoder_norms else 0.}
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_long_mem_akt(model, loader, save_path="", student_id=None):
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    ys, ps, records = [], [], []
    residual = model.phase != "base"
    bases, alphas, reasons = [], [], []
    slot_counts, changes = [], []
    errors = {"block_start": [], "other": [], "prior_block_same_kc": []}
    events, timestamp_blocks, event_distance_blocks = 0, 0, 0
    for raw in loader:
        uid = _uid(raw)
        if student_id is not None and uid != str(student_id):
            continue
        student = _student(raw, device)
        memory = model.initial_memory(1, device)
        previous_concepts = set()
        completed_valid_blocks = 0
        for absolute_start, block in _blocks(student, model.block_size):
            pred, cache = model.predict_block(block["cseqs"].long(), block["qseqs"].long(),
                                              block["rseqs"], block["valid"], memory,
                                              has_history=completed_valid_blocks > 0,
                                              uid=uid, block_index=absolute_start//model.block_size)
            select = _select(model, block, absolute_start)
            ys.extend(block["rseqs"][select].cpu().tolist())
            ps.extend(pred[select].cpu().tolist())
            events += int(block["valid"].sum())
            if residual:
                bases.extend(cache["base_probability"][select].cpu().tolist())
                alphas.extend(cache["residual_applied"][select].cpu().tolist())
                reasons.extend(cache["fallback_reason"][select].cpu().tolist())
            seen_in_block = set()
            for offset in range(pred.size(1)):
                concept = int(block["cseqs"][0, offset])
                if bool(select[0, offset]):
                    error = float((pred[0, offset] - block["rseqs"][0, offset]).square())
                    errors["block_start" if offset == 0 else "other"].append(error)
                    if concept in previous_concepts and concept not in seen_in_block:
                        errors["prior_block_same_kc"].append(error)
                if bool(block["valid"][0, offset]):
                    seen_in_block.add(concept)
            previous_concepts.update(seen_in_block)
            if save_path:
                for offset in range(pred.size(1)):
                    records.append([uid, absolute_start//model.block_size, absolute_start+offset,
                                    float(block["rseqs"][0, offset]), float(pred[0, offset]),
                                    bool(select[0, offset]), model.score_protocol])
                    if residual:
                        for key in RESIDUAL_COLUMNS:
                            value = cache[key][0, offset].item()
                            if key == "fallback_reason":
                                value = FALLBACK_REASONS[value]
                            elif isinstance(value, float) and not np.isfinite(value):
                                value = ""
                            records[-1].append(value)
            summary = model.summarize(block["rseqs"], block["valid"], block["tseqs"], cache)
            timestamp_blocks += int(summary["uses_timestamps"].sum())
            event_distance_blocks += int((~summary["uses_timestamps"] & summary["has_valid"]).sum())
            slot_counts.append(summary["raw_count"].sum(0).cpu().numpy())
            new_memory = model.write(memory, summary)
            changes.append(float((new_memory - memory).norm()))
            memory = new_memory
            completed_valid_blocks += int(block["valid"].any())
    ys, ps = np.asarray(ys), np.asarray(ps)
    if save_path:
        with open(save_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["uid", "block", "position", "target", "probability", "scored", "protocol"]
                            + (RESIDUAL_COLUMNS if residual else []))
            writer.writerows(records)
    if not len(ys):
        raise ValueError("No selected labels to evaluate (check student_id / scoring masks)")
    auc = metrics.roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    counts = np.sum(slot_counts, axis=0)
    diagnostics = {"protocol": model.score_protocol, "count": len(ys),
                   "logloss": metrics.log_loss(ys, ps, labels=[0, 1]),
                   "brier": float(np.mean((ys-ps)**2)),
                   "interactions_per_second": events / (time.perf_counter()-started),
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   "slot_access_fraction": (counts/counts.sum()).tolist(),
                   "memory_update_norm": float(np.mean(changes)),
                   "time_weight_enabled": model.time_weight,
                   "timestamp_blocks": timestamp_blocks, "event_distance_blocks": event_distance_blocks}
    if residual:
        bases, alphas, reasons = np.asarray(bases), np.asarray(alphas), np.asarray(reasons)
        active = alphas > 0
        difference = (ys-ps)**2 - (ys-bases)**2
        subset_logloss = (metrics.log_loss(ys[active], ps[active], labels=[0, 1])
                          - metrics.log_loss(ys[active], bases[active], labels=[0, 1])) if active.any() else None
        diagnostics.update(base_brier=float(np.mean((ys-bases)**2)),
                           base_logloss=metrics.log_loss(ys, bases, labels=[0, 1]),
                           brier_change=float(difference.mean()),
                           logloss_change=float(diagnostics["logloss"]-metrics.log_loss(ys, bases, labels=[0, 1])),
                           intervention_brier_change=float(difference[active].mean()) if active.any() else None,
                           intervention_logloss_change=subset_logloss,
                           intervention_rate=float(active.mean()), residual_fraction=float(alphas.mean()),
                           mean_probability_change=float(np.abs(ps-bases).mean()),
                           changed_probability_count=int((ps != bases).sum()),
                           numerical_fallback_count=int((reasons == 2).sum()),
                           wrong_to_right=int((((bases >= .5) != ys) & ((ps >= .5) == ys)).sum()),
                           right_to_wrong=int((((bases >= .5) == ys) & ((ps >= .5) != ys)).sum()))
    for subset, values in errors.items():
        diagnostics[subset+"_count"] = len(values)
        diagnostics[subset+"_brier"] = float(np.mean(values)) if values else None
    model.last_eval_diagnostics = diagnostics
    return auc, metrics.accuracy_score(ys, ps >= .5), diagnostics
