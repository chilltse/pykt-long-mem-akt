"""Ordered student-level training and evaluation for :mod:`long_mem_akt`."""
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


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
    model.train()
    losses = []
    writer_norms, encoder_norms = [], []
    device = next(model.parameters()).device
    for raw in loader:
        student = _student(raw, device)
        memory = model.initial_memory(1, device)
        pending = None
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
                                                  block["rseqs"], block["valid"], memory)
                select = _select(model, block, absolute_start)
                count = select.sum()
                block_loss = F.binary_cross_entropy_with_logits(cache["logits"][select], block["rseqs"][select], reduction="sum")
                total = total + block_loss + cache["reg_loss"]
                nlabels = nlabels + count
                summary = model.summarize(block["rseqs"], block["valid"], block["tseqs"], cache)
                if pos == len(group) - 1:
                    next_pending = {"memory": memory, "summary": summary}
                else:
                    memory = model.write(memory, summary)
            optimizer.zero_grad(set_to_none=True)
            if nlabels:
                loss = total / nlabels
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
    gates, slot_counts, changes = [], [], []
    errors = {"block_start": [], "other": [], "prior_block_same_kc": []}
    events, timestamp_blocks, event_distance_blocks = 0, 0, 0
    for raw in loader:
        uid = str(raw["uid"][0]) if isinstance(raw["uid"], (list, tuple)) else str(raw["uid"])
        if student_id is not None and uid != str(student_id):
            continue
        student = _student(raw, device)
        memory = model.initial_memory(1, device)
        previous_concepts = set()
        for absolute_start, block in _blocks(student, model.block_size):
            pred, cache = model.predict_block(block["cseqs"].long(), block["qseqs"].long(),
                                              block["rseqs"], block["valid"], memory)
            select = _select(model, block, absolute_start)
            ys.extend(block["rseqs"][select].cpu().tolist())
            ps.extend(pred[select].cpu().tolist())
            events += int(block["valid"].sum())
            gates.extend(cache["gate"][block["valid"]].flatten().cpu().tolist())
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
            summary = model.summarize(block["rseqs"], block["valid"], block["tseqs"], cache)
            timestamp_blocks += int(summary["uses_timestamps"].sum())
            event_distance_blocks += int((~summary["uses_timestamps"] & summary["has_valid"]).sum())
            slot_counts.append(summary["raw_count"].sum(0).cpu().numpy())
            new_memory = model.write(memory, summary)
            changes.append(float((new_memory - memory).norm()))
            memory = new_memory
    ys, ps = np.asarray(ys), np.asarray(ps)
    if save_path:
        with open(save_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["uid", "block", "position", "target", "probability", "scored", "protocol"])
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
                   "gate_quantiles": np.quantile(gates, [0, .25, .5, .75, 1]).tolist(),
                   "slot_access_fraction": (counts/counts.sum()).tolist(),
                   "memory_update_norm": float(np.mean(changes)),
                   "time_weight_enabled": model.time_weight,
                   "timestamp_blocks": timestamp_blocks, "event_distance_blocks": event_distance_blocks}
    for subset, values in errors.items():
        diagnostics[subset+"_count"] = len(values)
        diagnostics[subset+"_brier"] = float(np.mean(values)) if values else None
    model.last_eval_diagnostics = diagnostics
    return auc, metrics.accuracy_score(ys, ps >= .5), diagnostics
