"""Field-batch adapters for pyKT's shared training and evaluation entry points."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def forward_field_batch(model, data):
    device = next(model.parameters()).device
    batch = {key: data[key].unsqueeze(0).to(device)
             for key in ("attrs", "response", "pid", "loss_mask")}
    selected = batch["loss_mask"] & (batch["response"] != -1)
    inputs = batch["pid"] if model.model_name == "akt" else batch["attrs"]
    pred, reg = model(inputs, batch["response"], batch["pid"], loss_mask=selected)
    return pred[selected], batch["response"][selected].float(), reg


def field_batch_loss(model, data):
    pred, labels, reg = forward_field_batch(model, data)
    return F.binary_cross_entropy(pred, labels) + reg


def evaluate_field_model(model, loader, save_path=""):
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for data in loader:
            pred, target, _ = forward_field_batch(model, data)
            labels.extend(target.cpu().tolist())
            predictions.extend(pred.cpu().tolist())
    if not labels:
        raise ValueError("No events in evaluation split")
    if save_path:
        Path(save_path).write_text(json.dumps({"labels": labels, "predictions": predictions}))
    return {"bce": F.binary_cross_entropy(torch.tensor(predictions), torch.tensor(labels)).item(),
            "events": len(labels),
            "auc": roc_auc_score(labels, predictions) if len(set(labels)) > 1 else -1.0,
            "accuracy": sum((p >= .5) == y for p, y in zip(predictions, labels)) / len(labels)}


def load_fmkc_checkpoint(path, device="cpu"):
    """Restore standard pyKT weights plus sibling config.json and encodings.

    Legacy bundled checkpoints remain readable for previous local experiments.
    """
    from .akt_fmkc import AKTFMKC
    path = Path(path)
    weights = torch.load(path, map_location=device, weights_only=False)
    if "model_state" in weights:
        metadata = weights
        config = metadata["model_config"]
        weights = metadata["model_state"]
    else:
        metadata = json.loads((path.parent / "config.json").read_text())
        config = dict(metadata["model_config"])
        config.update(field_sizes=metadata["data_config"]["field_sizes"],
                      n_pid=metadata["data_config"]["num_q"])
    if metadata.get("params", {}).get("model_name") == "akt":
        from .init_model import init_model
        model = init_model("akt", metadata["model_config"], metadata["data_config"], "qid").to(device)
    else:
        model = AKTFMKC(**config).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model, metadata
