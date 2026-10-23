"""Training workflow helpers for the single LongMemAKT model.

Artifacts use the usual config.json/qid_model.ckpt layout plus a checked manifest.
All role selection is explicit; test students are consumed only by evaluate.
"""
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn import metrics

from ..datasets.long_mem_akt_dataset import LongMemAKTDataset
from .long_mem_akt import LongMemAKT, smoothstep
from .long_mem_akt_runner import _student, _blocks, _select, _uid, evaluate_long_mem_akt


VERSION = 2
ROLES = ("fit", "dev", "selector_fit", "calibration", "test")


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    with open(path, "w") as handle:
        json.dump(clean(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def validate_roles(roles):
    seen = set()
    for role in ROLES:
        ids = roles[role]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Data roles must be nonempty, unique, student-disjoint: " + role)
        seen.update(ids)


def create_roles(data_config, fold, seed):
    config = copy.deepcopy(data_config)
    config["dpath"] = str(Path(config["dpath"]).resolve())
    train_path = Path(config["dpath"]) / config["train_valid_original_file"]
    test_path = Path(config["dpath"]) / config["test_original_file"]
    train = pd.read_csv(train_path, dtype={"uid": str})
    test = pd.read_csv(test_path, dtype={"uid": str})
    train = train[train.fold.isin(config["folds"])]
    test = test[test.fold == -1]
    if train.uid.duplicated().any() or test.uid.duplicated().any():
        raise ValueError("Expected unique original student rows")
    pool = sorted(train.loc[train.fold != fold, "uid"].tolist())
    np.random.RandomState(seed).shuffle(pool)
    # Retain the original held-out fold as dev. Reserve only from training folds.
    if len(pool) < 6:
        raise ValueError("Need >=6 training students for fit/selector_fit/calibration")
    n = max(2, int(len(pool)*.2))
    roles = dict(fit=pool[2*n:], selector_fit=pool[:n], calibration=pool[n:2*n],
                 dev=sorted(train.loc[train.fold == fold, "uid"].tolist()),
                 test=sorted(test.uid.tolist()))
    validate_roles(roles)
    return dict(version=VERSION, roles=roles, data_config=config, fold=fold, seed=seed,
                source_sha256={"train": file_digest(train_path), "test": file_digest(test_path)})


def role_loader(manifest, role):
    validate_roles(manifest["roles"])
    config = manifest["data_config"]
    is_test = role == "test"
    path = Path(config["dpath"]) / config["test_original_file" if is_test else "train_valid_original_file"]
    if file_digest(path) != manifest["source_sha256"]["test" if is_test else "train"]:
        raise ValueError("Source data changed since role assignment; create a new experiment")
    dataset = LongMemAKTDataset(str(path), config["input_type"], {-1} if is_test else set(config["folds"]),
                               config.get("maxlen", 200), config.get("min_seq_len", 3))
    wanted = set(manifest["roles"][role])
    indices = [i for i, row in enumerate(dataset.records) if row["uid"] in wanted]
    if len(indices) != len(wanted):
        raise ValueError("Role UID absent from its assigned source")
    return DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False)


def save_artifact(directory, model, model_config, roles, stage, provenance=None, calibration_report=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(model_config=model_config, data_config=roles["data_config"],
                  params=dict(model_name="long_mem_akt", emb_type="qid",
                              dataset_name=roles["data_config"]["dataset_name"], fold=roles["fold"]))
    torch.save(model.state_dict(), directory / "qid_model.ckpt")
    write_json(directory / "config.json", config)
    write_json(directory / "roles.json", roles)
    manifest = dict(version=VERSION, stage=stage, weights_sha256=file_digest(directory / "qid_model.ckpt"),
                    config_sha256=json_digest(config), roles_sha256=json_digest(roles),
                    provenance=provenance or {}, mc_scope="AKT nn.Dropout only",
                    seed_strategy="sha256(mc_seed:uid:block_index)",
                    calibration_report=calibration_report)
    write_json(directory / "selective_manifest.json", manifest)


def verify_artifact(directory, model_config=None):
    directory = Path(directory)
    if not (directory / "selective_manifest.json").is_file():
        raise ValueError("Old LongMemAKT checkpoints are unsupported; retrain with --stage train")
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "selective_manifest.json").read_text())
    roles = json.loads((directory / "roles.json").read_text())
    validate_roles(roles["roles"])
    if (manifest["version"] != VERSION or manifest["config_sha256"] != json_digest(config)
            or manifest["weights_sha256"] != file_digest(directory / "qid_model.ckpt")
            or manifest["roles_sha256"] != json_digest(roles)):
        raise ValueError("Selective artifact/config/roles checksum mismatch; recalibration required")
    if model_config is not None and model_config != config["model_config"]:
        raise ValueError("Model configuration differs from calibrated artifact")
    return config, manifest, roles


def load_artifact(directory, device="cpu"):
    config, manifest, roles = verify_artifact(directory)
    dc = config["data_config"]
    model = LongMemAKT(dc["num_c"], dc["num_q"], **config["model_config"]).to(device)
    model.load_state_dict(torch.load(Path(directory) / "qid_model.ckpt", map_location=device))
    if config["model_config"]["phase"] == "inference":
        if not bool(model.selector_ready) or not torch.isfinite(model.calibration).all():
            raise ValueError("Incomplete selector/calibration artifact")
        if manifest["calibration_report"] is not None:
            expected = model.calibration.new_tensor(manifest["calibration_report"]["E3"]["selected"]["thresholds"])
            if not torch.equal(model.calibration, expected):
                raise ValueError("Calibration thresholds differ between state and manifest")
    return model, config["model_config"], manifest, roles


@torch.no_grad()
def collect_predictions(model, loader):
    """CPU, graph-free, scored rows; labels never enter features or online selection."""
    model.eval()
    device = next(model.parameters()).device
    rows = {k: [] for k in ("features", "base_logits", "delta", "uncertainty", "history", "y")}
    ids = []
    for raw in loader:
        uid = _uid(raw)
        student = _student(raw, device)
        memory = model.initial_memory(1, device)
        completed = 0
        for start, block in _blocks(student, model.block_size):
            _, cache = model.predict_block(block["cseqs"], block["qseqs"], block["rseqs"],
                                           block["valid"], memory, has_history=completed > 0)
            u = model.mc_uncertainty(block["cseqs"], block["qseqs"], block["rseqs"], block["valid"],
                                     uid, start//model.block_size) if completed else torch.zeros_like(block["rseqs"])
            selected = _select(model, block, start)
            values = dict(features=cache["benefit_input"], base_logits=cache["base_logits"],
                          delta=cache["residual_logit"], uncertainty=u,
                          history=cache["has_history"], y=block["rseqs"])
            for key, value in values.items():
                rows[key].append(value[selected].detach().cpu())
            ids.extend((uid, start+offset) for offset in selected[0].nonzero(as_tuple=True)[0].tolist())
            summary = model.summarize(block["rseqs"], block["valid"], block["tseqs"], cache)
            memory = model.write(memory, summary)
            completed += int(block["valid"].any())
    if not ids:
        raise ValueError("No scored targets in this role")
    result = {k: torch.cat(v) for k, v in rows.items()}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite predictor data: stop fitting/calibration and inspect candidate")
    result["ids"] = ids
    return result


def fit_benefit(model, data, epochs=100, learning_rate=1e-3, seed=42):
    eligible = data["history"]
    uids = sorted({uid for (uid, _), use in zip(data["ids"], eligible.tolist()) if use})
    if len(uids) < 2:
        raise ValueError("Need >=2 selector_fit students with scored history for student-level early stopping")
    np.random.RandomState(seed).shuffle(uids)
    held_out = set(uids[:max(1, len(uids)//5)])
    dev = torch.tensor([uid in held_out for uid, _ in data["ids"]]) & eligible
    fit = eligible & ~dev
    target = (data["y"]-data["base_logits"].sigmoid()).square() - (
        data["y"]-(data["base_logits"]+data["delta"]).sigmoid()).square()
    model.set_stage("benefit")
    model.train()
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.benefit_head.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(data["features"][fit], target[fit]), batch_size=256,
                        shuffle=True, generator=generator)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        for features, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.benefit_head(features.to(device)).squeeze(-1), y.to(device))
            loss.backward()
            opt.step()
        with torch.no_grad():
            dev_loss = float(F.mse_loss(model.benefit_head(data["features"][dev].to(device)).squeeze(-1),
                                       target[dev].to(device)))
        history.append(dict(epoch=epoch+1, dev_mse=dev_loss))
        if dev_loss < best:
            best, best_state, stale = dev_loss, copy.deepcopy(model.benefit_head.state_dict()), 0
        else:
            stale += 1
        if stale >= 10:
            break
    if best_state is None:
        raise ValueError("Benefit training produced no finite validation loss")
    model.benefit_head.load_state_dict(best_state)
    model.selector_ready.fill_(True)
    model.set_stage("inference")
    return dict(history=history, fit_students=sorted(set(uids)-held_out), dev_students=sorted(held_out),
                target="base Brier - full candidate Brier", target_ids_sha256=json_digest(data["ids"]))


def probability_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(brier=float(np.mean((y-p)**2)), logloss=float(metrics.log_loss(y, p, labels=[0, 1])),
                auc=float(metrics.roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None)


@torch.no_grad()
def calibrate(model, data, logloss_tolerance=0., uncertainty_only=False):
    if not bool(model.selector_ready):
        raise ValueError("Train benefit before calibration")
    if logloss_tolerance < 0:
        raise ValueError("logloss_tolerance must be nonnegative")
    model.eval()
    device = next(model.parameters()).device
    gain = torch.cat([model.benefit_head(chunk.to(device)).squeeze(-1).cpu()
                      for chunk in data["features"].split(1024)])
    if not torch.isfinite(gain).all():
        raise ValueError("Nonfinite benefit output during calibration")
    u, h, z, delta, y = [data[k] for k in ("uncertainty", "history", "base_logits", "delta", "y")]
    baseline = probability_metrics(y.numpy(), z.sigmoid().numpy())
    # A variance of a [0,1] variable is <= .25: this is the no-intervention policy.
    policies = [(0.25, 1., 0., 1.)]
    if h.any():
        thresholds = np.quantile(u[h].numpy(), [.8, .9, .95])
        widths = [max(float(u[h].max())*.1, 1e-8), max(float(u[h].max())*.5, 1e-8)]
        policies += list(itertools.product(thresholds.tolist(), widths,
                                           [0.] if uncertainty_only else [0., .01], [.01, .05]))
    results, best = [], None
    for tu, wu, tg, wg in policies:
        alpha = smoothstep(u, tu, wu) * h
        if not uncertainty_only:
            alpha *= smoothstep(gain, tg, wg)
        p = (z + alpha*delta).sigmoid()
        result = dict(thresholds=[tu, wu, tg, wg], **probability_metrics(y.numpy(), p.numpy()),
                      intervention_rate=float((alpha > 0).float().mean()), mean_alpha=float(alpha.mean()))
        results.append(result)
        if result["logloss"] <= baseline["logloss"]+logloss_tolerance and (
                best is None or result["brier"] < best["brier"]):
            best = result
    if best is None:
        raise ValueError("No feasible policy (including fallback); check numeric inputs")
    model.set_calibration(*best["thresholds"], uncertainty_only=uncertainty_only)
    return dict(objective="brier", logloss_tolerance=logloss_tolerance, baseline=baseline,
                selected=best, policies=results, uncertainty_only=uncertainty_only,
                target_ids_sha256=json_digest(data["ids"]),
                note="Calibration selection is not an independent statistical guarantee")
