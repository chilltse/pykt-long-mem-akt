"""Training workflow helpers for the single LongMemAKT model.

Artifacts use the usual config.json/qid_model.ckpt layout plus a checked manifest.
All role selection is explicit; test students are consumed only by evaluate.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ..datasets.long_mem_akt_dataset import LongMemAKTDataset
from .long_mem_akt import LongMemAKT


VERSION = 3
ROLES = ("train", "valid", "test")


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
    if set(roles) != set(ROLES):
        raise ValueError("Expected only train, valid and test roles")
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
    if fold not in config["folds"]:
        raise ValueError("Validation fold is not in configured folds")
    roles = dict(train=pool,
                 valid=sorted(train.loc[train.fold == fold, "uid"].tolist()),
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


def save_artifact(directory, model, model_config, roles, stage, provenance=None):
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
                    provenance=provenance or {})
    write_json(directory / "long_mem_manifest.json", manifest)


def verify_artifact(directory, model_config=None):
    directory = Path(directory)
    if not (directory / "long_mem_manifest.json").is_file():
        raise ValueError("Old LongMemAKT checkpoints are unsupported; retrain with --stage train")
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "long_mem_manifest.json").read_text())
    roles = json.loads((directory / "roles.json").read_text())
    validate_roles(roles["roles"])
    if (manifest["version"] != VERSION or manifest["config_sha256"] != json_digest(config)
            or manifest["weights_sha256"] != file_digest(directory / "qid_model.ckpt")
            or manifest["roles_sha256"] != json_digest(roles)):
        raise ValueError("LongMem artifact/config/roles checksum mismatch")
    if model_config is not None and model_config != config["model_config"]:
        raise ValueError("Model configuration differs from saved artifact")
    return config, manifest, roles


def load_artifact(directory, device="cpu"):
    config, manifest, roles = verify_artifact(directory)
    dc = config["data_config"]
    model = LongMemAKT(dc["num_c"], dc["num_q"], **config["model_config"]).to(device)
    model.load_state_dict(torch.load(Path(directory) / "qid_model.ckpt", map_location=device))
    return model, config["model_config"], manifest, roles
