"""Reusable prepared student-sequence loader and chronological block replay."""
import json
import random
from pathlib import Path
from collections import deque

import torch
from torch.utils.data import IterableDataset, DataLoader


def encode_student(events, encoding):
    attrs, pids, responses = [], [], []
    for values, pid, response in events:
        attrs.append([2 if value is None or value == "" else mapping.get(value, 1)
                      for mapping, value in zip(encoding["fields"], values)])
        pids.append(encoding["questions"].get(pid, 0))
        responses.append(response)
    return {"attrs": torch.tensor(attrs, dtype=torch.long),
            "pid": torch.tensor(pids, dtype=torch.long),
            "response": torch.tensor(responses, dtype=torch.long)}


class StudentBlockCache:
    """Bounded raw-event replay, owned by exactly one student and one pass.

    Recompute AKT representations under current weights: detached KV reuse would
    change its distance-based attention and multilayer context semantics.
    """
    def __init__(self, uid, cache_blocks):
        if cache_blocks < 0:
            raise ValueError("cache_blocks must be nonnegative")
        self.uid = uid
        self.blocks = deque(maxlen=cache_blocks)

    def prepend(self, uid, block):
        if uid != self.uid:
            raise ValueError("Cannot share history between students")
        offset = sum(len(item["response"]) for item in self.blocks)
        joined = {key: torch.cat([item[key] for item in self.blocks] + [value])
                  for key, value in block.items()}
        return joined, offset

    def append(self, uid, block):
        if uid != self.uid:
            raise ValueError("Cannot share history between students")
        self.blocks.append({key: value.detach().cpu().clone() for key, value in block.items()})


def iter_student_blocks(uid, events, block_size, cache_blocks):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    cache = StudentBlockCache(uid, cache_blocks)
    for start in range(0, len(events["response"]), block_size):
        block = {key: value[start:start + block_size] for key, value in events.items()}
        yield cache.prepend(uid, block)
        cache.append(uid, block)


class StudentBlockDataset(IterableDataset):
    """One chronological block per yield; shuffle students only, never events.

    batch_size=None and num_workers=0 keep each student's stream ordered. Each
    iteration constructs new caches, so epochs and splits cannot share history.
    """
    def __init__(self, students, ids, block_size, cache_blocks, shuffle=False, seed=0):
        super().__init__()
        self.students = students
        self.ids = list(ids)
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("StudentBlockDataset currently requires num_workers=0")
        order = list(self.ids)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
            self.epoch += 1
        for uid in order:
            for block, offset in iter_student_blocks(uid, self.students[uid],
                                                     self.block_size, self.cache_blocks):
                selected = torch.zeros_like(block["response"], dtype=torch.bool)
                selected[offset:] = True
                yield dict(block, loss_mask=selected, uid=uid)


def init_student_loaders(data_config, fold, block_size=128, cache_blocks=1,
                         seq_len=128, seed=3407):
    """Load prepared artifacts only. Never fit encodings or read source records."""
    if data_config.get("dataset_format") != "student_fields_v1":
        raise ValueError("Run data_preprocess.py --dataset_name ... --model_name yousician_akt first")
    if fold not in data_config["folds"]:
        raise ValueError(f"Unknown fold: {fold}")
    if block_size < 1 or cache_blocks < 0 or (seq_len is not None and
            (seq_len < 1 or block_size * cache_blocks < seq_len)):
        raise ValueError("Cache capacity must cover positive seq_len")
    root = Path(data_config["dpath"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / "examples" / root
    artifact = json.loads((root / data_config["fold_file"].format(fold=fold)).read_text())
    sequences = json.loads((root / data_config["student_file"]).read_text())
    encoding, splits = artifact["encoding"], artifact["splits"]
    groups = [set(splits[key]) for key in ("train", "valid", "test")]
    if any(not group for group in groups) or any(groups[i] & groups[j]
            for i in range(3) for j in range(i)):
        raise ValueError("Prepared student splits must be nonempty and disjoint")
    students = {uid: encode_student(events, encoding) for uid, events in sequences.items()}
    data_config.update(field_sizes=[len(v) + 3 for v in encoding["fields"]],
                       num_q=len(encoding["questions"]), selected_fold=fold,
                       encoding=encoding, splits=splits, block_size=block_size,
                       cache_blocks=cache_blocks, seq_len=seq_len)
    return tuple(DataLoader(StudentBlockDataset(students, splits[key], block_size,
                     cache_blocks, shuffle=key == "train", seed=seed), batch_size=None,
                     num_workers=0) for key in ("train", "valid", "test"))
