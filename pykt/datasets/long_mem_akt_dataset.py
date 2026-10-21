"""Student ordered data for LongMemAKT.

This reads the existing original split CSV. Generated sequence files may have
dropped short tails; the default KTDataset also discards their student IDs.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset


class LongMemAKTDataset(Dataset):
    def __init__(self, path, input_type, folds, legacy_maxlen=200, min_seq_len=3):
        df = pd.read_csv(path, dtype={"uid": str})
        df = df[df["fold"].isin(folds)].reset_index(drop=True)
        if df.uid.duplicated().any():
            raise ValueError("Expected one original ordered row per student, not pre-cut blocks")
        if "concepts" not in input_type:
            raise ValueError("Original AKT requires concept IDs; do not silently replace them with zero")
        self.legacy_maxlen, self.min_seq_len = legacy_maxlen, min_seq_len
        self.records = []
        for _, row in df.iterrows():
            responses = [int(x) for x in str(row.responses).split(",")]
            concepts = ([int(x) for x in str(row.concepts).split(",")]
                        if "concepts" in input_type else [0] * len(responses))
            questions = ([int(x) for x in str(row.questions).split(",")]
                         if "questions" in input_type and "questions" in row and str(row.questions) != "nan"
                         else [0] * len(responses))
            if "timestamps" in row and str(row.timestamps) != "nan":
                timestamps = [float(x) for x in str(row.timestamps).split(",")]
            else:
                timestamps = [float("nan")] * len(responses)
            if not (len(concepts) == len(questions) == len(timestamps) == len(responses)):
                raise ValueError("LongMemAKT requires aligned question/concept/response/timestamp fields")
            if not responses or any(r not in (0, 1) for r in responses) or min(concepts) < 0:
                raise ValueError("Expected an unpadded binary-response original student sequence")
            # Reproduce generate_sequences' score selection, independently of
            # the memory block size: each original segment drops its first label;
            # a too-short original tail contributes no labels.
            selected = torch.ones(len(responses), dtype=torch.bool)
            selected[::legacy_maxlen] = False
            tail = len(responses) % legacy_maxlen
            if 0 < tail < min_seq_len:
                selected[-tail:] = False
            if "selectmasks" in row and pd.notna(row.selectmasks):
                original_mask = torch.tensor([int(x) == 1 for x in str(row.selectmasks).split(",")])
                if len(original_mask) != len(responses):
                    raise ValueError("Misaligned selectmasks")
            else:
                original_mask = torch.ones(len(responses), dtype=torch.bool)
            self.records.append({"uid": str(row.uid), "cseqs": torch.tensor(concepts),
                                 "qseqs": torch.tensor(questions), "rseqs": torch.tensor(responses, dtype=torch.float),
                                 "tseqs": torch.tensor(timestamps, dtype=torch.float64),
                                 "smasks": selected & original_mask, "allowed": original_mask})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
