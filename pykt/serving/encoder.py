"""The training vocabulary is immutable; raw JSON scalar types are preserved."""
import copy
import hashlib
import json
from dataclasses import asdict, dataclass

from pykt.preprocess.vector_kc_preprocess import category


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True,
                      allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class EncodedQuestion:
    attrs: list
    pid: int
    unknown_fields: list
    unknown_question: bool

    def as_dict(self):
        return asdict(self)


class KCEncoder:
    def __init__(self, encoding, field_sizes, n_pid):
        self.encoding = copy.deepcopy(encoding)
        names = encoding.get('field_names')
        fields = encoding.get('fields')
        questions = encoding.get('questions')
        if (not isinstance(names, list) or not names or
                any(not isinstance(n, str) or not n for n in names) or
                len(set(names)) != len(names)):
            raise ValueError('encoding.field_names must contain unique nonempty strings')
        if not isinstance(fields, list) or len(fields) != len(names) or len(fields) != len(field_sizes):
            raise ValueError('Encoding width does not match model')
        for mapping, size in zip(fields, field_sizes):
            if not isinstance(mapping, dict) or sorted(mapping.values()) != list(range(3, size)):
                raise ValueError('Field vocabulary IDs must match embedding range starting at 3')
            for token, index in mapping.items():
                if type(index) is not int or not isinstance(token, str) or category(json.loads(token)) != token:
                    raise ValueError('Invalid typed JSON scalar vocabulary token')
        if (not isinstance(questions, dict) or any(type(i) is not int for i in questions.values())
                or sorted(questions.values()) != list(range(1, n_pid + 1))):
            raise ValueError('Question vocabulary IDs do not match model')
        for identity in questions:
            tokens = json.loads(identity)
            if not isinstance(tokens, list) or len(tokens) != len(names):
                raise ValueError('Invalid question identity width')
            for token, mapping in zip(tokens, fields):
                if token is not None and (not isinstance(token, str) or token not in mapping):
                    raise ValueError('Question identity contains an unknown training token')
            if json.dumps(tokens, ensure_ascii=False, separators=(',', ':')) != identity:
                raise ValueError('Noncanonical question identity')
        self.field_names = list(names)
        self.version = fingerprint(self.encoding)

    def check_names(self, names):
        if names != self.field_names:
            raise ValueError('kc_feature_names must exactly match checkpoint field names and order')

    def encode(self, vector):
        if not isinstance(vector, list) or len(vector) != len(self.field_names):
            raise ValueError('KC vector width must match checkpoint field_names')
        # category() enforces finite JSON scalars, retaining bool/int/float distinctions.
        tokens = [category(v) for v in vector]
        identity = json.dumps(tokens, ensure_ascii=False, separators=(',', ':'))
        attrs = [2 if token is None else mapping.get(token, 1)
                 for token, mapping in zip(tokens, self.encoding['fields'])]
        pid = self.encoding['questions'].get(identity, 0)
        return EncodedQuestion(attrs, pid, [n for n, i in zip(self.field_names, attrs) if i == 1], pid == 0)
