"""External, detached, student-isolated memory with read/commit transactions."""
from dataclasses import dataclass
import torch
from .akt_memory_index import HNSWIPIndex


@dataclass(frozen=True)
class Records:
    pos: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    version: torch.Tensor

    def __len__(self):
        return self.pos.numel()

    def take(self, indices):
        return Records(*(getattr(self, name)[indices].clone() for name in ('pos', 'key', 'value', 'version')))

    @staticmethod
    def join(left, right):
        if left is None:
            return right
        return Records(*(torch.cat((getattr(left, name), getattr(right, name)))
                         for name in ('pos', 'key', 'value', 'version')))

    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.pos, self.key, self.value, self.version))


class MemoryBank:
    def __init__(self, model):
        self.model = model
        self.students = {}

    def clear(self, student_id=None):
        if student_id is None:
            self.students.clear()
        else:
            self.students.pop(str(student_id), None)

    def snapshot(self, student_ids):
        ids = [str(uid) for uid in student_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("A batch cannot contain dependent blocks from the same student")
        result = []
        for uid in ids:
            state = self.students.get(uid, dict(block_id=0, last_pos=-1, layers={}))
            result.append(dict(student_id=uid, block_id=state['block_id'], last_pos=state['last_pos'],
                               layers={name: dict(layer) for name, layer in state['layers'].items()}))
        return result

    def commit(self, pending, parameter_version=0):
        ids = [p['student_id'] for p in pending]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate student transaction")
        # Validate the entire batch before modifying any student's state.
        for item in pending:
            state = self.students.get(item['student_id'], dict(block_id=0, last_pos=-1))
            pos = item['pos']
            if item['block_id'] != state['block_id']:
                raise ValueError("Out-of-order block or duplicate commit")
            if len(pos) and (int(pos[0]) <= state['last_pos'] or not bool((pos[1:] > pos[:-1]).all())):
                raise ValueError("Position rollback or record ID conflict")
            expected = set(self.model.layer_names()) if self.model.memory_mode != 'off' else set()
            if set(item['layers']) != expected:
                raise ValueError("Incomplete layer transaction")
            for record in item['layers'].values():
                if not torch.equal(record.pos, pos) or record.key.grad_fn or record.value.grad_fn or record.key.requires_grad or record.value.requires_grad:
                    raise ValueError("Layer records must be aligned and detached")
        for item in pending:
            uid = item['student_id']
            state = self.students.get(uid, dict(block_id=0, last_pos=-1, layers={}))
            layers = {}
            for name, incoming in item['layers'].items():
                previous = state['layers'].get(name, {})
                merged = Records.join(previous.get('recent'), incoming)
                r = self.model.recent_capacity
                split = max(0, len(merged)-r)
                recent = merged.take(slice(split, None))
                layer = dict(recent=recent)
                if self.model.is_retriever(name) and self.model.memory_mode in ('retrieval', 'dense_oracle'):
                    remote = Records.join(previous.get('remote'), merged.take(slice(0, split)))
                    remote = remote.take(slice(max(0, len(remote)-self.model.remote_capacity), None))
                    layer['remote'] = remote
                    if self.model.retrieval_backend == 'hnsw_ip' and self.model.memory_mode == 'retrieval':
                        attention = self.model.attention_layer(name)
                        index = previous.get('index') or HNSWIPIndex(
                            attention.h, attention.d_k, self.model.hnsw_connections,
                            self.model.hnsw_ef_construction, self.model.hnsw_ef_search)
                        index.update(remote, attention, parameter_version,
                                     refresh=(state['block_id']+1) % self.model.index_refresh_blocks == 0)
                        layer['index'] = index
                layers[name] = layer
            self.students[uid] = dict(block_id=state['block_id']+1,
                                     last_pos=int(item['pos'][-1]) if len(item['pos']) else state['last_pos'],
                                     layers=layers)

    def stats(self):
        values = dict(bank_bytes=0, index_bytes=0, refreshes=0, fallbacks=0, searches=0,
                      index_update_seconds=0., index_transfer_seconds=0., index_search_seconds=0.)
        for state in self.students.values():
            for layer in state['layers'].values():
                for name in ('recent', 'remote'):
                    if name in layer:
                        values['bank_bytes'] += layer[name].bytes()
                index = layer.get('index')
                if index:
                    values['index_bytes'] += index.bytes()
                    for name in ('refreshes', 'fallbacks', 'searches'):
                        values[name] += getattr(index, name)
                    for name in ('update', 'transfer', 'search'):
                        values['index_'+name+'_seconds'] += getattr(index, name+'_seconds')
        return values
