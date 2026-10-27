"""Original AKT parameters/head plus detached recent caches and remote retrieval.

The inherited forward is the unchanged stateless AKT API. Stateful runners call
forward_with_memory and commit its detached records only after the full block.
"""
import torch
from .akt import AKT
from .akt_memory_bank import Records
from .akt_candidate_attention import candidate_attention, residual_memory_attention


class LongMemTopKAKT(AKT):
    def __init__(self, n_question, n_pid, d_model=128, n_blocks=2, dropout=.1,
                 memory_mode='retrieval', block_len=200, recent_capacity=200,
                 remote_capacity=8192, remote_topk=32, retrieval_backend='exact',
                 retrieval_query_chunk=32, retrieval_memory_chunk=1024,
                 index_refresh_blocks=8, freeze_backbone=False, memory_detach=True,
                 hnsw_connections=16, hnsw_ef_construction=100, hnsw_ef_search=64,
                 score_protocol='legacy', collect_diagnostics=False, diagnose_dense=False,
                 architecture_variant='legacy_all_layers', memory_decay_mode='legacy_mul',
                 training_stage='full', gate_lr=1e-3, **kwargs):
        if architecture_variant not in ('legacy_all_layers', 'v2_last_retriever_residual'):
            raise ValueError('Unknown architecture_variant')
        if memory_decay_mode not in ('legacy_mul', 'log_additive'):
            raise ValueError('Unknown memory_decay_mode')
        if architecture_variant == 'v2_last_retriever_residual' and retrieval_backend != 'exact':
            raise ValueError('V2 currently requires exact retrieval; ANN is deferred')
        self.architecture_variant = architecture_variant
        self.memory_decay_mode = memory_decay_mode
        self.training_stage = training_stage
        self.gate_lr = gate_lr
        if memory_mode not in ('off', 'recent', 'retrieval', 'dense_oracle'):
            raise ValueError('Unknown memory_mode')
        if retrieval_backend not in ('exact', 'hnsw_ip'):
            raise ValueError('Unknown retrieval_backend')
        if min(recent_capacity, remote_capacity, remote_topk) < 0 or min(
                block_len, retrieval_query_chunk, retrieval_memory_chunk, index_refresh_blocks,
                hnsw_connections, hnsw_ef_construction, hnsw_ef_search) < 1:
            raise ValueError('Invalid memory capacities/chunk/index sizes')
        if not memory_detach:
            raise ValueError('Memory requires detached historical features')
        if score_protocol not in ('legacy', 'student'):
            raise ValueError('Unknown score_protocol')
        emb_type = kwargs.get('emb_type', 'qid')
        if memory_mode != 'off' and emb_type not in ('qid', 'qid_pdiff'):
            raise ValueError('Memory attention supports only qid and qid_pdiff; not avgpool/linear')
        if d_model % kwargs.get('num_attn_heads', 8):
            raise ValueError('d_model must be divisible by num_attn_heads')
        super().__init__(n_question, n_pid, d_model, n_blocks, dropout, **kwargs)
        self.model_name = 'long_mem_topk_akt'
        for name in ('memory_mode', 'block_len', 'recent_capacity', 'remote_capacity', 'remote_topk',
                     'retrieval_backend', 'retrieval_query_chunk', 'retrieval_memory_chunk',
                     'index_refresh_blocks', 'hnsw_connections', 'hnsw_ef_construction',
                     'hnsw_ef_search', 'score_protocol', 'collect_diagnostics', 'diagnose_dense'):
            setattr(self, name, locals()[name])
        if architecture_variant == 'v2_last_retriever_residual':
            if emb_type not in ('qid', 'qid_pdiff') or n_blocks < 1:
                raise ValueError('V2 requires qid/qid_pdiff and at least one block')
            self.memory_alpha = torch.nn.Parameter(torch.zeros(kwargs.get('num_attn_heads', 8)))
        self.feature_version = 0
        self.freeze_backbone = freeze_backbone
        self.configure_training(training_stage)
        if freeze_backbone:
            self.requires_grad_(False)

    @property
    def last_retriever_name(self):
        return 'decoder.' + str(len(self.model.blocks_2)-1)

    def configure_training(self, stage):
        if stage not in ('gate_only', 'last_layer', 'full'):
            raise ValueError('Unknown training_stage')
        if self.architecture_variant != 'v2_last_retriever_residual' and stage != 'full':
            raise ValueError('Staged training requires V2')
        self.training_stage = stage
        self.requires_grad_(stage == 'full')
        if hasattr(self, 'memory_alpha'):
            self.memory_alpha.requires_grad_(True)
        if stage == 'last_layer':
            self.model.blocks_2[-1].requires_grad_(True)
            self.out.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.training_stage != 'full':
            self.model.eval()
            self.out.eval()
            if self.training_stage == 'last_layer':
                self.model.blocks_2[-1].train()
                self.out.train()
        return self

    def optimizer_groups(self, learning_rate):
        gate = getattr(self, 'memory_alpha', None)
        shared = [p for p in self.parameters() if p.requires_grad and p is not gate]
        groups = []
        if shared:
            groups.append(dict(params=shared, lr=learning_rate))
        if gate is not None and gate.requires_grad:
            groups.append(dict(params=[gate], lr=self.gate_lr))
        return groups

    def load_baseline_state_dict(self, state):
        expected = set(self.state_dict()) - ({'memory_alpha'} if hasattr(self, 'memory_alpha') else set())
        if set(state) != expected:
            raise RuntimeError('Baseline checkpoint keys differ: ' + str(set(state) ^ expected))
        merged = dict(state)
        if hasattr(self, 'memory_alpha'):
            merged['memory_alpha'] = torch.zeros_like(self.memory_alpha)
        return self.load_state_dict(merged, strict=True)

    def layer_names(self):
        if self.architecture_variant == 'v2_last_retriever_residual':
            return [self.last_retriever_name]
        return (['qa.'+str(i) for i in range(len(self.model.blocks_1))] +
                ['decoder.'+str(i) for i in range(len(self.model.blocks_2))])

    @staticmethod
    def is_retriever(name):
        return name.startswith('decoder.') and int(name.split('.')[1]) % 2 == 1

    def attention_layer(self, name):
        kind, index = name.split('.')
        blocks = self.model.blocks_1 if kind == 'qa' else self.model.blocks_2
        return blocks[int(index)].masked_attn_head

    def forward_with_memory(self, q_data, target, pid_data, abs_pos, valid_mask,
                            memory_snapshot, block_ids=None):
        if q_data.ndim != 2 or target.shape != q_data.shape or abs_pos.shape != q_data.shape or valid_mask.shape != q_data.shape:
            raise ValueError('Expected aligned [batch, block] input tensors')
        if abs_pos.dtype != torch.long or valid_mask.dtype != torch.bool:
            raise ValueError('abs_pos must be int64 and valid_mask must be bool')
        if len(memory_snapshot) != q_data.size(0) or q_data.size(1) > self.block_len:
            raise ValueError('Snapshot batch mismatch or block exceeds block_len')
        if len({s['student_id'] for s in memory_snapshot}) != len(memory_snapshot):
            raise ValueError('Duplicate student in batch')
        diagnostics = {key: 0. for key in (
            'projection_seconds', 'retrieval_seconds', 'attention_seconds', 'remote_mass_sum',
            'query_heads', 'remote_hits', 'candidate_age_sum', 'remote_candidates', 'feature_age_sum',
            'recall_sum', 'recall_rows', 'retained_mass_sum', 'dense_output_l2_sum', 'dense_rows',
            'rescore_abs_sum', 'index_age_sum', 'indexed_candidates')}
        predictions, pending = [], []
        regularization = self.out[-1].weight.sum()*0
        for row, snapshot in enumerate(memory_snapshot):
            selected = valid_mask[row].nonzero(as_tuple=True)[0]
            positions = abs_pos[row, selected]
            if block_ids is not None and int(block_ids[row]) != snapshot['block_id']:
                raise ValueError('Out-of-order block')
            if len(positions) and (int(positions[0]) <= snapshot['last_pos'] or not bool((positions[1:] > positions[:-1]).all())):
                raise ValueError('Absolute positions must strictly increase across blocks')
            for layer in snapshot['layers'].values():
                for kind in ('recent', 'remote'):
                    record = layer.get(kind)
                    if record is not None and len(record) and int(record.pos.max()) > snapshot['last_pos']:
                        raise ValueError('Snapshot includes uncommitted/future records')
            transaction = dict(student_id=snapshot['student_id'], block_id=snapshot['block_id'],
                               pos=positions.detach().cpu().clone(), layers={})
            padded = self.out[-1].weight.new_zeros(q_data.size(1))
            if not len(selected):
                # Empty records still advance the transaction block id.
                if self.memory_mode != 'off':
                    for name in self.layer_names():
                        empty = self.out[-1].weight.new_empty((0, self.model.d_model)).cpu()
                        transaction['layers'][name] = Records(transaction['pos'], empty, empty.clone(), torch.empty(0, dtype=torch.long))
                padded = padded + regularization * 0
                if hasattr(self, 'memory_alpha'):
                    padded = padded + self.memory_alpha.sum() * 0
                predictions.append(padded)
                pending.append(transaction)
                continue
            c, r = q_data[row:row+1, selected].long(), target[row:row+1, selected].long()
            pid = pid_data[row:row+1, selected].long() if pid_data is not None else None
            if self.memory_mode == 'off':
                pred, reg = super().forward(c, r, pid)
                regularization = regularization + reg
            else:
                qe, qa, pdiff, reg = self.encode_embeddings(c, r, pid)
                regularization = regularization + reg
                x, y = qe[0], qa[0]
                difficulty = pdiff[0] if pdiff is not None and 'pdiff' in self.emb_type else None

                def layer_forward(name, layer, query, key, value, strict, apply_pos):
                    if self.architecture_variant == 'v2_last_retriever_residual' and name != self.last_retriever_name:
                        return layer(mask=0 if strict else 1, query=query[None], key=key[None],
                                     values=value[None], apply_pos=apply_pos,
                                     pdiff=pdiff)[0]
                    transaction['layers'][name] = Records(
                        transaction['pos'], key.detach().cpu().clone(), value.detach().cpu().clone(),
                        torch.full((len(selected),), self.feature_version, dtype=torch.long))
                    if self.architecture_variant == 'v2_last_retriever_residual':
                        result = residual_memory_attention(layer.masked_attn_head, query, key, value, positions,
                                                           snapshot['layers'].get(name, {}), difficulty, self, diagnostics)
                    else:
                        result = candidate_attention(layer.masked_attn_head, query, key, value, positions,
                                                     snapshot['layers'].get(name, {}), strict, difficulty, self, diagnostics)
                    result = layer.layer_norm1(query + layer.dropout1(result))
                    if apply_pos:
                        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(result))))
                        result = layer.layer_norm2(result + layer.dropout2(ff))
                    return result

                for i, layer in enumerate(self.model.blocks_1):
                    y = layer_forward('qa.'+str(i), layer, y, y, y, False, True)
                for i, layer in enumerate(self.model.blocks_2):
                    strict = i % 2 == 1
                    x = layer_forward('decoder.'+str(i), layer, x, x, y if strict else x, strict, strict)
                pred = self.out(torch.cat((x, qe[0]), -1)).squeeze(-1).sigmoid()[None]
            predictions.append(padded.index_copy(0, selected, pred[0]))
            pending.append(transaction)
        self.last_memory_diagnostics = diagnostics
        return torch.stack(predictions), regularization, pending
