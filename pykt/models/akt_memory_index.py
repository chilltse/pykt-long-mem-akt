"""Student/layer-local raw inner-product search. No normalized/cosine keys."""
import time
import warnings
import numpy as np
import torch


def heads(projection, features, h, d):
    return projection(features).reshape(-1, h, d).transpose(0, 1)


class ExactIPIndex:
    """Current-parameter, chunked exhaustive search; O(Q*M*d), not ANN."""
    @staticmethod
    @torch.no_grad()
    def search(query, records, attention, k, memory_chunk=1024):
        h, length, _ = query.shape
        k = min(k, len(records))
        indices = torch.empty(h, length, 0, dtype=torch.long, device=query.device)
        scores = torch.empty(h, length, 0, device=query.device)
        if not k:
            return indices
        for start in range(0, len(records), memory_chunk):
            feature = records.key[start:start+memory_chunk].to(query.device, dtype=query.dtype)
            keys = heads(attention.k_linear, feature, h, attention.d_k)
            with torch.autocast(device_type=query.device.type, enabled=False):
                similarity = query.float() @ keys.float().transpose(-1, -2)
            ids = torch.arange(start, start+keys.size(1), device=query.device)
            ids = ids[None, None].expand(h, length, -1)
            scores = torch.cat((scores, similarity), -1)
            indices = torch.cat((indices, ids), -1)
            scores, chosen = scores.topk(min(k, scores.size(-1)), -1)
            indices = indices.gather(-1, chosen)
        return indices


class HNSWIPIndex:
    """One Faiss graph per head, stable position IDs, bounded tombstones.

    Mutated only at commit, after all queries have read the block snapshot.
    Missing Faiss is an explicit, warned exact fallback (never mislabeled ANN).
    """
    def __init__(self, h, d, connections=16, ef_construction=100, ef_search=64):
        self.h, self.d = h, d
        self.connections, self.ef_construction, self.ef_search = connections, ef_construction, ef_search
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            self.faiss = None
            warnings.warn("faiss-cpu is unavailable: hnsw_ip uses exact fallback; install requirements-memory.txt")
        self.graphs = []
        self.ids = np.empty(0, dtype=np.int64)
        self.keys = np.empty((h, 0, d), dtype=np.float32)
        self.versions = np.empty(0, dtype=np.int64)
        self.index_version = -1
        self.refreshes = 0
        self.fallbacks = 0
        self.searches = 0
        self.update_seconds = 0.
        self.transfer_seconds = 0.
        self.search_seconds = 0.

    @torch.no_grad()
    def update(self, records, attention, version, refresh=False):
        started = time.perf_counter()
        active = set(records.pos.tolist())
        dead = sum(int(i) not in active for i in self.ids)
        rebuild = refresh or not self.graphs or (len(self.ids) and dead / len(self.ids) > .2)
        if self.faiss is None:
            self.update_seconds += time.perf_counter()-started
            return
        if rebuild:
            self.graphs = [self.faiss.IndexHNSWFlat(self.d, self.connections,
                                                  self.faiss.METRIC_INNER_PRODUCT) for _ in range(self.h)]
            for graph in self.graphs:
                graph.hnsw.efConstruction = self.ef_construction
                graph.hnsw.efSearch = self.ef_search
            self.ids = np.empty(0, dtype=np.int64)
            self.keys = np.empty((self.h, 0, self.d), dtype=np.float32)
            self.versions = np.empty(0, dtype=np.int64)
            self.index_version = version
            self.refreshes += 1
        existing = set(self.ids.tolist())
        new = [i for i, pos in enumerate(records.pos.tolist()) if pos not in existing]
        if new:
            device = attention.k_linear.weight.device
            transfer = time.perf_counter()
            feature = records.key[new].to(device, dtype=attention.k_linear.weight.dtype)
            keys = heads(attention.k_linear, feature, self.h, self.d).float().cpu().numpy()
            self.transfer_seconds += time.perf_counter()-transfer
            for head, graph in enumerate(self.graphs):
                graph.add(np.ascontiguousarray(keys[head]))
            self.keys = np.concatenate((self.keys, keys), 1)
            self.ids = np.concatenate((self.ids, records.pos[new].numpy()))
            self.versions = np.concatenate((self.versions, np.full(len(new), version, dtype=np.int64)))
        self.update_seconds += time.perf_counter()-started

    @torch.no_grad()
    def search(self, query, records, attention, k, memory_chunk=1024):
        self.searches += 1
        k = min(k, len(records))
        if not k or self.faiss is None or not self.graphs:
            self.fallbacks += int(k > 0)
            return ExactIPIndex.search(query, records, attention, k, memory_chunk)
        started = time.perf_counter()
        q = query.detach().float().cpu().numpy()
        self.transfer_seconds += time.perf_counter()-started
        started = time.perf_counter()
        active = {int(pos): i for i, pos in enumerate(records.pos.tolist())}
        result = np.full((self.h, query.size(1), k), -1, dtype=np.int64)
        # Batched queries, over-recall to tolerate tombstones. Retry before fallback.
        count = min(len(self.ids), max(k*2, k+16))
        while True:
            for head, graph in enumerate(self.graphs):
                graph.hnsw.efSearch = max(self.ef_search, 2*count)
                _, neighbors = graph.search(np.ascontiguousarray(q[head]), count)
                for row, candidates in enumerate(neighbors):
                    selected = []
                    for idx in candidates:
                        if idx < 0:
                            continue
                        if idx >= len(self.ids):
                            raise ValueError("HNSW returned an invalid internal ID")
                        pos = int(self.ids[idx])
                        if pos in active and active[pos] not in selected:
                            selected.append(active[pos])
                        if len(selected) == k:
                            break
                    result[head, row, :len(selected)] = selected
            if (result >= 0).all() or count == len(self.ids):
                break
            count = min(len(self.ids), count*2)
        self.search_seconds += time.perf_counter()-started
        if (result < 0).any():
            self.fallbacks += 1
            return ExactIPIndex.search(query, records, attention, k, memory_chunk)
        return torch.from_numpy(result).to(query.device)

    def bytes(self):
        graph_bytes = sum(len(self.faiss.serialize_index(g)) for g in self.graphs) if self.faiss else 0
        return graph_bytes + self.ids.nbytes + self.keys.nbytes + self.versions.nbytes
