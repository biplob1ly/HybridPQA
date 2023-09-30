import logging
import pickle
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger()


class DenseIndexer(object):

    def __init__(self, buffer_size: int = 500):
        self.buffer_size = buffer_size
        self.index = None

    def index_data(self, ctx_vectors: np.array, ctx_ids: np.array):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_ctx_count: int) -> Tuple[np.array, np.array]:
        raise NotImplementedError


class DenseFlatIndexer(DenseIndexer):

    def __init__(self, vector_sz: int, buffer_size: int = 500):
        super(DenseFlatIndexer, self).__init__(buffer_size=buffer_size)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vector_sz))

    def index_data(self, ctx_vectors: np.array, ctx_ids: np.array):
        ctx_vectors = ctx_vectors.astype('float32')
        ctx_ids = ctx_ids.astype('int64')
        n = len(ctx_vectors)
        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            vectors = ctx_vectors[i:i+self.buffer_size]
            ids = ctx_ids[i:i+self.buffer_size]
            self.index.add_with_ids(vectors, ids)

    def search_knn(self, query_vectors: np.array, top_ctx_count: int) -> Tuple[np.array, np.array]:
        query_vectors = query_vectors.astype('float32')
        scores_arr, ctx_ids_arr = self.index.search(query_vectors, top_ctx_count)
        return scores_arr, ctx_ids_arr
