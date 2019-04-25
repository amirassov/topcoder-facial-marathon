import nmslib
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class NMSLibNeighbours:
    def __init__(self, n_neighbours, space, n_jobs):
        self.n_neighbours = n_neighbours
        self.index = nmslib.init(method='hnsw', space=space, data_type=nmslib.DataType.DENSE_VECTOR)
        self.space = space
        self.n_jobs = n_jobs

    def fit(self, paths):
        for path in tqdm(paths):
            self.fit_sample(path)
        self.index.createIndex({'post': 2, 'indexThreadQty': self.n_jobs}, print_progress=True)

    def fit_sample(self, path):
        data = np.load(path)
        labels = data['labels']
        embeddings = data['embeddings']
        if len(embeddings):
            self.index.addDataPointBatch(data=embeddings, ids=labels)

    def predict(self, paths):
        predictions = defaultdict(list)
        for path in tqdm(paths):
            data = np.load(path)
            for i, (neighbours, distances) in enumerate(self.predict_sample(data)):
                predictions['neighbours'].append(neighbours)
                predictions['distances'].append(distances)
                predictions['labels'].append(data['labels'][i])
                predictions['ids'].append(data['ids'][i])
        return predictions

    def predict_sample(self, data: np.array):
        if len(data['embeddings']):
            return self.index.knnQueryBatch(data['embeddings'], k=self.n_neighbours, num_threads=self.n_jobs)
        else:
            return []

    def dump(self, index_path):
        self.index.saveIndex(index_path)

    def load(self, index_path):
        self.index.loadIndex(index_path)
