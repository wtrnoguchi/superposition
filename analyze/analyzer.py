import pickle

import matplotlib.pyplot as plt
import numpy
from sklearn.decomposition import PCA


class PCAAnalyzer(object):
    MARGIN_SCALE = 0.1

    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components, svd_solver='full')

    def fit(self, data):
        x = self.pca.fit_transform(data)
        margin = self.MARGIN_SCALE * (x.max(0) - x.min(0))
        _lim = (x.min(0) - margin, x.max(0) + margin)
        self.lim = numpy.array(_lim).T

    def get_lim(self, axis):
        return self.lim[axis]

    def transform_sequence(self, data):
        shp = data.shape
        data = data.reshape(-1, shp[-1])
        x = self.pca.transform(data)
        shp = list(shp)
        shp[-1] = x.shape[-1]
        shp = tuple(shp)

        x = x.reshape(*(shp))
        return x

    @property
    def contribution_ratio(self):
        return self.pca.explained_variance_ratio_

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump({'pca': self.pca, 'lim': self.lim}, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            loaded = pickle.load(f)
            self.pca = loaded['pca']
            self.lim = loaded['lim']
