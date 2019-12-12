import numpy as np
import torch.utils.data as data

import pdb

class RandomSampler(data.sampler.Sampler):
    def __init__(self, data_source, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.seed = seed
        self.identical_epochs = identical_epochs
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32 - 1)
        self.rng = np.random.RandomState(self.seed)
        self.total_pointclouds_count = len(self.data_source)
    def __iter__(self):
        if self.identical_epochs:
            self.rng.seed(self.seed)
        return iter(self.rng.choice(self.total_pointclouds_count, size=self.total_pointclouds_count, replace=False))
    def __len__(self):
        return self.total_pointclouds_count

class Sampler(data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.total_pointclouds_count = len(self.data_source)
    def __iter__(self):
        return iter(np.arange(0, self.total_pointclouds_count))
    def __len__(self):
        return self.total_pointclouds_count