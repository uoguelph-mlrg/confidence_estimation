import numpy as np
from torch.utils.data import Dataset


class GaussianNoise(Dataset):
    """Gaussian Noise Dataset"""

    def __init__(self, size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0):
        self.size = size
        self.n_samples = n_samples
        self.mean = mean
        self.variance = variance
        self.data = np.random.normal(loc=self.mean, scale=self.variance, size=(self.n_samples,) + self.size)
        self.data = np.clip(self.data, 0, 1)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


class UniformNoise(Dataset):
    """Uniform Noise Dataset"""

    def __init__(self, size=(3, 32, 32), n_samples=10000, low=0, high=1):
        self.size = size
        self.n_samples = n_samples
        self.low = low
        self.high = high
        self.data = np.random.uniform(low=self.low, high=self.high, size=(self.n_samples,) + self.size)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]