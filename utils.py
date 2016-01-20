import numpy as np


def to_onehot(labels, n_classes, dtype):
    n_dim = len(labels.shape)
    if n_dim == 2 and labels.shape[1] != 1:
        raise ValueError
    if n_dim > 2:
        raise ValueError
    n_samples = labels.size
    onehot = np.zeros((n_samples, n_classes), dtype=dtype)
    onehot[np.arange(n_samples), labels.flatten().astype(int)] = 1
    return onehot


class MeanCenterStdToOneNorm:
    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = np.sqrt(data.var(axis=0) + self.epsilon)

    def apply(self, data):
        r = (data-self.mean)/self.std
        if np.all(np.isfinite(r)):
            return r
        else:
            raise ValueError


def get_3_argument_model_builder(std_model_builder, model_params):
    return lambda a,b,c: std_model_builder(a, b, c, **model_params)
