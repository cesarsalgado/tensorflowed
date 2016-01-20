import numpy as np


class Dataset:
    def __init__(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError
        self.x = x
        self.y = y
        self.n_classes = y.shape[1]

    def size(self):
        return self.y.shape[0]

    def get_x(self, inds):
        return self.x[inds]

    def get_y(self, inds):
        return self.y[inds]

    def is_compatible(self, other):
        if self.n_classes != other.n_classes:
            return False
        elif self.x.shape[1:] != other.x.shape[1:]:
            return False
        elif self.y.shape[1:] != other.y.shape[1:]:
            return False
        else:
            return True


class TrainTestDatasets:
    def __init__(self, train, test, meta=None):
        self.train = train
        self.test = test
        self.meta = meta
        if not train.is_compatible(test):
            raise ValueError
        self.n_classes = train.n_classes

    def n_phases(self):
        return 2

    def datasets(self):
        yield self.train
        yield self.test


class TrainValTestDatasets:
    def __init__(self, train, val, test, meta=None):
        self.train = train
        self.val = val
        self.test = test
        self.meta = meta
        if not (train.is_compatible(val) and val.is_compatible(test)):
            raise ValueError
        self.n_classes = train.n_classes

    def n_phases(self):
        return 3

    def datasets(self):
        yield self.train
        yield self.val
        yield self.test


class OneEpochBatchIterator:
    def __init__(self, rand_state, batch_size, dataset):
        self.batch_size = batch_size
        self.dataset = dataset
        self.size = dataset.size()
        self.rem = self.size%batch_size
        self.n_batches = self.size//batch_size + bool(self.rem)
        self.batch_idx = 0
        self.perm = rand_state.permutation(self.size)

    def has_next(self):
        return self.batch_idx < self.n_batches

    def next(self):
        start = self.batch_idx*self.batch_size
        if self.batch_idx == self.n_batches-1 and self.rem > 0:
            end = start + self.rem
            n_to_fill = self.batch_size-self.rem
            left_inds = self.perm[start:]
            right_inds = self.perm[:n_to_fill]
            inds = np.hstack((left_inds, right_inds))
            n_used = self.rem
        else:
            inds = self.perm[start:start+self.batch_size]
            n_used = self.batch_size
        x = self.dataset.get_x(inds)
        y = self.dataset.get_y(inds)
        self.batch_idx += 1
        return x, y, n_used
