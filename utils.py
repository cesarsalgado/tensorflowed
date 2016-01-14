from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from models import simple_cnns
from models import residual_nets
from tensorflow.examples.tutorials.mnist import input_data
import ipdb

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

class TrainValTestDatasets:
    def __init__(self, train, val, test, meta=None):
        self.train = train
        self.val = val
        self.test = test
        self.meta = meta
        if not (train.is_compatible(val) and val.is_compatible(test)):
            raise ValueError
        self.n_classes = train.n_classes

class OneEpochBatchIterator(object):
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

def run_for_one_epoch(sess, rand_state, batch_size, dataset, x_pl, y_pl, predictions,
                      optimizer=None, loss=None, learning_rate=None):
    batch_iter = OneEpochBatchIterator(rand_state, batch_size, dataset)
    n_correct = 0
    count = 0
    while batch_iter.has_next():
        x, y_one_hot, n_used = batch_iter.next()
        feed_dict = {x_pl: x, y_pl: y_one_hot}
        if optimizer is None:
            y_pred_one_hot = predictions.eval(feed_dict=feed_dict)
            phase = 'Test'
        else:
            if None in [predictions, loss, learning_rate]:
                raise ValueError
            _, loss_v, lr, y_pred_one_hot = sess.run([optimizer, loss, learning_rate, predictions], feed_dict=feed_dict)
            phase = 'Train'
        y = y_one_hot.argmax(axis=1)
        y_pred = y_pred_one_hot.argmax(axis=1)
        current_n_correct = (y_pred[:n_used] == y[:n_used]).sum()
        n_correct += current_n_correct
        count += n_used
        if optimizer is not None:
            if batch_iter.batch_idx%10 == 0:
                sys.stdout.write('Minibatch: %d/%d (%.1f%%), accuracy: %.3f, loss: %.3f, learning rate: %.6f\r' % 
                    (batch_iter.batch_idx, batch_iter.n_batches, 100*batch_iter.batch_idx/float(batch_iter.n_batches),
                     current_n_correct/float(n_used), loss_v, lr))
                sys.stdout.flush()
    if optimizer is not None:
        print()
    print(phase, ' accuracy: %.3f' % (n_correct/float(count)))

# dataset dimensions should be in the right order.
# samples, height, width, channels
def get_dataset_placeholders(batch_size, dataset):
    x_pl = tf.placeholder(tf.float32, shape=(batch_size,)+dataset.x.shape[1:])
    y_pl = tf.placeholder(tf.float32, shape=(batch_size, dataset.n_classes))
    return (x_pl, y_pl)

# here the network is used for training and testing.
# This mean that the same batch_size is used in both phases.
def train_test01(params):
    # start get params
    logits_getter = params['logits_getter']
    loss_getter = params['loss_getter']
    batch_size = params['batch_size']

    weight_decay = params['weight_decay']  # TODO Not implemented.

    base_lr = params['base_lr']
    lr_decay = params['lr_decay']
    lr_decay_step = params['lr_decay_step']
    momentum = params['momentum']

    max_epochs = params['max_epochs']

    save_every_x_epoch = params['save_every_x_epoch']  # TODO Not implemented.

    rand_state = params['rand_state']

    train_test_datasets_getter = params['train_test_datasets_getter']
    # end get params

    ipdb.set_trace()
    train_test_ds = train_test_datasets_getter()
    n_classes = train_test_ds.n_classes

    x_pl, y_pl = get_dataset_placeholders(batch_size, train_test_ds.train)

    logits = logits_getter(x_pl, n_classes)
    
    loss = loss_getter(logits, y_pl)
    predictions = tf.nn.softmax(logits)

    batches_seen = tf.Variable(0, name='batches_seen', trainable=False)
    train_size = train_test_ds.train.size()
    learning_rate = tf.train.exponential_decay(base_lr, batches_seen*batch_size, lr_decay_step*train_size, lr_decay, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=batches_seen)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print('Variables Initialized!')
        for epoch in xrange(1, max_epochs+1):
            print("Epoch %d/%d. Starting training." % (epoch,max_epochs))
            run_for_one_epoch(sess, rand_state, batch_size, train_test_ds.train, x_pl, y_pl, predictions, optimizer=optimizer,
                              loss=loss, learning_rate=learning_rate)
            print("Epoch %d/%d. Starting testing." % (epoch,max_epochs))
            run_for_one_epoch(sess, rand_state, batch_size, train_test_ds.test, x_pl, y_pl, predictions)

def softmax_cross_entropy_with_logits_loss_getter(logits, y_pl):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_pl))

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

def mnist_train_test_datasets_getter():
    datasets = input_data.read_data_sets('/home/cesar/Dropbox/programming/learning/tensorflow/mnist/data', one_hot=True)
    n_samples, n_dims = datasets.train.images.shape
    side = int(np.sqrt(n_dims))
    train = Dataset(datasets.train.images.reshape(n_samples, side, side, 1), datasets.train.labels)
    n_samples, n_dims = datasets.test.images.shape
    test = Dataset(datasets.test.images.reshape(n_samples, side, side, 1), datasets.test.labels)
    train_mean = train.x.mean(axis=0)
    train_std = train.x.std(axis=0)
    train_std[train_std==0.0] = 1.0
    train.x = (train.x-train_mean)/train_std
    test.x = (test.x-train_mean)/train_std
    return TrainTestDatasets(train, test)

def test_train_test01():
    params = {}
    #logits_getter = lambda a, b: simple_cnns.get_cnn(imgs_pl=a, n_classes=b, initial_n_filters=32)
    logits_getter = lambda a, b: residual_nets.get_cnn_given_pl(imgs_pl=a, n_classes=b, initial_n_filters=16, initial_kernel_size=3, initial_subsample=1, residual_blks_per_subsample=1)
    params['logits_getter'] = logits_getter
    params['loss_getter'] = softmax_cross_entropy_with_logits_loss_getter
    params['batch_size'] = 128

    params['weight_decay'] = 0.01 # Still not working

    params['base_lr'] = 0.001
    params['lr_decay'] = 0.95
    params['lr_decay_step'] = 1
    params['momentum'] = 0.9

    params['max_epochs'] = 3

    params['save_every_x_epoch'] = 1

    params['rand_state'] = np.random.RandomState(1)

    params['train_test_datasets_getter'] = mnist_train_test_datasets_getter

    train_test01(params)
