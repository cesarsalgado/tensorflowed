import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflowed.models import residual_nets
from tensorflowed.data import Dataset, TrainTestDatasets, TrainValTestDatasets
from tensorflowed import learn
from tensorflowed import utils
import numpy as np
from os import path

MNIST_PATH = '/home/cesar/Dropbox/programming/learning/tensorflow/mnist/data'

def mnist_train_test_datasets_getter():
    datasets = input_data.read_data_sets(MNIST_PATH, one_hot=True)
    n_samples, n_dims = datasets.train.images.shape
    side = int(np.sqrt(n_dims))
    train = Dataset(datasets.train.images.reshape(n_samples, side, side, 1), datasets.train.labels)
    n_samples, n_dims = datasets.test.images.shape
    test = Dataset(datasets.test.images.reshape(n_samples, side, side, 1), datasets.test.labels)
    normalizer = utils.MeanCenterStdToOneNorm()
    normalizer.fit(train.x)
    train.x = normalizer.apply(train.x)
    test.x = normalizer.apply(test.x)
    return TrainTestDatasets(train, test, meta=normalizer)

# maybe it doesn't make much sense to apply the random transformation
# below after the data has been normalized, but I am putting here just
# to show how to use the 'data_augmenter' option.
def mnist_data_augmenter(image):
    image = tf.image.random_brightness(image, 0.1)
    return image

def set_params_and_run01():
    data_augment = False
    model_params = dict(
        initial_n_filters=16,
        initial_kernel_size=3,
        initial_subsample=1,
        residual_blks_per_subsample=3,
        min_map_side_for_conv=4,
        uniform=True,
        weight_decay=0.0001,
        dropout_prob=None,
        enable_global_avg_pool=True,
        activation=tf.nn.relu,
    )
    params = {}
    params['model'] = residual_nets.ResNet(**model_params)
    params['batch_size'] = 128

    params['base_lr'] = 0.01
    params['lr_decay'] = 0.95
    params['lr_decay_step'] = 1

    params['optimizer'] = tf.train.MomentumOptimizer
    params['optimizer_params'] = dict(momentum=0.9)

    params['max_epochs'] = 10

    params['checkpoint_epoch_step'] = 1
    params['checkpoint_path'] = MNIST_PATH

    params['summary_batch_step'] = 30
    params['console_print_step'] = 10

    params['train_test_datasets_getter'] = lambda : mnist_train_test_datasets_getter()
    params['data_augmenter'] = mnist_data_augmenter if data_augment else None

    params['random_seed'] = 1

    params['summaries_path'] = MNIST_PATH

    print model_params
    print params
    learn.train(params)

def set_params_and_run02():
    data_augment = False
    model_params = dict(
        initial_n_filters=16,
        initial_kernel_size=3,
        initial_subsample=1,
        residual_blks_per_subsample=3,
        min_map_side_for_conv=4,
        uniform=True,
        weight_decay=0.0001,
        dropout_prob=None,
        enable_global_avg_pool=True,
        activation=tf.nn.relu,
    )
    params = {}
    params['model'] = residual_nets.ResNet(**model_params)
    params['batch_size'] = 128

    params['base_lr'] = 0.001
    params['lr_decay'] = 1.0
    params['lr_decay_step'] = 1

    params['optimizer'] = tf.train.AdamOptimizer
    params['optimizer_params'] = None 

    params['max_epochs'] = 10

    params['checkpoint_epoch_step'] = 1
    params['checkpoint_path'] = MNIST_PATH

    params['summary_batch_step'] = 30
    params['console_print_step'] = 10

    params['train_test_datasets_getter'] = lambda : mnist_train_test_datasets_getter()
    params['data_augmenter'] = mnist_data_augmenter if data_augment else None

    params['random_seed'] = 1

    params['summaries_path'] = MNIST_PATH

    print model_params
    print params
    learn.train(params)
