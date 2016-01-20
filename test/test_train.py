from tensorflow.examples.tutorials.mnist import input_data
from tensorflowed.models import simple_cnns
from tensorflowed.models import residual_nets
from tensorflowed.data import Dataset, TrainTestDatasets, TrainValTestDatasets
from os import path

MNIST_PATH = '/home/cesar/Dropbox/programming/learning/tensorflow/mnist/data'

def mnist_train_test_datasets_getter():
    datasets = input_data.read_data_sets(MNIST_PATH, one_hot=True)
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
    #model = lambda a, b: simple_cnns.get_cnn(imgs_pl=a, n_classes=b, initial_n_filters=32)
    model = lambda a, b, c: residual_nets.get_cnn_given_pl(imgs_pl=a, y_pl=b, n_classes=c, initial_n_filters=16, initial_kernel_size=3, initial_subsample=1, residual_blks_per_subsample=1)
    params['model'] = model
    params['batch_size'] = 128

    params['base_lr'] = 0.001
    params['lr_decay'] = 0.95
    params['lr_decay_step'] = 1
    params['momentum'] = 0.9

    params['max_epochs'] = 3

    params['checkpoint_epoch_step'] = 1
    params['checkpoint_path'] = MNIST_PATH

    params['summary_batch_step'] = 30
    params['console_print_step'] = 10

    params['rand_state'] = np.random.RandomState(1)

    params['train_test_datasets_getter'] = mnist_train_test_datasets_getter

    params['graph_save_path'] = path.dirname(path.abspath(__file__))

    train_test01(params)
