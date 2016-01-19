from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflowed.data import OneEpochBatchIterator
import ipdb

# dataset dimensions should be in the right order.
# samples, height, width, channels
def get_dataset_placeholders(batch_size, dataset):
    x_pl = tf.placeholder(tf.float32, shape=(batch_size,)+dataset.x.shape[1:])
    y_pl = tf.placeholder(tf.float32, shape=(batch_size, dataset.n_classes))
    return (x_pl, y_pl)

def run_for_one_epoch(sess, rand_state, batch_size, dataset, x_pl, y_pl, logits,
                      optimizer=None, loss=None, learning_rate=None):
    batch_iter = OneEpochBatchIterator(rand_state, batch_size, dataset)
    n_correct = 0
    count = 0
    while batch_iter.has_next():
        x, y_one_hot, n_used = batch_iter.next()
        feed_dict = {x_pl: x, y_pl: y_one_hot}
        if optimizer is None:
            y_pred_one_hot = logits.eval(feed_dict=feed_dict)
            phase = 'Test'
        else:
            if None in [logits, loss, learning_rate]:
                raise ValueError
            _, total_loss, top_loss, wd_loss, lr, y_pred_one_hot = sess.run([optimizer, loss[0], loss[1], loss[2], learning_rate, logits], feed_dict=feed_dict)
            phase = 'Train'
        y = y_one_hot.argmax(axis=1)
        y_pred = y_pred_one_hot.argmax(axis=1)
        current_n_correct = (y_pred[:n_used] == y[:n_used]).sum()
        n_correct += current_n_correct
        count += n_used
        if optimizer is not None:
            if batch_iter.batch_idx%10 == 0:
                sys.stdout.write('Minibatch: %d/%d (%.1f%%), accuracy: %.3f, total_loss: %.3f, top_loss: %.3f, wd_loss: %f, learning rate: %.6f\r' % (batch_iter.batch_idx, batch_iter.n_batches, 100*batch_iter.batch_idx/float(batch_iter.n_batches), current_n_correct/float(n_used), total_loss, top_loss, wd_loss, lr))
                sys.stdout.flush()
    if optimizer is not None:
        print()
    print(phase, ' accuracy: %.3f' % (n_correct/float(count)))

# here the same model is used for training and testing.
# This mean that the same batch_size is used in both phases.
def train(params):
    # start get params
    model = params['model']
    batch_size = params['batch_size']

    base_lr = params['base_lr']
    lr_decay = params['lr_decay']
    lr_decay_step = params['lr_decay_step']
    momentum = params['momentum']

    max_epochs = params['max_epochs']

    save_every_x_epoch = params['save_every_x_epoch']  # TODO Not implemented.

    rand_state = params['rand_state']

    train_test_datasets_getter = params['train_test_datasets_getter']

    graph_save_path = params['graph_save_path']
    # end get params

    ops.reset_default_graph()

    ipdb.set_trace()
    train_test_ds = train_test_datasets_getter()
    n_classes = train_test_ds.n_classes

    ipdb.set_trace()
    x_pl, y_pl = get_dataset_placeholders(batch_size, train_test_ds.train)

    logits, loss = model(x_pl, y_pl, n_classes)
    
    batches_seen = tf.Variable(0, name='batches_seen', trainable=False)
    train_size = train_test_ds.train.size()
    learning_rate = tf.train.exponential_decay(base_lr, batches_seen*batch_size, lr_decay_step*train_size, lr_decay, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss[0], global_step=batches_seen)

    writer = None
    try:
        with tf.Session() as sess:
            if graph_save_path is not None:
                writer = tf.train.SummaryWriter(graph_save_path, sess.graph_def)
            tf.initialize_all_variables().run()
            print('Variables Initialized!')
            for epoch in xrange(1, max_epochs+1):
                print("Epoch %d/%d. Starting train." % (epoch,max_epochs))
                run_for_one_epoch(sess, rand_state, batch_size, train_test_ds.train, x_pl, y_pl, logits, optimizer=optimizer,
                                  loss=loss, learning_rate=learning_rate)
                print("Starting test.")
                run_for_one_epoch(sess, rand_state, batch_size, train_test_ds.test, x_pl, y_pl, logits)
    except KeyboardInterrupt:
        print("\n\nExiting!!\n")
    finally:
        if writer is not None:
            writer.close()
