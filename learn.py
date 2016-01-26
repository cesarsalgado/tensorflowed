from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflowed.data import OneEpochBatchIterator
from tensorflowed.models import common
import os
from os.path import join


def get_dataset_producer(batch_size, dataset, data_augmenter):
    with tf.name_scope('input'):
        with tf.device('/cpu:0'):
            x_init = tf.placeholder(tf.float32, shape=dataset.x.shape)
            y_init = tf.placeholder(tf.float32, shape=(dataset.y.shape[0], dataset.n_classes))
            x_var = tf.Variable(x_init, trainable=False, collections=[])
            y_var = tf.Variable(y_init, trainable=False, collections=[])

            image, label = tf.train.slice_input_producer([x_var, y_var])

            if data_augmenter is not None:
                image = data_augmenter(image)

            images, labels = tf.train.batch([image, label], batch_size=batch_size)
    return images, labels, x_init, y_init, x_var, y_var

# dataset dimensions should be in the right order.
# samples, height, width, channels
def get_dataset_placeholders(batch_size, dataset):
    x_pl = tf.placeholder(tf.float32, shape=(batch_size,)+dataset.x.shape[1:])
    y_pl = tf.placeholder(tf.float32, shape=(batch_size, dataset.n_classes))
    return (x_pl, y_pl)

def run_for_one_epoch(epoch, sess, rand_state, batch_size, dataset, x_var, y_var, logits,
                      console_print_step=10, train_op=None, losses=None,
                      learning_rate=None, summary_op=None, summary_writer=None, summary_batch_step=None):
    batch_iter = OneEpochBatchIterator(rand_state, batch_size, dataset, train_op is not None)
    n_correct = 0
    count = 0
    while batch_iter.has_next():
        if train_op is None:
            x, y_one_hot, n_used = batch_iter.next()
            feed_dict = {x_var: x, y_var: y_one_hot}
            y_pred_one_hot = logits.eval(feed_dict=feed_dict)
            phase = 'Test'
        else:
            batch_iter.next()
            if None in [logits, losses, learning_rate, summary_op, summary_writer, summary_batch_step]:
                raise ValueError
            _, total_loss, top_loss, wd_loss, lr, y_pred_one_hot, y_one_hot = sess.run([train_op, losses[0], losses[1], losses[2],
                                                                                        learning_rate, logits, y_var])
            assert not np.isnan(total_loss), 'Model diverged with loss = NaN'
            n_used = batch_size
            phase = 'Train'
        y = y_one_hot.argmax(axis=1)
        y_pred = y_pred_one_hot.argmax(axis=1)
        current_n_correct = (y_pred[:n_used] == y[:n_used]).sum()
        n_correct += current_n_correct
        count += n_used
        if train_op is not None:
            if batch_iter.batch_idx%summary_batch_step == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, (epoch-1)*batch_iter.n_batches+batch_iter.batch_idx)
            if batch_iter.batch_idx%console_print_step == 0:
                sys.stdout.write('Minibatch: %d/%d (%.1f%%), accuracy: %.3f, total_loss: %.3f, top_loss: %.3f, wd_loss: %f, learning rate: %.6f\r' % (batch_iter.batch_idx, batch_iter.n_batches, 100*batch_iter.batch_idx/float(batch_iter.n_batches), current_n_correct/float(n_used), total_loss, top_loss, wd_loss, lr))
                sys.stdout.flush()
    if train_op is not None:
        print()
    accu = n_correct/float(count)
    print(phase, ' accuracy: %.3f' % accu)
    return accu

def get_train_op(losses, batches_seen, batch_size, train_size, base_lr, momentum, lr_decay_step, lr_decay):
    lr = tf.train.exponential_decay(base_lr,
                                    batches_seen*batch_size,
                                    lr_decay_step*train_size,
                                    lr_decay,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    common.add_loss_summaries(losses)

    opt = tf.train.MomentumOptimizer(lr, momentum)
    grads = opt.compute_gradients(losses[0])

    apply_gradient_op = opt.apply_gradients(grads, global_step=batches_seen)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    return apply_gradient_op, lr

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

    checkpoint_epoch_step = params['checkpoint_epoch_step']
    checkpoint_path = params['checkpoint_path']

    summary_batch_step = params['summary_batch_step']
    console_print_step = params['console_print_step']

    random_seed = params['random_seed']

    train_test_datasets_getter = params['train_test_datasets_getter']

    data_augmenter = params['data_augmenter']

    summaries_path = params['summaries_path']
    # end get params

    rand_state = np.random.RandomState(random_seed)
    tf.set_random_seed(rand_state.randint(1000000))

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.isdir(summaries_path):
        os.makedirs(summaries_path)

    os.system("rm %s"%join(summaries_path, "events.out.tfevents.*"))

    ops.reset_default_graph()

    train_test_ds = train_test_datasets_getter()
    n_classes = train_test_ds.n_classes

    # using queue instead of vanilla placeholders for training in case you
    # want to use data augmentation (e.g. tf.image.random_flip...)
    x_train, y_train, x_init, y_init, x_var, y_var = get_dataset_producer(batch_size, train_test_ds.train, data_augmenter)

    x_test, y_test = get_dataset_placeholders(batch_size, train_test_ds.test)

    # TODO: How to reuse variable without putting the two models under the same scope.
    # The way is now many layers are held under the same tab in ternsorboard and
    # that is bad.
    with tf.variable_scope("train_and_eval") as scope:
        logits_train, losses = model.logits_train(x_train, y_train)
        scope.reuse_variables()
        logits_eval, _ = model.logits_eval(x_test, y_test)
    
    batches_seen = tf.Variable(0, name='batches_seen', trainable=False)
    train_size = train_test_ds.train.size()
    train_op, learning_rate = get_train_op(list(losses), batches_seen, batch_size, train_size, base_lr, momentum, lr_decay_step, lr_decay)

    tf.image_summary('images', x_train, max_images=10)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        summary_writer = None
        coords = None
        try:
            tf.initialize_all_variables().run()
            sess.run(x_var.initializer, feed_dict={x_init: train_test_ds.train.x})
            sess.run(y_var.initializer, feed_dict={y_init: train_test_ds.train.y})
            print('Variables Initialized!')

            summary_writer = tf.train.SummaryWriter(summaries_path, sess.graph_def)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for epoch in xrange(1, max_epochs+1):
                print("Epoch %d/%d. Starting train." % (epoch,max_epochs))
                train_accu = run_for_one_epoch(epoch, sess, rand_state, batch_size, train_test_ds.train, x_train, y_train, logits_train,
                                  console_print_step=console_print_step, train_op=train_op,
                                  losses=losses, learning_rate=learning_rate, summary_op=summary_op,
                                  summary_writer=summary_writer, summary_batch_step=summary_batch_step)
                print("Starting test.")
                val_accu = run_for_one_epoch(epoch, sess, rand_state, batch_size, train_test_ds.test, x_test, y_test,
                                             logits_eval, console_print_step=console_print_step)
                if epoch%checkpoint_epoch_step == 0:
                    val_accu_str = str(val_accu).replace('.', '_')
                    train_accu_str = str(train_accu).replace('.', '_')
                    checkpoint_file_path = join(checkpoint_path, 'epochs_completed_%03d_train_accu_%s_val_accu_%s.ckpt'%(epoch, train_accu_str, val_accu_str))
                    actual = saver.save(sess, checkpoint_file_path, global_step=batches_seen)
                    if actual:
                        print("\nModel Saved at: %s\n"%actual)
        except KeyboardInterrupt:
            print("\n\nExiting!!\n")
            #TODO: how to save at this point correctly? The commented code below is not working.
            #batches_per_epoch = np.ceil(float(train_size)/batch_size)
            #partial_epochs = ("%.2f"%(batches_seen.eval(sess)/batches_per_epoch)).replace('.', '_')
            #checkpoint_file_path = join(checkpoint_path, 'epochs_partially_completed_%s.ckpt'%partial_epochs)
            #actual = saver.save(sess, checkpoint_path, global_step=batches_seen)
            #if actual:
            #    print("\nModel Saved at: %s\n"%actual)
        finally:
            if coord is not None:
                coord.request_stop()
            if summary_writer is not None:
                summary_writer.close()
            if coord is not None:
                coord.join(threads)
