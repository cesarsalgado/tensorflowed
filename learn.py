from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflowed.data import OneEpochBatchIterator
from tensorflowed.models import common
import os
from os.path import join
import ipdb

# dataset dimensions should be in the right order.
# samples, height, width, channels
def get_dataset_placeholders(batch_size, dataset):
    x_pl = tf.placeholder(tf.float32, shape=(batch_size,)+dataset.x.shape[1:])
    y_pl = tf.placeholder(tf.float32, shape=(batch_size, dataset.n_classes))
    return (x_pl, y_pl)

def run_for_one_epoch(epoch, sess, rand_state, batch_size, dataset, x_pl, y_pl, logits,
                      dropout_keep_prob, console_print_step=10, train_op=None, loss=None,
                      learning_rate=None, summary_op=None, summary_writer=None, summary_batch_step=None):
    batch_iter = OneEpochBatchIterator(rand_state, batch_size, dataset)
    n_correct = 0
    count = 0
    while batch_iter.has_next():
        x, y_one_hot, n_used = batch_iter.next()
        feed_dict = {x_pl: x, y_pl: y_one_hot}
        if train_op is None:
            if dropout_keep_prob is not None:
                feed_dict[dropout_keep_prob] = 1.0
            y_pred_one_hot = logits.eval(feed_dict=feed_dict)
            phase = 'Test'
        else:
            if dropout_keep_prob is not None:
                feed_dict[dropout_keep_prob] = 0.5
            if None in [logits, loss, learning_rate, summary_op, summary_writer, summary_batch_step]:
                raise ValueError
            _, total_loss, top_loss, wd_loss, lr, y_pred_one_hot = sess.run([train_op, loss[0], loss[1], loss[2], learning_rate, logits], feed_dict=feed_dict)
            assert not np.isnan(total_loss), 'Model diverged with loss = NaN'
            phase = 'Train'
        y = y_one_hot.argmax(axis=1)
        y_pred = y_pred_one_hot.argmax(axis=1)
        current_n_correct = (y_pred[:n_used] == y[:n_used]).sum()
        n_correct += current_n_correct
        count += n_used
        if train_op is not None:
            if batch_iter.batch_idx%summary_batch_step == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, (epoch-1)*batch_iter.n_batches+batch_iter.batch_idx)
            if batch_iter.batch_idx%console_print_step == 0:
                sys.stdout.write('Minibatch: %d/%d (%.1f%%), accuracy: %.3f, total_loss: %.3f, top_loss: %.3f, wd_loss: %f, learning rate: %.6f\r' % (batch_iter.batch_idx, batch_iter.n_batches, 100*batch_iter.batch_idx/float(batch_iter.n_batches), current_n_correct/float(n_used), total_loss, top_loss, wd_loss, lr))
                sys.stdout.flush()
    if train_op is not None:
        print()
    accu = n_correct/float(count)
    print(phase, ' accuracy: %.3f' % accu)
    return accu

def get_train_op(loss, batches_seen, batch_size, train_size, base_lr, momentum, lr_decay_step, lr_decay):
    lr = tf.train.exponential_decay(base_lr,
                                    batches_seen*batch_size,
                                    lr_decay_step*train_size,
                                    lr_decay,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    common.add_loss_summaries(loss)

    opt = tf.train.MomentumOptimizer(lr, momentum)
    grads = opt.compute_gradients(loss[0])

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

    rand_state = params['rand_state']

    train_test_datasets_getter = params['train_test_datasets_getter']

    graph_save_path = params['graph_save_path']
    # end get params

    os.system("rm %s"%join(graph_save_path, "events.out.tfevents.*"))

    ops.reset_default_graph()

    train_test_ds = train_test_datasets_getter()
    n_classes = train_test_ds.n_classes

    x_pl, y_pl = get_dataset_placeholders(batch_size, train_test_ds.train)

    logits, loss, dropout_keep_prob = model(x_pl, y_pl, n_classes)
    
    batches_seen = tf.Variable(0, name='batches_seen', trainable=False)
    train_size = train_test_ds.train.size()
    train_op, learning_rate = get_train_op(list(loss), batches_seen, batch_size, train_size, base_lr, momentum, lr_decay_step, lr_decay)

    tf.image_summary('images', x_pl, max_images=10)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.merge_all_summaries()

    batches_per_epoch = np.ceil(float(train_size)/batch_size)

    with tf.Session() as sess:
        summary_writer = None
        try:
            summary_writer = tf.train.SummaryWriter(graph_save_path, sess.graph_def)
            tf.initialize_all_variables().run()
            print('Variables Initialized!')
            for epoch in xrange(1, max_epochs+1):
                print("Epoch %d/%d. Starting train." % (epoch,max_epochs))
                run_for_one_epoch(epoch, sess, rand_state, batch_size, train_test_ds.train, x_pl, y_pl, logits,
                                  dropout_keep_prob, console_print_step=console_print_step, train_op=train_op,
                                  loss=loss, learning_rate=learning_rate, summary_op=summary_op,
                                  summary_writer=summary_writer, summary_batch_step=summary_batch_step)
                print("Starting test.")
                val_accu = run_for_one_epoch(epoch, sess, rand_state, batch_size, train_test_ds.test, x_pl, y_pl,
                                             logits, dropout_keep_prob, console_print_step=console_print_step)
                if epoch%checkpoint_epoch_step == 0:
                    val_accu_str = str(val_accu).replace('.', '_')
                    checkpoint_file_path = join(checkpoint_path, 'epochs_completed_%03d_val_accu_%s.ckpt'%(epoch, val_accu_str))
                    saver.save(sess, checkpoint_file_path, global_step=batches_seen)
                    print("\nModel Saved at: %s\n"%checkpoint_file_path)
        except KeyboardInterrupt:
            print("\n\nExiting!!\n")
        finally:
            if summary_writer is not None:
                summary_writer.close()
            #partial_epochs = ("%.2f"%(batches_seen.eval(sess)/batches_per_epoch)).replace('.', '_')
            #checkpoint_file_path = join(checkpoint_path, 'epochs_partially_completed_%s.ckpt'%partial_epochs)
            #saver.save(sess, checkpoint_path, global_step=batches_seen)
            #print("\nModel Saved at: %s\n"%checkpoint_file_path)
