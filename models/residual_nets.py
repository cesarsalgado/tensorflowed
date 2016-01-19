import tensorflow as tf
from tensorflow.python.ops import attention_ops # it is needed to avoid bug when calling extract_glimpse
from tensorflowed.models import common
import numpy as np
import math
import ipdb


def get_residual_block(idx, input_, kernel_size, n_in_channels, n_out_channels, uniform=True, weight_decay=None, residual=True, suffix=''):
    subsample = n_out_channels/n_in_channels
    if subsample not in (1, 2):
        raise ValueError
    if residual:
        prefix = ''
    else:
        prefix = 'non_'
    with tf.variable_scope("%sresidual%02d_%s" % (prefix, idx, suffix)):
        with tf.variable_scope("conv1"):
            n_channels = n_in_channels
            n_filters = n_out_channels
            conv_plus_bias = common.get_conv_plus_bias_block(input_, kernel_size, n_channels, n_filters, uniform=uniform,
                                                             weight_decay=weight_decay, subsample=subsample)
            relu1 = tf.nn.relu(conv_plus_bias)
        if residual:
            if subsample != 1:
                with tf.variable_scope("shortcut"):
                    n_channels = n_in_channels
                    n_filters = n_out_channels
                    h, w = relu1.get_shape().as_list()[1:3]
                    subsampled = tf.image.resize_nearest_neighbor(input_, [h, w])
                    weight_init = tf.learn.xavier_initializer(n_in_channels, n_out_channels, uniform=uniform)
                    kernel = common.variable_with_weight_decay('weights', shape=[1, 1, n_channels, n_filters],
                                                               weight_init=weight_init, weight_decay=weight_decay)
                    shortcut = tf.nn.conv2d(subsampled, kernel, [1, 1, 1, 1], padding='SAME')
            else:
                shortcut = input_
        with tf.variable_scope("conv2"):
            n_channels = n_out_channels
            n_filters = n_out_channels
            conv_plus_bias = common.get_conv_plus_bias_block(relu1, kernel_size, n_channels, n_filters, uniform=uniform,
                                                             weight_decay=weight_decay, subsample=1)
            if residual:
                conv_plus_bias = conv_plus_bias + shortcut
            relu2 = tf.nn.relu(conv_plus_bias)
    return relu2

# suppose square input
def get_cnn_common(imgs_pl, y_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size, initial_subsample,
                   residual_blks_per_subsample=2, min_map_side_for_conv=6, uniform=True, weight_decay=None, crop_level=None):
    current_n_filters = initial_n_filters
    kernel_size = 3
    print("'%s' - shape: %s"%(imgs_pl.name, str(imgs_pl.get_shape())))
    with tf.variable_scope('conv0'):
        conv_plus_bias = common.get_conv_plus_bias_block(imgs_pl, initial_kernel_size, img_channels, current_n_filters,
                                                         uniform=uniform, weight_decay=weight_decay, subsample=initial_subsample)
        relu0 = tf.nn.relu(conv_plus_bias)
        print("'%s' - shape: %s"%(relu0.name, str(relu0.get_shape())))
    current_map_side = img_side/initial_subsample
    last_output = relu0
    count = 1
    subsample_count = 0
    last_output_crop = None
    while True:
        for i in xrange(1, residual_blks_per_subsample+1):
            last_output = get_residual_block(count, last_output, 3, current_n_filters, current_n_filters, uniform=uniform, weight_decay=weight_decay)
            if last_output_crop is not None:
                last_output_crop = get_residual_block(count, last_output_crop, 3, current_n_filters,
                                                      current_n_filters, suffix='crop', uniform=uniform, weight_decay=weight_decay)
            count += 1
        if current_map_side/2 < min_map_side_for_conv:
            break
        next_n_filters = 2*current_n_filters
        last_output = get_residual_block(count, last_output, 3, current_n_filters, next_n_filters, uniform=uniform, weight_decay=weight_decay)
        if last_output_crop is not None:
            last_output_crop = get_residual_block(count, last_output_crop, 3, current_n_filters, next_n_filters, suffix='crop', uniform=uniform, weight_decay=weight_decay)
        print("'%s' - shape: %s"%(last_output.name, str(last_output.get_shape())))
        count += 1
        subsample_count += 1
        current_n_filters = next_n_filters
        current_map_side = last_output.get_shape().as_list()[1]
        if crop_level is not None:
            if subsample_count == crop_level:
                crop = tf.image.extract_glimpse(imgs_pl, [current_map_side, current_map_side], np.zeros((batch_size,2), np.float32))
                #crop = tf.image.extract_glimpse(imgs_pl, tf.cast([current_map_side, current_map_side], tf.int32), tf.zeros((batch_size,2)))
                with tf.variable_scope('inc_filters_crop'):
                    last_output_crop = common.get_conv_plus_bias_block(crop, 3, img_channels, current_n_filters,
                                                                       uniform=uniform, weight_decay=weight_decay)
    # global avg pooling
    if last_output_crop is not None:
        merge = last_output + last_output_crop
        last_output = get_residual_block(count, merge, 3, current_n_filters, current_n_filters, uniform=uniform, weight_decay=weight_decay)
    ksize = strides = [1, current_map_side, current_map_side, 1]
    avg_pool = tf.nn.avg_pool(last_output, ksize, strides, padding='VALID')
    print("'%s' - shape: %s"%(avg_pool.name, str(avg_pool.get_shape())))
    reshape = tf.reshape(avg_pool, [batch_size, current_n_filters])
    print("'%s' - shape: %s"%(reshape.name, str(reshape.get_shape())))
    # fully connected and softmax
    weight_init = tf.learn.xavier_initializer(current_n_filters, n_classes, uniform=uniform)
    fc_weights = common.variable_with_weight_decay('fc_weights', shape=[current_n_filters, n_classes],
                                                   weight_init=weight_init, weight_decay=weight_decay)
    fc_biases = tf.get_variable('fc_biases', [n_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.nn.xw_plus_b(reshape, fc_weights, fc_biases)
    print("'%s' - shape: %s"%(logits.name, str(logits.get_shape())))
    return logits, common.loss(logits, y_pl)

def get_cnn(n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size, initial_subsample,
            residual_blks_per_subsample=2, min_map_side_for_conv=6, uniform=True, weight_decay=None, crop_level=None):
    imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_side, img_side, img_channels))
    y_pl = tf.placeholder(tf.float32, shape=(batch_size, n_classes))
    logits, loss = get_cnn_common(imgs_pl, y_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size,
                                  initial_subsample, residual_blks_per_subsample, min_map_side_for_conv, uniform, weight_decay, crop_level)
    return imgs_pl, y_pl, logits, loss

def get_cnn_given_pl(imgs_pl, y_pl, n_classes, initial_n_filters, initial_kernel_size, initial_subsample,
                     residual_blks_per_subsample=2, min_map_side_for_conv=6, uniform=True, weight_decay=None, crop_level=None):
    batch_size, img_side, img_side, img_channels = imgs_pl.get_shape().as_list()
    return get_cnn_common(imgs_pl, y_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size, initial_subsample,
                          residual_blks_per_subsample, min_map_side_for_conv, uniform, weight_decay, crop_level)

def gen_graph01():
    sess = tf.Session()
    logits = get_residual_cnn(10, 128, 1, 28, 16, 3, 1)
    writer = tf.train.SummaryWriter('/tmp/residual01', sess.graph_def)
    tf.initialize_all_variables().run(session=sess)
    writer.close()
    sess.close()
