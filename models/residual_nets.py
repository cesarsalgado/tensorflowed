import tensorflow as tf
from tensorflow.python.ops import attention_ops # it is needed to avoid bug when calling extract_glimpse
from tensorflowed.models import common
import numpy as np
import math
import ipdb


def get_residual_block(train, reuse, activation, idx, input_, kernel_size, n_in_channels, n_out_channels, subsample=1, uniform=True, weight_decay=None, residual=True, suffix=''):
    if subsample not in (1, 2):
        raise ValueError
    if residual:
        prefix = ''
    else:
        prefix = 'non_'
    with tf.variable_scope("%sresidual%02d_%s" % (prefix, idx, suffix), reuse=reuse):
        with tf.variable_scope("conv1"):
            n_channels = n_in_channels
            n_filters = n_out_channels
            conv_plus_bias = common.get_conv_plus_bias_block(train, input_, kernel_size, n_channels, n_filters, uniform=uniform,
                                                             weight_decay=weight_decay, subsample=subsample)
            acti1 = activation(conv_plus_bias)
        if residual:
            if n_in_channels != n_out_channels or subsample != 1:
                with tf.variable_scope("shortcut"):
                    n_channels = n_in_channels
                    n_filters = n_out_channels
                    if subsample != 1:
                        h, w = acti1.get_shape().as_list()[1:3]
                        subsampled = tf.image.resize_nearest_neighbor(input_, [h, w])
                    else:
                        subsampled = input_
                    if n_in_channels != n_out_channels:
                        weight_init = tf.learn.xavier_initializer(n_in_channels, n_out_channels, uniform=uniform)
                        kernel = common.variable_with_weight_decay('weights', shape=[1, 1, n_channels, n_filters],
                                                                   weight_init=weight_init, weight_decay=weight_decay, train=train)
                        shortcut = tf.nn.conv2d(subsampled, kernel, [1, 1, 1, 1], padding='SAME')
                    else:
                        shortcut = subsampled
                    common.activation_summary(shortcut, train)
            else:
                shortcut = input_
        with tf.variable_scope("conv2"):
            n_channels = n_out_channels
            n_filters = n_out_channels
            conv_plus_bias = common.get_conv_plus_bias_block(train, acti1, kernel_size, n_channels, n_filters, uniform=uniform,
                                                             weight_decay=weight_decay, subsample=1)
            common.activation_summary(conv_plus_bias, train)
            if residual:
                conv_plus_bias = conv_plus_bias + shortcut
            acti2 = activation(conv_plus_bias)
            common.activation_summary(acti2, train)
            common.activation_image_summary(acti2, acti2.op.name+'/acti2', 3, train)
    return acti2

# suppose square input
def get_cnn_common(train, images, labels, n_classes, batch_size, img_channels, img_side, initial_n_filters,
                   initial_kernel_size, initial_subsample, residual_blks_per_subsample,
                   min_map_side_for_conv, uniform, weight_decay,
                   dropout_prob, enable_global_avg_pool, activation, reuse):
    current_n_filters = initial_n_filters
    kernel_size = 3
    common.print_shape(images)
    with tf.variable_scope('conv0', reuse=reuse):
        conv_plus_bias = common.get_conv_plus_bias_block(train, images, initial_kernel_size, img_channels, current_n_filters,
                                                         uniform=uniform, weight_decay=weight_decay, subsample=initial_subsample)
        acti0 = activation(conv_plus_bias)
        common.activation_image_summary(acti0, 'acti0', 3, train)
        common.activation_summary(acti0, train)
        common.print_shape(acti0)
    current_map_side = img_side/initial_subsample
    last_output = acti0
    count = 1
    subsample_count = 0
    while True:
        for i in xrange(1, residual_blks_per_subsample+1):
            last_output = get_residual_block(train, reuse, activation, count, last_output, 3, current_n_filters, current_n_filters, uniform=uniform, weight_decay=weight_decay)
            count += 1
        if current_map_side/2 < min_map_side_for_conv:
            break
        next_n_filters = 2*current_n_filters
        last_output = get_residual_block(train, reuse, activation, count, last_output, 3, current_n_filters, next_n_filters, subsample=2, uniform=uniform, weight_decay=weight_decay)
        common.print_shape(last_output)
        count += 1
        subsample_count += 1
        current_n_filters = next_n_filters
        current_map_side = last_output.get_shape().as_list()[1]
    # global avg pooling
    if enable_global_avg_pool:
        ksize = strides = [1, current_map_side, current_map_side, 1]
        last_output = tf.nn.avg_pool(last_output, ksize, strides, padding='VALID')
        common.activation_summary(last_output, train)
        common.print_shape(last_output)
    else:
        old_n_fileters = current_n_filters
        current_n_filters = current_n_filters*(current_map_side**2)
    reshape = tf.reshape(last_output, [batch_size, current_n_filters])
    common.print_shape(reshape)
    # fully connected and softmax
    if dropout_prob is not None:
        reshape = tf.nn.dropout(reshape, dropout_prob if train else 1.0)
        common.activation_summary(reshape, train)
        if not enable_global_avg_pool:
            common.activation_image_summary(tf.reshape(reshape, [batch_size, current_map_side, current_map_side, old_n_fileters]), reshape.op.name, 3, train)
    with tf.variable_scope('logits', reuse=reuse):
        weight_init = tf.learn.xavier_initializer(current_n_filters, n_classes, uniform=uniform)
        fc_weights = common.variable_with_weight_decay('fc_weights', shape=[current_n_filters, n_classes],
                                                       weight_init=weight_init, weight_decay=weight_decay, train=train)
        fc_biases = tf.get_variable('fc_biases', [n_classes], initializer=tf.constant_initializer(0.0))
        logits = tf.nn.xw_plus_b(reshape, fc_weights, fc_biases)
        common.activation_summary(logits, train)
        common.print_shape(logits)
    return logits, common.cross_entropy_weight_decay_losses(logits, labels) if train else None

def get_cnn(train, images, labels, initial_n_filters, initial_kernel_size, initial_subsample,
            residual_blks_per_subsample, min_map_side_for_conv, uniform, weight_decay,
            dropout_prob, enable_global_avg_pool, activation, reuse):
    batch_size, img_side, img_side, img_channels = images.get_shape().as_list()
    batch_size, n_classes = labels.get_shape().as_list()
    return get_cnn_common(train, images, labels, n_classes, batch_size, img_channels, img_side, initial_n_filters,
                          initial_kernel_size, initial_subsample, residual_blks_per_subsample, min_map_side_for_conv,
                          uniform, weight_decay, dropout_prob, enable_global_avg_pool, activation, reuse)

class ResNet:
    def __init__(self, initial_n_filters=16, initial_kernel_size=3, initial_subsample=1,
                 residual_blks_per_subsample=2, min_map_side_for_conv=6, uniform=True,
                 weight_decay=None, dropout_prob=None, enable_global_avg_pool=True,
                 activation=tf.nn.relu):
                    self.params = {}
                    self.params.update(locals())
                    del self.params["self"]

    def logits_train(self, images, labels, reuse=False):
        self.params['train'] = True
        self.params['images'] = images
        self.params['labels'] = labels
        self.params['reuse'] = reuse
        return get_cnn(**self.params)

    def logits_eval(self, images, labels, reuse=False):
        self.params['train'] = False
        self.params['images'] = images
        self.params['labels'] = labels
        self.params['reuse'] = reuse
        return get_cnn(**self.params)
