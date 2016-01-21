import tensorflow as tf
from tensorflow.python.ops import attention_ops # it is needed to avoid bug when calling extract_glimpse
from tensorflowed.models import common
import numpy as np
import math
import ipdb


def get_residual_block(activation, idx, input_, kernel_size, n_in_channels, n_out_channels, subsample=1, uniform=True, weight_decay=None, residual=True, suffix=''):
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
            acti1 = activation(conv_plus_bias)
            common.activation_summary(acti1)
            common.activation_image_summary(acti1, acti1.op.name+'/acti1', 3)
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
                                                                   weight_init=weight_init, weight_decay=weight_decay)
                        shortcut = tf.nn.conv2d(subsampled, kernel, [1, 1, 1, 1], padding='SAME')
                    else:
                        shortcut = subsampled
                    common.activation_summary(shortcut)
            else:
                shortcut = input_
        with tf.variable_scope("conv2"):
            n_channels = n_out_channels
            n_filters = n_out_channels
            conv_plus_bias = common.get_conv_plus_bias_block(acti1, kernel_size, n_channels, n_filters, uniform=uniform,
                                                             weight_decay=weight_decay, subsample=1)
            common.activation_summary(conv_plus_bias)
            if residual:
                conv_plus_bias = conv_plus_bias + shortcut
            common.activation_summary(conv_plus_bias)
            acti2 = activation(conv_plus_bias)
            common.activation_summary(acti2)
            common.activation_image_summary(acti2, acti2.op.name+'/acti2', 3)
    return acti2

# suppose square input
def get_cnn_common(imgs_pl, y_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters,
                   initial_kernel_size, initial_subsample, residual_blks_per_subsample=2,
                   min_map_side_for_conv=6, uniform=True, weight_decay=None, crop_level=None, concat_center_pixel=False,
                   enable_dropout=False, enable_global_avg_pool=True, activation=tf.nn.relu):
    if concat_center_pixel and crop_level is not None:
        raise ValueError
    current_n_filters = initial_n_filters
    kernel_size = 3
    common.print_shape(imgs_pl)
    with tf.variable_scope('conv0'):
        conv_plus_bias = common.get_conv_plus_bias_block(imgs_pl, initial_kernel_size, img_channels, current_n_filters,
                                                         uniform=uniform, weight_decay=weight_decay, subsample=initial_subsample)
        acti0 = activation(conv_plus_bias)
        common.activation_image_summary(acti0, 'acti0', 3)
        common.activation_summary(acti0)
        common.print_shape(acti0)
    current_map_side = img_side/initial_subsample
    last_output = acti0
    count = 1
    subsample_count = 0
    last_output_crop = None
    while True:
        for i in xrange(1, residual_blks_per_subsample+1):
            last_output = get_residual_block(activation, count, last_output, 3, current_n_filters, current_n_filters, uniform=uniform, weight_decay=weight_decay)
            if last_output_crop is not None:
                last_output_crop = get_residual_block(activation, count, last_output_crop, 3, current_n_filters,
                                                      current_n_filters, suffix='crop', uniform=uniform, weight_decay=weight_decay)
            count += 1
        if current_map_side/2 < min_map_side_for_conv:
            if crop_level == -1:
                crop = tf.image.extract_glimpse(imgs_pl, [current_map_side, current_map_side], np.zeros((batch_size,2), np.float32))
                common.activation_summary(crop)
                last_output_crop = get_residual_block(activation, count, crop, 3, img_channels, current_n_filters,
                                                      suffix='crop', uniform=uniform, weight_decay=weight_decay)
                count += 1
                for i in xrange(1, residual_blks_per_subsample):
                    last_output_crop = get_residual_block(activation, count, last_output_crop, 3, current_n_filters,
                                                          current_n_filters, suffix='crop', uniform=uniform, weight_decay=weight_decay)
                    count += 1
            break
        next_n_filters = 2*current_n_filters
        last_output = get_residual_block(activation, count, last_output, 3, current_n_filters, next_n_filters, subsample=2, uniform=uniform, weight_decay=weight_decay)
        if last_output_crop is not None:
            last_output_crop = get_residual_block(activation, count, last_output_crop, 3, current_n_filters, next_n_filters, subsample=2, suffix='crop', uniform=uniform, weight_decay=weight_decay)
        common.print_shape(last_output)
        count += 1
        subsample_count += 1
        current_n_filters = next_n_filters
        current_map_side = last_output.get_shape().as_list()[1]
        if crop_level is not None:
            if subsample_count == crop_level:
                crop = tf.image.extract_glimpse(imgs_pl, [current_map_side, current_map_side], np.zeros((batch_size,2), np.float32))
                common.activation_summary(crop)
                with tf.variable_scope('inc_filters_crop'):
                    last_output_crop = common.get_conv_plus_bias_block(crop, 3, img_channels, current_n_filters,
                                                                       uniform=uniform, weight_decay=weight_decay)
                    common.activation_summary(last_output_crop)
    # global avg pooling
    if enable_global_avg_pool:
        ksize = strides = [1, current_map_side, current_map_side, 1]
        last_output = tf.nn.avg_pool(last_output, ksize, strides, padding='VALID')
        common.activation_summary(last_output)
        common.print_shape(last_output)
        if last_output_crop is not None:
            last_output_crop = tf.nn.avg_pool(last_output_crop, ksize, strides, padding='VALID')
            common.activation_summary(last_output_crop)
            common.print_shape(last_output_crop)
    else:
        current_n_filters = current_n_filters*(current_map_side**2)
    reshape = tf.reshape(last_output, [batch_size, current_n_filters])
    common.print_shape(reshape)
    if last_output_crop is not None:
        crop_reshape = tf.reshape(last_output_crop, [batch_size, current_n_filters])
        common.print_shape(crop_reshape)
    if concat_center_pixel:
        crop = tf.image.extract_glimpse(imgs_pl, [1, 1], np.zeros((batch_size,2), np.float32))
        crop_reshape = tf.reshape(crop, [batch_size, img_channels])
        current_n_filters = img_channels
        n_hid = 10
        weight_init = tf.learn.xavier_initializer(current_n_filters, n_hid, uniform=uniform)
        weights = common.variable_with_weight_decay('concat_weights', shape=[current_n_filters, n_hid],
                                                       weight_init=weight_init, weight_decay=weight_decay)
        biases = tf.get_variable('concat_biases', [n_hid], initializer=tf.constant_initializer(0.0))
        crop_reshape = activation(tf.nn.xw_plus_b(crop_reshape, weights, biases))
        common.print_shape(crop_reshape)
    if concat_center_pixel or last_output_crop is not None:
        reshape = tf.concat(1, [reshape, crop_reshape])
        common.print_shape(reshape)
        current_n_filters = reshape.get_shape().as_list()[1]
    # fully connected and softmax
    if enable_dropout:
        keep_prob = tf.placeholder(tf.float32)
        reshape = tf.nn.dropout(reshape, keep_prob)
    else:
        keep_prob = None
    weight_init = tf.learn.xavier_initializer(current_n_filters, n_classes, uniform=uniform)
    fc_weights = common.variable_with_weight_decay('fc_weights', shape=[current_n_filters, n_classes],
                                                   weight_init=weight_init, weight_decay=weight_decay)
    fc_biases = tf.get_variable('fc_biases', [n_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.nn.xw_plus_b(reshape, fc_weights, fc_biases)
    common.activation_summary(logits)
    common.print_shape(logits)
    return logits, common.loss(logits, y_pl), keep_prob

def get_cnn(n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size, initial_subsample,
            residual_blks_per_subsample=2, min_map_side_for_conv=6, uniform=True, weight_decay=None, crop_level=None, concat_center_pixel=False,
            enable_dropout=False, enable_global_avg_pool=True, activation=tf.nn.relu):
    imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_side, img_side, img_channels))
    y_pl = tf.placeholder(tf.float32, shape=(batch_size, n_classes))
    logits, loss = get_cnn_common(imgs_pl, y_pl, n_classes, batch_size, img_channels, img_side,
                                  initial_n_filters, initial_kernel_size, initial_subsample,
                                  residual_blks_per_subsample, min_map_side_for_conv, uniform,
                                  weight_decay, crop_level, concat_center_pixel, enable_dropout, enable_global_avg_pool, activation)
    return imgs_pl, y_pl, logits, loss

def get_cnn_given_pl(imgs_pl, y_pl, n_classes, initial_n_filters, initial_kernel_size, initial_subsample,
                     residual_blks_per_subsample=2, min_map_side_for_conv=6, uniform=True, weight_decay=None,
                     crop_level=None, concat_center_pixel=False, enable_dropout=False, enable_global_avg_pool=True, activation=tf.nn.relu):
    batch_size, img_side, img_side, img_channels = imgs_pl.get_shape().as_list()
    return get_cnn_common(imgs_pl, y_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters,
                          initial_kernel_size, initial_subsample, residual_blks_per_subsample, min_map_side_for_conv,
                          uniform, weight_decay, crop_level, concat_center_pixel, enable_dropout, enable_global_avg_pool, activation)

def gen_graph01():
    sess = tf.Session()
    logits = get_residual_cnn(10, 128, 1, 28, 16, 3, 1)
    writer = tf.train.SummaryWriter('/tmp/residual01', sess.graph_def)
    tf.initialize_all_variables().run(session=sess)
    writer.close()
    sess.close()
