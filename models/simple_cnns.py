import tensorflow as tf
import ipdb


def var_truncated_normal(name, shape, stddev):
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    return var

def get_conv_block(input_, kernel_size, n_channels, n_filters, conv_stride=1, stddev=0.1, activation=tf.nn.relu, pool_ksize=None, pool_stride=None, pool=tf.nn.max_pool):
    kernel = var_truncated_normal('weights', shape=[kernel_size, kernel_size, n_channels, n_filters],
                                  stddev=stddev)
    biases = tf.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input_, kernel, [1, conv_stride, conv_stride, 1], padding='SAME')
    conv_plus_bias = tf.nn.bias_add(conv, biases)
    act = activation(conv_plus_bias)
    if pool_ksize is not None:
        if pool_stride is None:
            pool_stride = pool_ksize
        result = pool(act, ksize=[1, pool_ksize, pool_ksize, 1],
                      strides=[1, pool_stride, pool_stride, 1],
                      padding='SAME', name='pool')
    else:
        result = act
    return result

def get_cnn(imgs_pl, n_classes, initial_n_filters, n_conv_blocks=2):
    kernel_size = 3
    batch_size, h, w, n_channels = imgs_pl.get_shape().as_list()
    prev_n_filters = n_channels
    current_n_filters = initial_n_filters
    block_result = imgs_pl
    for i in xrange(n_conv_blocks):
        with tf.variable_scope('conv%d'%(i+1)):
            block_result = get_conv_block(block_result, kernel_size, prev_n_filters, current_n_filters, pool_ksize=2)
        prev_n_filters = current_n_filters
        current_n_filters = prev_n_filters*2
    _, map_h, map_w, map_n_channels = block_result.get_shape().as_list()
    vec_size = map_h*map_w*map_n_channels
    reshape = tf.reshape(block_result, [batch_size, vec_size])
    fc_weights = tf.get_variable('fc_weights', [vec_size, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc_biases = tf.get_variable('fc_biases', [n_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(reshape, fc_weights) + fc_biases
    return logits
