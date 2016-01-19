import tensorflow as tf


def add_loss_summaries(total_loss):
    losses = tf.get_collection('wd_losses')+[total_loss]
    for l in losses:
        tf.scalar_summary(l.op.name, l)
    return loss_averages_op

def activation_summary(x):
    op_name =  x.op.name
    tf.histogram_summary(op_name + '/activations', x)
    tf.scalar_summary(op_name + '/sparsity', tf.nn.zero_fraction(x))

def loss(logits, onehot_labels):
    top_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels))
    wd_loss = tf.add_n(tf.get_collection('wd_losses'), name='wd_loss')
    total_loss = top_loss + wd_loss
    return total_loss, top_loss, wd_loss

def variable_with_weight_decay(name, shape, weight_init, weight_decay):
    var = tf.get_variable(name, shape, initializer=weight_init)
    if weight_decay is not None and weight_decay > 0:
        weight_decay_loss = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('wd_losses', weight_decay)
    return var

def get_conv_plus_bias_block(input_, kernel_size, n_channels, n_filters, uniform=True, weight_decay=None, subsample=1):
    patch_size = kernel_size**2
    weight_init = tf.learn.xavier_initializer(n_channels*patch_size, n_filters*patch_size)
    kernel = variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, n_channels, n_filters],
                                        weight_init=weight_init, weight_decay=weight_decay)
    biases = tf.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input_, kernel, [1, subsample, subsample, 1], padding='SAME')
    conv_plus_bias = tf.nn.bias_add(conv, biases)
    return conv_plus_bias
