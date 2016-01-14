import tensorflow as tf
import ipdb

def variable_with_weight_decay(name, shape, stddev, weight_decay):
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    if weight_decay:
        weight_decay_loss = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def get_conv_plus_bias_block(input_, kernel_size, n_channels, n_filters, subsample=1, stddev=0.1, weight_decay=0.0):
    kernel = variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, n_channels, n_filters],
                                        stddev=stddev, weight_decay=weight_decay)
    biases = tf.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input_, kernel, [1, subsample, subsample, 1], padding='SAME')
    conv_plus_bias = tf.nn.bias_add(conv, biases)
    return conv_plus_bias

def get_residual_block(idx, input_, kernel_size, n_in_channels, n_out_channels, stddev=0.1, weight_decay=0.0, residual=True):
    subsample = n_out_channels/n_in_channels
    if subsample not in (1, 2):
        raise ValueError
    if residual:
        prefix = ''
    else:
        prefix = 'non_'
    with tf.variable_scope("%sresidual%02d" % (prefix, idx)):
        with tf.variable_scope("conv1"):
            n_channels = n_in_channels
            n_filters = n_out_channels
            conv_plus_bias = get_conv_plus_bias_block(input_, kernel_size, n_channels, n_filters, subsample=subsample,
                                                      stddev=stddev, weight_decay=weight_decay)
            relu1 = tf.nn.relu(conv_plus_bias)
        if residual:
            if subsample != 1:
                with tf.variable_scope("shortcut"):
                    n_channels = n_in_channels
                    n_filters = n_out_channels
                    h, w = relu1.get_shape().as_list()[1:3]
                    subsampled = tf.image.resize_nearest_neighbor(input_, [h, w])
                    kernel = variable_with_weight_decay('weights', shape=[1, 1, n_channels, n_filters],
                                                        stddev=stddev, weight_decay=weight_decay)
                    shortcut = tf.nn.conv2d(subsampled, kernel, [1, 1, 1, 1], padding='SAME')
            else:
                shortcut = input_
        with tf.variable_scope("conv2"):
            n_channels = n_out_channels
            n_filters = n_out_channels
            conv_plus_bias = get_conv_plus_bias_block(relu1, kernel_size, n_channels, n_filters, subsample=1,
                                                      stddev=stddev, weight_decay=weight_decay)
            if residual:
                conv_plus_bias = conv_plus_bias + shortcut
            relu2 = tf.nn.relu(conv_plus_bias)
    return relu2

def get_cnn_common(imgs_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size, initial_subsample,
                     residual_blks_per_subsample=2, min_map_side_for_conv=6):
    current_n_filters = initial_n_filters
    kernel_size = 3
    with tf.variable_scope('conv0'):
        conv_plus_bias = get_conv_plus_bias_block(imgs_pl, initial_kernel_size, img_channels, current_n_filters, subsample=initial_subsample)
        relu0 = tf.nn.relu(conv_plus_bias)
    current_map_side = img_side/initial_subsample
    last_output = relu0
    count = 1
    while True:
        for i in xrange(1, residual_blks_per_subsample+1):
            last_output = get_residual_block(count, last_output, 3, current_n_filters, current_n_filters)
            count += 1
        if current_map_side/2 < min_map_side_for_conv:
            break
        next_n_filters = 2*current_n_filters
        last_output = get_residual_block(count, last_output, 3, current_n_filters, next_n_filters)
        count += 1
        current_n_filters = next_n_filters
        current_map_side = last_output.get_shape().as_list()[1]
    # global avg pooling
    ksize = strides = [1, current_map_side, current_map_side, 1]
    avg_pool = tf.nn.avg_pool(last_output, ksize, strides, padding='VALID')
    reshape = tf.reshape(avg_pool, [batch_size, current_n_filters])
    # fully connected and softmax
    fc_weights = tf.get_variable('fc_weights', [current_n_filters, n_classes], initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc_biases = tf.get_variable('fc_biases', [n_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(reshape, fc_weights) + fc_biases
    return logits

def get_cnn(n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size, initial_subsample,
                     residual_blks_per_subsample=2, min_map_side_for_conv=6):
    imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_side, img_side, img_channels))
    logits = get_cnn_common(imgs_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size,
                               initial_subsample, residual_blks_per_subsample, min_map_side_for_conv)
    return imgs_pl, logits

def get_cnn_given_pl(imgs_pl, n_classes, initial_n_filters, initial_kernel_size, initial_subsample,
                      residual_blks_per_subsample=2, min_map_side_for_conv=6):
    batch_size, img_side, img_side, img_channels = imgs_pl.get_shape().as_list()
    return get_cnn_common(imgs_pl, n_classes, batch_size, img_channels, img_side, initial_n_filters, initial_kernel_size, initial_subsample,
                          residual_blks_per_subsample, min_map_side_for_conv)

def gen_graph01():
    sess = tf.Session()
    logits = get_residual_cnn(10, 128, 1, 28, 16, 3, 1)
    writer = tf.train.SummaryWriter('/tmp/residual01', sess.graph_def)
    tf.initialize_all_variables().run(session=sess)
    writer.close()
    sess.close()
