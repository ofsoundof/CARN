__author__ = 'yawli'

import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.io as sio
import numpy as np

def rgb_to_ycbcr(image_rgb): #batch x H x W x C
    """
    convert image from rgb to ycbcr
    :param image_rgb:
    :return:
    """
    image_r = tf.squeeze(tf.slice(image_rgb, [0, 0, 0, 0], [-1, -1, -1, 1]), axis=3)
    image_g = tf.squeeze(tf.slice(image_rgb, [0, 0, 0, 1], [-1, -1, -1, 1]), axis=3)
    image_b = tf.squeeze(tf.slice(image_rgb, [0, 0, 0, 2], [-1, -1, -1, 1]), axis=3)
    image_y = 16 + (65.738 * image_r + 129.057 * image_g + 25.064 * image_b)/256
    image_cb = 128 + (-37.945 * image_r - 74.494 * image_g + 112.439 * image_b)/256
    image_cr = 128 + (112.439 * image_r - 94.154 * image_g - 18.285 * image_b)/256
    image_ycbcr = tf.stack([image_y, image_cb, image_cr], axis=3)
    return image_ycbcr

def ycbcr_to_rgb(image_ycbcr):
    """
    convert image from ycbcr back to rgb
    :param image_ycbcr:
    :return:
    """
    image_y = tf.squeeze(tf.slice(image_ycbcr, [0, 0, 0, 0], [-1, -1, -1, 1]), axis=3)
    image_cb = tf.squeeze(tf.slice(image_ycbcr, [0, 0, 0, 1], [-1, -1, -1, 1]), axis=3)
    image_cr = tf.squeeze(tf.slice(image_ycbcr, [0, 0, 0, 2], [-1, -1, -1, 1]), axis=3)
    image_r = (298.082 * image_y + 408.583 * image_cr)/256 - 222.921
    image_g = (298.082 * image_y - 100.291 * image_cb - 208.120 * image_cr)/256 + 135.576
    image_b = (298.082 * image_y + 516.412 * image_cb)/256 - 276.836

    image_r = tf.maximum(0.0, tf.minimum(255.0, image_r))
    image_g = tf.maximum(0.0, tf.minimum(255.0, image_g))
    image_b = tf.maximum(0.0, tf.minimum(255.0, image_b))

    image_rgb = tf.stack([image_r, image_g, image_b], axis=3)
    return image_rgb

def pixelShuffler(inputs, scale):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)

def carn_rgb(data_mid):
    """
    carn for super-resolution. For PIRM challenge.
    :param data_mid: interpolated RGB image
    :return: RGB super-resolved image
    """
    num_anchor = 16 #FLAGS.deep_anchor
    inner_channel = 16 #FLAGS.deep_channel
    deep_feature_layer = 3
    deep_kernel = 3
    deep_layer = 7
    upscale = 4

    with slim.arg_scope([slim.conv2d], stride=1,
                        weights_initializer=tf.keras.initializers.he_normal(),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
        feature = slim.conv2d(data_mid, 64, [3, 3], stride=1, scope='feature_layer1', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, 64, [3, 3], stride=2, scope='feature_layer2', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, inner_channel, [3, 3], stride=2, scope='feature_layer3', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        reshape_size = tf.shape(feature)
        kernel_size = deep_kernel

        def regression_layer(input_feature, r, k, dim, num, flag_shortcut, l):
            """
            the regression block
            :param input_feature:
            :param r: reshape size
            :param k: kernel size
            :param dim: dimension of inner channel
            :param num: number of anchors
            :param flag_shortcut: short cut flag
            :param l: regression block identifier
            :return: regression result (output feature)
            """
            with tf.name_scope('regression_layer' + l):
                result = slim.conv2d(input_feature, num*dim, [k, k], scope='regression_' + l, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
                result = tf.reshape(result, [r[0], r[1], r[2], num, dim])   # B x H x W 2^R x s^2
                alpha = slim.conv2d(input_feature, num, [k, k], scope='alpha_' + l, activation_fn=tf.nn.softmax)  # B x H x W x R  filter k_size x k_size x Cin x R
                alpha = tf.expand_dims(alpha, 4)
                output_feature = tf.reduce_sum(result * alpha, axis=3)
                if flag_shortcut:
                    return output_feature + input_feature
                else:
                    return output_feature
        if deep_layer == 1:
            regression = regression_layer(feature, reshape_size, kernel_size, upscale ** 2 * 3, num_anchor, inner_channel == upscale**2, '1')
        else:
            regression = regression_layer(feature, reshape_size, kernel_size, inner_channel, num_anchor, True, '1')
            for i in range(2, deep_layer):
                regression = regression_layer(regression, reshape_size, kernel_size, inner_channel, num_anchor, True, str(i))
            regression = regression_layer(regression, reshape_size, kernel_size, upscale ** 2 * 3, num_anchor, inner_channel == upscale ** 2 * 3, str(deep_layer))

    sr_space = tf.depth_to_space(regression, upscale, name='sr_space')
    sr = sr_space + data_mid

    return sr

def carn_rgb_y(data_mid):
    """
    different from carn_rgb. This function only refines the Y channel.For PIRM challenge.
    :param interpolated RGB image
    :return: RGB image with only Y channel super-resolved
    """
    num_anchor = 16 #FLAGS.deep_anchor
    inner_channel = 16 #FLAGS.deep_channel
    deep_feature_layer = 3
    deep_kernel = 3
    deep_layer = 3
    upscale = 4

    data_mid_ycbcr = rgb_to_ycbcr(data_mid*255)
    data_mid_y = tf.slice(data_mid_ycbcr, [0, 0, 0, 0], [-1, -1, -1, 1])/255

    # activation = tf.keras.layers.PReLU(shared_axes=[1, 2]) #if FLAGS.activation_regression == 1 else None
    # biases_add = tf.zeros_initializer() #if FLAGS.biases_add_regression == 1 else None

    with slim.arg_scope([slim.conv2d], stride=1,
                        weights_initializer=tf.keras.initializers.he_normal(),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
        feature = slim.conv2d(data_mid_y, 64, [3, 3], stride=1, scope='feature_layer1')#, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, 64, [3, 3], stride=2, scope='feature_layer2')#, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, inner_channel, [3, 3], stride=2, scope='feature_layer3')#, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        reshape_size = tf.shape(feature)
        kernel_size = deep_kernel

        def regression_layer(input_feature, r, k, dim, num, flag_shortcut, l):
            with tf.name_scope('regression_layer' + l):
                result = slim.conv2d(input_feature, num*dim, [k, k], scope='regression_' + l, activation_fn=None)  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
                result = tf.reshape(result, [r[0], r[1], r[2], num, dim])   # B x H x W 2^R x s^2
                alpha = slim.conv2d(input_feature, num, [k, k], scope='alpha_' + l, activation_fn=tf.nn.softmax)  # B x H x W x R  filter k_size x k_size x Cin x R
                alpha = tf.expand_dims(alpha, 4)
                output_feature = tf.reduce_sum(result * alpha, axis=3)
                if flag_shortcut:
                    return output_feature + input_feature
                else:
                    return output_feature
        if deep_layer == 1:
            regression = regression_layer(feature, reshape_size, kernel_size, upscale ** 2, num_anchor, inner_channel==upscale**2, '1')
        else:
            regression = regression_layer(feature, reshape_size, kernel_size, inner_channel, num_anchor, True, '1')
            for i in range(2, deep_layer):
                regression = regression_layer(regression, reshape_size, kernel_size, inner_channel, num_anchor, True, str(i))
            regression = regression_layer(regression, reshape_size, kernel_size, upscale ** 2, num_anchor, inner_channel == upscale ** 2, str(deep_layer))
        # regression = slim.conv2d(regression, inner_channel, [3, 3], scope='feature_layer9', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        # regression = slim.conv2d(regression, inner_channel, [3, 3], scope='feature_layer10', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1

    sr_space = tf.depth_to_space(regression, upscale, name='sr_space')
    sr_y = sr_space + data_mid_y
    sr_ycbcr = tf.concat([sr_y*255, tf.slice(data_mid_ycbcr, [0, 0, 0, 1], [-1, -1, -1, -1])], axis=3)
    sr = ycbcr_to_rgb(sr_ycbcr)/255

    return sr

def carn_sr_y(data_mid):
    """
    this function only refines the y channel. For PIRM challenge.
    :param data_mid: luminance image
    :return: super-resolved luminance image.
    """
    num_anchor = 16 #FLAGS.deep_anchor
    inner_channel = 16 #FLAGS.deep_channel
    deep_feature_layer = 3
    deep_kernel = 3
    deep_layer = 3
    upscale = 4

    with slim.arg_scope([slim.conv2d], stride=1,
                        weights_initializer=tf.keras.initializers.he_normal(),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
        feature = slim.conv2d(data_mid, 64, [3, 3], stride=1, scope='feature_layer1')#, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, 64, [3, 3], stride=2, scope='feature_layer2')#, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, inner_channel, [3, 3], stride=2, scope='feature_layer3')#, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        reshape_size = tf.shape(feature)
        kernel_size = deep_kernel

        def regression_layer(input_feature, r, k, dim, num, flag_shortcut, l):
            with tf.name_scope('regression_layer' + l):
                result = slim.conv2d(input_feature, num*dim, [k, k], scope='regression_' + l, activation_fn=None)  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
                result = tf.reshape(result, [r[0], r[1], r[2], num, dim])   # B x H x W 2^R x s^2
                alpha = slim.conv2d(input_feature, num, [k, k], scope='alpha_' + l, activation_fn=tf.nn.softmax)  # B x H x W x R  filter k_size x k_size x Cin x R
                alpha = tf.expand_dims(alpha, 4)
                output_feature = tf.reduce_sum(result * alpha, axis=3)
                if flag_shortcut:
                    return output_feature + input_feature
                else:
                    return output_feature
        if deep_layer == 1:
            regression = regression_layer(feature, reshape_size, kernel_size, upscale ** 2, num_anchor, inner_channel==upscale**2, '1')
        else:
            regression = regression_layer(feature, reshape_size, kernel_size, inner_channel, num_anchor, True, '1')
            for i in range(2, deep_layer):
                regression = regression_layer(regression, reshape_size, kernel_size, inner_channel, num_anchor, True, str(i))
            regression = regression_layer(regression, reshape_size, kernel_size, upscale ** 2, num_anchor, inner_channel == upscale ** 2, str(deep_layer))

    sr_space = tf.depth_to_space(regression, upscale, name='sr_space')
    sr = sr_space + data_mid

    return sr

def carn(data, data_mid, FLAGS, step):
    """
    CARN (convolutional anchored regression network) for single image super-resolution.
    :param data: low-resolution image
    :param data_mid: interpolated image
    :param FLAGS:
    :param step:
    :return: sr: super-resolved image
    """
    num_anchor = FLAGS.deep_anchor
    inner_channel = FLAGS.deep_channel
    deep_kernel = FLAGS.deep_kernel
    deep_layer = FLAGS.deep_layer
    upscale = FLAGS.upscale
    # activation = tf.keras.layers.PReLU(shared_axes=[1, 2]) if FLAGS.activation_regression == 1 else None
    # biases_add = tf.zeros_initializer() if FLAGS.biases_add_regression == 1 else None
    with slim.arg_scope([slim.conv2d], stride=1,
                        weights_initializer=tf.keras.initializers.he_normal(),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
        feature = slim.conv2d(data, 64, [3, 3], stride=1, scope='feature_layer1', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
        feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer2', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
        feature = slim.conv2d(feature, inner_channel, [3, 3], stride=1, scope='feature_layer3', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
        reshape_size = tf.shape(feature)
        kernel_size = deep_kernel

        def regression_layer(input_feature, r, k, dim, num, flag_shortcut, l):
            """
            the regression block
            :param input_feature:
            :param r: reshape size
            :param k: kernel size
            :param dim: dimension of inner channel
            :param num: number of anchors
            :param flag_shortcut: short cut flag
            :param l: regression block identifier
            :return: regression result (output feature)
            """
            with tf.name_scope('regression_layer' + l):
                # PReLU activation function used in the regression block
                result = slim.conv2d(input_feature, num*dim, [k, k], scope='regression_' + l, activation_fn=None)
                result = tf.reshape(result, [r[0], r[1], r[2], num, dim])

                # bias used in the similarity layer
                alpha = slim.conv2d(input_feature, num, [k, k], scope='alpha_' + l, activation_fn=tf.nn.softmax)
                alpha = tf.expand_dims(alpha, 4)
                output_feature = tf.reduce_sum(result * alpha, axis=3)
                if flag_shortcut:
                    return output_feature + input_feature
                else:
                    return output_feature
        if deep_layer == 1:
            regression = regression_layer(feature, reshape_size, kernel_size, upscale ** 2, num_anchor, inner_channel == upscale**2, '1')
        else:
            regression = regression_layer(feature, reshape_size, kernel_size, inner_channel, num_anchor, True, '1')
            for i in range(2, deep_layer):
                regression = regression_layer(regression, reshape_size, kernel_size, inner_channel, num_anchor, True, str(i))
            regression = regression_layer(regression, reshape_size, kernel_size, upscale ** 2, num_anchor, inner_channel == upscale ** 2, str(deep_layer))

    sr_space = tf.depth_to_space(regression, upscale, name='sr_space')
    sr = sr_space + data_mid
    return sr, reshape_size[0]

def vdsr(data, data_mid, FLAGS, step):
    feature = slim.conv2d(data, 64, [3, 3], stride=1, scope='feature_layer1', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer2', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer3', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))
    for i in range(4, 11):
        feature = slim.conv2d(feature, 16, [3, 3], stride=1, scope='feature_layer{}'.format(i), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))
    #feature = slim.conv2d(feature, 1, [3, 3], stride=1, scope='feature_layer20', activation_fn=None, weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))
    sr_space = tf.depth_to_space(feature, 4, name='to_space')
    image = sr_space + data_mid
    return image, step

def srcnn(data, data_mid, FLAGS, global_step):
    net = slim.conv2d(data_mid, 64, [9, 9], stride=1, scope='extraction')
    net = slim.conv2d(net, 32, [5, 5], stride=1, scope='mapping')
    net = slim.conv2d(net, 1, [5, 5], stride=1, scope='reconstruction', activation_fn=None)
    return net, tf.shape(net)[0]

def espcn_comp(data, data_mid, FLAGS, global_step):
    with slim.arg_scope([slim.conv2d], stride=1,
                        weights_initializer=tf.keras.initializers.he_normal(),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
        net = slim.conv2d(data, 64, [5, 5], scope='feature_layer1', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
        for i in range(2, FLAGS.deep_feature_layer):
            net = slim.conv2d(net, 32, [3, 3], scope='feature_layer' + str(i), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
        net = slim.conv2d(net, FLAGS.upscale ** 2, [3, 3], scope='output_layer', activation_fn=None)
    sr_space = tf.depth_to_space(net, FLAGS.upscale, name='sr_space')
    sr = sr_space + data_mid
    return sr, global_step

def espcn(data, data_mid, FLAGS, global_step):

    net = slim.conv2d(data, 64, [5, 5], stride=1, scope='layer_one')
    net = slim.conv2d(net, 32, [3, 3], stride=1, scope='layer_two')
    net = slim.conv2d(net, FLAGS.upscale**2, [3, 3], stride=1, scope='layer_three', activation_fn=tf.nn.tanh)
    sr_space = pixelShuffler(net, FLAGS.upscale)
    # sr_space = tf.depth_to_space(net, FLAGS.upscale, name='sr_space')
    sr = sr_space + data_mid
    return sr, tf.shape(net)[0]

def srresnet(data, data_mid, FLAGS, global_step):

    gen_output_channels = 3
    is_training = len(FLAGS.test_dir) == 0
    num_resblock = FLAGS.deep_feature_layer
    def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
        # kernel: An integer specifying the width and height of the 2D convolution window
        with tf.variable_scope(scope):
            if use_bias:
                return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                                activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            else:
                return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                                activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=None)
    def batchnorm(inputs, is_training):
        return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                            scale=False, fused=True, is_training=is_training)

    def prelu_tf(inputs, name='Prelu'):
        with tf.variable_scope(name):
            alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5

        return pos + neg
    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, is_training)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, is_training)
            net = net + inputs

        return net

    def pixelShuffler(inputs, scale=2):
        size = tf.shape(inputs)
        batch_size = size[0]
        h = size[1]
        w = size[2]
        c = inputs.get_shape().as_list()[-1]

        # Get the target channel size
        channel_target = c // (scale * scale)
        channel_factor = c // channel_target

        shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
        shape_2 = [batch_size, h * scale, w * scale, 1]

        # Reshape and transpose for periodic shuffling for each channel
        input_split = tf.split(inputs, channel_target, axis=3)
        output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

        return output


    def phaseShift(inputs, scale, shape_1, shape_2):
        # Tackle the condition when the batch is None
        X = tf.reshape(inputs, shape_1)
        X = tf.transpose(X, [0, 1, 3, 2, 4])

        return tf.reshape(X, shape_2)

    with tf.variable_scope('generator_unit', reuse=tf.AUTO_REUSE):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(data, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, is_training)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net, tf.shape(net)[0]

def fsrcnn(data, data_mid, FLAGS, global_step):
    d = 56
    s = 12
    m = 4
    with slim.arg_scope([slim.conv2d], stride=1, weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001)):
        net = slim.conv2d(data, d, [5, 5], activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), scope='extraction')
        net = slim.conv2d(net, s, [1, 1], activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), scope='shrinking')
        net = slim.repeat(net, m, slim.conv2d, s, [3, 3], activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), scope='mapping')
        net = slim.conv2d(net, d, [1, 1], activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), scope='expansion')
        net = slim.conv2d_transpose(net, 1, [9, 9], stride=FLAGS.upscale, activation_fn=None, biases_initializer=None)

    return net, tf.shape(net)[0]

def gen_anchors(num_anchor):
    #generate fixed anchors
    if num_anchor == 1:
        return [[-1], [1]]
    else:
         anchors = []
         anchors.extend([-1] + a for a in gen_anchors(num_anchor-1))
         anchors.extend([1] + a for a in gen_anchors(num_anchor-1))
         return anchors

def hardmax(x):
    #Softmax returns a 'soft' one-hot encoding of the largest value
    # E.g. softmax([1,2,5,3,1]) should be sth like [0.05,0.1,0.7,0.15,0.05] where the numbers add up to 1
    #Hardmax is defined by Eirikur as softmax(K*x) when K-> infty, so it should one-hot encode the largest value
    # E.g. hardmax([1,2,5,3,1]) = [0,0,1,0,0]

    #Here is one implementation of argmax, which is correct for all x which has the difference between the largest and 2nd largest entry to be more than  circa 1E-15
    return tf.round(tf.nn.softmax(1E20*x))

    #The rock solid implementation, actually uses argmax to find the index of the largest value, and then uses one-hot to convert that to [0,0,0,0,0,1,0,0,0,0].
    # [ 1E-10,2E-10,3E-10,4E-10]
    # [0,0,0.001,0.999]

def fast_hashnet(data, data_mid, FLAGS, global_step):

    with tf.name_scope("generate_anchors"):
        anchors = np.array(gen_anchors(FLAGS.regression))                                                           #anchor_num x anchor_dim    2^R x R
        anchors = tf.transpose(tf.constant(anchors, tf.float32), [1, 0])                                            # R x 2^R
        anchors = tf.Variable(anchors, name='anchors', trainable=False)                                             # R x 2^R
        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0)                                           # 1 x 1 x R x 2^R

    feature = slim.conv2d(data, 32, [3, 3], stride=1, scope='feature_layer1', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x Cin
    feature = slim.conv2d(feature, FLAGS.regression, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x R  filter k_size x k_size x Cin x R

    #regression
    k_size = 1
    regression = slim.conv2d(feature, 2**FLAGS.regression*FLAGS.upscale**2, [k_size, k_size], stride=1, scope='regression_layer', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
    r_size = tf.shape(regression)
    regression = tf.reshape(regression, [r_size[0], r_size[1], r_size[2], 2**FLAGS.regression, FLAGS.upscale**2])   # B x H x W 2^R x s^2

    weights_regression = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression_layer/weights') # k x k x Cin x 2^R x s^2
    biases_regression = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression_layer/biases')
    weights_regression = tf.squeeze(weights_regression, axis=[0, 1], name='weights_squeeze')                   # Cin x 2^R x s^2
    weights_regression = tf.reshape(weights_regression, [FLAGS.regression, 2**FLAGS.regression, FLAGS.upscale**2], name='weights_reshape')  # k x k x Cin x 2^R x s^2
    weights_regression = tf.Variable(tf.transpose(weights_regression, [1, 0, 2]), trainable=False, name='final_weights')       # 2^R x k x k x Cin x s^2
    biases_regression = tf.Variable(tf.reshape(biases_regression, [2**FLAGS.regression, FLAGS.upscale**2]), trainable=False, name='final_biases')  # 2^R x s^2

    #similar coefficient
    hashcode = tf.nn.conv2d(feature, anchors, strides=[1, 1, 1, 1], padding='SAME', name='similarity_layer')       # B x H x W x 2^R  anchor_size 1 x 1 x R x 2^R, anchor as the filter
    # sigma = 1E20
    # global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.float32)
    # if len(FLAGS.test_dir):
    #     annealing_factor = global_step
    # else:
    #     # annealing_factor = tf.cond(tf.less_equal(global_step, 50000), lambda: global_step, lambda: 50000)
    f1 = lambda: tf.constant(0)
    f2 = lambda: tf.constant(50000)
    f3 = lambda: global_step-150000
    annealing_factor = tf.case({tf.less_equal(global_step, 150000): f1, tf.greater_equal(global_step, 200000): f2}, default=f3, exclusive=True)
        # if tf.less_equal(global_step, 50000):
        #     global_step_increment = tf.assign(global_step, global_step+1)
        # else:
        #     global_step_increment = global_step
    # sigma = 0.9997697679981565 ** global_step_increment
    # sigma = 0.999539589003088 ** global_step_increment
    sigma = 0.999079389984462 ** tf.cast(annealing_factor, dtype=tf.float32)
    alpha = (tf.nn.softmax(hashcode/sigma))   # B x H x W x 2^R
    # alpha = tf.nn.softmax(hashcode*sigma)
    alpha = tf.expand_dims(alpha, 4)                                                                                # B x H x W x 2^R x 1

    #final results
    sr_depth = tf.reduce_sum(regression * alpha, axis=3, name='sr_depth')
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale,  name='sr_space')
    sr = sr_space + data_mid
    # global_step_increment = global_step
    return sr, annealing_factor

def fast_hashnet_restore(data, data_mid, FLAGS, global_step):

    feature = slim.conv2d(data, 32, [3, 3], stride=1, scope='feature_layer1', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]), weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x Cin
    feature = slim.conv2d(feature, FLAGS.regression, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_initializer=tf.keras.initializers.he_normal(), weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x R  filter k_size x k_size x Cin x R

    #hash code
    Cin = FLAGS.regression
    k_size = 1
    hashcode_hard = tf.cast((tf.sign(feature) + 1)/2, dtype=tf.int32)                                                                                                            # B x H x W x R
    #indices
    basis = [2**i for i in range(FLAGS.regression-1, -1, -1)]
    basis = tf.reshape(tf.constant(basis, dtype=tf.int32), [1, 1, 1, FLAGS.regression])  # 1 x 1 x 1 x R
    indices = tf.reduce_sum(hashcode_hard*basis, axis=3, keep_dims=True) # B x H x W x 1
    # weights and biases
    weights_regression = tf.Variable(tf.random_normal([2**FLAGS.regression, Cin, FLAGS.upscale**2], stddev=1e-3), name="final_weights") #k_size x k_size x Cin x 2^Rxs^2
    biases_regression = tf.Variable(tf.zeros([2**FLAGS.regression, FLAGS.upscale**2]), name="final_biases") #2^Rxs^2

    # weights_regression = tf.Variable(tf.random_normal([k_size, k_size, Cin, 2**FLAGS.regression*FLAGS.upscale**2], stddev=1e-3), name="regression_layer/weights") #k_size x k_size x Cin x 2^Rxs^2
    # biases_regression = tf.Variable(tf.zeros([2**FLAGS.regression* FLAGS.upscale**2]), name="regression_layer/biases") #2^Rxs^2
    # w_shape = tf.shape(weights_regression)
    # weights_regression = tf.reshape(weights_regression, [w_shape[0], w_shape[1], w_shape[2], 2**FLAGS.regression, FLAGS.upscale**2])  # k x k x Cin x 2^R x s^2
    # weights_regression = tf.transpose(weights_regression, [3, 0, 1, 2, 4])       # 2^R x k x k x Cin x s^2
    # biases_regression = tf.reshape(biases_regression, [2**FLAGS.regression, FLAGS.upscale**2])  # 2^R x s^2
    weights_gathered = tf.gather_nd(weights_regression, indices)                 # B x H x W x k x k x Cin x s^2
    # weights_gathered = tf.squeeze(weights_gathered, [3, 4])                      # B x H x W x Cin x s^2
    biases_gathered = tf.gather_nd(biases_regression, indices)                   # B x H x W x s^2
    #feature
    feature = tf.expand_dims(feature, axis=4)                                    # B x H x W x Cin x 1
    sr_depth = tf.reduce_sum(feature*weights_gathered, axis=3) + biases_gathered
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale)
    sr = sr_space + data_mid
    return sr, global_step
    #, time_part#, weights_gathered, biases_gathered, data_mid, indices








########################################################################################################################
# The followings are the codes that shows the development curve of the project. The project developed from simple to
# hard. First, a simple architecture is implemented to explore the basic techniques in learning algorithms. Second, ARN
# is implemented using convolutional layers with A+ anchors, learned anchors, and {-1,1} anchors. The problem in this
# stage was that the PSNR of the SR image is in a low level. Since the main aim is to develop faster algorithms than
# RAISR, the idea is first to implement ARN with conv layers and achieve high PSNR scores. Then replace the learned
# anchors with {-1,1} anchors and use hard assignment from one feature to a single anchor to make the algorithm even
# faster than RAISR. Third, fast implementations were developed according the above idea.

def simple_arch(data, data_mid, FLAGS):
    #data_shape = tf.shape(data)
    #bilinear = tf.image.resize_bilinear(data,[data_shape[1]*FLAGS.upscale,data_shape[2]*FLAGS.upscale])
    net = slim.conv2d(data, FLAGS.hidden1, [3, 3], stride=1, scope='conv1')
    net = slim.batch_norm(net)
    net = slim.conv2d(net, FLAGS.hidden2, [3, 3], stride=1, scope='conv2')
    net = slim.batch_norm(net)
    net = slim.conv2d_transpose(net, 3, [5, 5], stride=2, scope='deconv1', activation_fn=None)
    net = net + data_mid
    return net

def global_regression(data_mid):
    #Only use one anchor, the anchor is randomly initialized
    feature = slim.conv2d(data_mid, 30, [5, 5], stride=1, scope='feature1', activation_fn=None)  # B x H x W x 64
    gr = slim.conv2d(feature, 1, [1, 1], stride=1, scope='gr', activation_fn=None)  # B x H x W x 64
    sr = gr + data_mid
    return sr

def global_regression_anchor(data_mid):
    #Try to use more randomly initialized anchors
    feature = slim.conv2d(data_mid, 30, [5, 5], stride=1, scope='feature1', activation_fn=None)  # B x H x W x 64
    gr = slim.conv2d(feature, 16, [1, 1], stride=1, scope='gr', activation_fn=None)  # B x H x W x 2
    alpha = slim.conv2d(feature, 16, [1, 1], stride=1, scope='alpha', activation_fn=tf.nn.softmax) # B x H x W x 2
    gr = tf.reduce_sum(gr * alpha, 3)
    gr = tf.expand_dims(gr, 3)
    sr = gr + data_mid
    return sr

def srcnn_res(data_mid):
    net = slim.conv2d(data_mid, 64, [9, 9], stride=1, scope='extraction')
    net = slim.conv2d(net, 32, [3, 3], stride=1, scope='mapping')
    net = slim.conv2d(net, 1, [5, 5], stride=1, scope='reconstruction', activation_fn=None)
    net = net + data_mid
    return net

# Use A+ anchors
def arn_aplus_anchors(data, data_mid, FLAGS):
    """
    use fixed A+ anchors generated by radu
    :param data:
    :param data_mid:
    :param FLAGS:
    :return:
    """
    #load anchors from Matlat mat file generated by Aplus
    with tf.name_scope("anchor_load"):
        mat = sio.loadmat('/home/yawli/Downloads/AplusCodes_SR/conf_Zeyde_1024_finalx2.mat', struct_as_record=False)
        conf = mat['conf']
        dict_low = conf[0, 0].dict_lores
        dict_size = dict_low.shape
        anchors = tf.constant(dict_low, tf.float32)
        anchors = tf.Variable(anchors, name='anchors', trainable=False) # 28 x 16
        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0) # 1 x 1 x 28 x 16
    #feature = slim.conv2d(data, 64)
    #regress
    feature = slim.conv2d(data, dict_size[0], [5, 5], stride=1, scope='feature_layer')  # B x H x W x 28  filter 5 x 5 x 1 x 28
    regression = slim.conv2d(feature, dict_size[1]*FLAGS.upscale**2, [1, 1], stride=1, scope='regression_layer', activation_fn=None) # B x H x W x 16x4  filter 1 x 1 x 28 x 16x4
    r_size = tf.shape(regression)
    regression = tf.reshape(regression, [r_size[0], r_size[1], r_size[2], dict_size[1], FLAGS.upscale**2]) # B x H x W x 16 x 4
    #coefficients for different anchors
    alpha = tf.nn.softmax(tf.nn.conv2d(feature, anchors, strides=[1, 1, 1, 1], padding='SAME', name='similarity_layer')) # B x H x W x 16  anchor as the filter
    alpha = tf.expand_dims(alpha, 4) # B x H x W x 16 x 1
    sr_depth = tf.reduce_sum(regression * alpha, 3, name='sr_depth')
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale, name='sr_space')
    with tf.name_scope("sr_image"):
        sr = sr_space + data_mid
    return sr

def arn_aplus_anchors_more_layers(data, data_mid, FLAGS):
    #use fixed anchors generated by radu
    #load anchors from Matlat mat file generated by Aplus
    with tf.name_scope("anchor_load"):
        mat = sio.loadmat('/home/yawli/Downloads/AplusCodes_SR/conf_Zeyde_16_finalx2.mat', struct_as_record=False)
        conf = mat['conf']
        dict_low = conf[0, 0].dict_lores
        dict_size = dict_low.shape
        anchors = tf.constant(dict_low, tf.float32)
        anchors = tf.Variable(anchors, name='anchors') # 28 x 16
        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0) # 1 x 1 x 28 x 16
    #feature = slim.conv2d(data, 64)
    #regress
    # feature = slim.conv2d(data, 64, [5, 5], stride=1, scope='feature_layer1')  # B X H X W x 64   filter 5 x 5 x 1 x 64
    # feature = slim.conv2d(feature, dict_size[0], [3, 3], stride=1, scope='feature_layer2')  # B x H x W x 28  filter 5 x 5 x 64 x 28
    feature = slim.conv2d(data, 64, [5, 5], stride=1, scope='feature_layer1')  # B X H X W x 64   filter 5 x 5 x 1 x 64
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer2')  # B X H X W x 64   filter 5 x 5 x 64 x 64
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer3')  # B X H X W x 64   filter 5 x 5 x 64 x 64
    feature = slim.conv2d(feature, 32, [3, 3], stride=1, scope='feature_layer4')  # B X H X W x 64   filter 5 x 5 x 64 x 32
    #feature = slim.conv2d(feature, 32, [3, 3], stride=1, scope='feature_layer5')  # B X H X W x 64   filter 5 x 5 x 32 x 32
    # feature = slim.conv2d(feature, 32, [3, 3], stride=1, scope='feature_layer6')  # B X H X W x 64   filter 5 x 5 x 32 x 32
    feature = slim.conv2d(feature, dict_size[0], [3, 3], stride=1, scope='feature_layer5')  # B x H x W x 28  filter 5 x 5 x 64 x 28
    regression = slim.conv2d(feature, dict_size[1]*FLAGS.upscale**2, [1, 1], stride=1, scope='regression_layer', activation_fn=None) # B x H x W x 16x4  filter 1 x 1 x 28 x 16x4
    r_size = tf.shape(regression)
    regression = tf.reshape(regression, [r_size[0], r_size[1], r_size[2], dict_size[1], FLAGS.upscale**2]) # B x H x W x 16 x 4
    #coefficients for different anchors
    alpha = tf.nn.softmax(tf.nn.conv2d(feature, anchors, strides=[1, 1, 1, 1], padding='SAME', name='similarity_layer')) # B x H x W x 16  anchor as the filter
    alpha = tf.expand_dims(alpha, 4) # B x H x W x 16 x 1
    sr_depth = tf.reduce_sum(regression * alpha, 3, name='sr_depth')
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale, name='sr_space')
    with tf.name_scope("sr_image"):
        sr = sr_space + data_mid
    return sr

# Use learned anchors
def arn_conv(data, data_mid, FLAGS):
    """
    use random anchors
    :param data:
    :param data_mid:
    :param FLAGS:
    :return:
    """
    data_shape = tf.shape(data)
    #feature extraction
    feature = slim.conv2d(data, 8, [3, 3], stride=1, scope='feature_layer1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x
    feature = slim.conv2d(feature, 8, [3, 3], stride=1, scope='pca', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H
    feature = slim.conv2d(feature, 8, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x
    # feature = slim.conv2d(feature, 4, [3, 3], stride=1, scope='feature_layer3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x
    # feature = slim.conv2d(feature, 4, [3, 3], stride=1, scope='feature_layer4', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x

    #regression layer
    regression = slim.conv2d(feature, FLAGS.regression*FLAGS.upscale**2, [3, 3], stride=1, scope='regression_layer', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 1024x4
    regression = tf.reshape(regression, [data_shape[0], data_shape[1], data_shape[2], FLAGS.regression, FLAGS.upscale**2])  # B x W x H x 1024 x 4
    #compute the coefficient alpha
    #dimension of alpha_vector [batch_size, H, W, FLAGS.regression]
    alpha_vector = slim.conv2d(feature, FLAGS.regression, [3, 3], stride=1, scope='similarity_layer', activation_fn=tf.nn.softmax)   #B x W x H x 1024
    alpha_vector = tf.reshape(alpha_vector, [data_shape[0], data_shape[1], data_shape[2], FLAGS.regression, 1])  #B x W x H x 1024 x 1
    #dimension of alpha_vector [batch_size, H, W, FLAGS.regression, FLAGS.upscale**2]

    #Hadamard product and reduce sum
    sr_depth = tf.reduce_sum(regression*alpha_vector, 3)
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale) #B x 2H x 2W x 1
    sr = 1*sr_space + data_mid
    return sr

def arn_compare(data, data_mid, FLAGS):
    """
    compare with arn_conv to test whether the regression layer can improve the performance.
    :param data:
    :param data_mid:
    :param FLAGS:
    :return:
    """
    #feature
    feature = slim.conv2d(data, 16, [5, 5], stride=1, scope='feature_layer1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))
    feature = slim.conv2d(feature, 8, [3, 3], stride=1, scope='pca', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))
    sr_depth = slim.conv2d(feature, FLAGS.upscale**2, [3, 3], stride=1, scope='regression_layer', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))

    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale, name='sr_space')
    with tf.name_scope("sr_image"):
        sr = sr_space + data_mid
    return sr

# Try to implement the census filter used by RAISR
def census_filter(size):
    c_filter = np.zeros([size, size, 1, size**2-1])
    for i in range(size**2-1):
        if i > (size**2 - 1)/2:
            pixel_index = i + 1
        else:
            pixel_index = i
        y = int(np.ceil(pixel_index / float(size)))
        x = pixel_index - size * y
        c_filter[y-1, x-1, 0, i-1] = 1.0
        c_filter[1, 1, 0, i -1] = -1.0
    return c_filter

# A hash layer is used to learn the mapping from features to hashcodes
def hashnet_old(data, data_mid, FLAGS):
    """
    Use fixed anchors generated by gen_anchors.
    A hash layer is used to learn the mapping from features to hashcodes

    :param data:
    :param data_mid:
    :param FLAGS:
    :return:
    """
    with tf.name_scope("generate_anchors"):
        anchors = np.array(gen_anchors(4)) #anchor_num x anchor_dim    16 x 4
        anchor_size = anchors.shape
        anchor_size = [anchor_size[1], anchor_size[0]]  #anchor_dim x anchor_num  4 x 16
        anchors = tf.transpose(tf.constant(anchors, tf.float32), [1, 0]) # 4 x 16
        anchors = tf.Variable(anchors, name='anchors', trainable=False) # 4 x 16
        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0) # 1 x 1 x 4 x 16
    #feature
    feature = slim.conv2d(data, 64, [5, 5], stride=1, scope='feature_layer1')  # B x H x W x 64  filter 5 x 5 x 1 x 64
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer2')  # B x H x W x 32  filter 5 x 5 x 64 x 64
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer3')  # B x H x W x 32  filter 5 x 5 x 64 x 64
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer4')  # B x H x W x 32  filter 5 x 5 x 64 x 64
    feature = slim.conv2d(feature, 32, [3, 3], stride=1, scope='feature_layer5')  # B x H x W x 32  filter 5 x 5 x 64 x 32
    regression = slim.conv2d(feature, anchor_size[1]*FLAGS.upscale**2, [1, 1], stride=1, scope='regression_layer', activation_fn=None) # B x H x W x 16x4  filter 1 x 1 x 32 x 16x4
    r_size = tf.shape(regression)
    regression = tf.reshape(regression, [r_size[0], r_size[1], r_size[2], anchor_size[1], FLAGS.upscale**2]) # B x H x W x 16 x 4
    #hashcode
    hashcode = slim.conv2d(feature, 20, [1, 1], stride=1, scope='hash_layer1') # B x H x W x 20  filter 1 x 1 x 32 x 20
    hashcode = slim.conv2d(hashcode, 10, [1, 1], stride=1, scope='hash_layer2') # B x H x W x 10  filter 1 x 1 x 20 x 10
    hashcode = slim.conv2d(hashcode, anchor_size[0], [1, 1], stride=1, scope='hash_layer3', activation_fn=tf.nn.tanh) # B x H x W x 4  filter 1 x 1 x 10 x 4
    #coefficients for different anchors, the coefficients represent similarity
    #sigma = tf.Variable(1.0)
    alpha = tf.nn.softmax(tf.nn.conv2d(hashcode, anchors, strides=[1, 1, 1, 1], padding='SAME', name='similarity_layer')) # B x H x W x 16  anchor_size 1 x 1 x 4 x 16, anchor as the filter
    #alpha = tf.stop_gradient(hardmax(hashcode) - tf.nn.softmax(hashcode)) + tf.nn.softmax(hashcode)
    alpha = tf.expand_dims(alpha, 4) # B x H x W x 16 x 1
    sr_depth = tf.reduce_sum(regression * alpha, 3, name='sr_depth')
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale, name='sr_space')
    with tf.name_scope("sr_image"):
        sr = sr_space + data_mid
    return sr

# Try to separate conv layers in order to use multicores of cpu.
def feature_layer(data, channel, scope):
    feature1 = slim.conv2d(data, channel, [5, 1], stride=1, scope=scope+'1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature2 = slim.conv2d(data, channel, [1, 5], stride=1, scope=scope+'2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature3 = slim.conv2d(data, channel, [3, 3], stride=1, scope=scope+'3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature4 = slim.conv2d(data, channel, [3, 3], stride=1, scope=scope+'4', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = tf.concat([feature1, feature2, feature3, feature4], axis=3)
    return feature

# Develop an efficient way of indexing the anchors with the maximum similarity to the features.

# Operate on low or high resolution grid.
def hashnet1(data_mid, FLAGS):
    with tf.name_scope("generate_anchors"):
        anchors = np.array(gen_anchors(FLAGS.regression)) #anchor_num x anchor_dim    16 x 4
        anchor_size = anchors.shape
        anchor_size = [anchor_size[1], anchor_size[0]]  #anchor_dim x anchor_num  4 x 16
        anchors = tf.transpose(tf.constant(anchors, tf.float32), [1, 0]) # 4 x 16
        anchors = tf.Variable(anchors, name='anchors', trainable=False) # 4 x 16
        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0) # 1 x 1 x 4 x 16
    #feature
    Cin = 64
    feature = slim.conv2d(data_mid, 64, [5, 5], stride=1, scope='feature_layer1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 64  filter 5 x 5 x 1 x 64
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 32  filter 5 x 5 x 64 x 64
    feature = slim.conv2d(feature, Cin, [3, 3], stride=1, scope='feature_layer3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 32  filter 5 x 5 x 64 x 64
    k_size = 1
    regression = slim.conv2d(feature, anchor_size[1], [k_size, k_size], stride=1, scope='regression_layer', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001)) # B x H x W x 16  filter 1 x 1 x 32 x 16

    #hashcode
    hashcode = slim.conv2d(feature, anchor_size[0], [k_size, k_size], stride=1, scope='hash_layer', activation_fn=tf.nn.tanh, weights_regularizer=slim.l2_regularizer(0.0001)) # B x H x W x 4  filter 1 x 1 x 10 x 4
    #coefficients for different anchors, the coefficients represent similarity
    #sigma = tf.Variable(1.0)
    similarity = tf.nn.conv2d(hashcode, anchors, strides=[1, 1, 1, 1], padding='SAME', name='similarity_layer') # B x H x W x 16  anchor_size 1 x 1 x 4 x 16, anchor as the filter
    # alpha = tf.nn.softmax(similarity)
    alpha = tf.stop_gradient(hardmax(similarity) - tf.nn.softmax(similarity)) + tf.nn.softmax(similarity) # B x H x W x 16
    sr_res = tf.reduce_sum(regression * alpha, axis=3, keep_dims=True, name='sr_res')
    sr = sr_res + data_mid
    return sr

# Try to allocate computations on multiple GPUs when a single GPU runs out of memory.
# Move all the reshape and transpose ops to the training end, thus, saving some testing time.
def hashnet2(data, data_mid, FLAGS):
    # with tf.device('/device:GPU:'+FLAGS.sge_gpu_all[2]):
    # tf.add_to_collection('data', data)
    # tf.add_to_collection('data_mid', data_mid)
    with tf.name_scope("generate_anchors"):
        anchors = np.array(gen_anchors(FLAGS.regression))                                                           #anchor_num x anchor_dim    2^R x R
        anchors = tf.transpose(tf.constant(anchors, tf.float32), [1, 0])                                            # R x 2^R
        anchors = tf.Variable(anchors, name='anchors', trainable=False)                                             # R x 2^R
        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0)                                           # 1 x 1 x R x 2^R
# with tf.device('/device:GPU:' + FLAGS.sge_gpu_all[0]):
    #feature
    Cin = 10
    feature = slim.conv2d(data, 10, [3, 3], stride=1, scope='feature_layer1', weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = slim.conv2d(feature, Cin, [3, 3], stride=1, scope='feature_layer2', weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x Cin

    #regression
    k_size = 1
    regression = slim.conv2d(feature, 2**FLAGS.regression*FLAGS.upscale**2, [k_size, k_size], stride=1, scope='regression_layer', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
    r_size = tf.shape(regression)
    regression = tf.reshape(regression, [r_size[0], r_size[1], r_size[2], 2**FLAGS.regression, FLAGS.upscale**2])   # B x H x W 2^R x s^2

    #
    weights_regression = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression_layer/weights') # k x k x Cin x 2^R x s^2
    biases_regression = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression_layer/biases')
    weights_regression = tf.squeeze(weights_regression, axis=[0, 1], name='weights_squeeze')                   # Cin x 2^R x s^2
    weights_regression = tf.reshape(weights_regression, [Cin, 2**FLAGS.regression, FLAGS.upscale**2], name='weights_reshape')  # k x k x Cin x 2^R x s^2
    weights_regression = tf.Variable(tf.transpose(weights_regression, [1, 0, 2]), trainable=False, name='weights_transpose')       # 2^R x k x k x Cin x s^2
    biases_regression = tf.Variable(tf.reshape(biases_regression, [2**FLAGS.regression, FLAGS.upscale**2]), trainable=False, name='biases_reshape')  # 2^R x s^2

    # tf.add_to_collection(name='weights_transpose', value=weights_regression)
    # tf.add_to_collection(name='biases_reshape', value=biases_regression)

    #hashcode
    hashcode = slim.conv2d(feature, FLAGS.regression, [k_size, k_size], stride=1, scope='hash_layer', activation_fn=tf.nn.tanh, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x R  filter k_size x k_size x Cin x R
    #  with tf.device('/device:GPU:'+FLAGS.sge_gpu_all[2])

    #similar coefficient
    sigma = tf.Variable(1.0)
    hashcode = tf.nn.conv2d(hashcode, anchors, strides=[1, 1, 1, 1], padding='SAME', name='similarity_layer')       # B x H x W x 2^R  anchor_size 1 x 1 x R x 2^R, anchor as the filter

    alpha = tf.stop_gradient(hardmax(hashcode) - tf.nn.softmax(hashcode*sigma)) + tf.nn.softmax(hashcode*sigma)     # B x H x W x 2^R
    alpha = tf.expand_dims(alpha, 4)
    sr_depth = tf.reduce_sum(regression * alpha, axis=3, name='sr_depth')
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale)
    sr = sr_space + data_mid
    return sr

# Try to use the filters used by A+ to extract features.
# Try to learn the filters with the same dimension with those used by A+
def hashnet3(data, data_mid, FLAGS):
    # with tf.device('/device:GPU:'+FLAGS.sge_gpu_all[2]):
    # tf.add_to_collection('data', data)
    # tf.add_to_collection('data_mid', data_mid)
    with tf.name_scope("generate_anchors"):
        anchors = np.array(gen_anchors(FLAGS.regression))                                                           #anchor_num x anchor_dim    2^R x R
        anchors = tf.transpose(tf.constant(anchors, tf.float32), [1, 0])                                            # R x 2^R
        anchors = tf.Variable(anchors, name='anchors', trainable=False)                                             # R x 2^R
        anchors = tf.expand_dims(tf.expand_dims(anchors, axis=0), axis=0)                                           # 1 x 1 x R x 2^R
    #feature
    # filter1 = tf.constant([[[[1]]], [[[0]]], [[[-1]]]], tf.float32)
    # filter2 = tf.transpose(filter1, [1, 0, 2, 3])
    # filter3 = tf.constant([[[[1]]], [[[0]]], [[[-1]]], [[[0]]], [[[1]]]], tf.float32)/2
    # filter4 = tf.transpose(filter3, [1, 0, 2, 3])
    # data1 = tf.nn.conv2d(data, filter1, strides=[1, 1, 1, 1], padding='SAME', name='filtering1')
    # data2 = tf.nn.conv2d(data, filter2, strides=[1, 1, 1, 1], padding='SAME', name='filtering2')
    # data3 = tf.nn.conv2d(data, filter3, strides=[1, 1, 1, 1], padding='SAME', name='filtering3')
    # data4 = tf.nn.conv2d(data, filter4, strides=[1, 1, 1, 1], padding='SAME', name='filtering4')
    # data = tf.concat([data, data1, data2, data3, data4], axis=3)

    #use learned filters instead of fixed filters in the first layer
    # data1 = slim.conv2d(data, 1, [3, 1], stride=1, scope='data1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))
    # data2 = slim.conv2d(data, 1, [1, 3], stride=1, scope='data2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))
    # data3 = slim.conv2d(data, 1, [5, 1], stride=1, scope='data3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))
    # data4 = slim.conv2d(data, 1, [1, 5], stride=1, scope='data4', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))
    # data = tf.concat([data, data1, data2, data3, data4], axis=3)

    # #use learned two-dimensional filters 3X3
    # Cin = 10
    # feature = slim.conv2d(data, 4, [3, 3], stride=1, scope='feature_layer1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))
    # feature = slim.conv2d(feature, 10, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    # feature = slim.conv2d(feature, Cin, [3, 3], stride=1, scope='feature_layer3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x Cin

    Cin = 10
    feature = slim.conv2d(data, 4, [3, 3], stride=1, scope='feature_layer1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = slim.conv2d(feature, 4, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = slim.conv2d(feature, 4, [3, 3], stride=1, scope='feature_layer3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = slim.conv2d(feature, Cin, [3, 3], stride=1, scope='feature_layer4', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x Cin

    #regression
    k_size = 1
    regression = slim.conv2d(feature, 2**FLAGS.regression*FLAGS.upscale**2, [k_size, k_size], stride=1, scope='regression_layer', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
    r_size = tf.shape(regression)
    regression = tf.reshape(regression, [r_size[0], r_size[1], r_size[2], 2**FLAGS.regression, FLAGS.upscale**2])   # B x H x W 2^R x s^2

    # #
    # weights_regression = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression_layer/weights') # k x k x Cin x 2^R x s^2
    # biases_regression = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression_layer/biases')
    # weights_regression = tf.squeeze(weights_regression, axis=[0, 1], name='weights_squeeze')                   # Cin x 2^R x s^2
    # weights_regression = tf.reshape(weights_regression, [Cin, 2**FLAGS.regression, FLAGS.upscale**2], name='weights_reshape')  # k x k x Cin x 2^R x s^2
    # weights_regression = tf.Variable(tf.transpose(weights_regression, [1, 0, 2]), trainable=False, name='weights_transpose')       # 2^R x k x k x Cin x s^2
    # biases_regression = tf.Variable(tf.reshape(biases_regression, [2**FLAGS.regression, FLAGS.upscale**2]), trainable=False, name='biases_reshape')  # 2^R x s^2

    #hashcode
    hashcode = slim.conv2d(feature, FLAGS.regression, [k_size, k_size], stride=1, scope='hash_layer', activation_fn=tf.nn.tanh, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x R  filter k_size x k_size x Cin x R
    #  with tf.device('/device:GPU:'+FLAGS.sge_gpu_all[2])

    #similar coefficient
    sigma = tf.Variable(1.0)
    hashcode = tf.nn.conv2d(hashcode, anchors, strides=[1, 1, 1, 1], padding='SAME', name='similarity_layer')       # B x H x W x 2^R  anchor_size 1 x 1 x R x 2^R, anchor as the filter

    # alpha = tf.nn.softmax(hashcode)
    alpha = tf.stop_gradient(hardmax(hashcode) - tf.nn.softmax(hashcode*sigma)) + tf.nn.softmax(hashcode*sigma)     # B x H x W x 2^R
    alpha = tf.expand_dims(alpha, 4)                                                                                # B x H x W x 2^R x 1

    #final results
    sr_depth = tf.reduce_sum(regression * alpha, axis=3, name='sr_depth')
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale,  name='sr_space')
    sr = sr_space + data_mid
    # from IPython import embed; embed(); exit()
    return sr

def hashnet_restore(data_mid, FLAGS):

    anchor_size = [FLAGS.regression, 2**FLAGS.regression] # 4 x 16
    #feature
    Cin = 64
    feature = slim.conv2d(data_mid, 64, [5, 5], stride=1, scope='feature_layer1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 64  filter 5 x 5 x 1 x 64
    feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 32  filter 5 x 5 x 64 x 64
    feature = slim.conv2d(feature, Cin, [3, 3], stride=1, scope='feature_layer3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x 32  filter 5 x 5 x 64 x 64
    #hash code
    k_size = 1
    hashcode = slim.conv2d(feature, anchor_size[0], [k_size, k_size], stride=1, scope='hash_layer', activation_fn=tf.nn.tanh, weights_regularizer=slim.l2_regularizer(0.0001)) # B x H x W x 4  filter 1 x 1 x 10 x 4
    hashcode_hard = tf.cast((tf.sign(hashcode) + 1)/2, dtype=tf.int32)
    #indices
    basis = [2**i for i in range(FLAGS.regression-1, -1, -1)]
    base = tf.reshape(tf.constant(basis, dtype=tf.int32), [1, 1, 1, FLAGS.regression])
    index = tf.reduce_sum(hashcode_hard*base, axis=3) # B x H x W
    indices = tf.expand_dims(index, axis=3)            # B x H x W x 1

    # weights and biases
    weights_regression = tf.Variable(tf.random_normal([k_size, k_size, Cin, anchor_size[1]], stddev=1e-3), name="regression_layer/weights") #k_size x k_size x Cin x Cout
    biases_regression = tf.Variable(tf.zeros([anchor_size[1]]), name="regression_layer/biases") #Cout
    #gather filter weights and biases
    weights_regression = tf.transpose(weights_regression, [3, 0, 1, 2]) #Cout x k_size x k_size x Cin
    # from IPython import embed; embed(); exit()
    weights_gathered = tf.gather_nd(weights_regression, indices)        #B x H x W x k_size x k_size x Cin
    weights_gathered = tf.squeeze(weights_gathered, [3, 4])             #B x H x W x Cin
    biases_gathered = tf.gather_nd(biases_regression, indices)          #B x H x W
    #regression
    regression = tf.expand_dims(tf.reduce_sum(feature*weights_gathered, axis=3) + biases_gathered, 3)

    sr = regression + data_mid
    return sr, regression, weights_gathered, biases_gathered, data_mid, index

def hashnet_restore_low(data, data_mid, FLAGS):

    Cin = 10
    feature = slim.conv2d(data, 4, [3, 3], stride=1, scope='feature_layer1', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = slim.conv2d(feature, 4, [3, 3], stride=1, scope='feature_layer2', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = slim.conv2d(feature, 4, [3, 3], stride=1, scope='feature_layer3', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x f1
    feature = slim.conv2d(feature, Cin, [3, 3], stride=1, scope='feature_layer4', activation_fn=None, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x Cin

    # channel = 5
    # feature = feature_layer(data, channel, 'feature_layer1')
    # feature = feature_layer(feature, channel, 'feature_layer2')
    # Cin = channel * 4

    #hash code
    k_size = 1
    hashcode = slim.conv2d(feature, FLAGS.regression, [k_size, k_size], stride=1, scope='hash_layer', activation_fn=tf.nn.tanh, weights_regularizer=slim.l2_regularizer(0.0001))  # B x H x W x R
    hashcode_hard = tf.cast((tf.sign(hashcode) + 1)/2, dtype=tf.int32)                                                                                                            # B x H x W x R
    #indices
    basis = [2**i for i in range(FLAGS.regression-1, -1, -1)]
    basis = tf.reshape(tf.constant(basis, dtype=tf.int32), [1, 1, 1, FLAGS.regression])  # 1 x 1 x 1 x R
    indices = tf.reduce_sum(hashcode_hard*basis, axis=3, keep_dims=True) # B x H x W x 1
    # weights and biases
    weights_regression = tf.Variable(tf.random_normal([k_size, k_size, Cin, 2**FLAGS.regression*FLAGS.upscale**2], stddev=1e-3), name="regression_layer/weights") #k_size x k_size x Cin x 2^Rxs^2
    biases_regression = tf.Variable(tf.zeros([2**FLAGS.regression* FLAGS.upscale**2]), name="regression_layer/biases") #2^Rxs^2
    w_shape = tf.shape(weights_regression)
    weights_regression = tf.reshape(weights_regression, [w_shape[0], w_shape[1], w_shape[2], 2**FLAGS.regression, FLAGS.upscale**2])  # k x k x Cin x 2^R x s^2
    weights_regression = tf.transpose(weights_regression, [3, 0, 1, 2, 4])       # 2^R x k x k x Cin x s^2
    biases_regression = tf.reshape(biases_regression, [2**FLAGS.regression, FLAGS.upscale**2])  # 2^R x s^2
    weights_gathered = tf.gather_nd(weights_regression, indices)                 # B x H x W x k x k x Cin x s^2
    weights_gathered = tf.squeeze(weights_gathered, [3, 4])                      # B x H x W x Cin x s^2
    biases_gathered = tf.gather_nd(biases_regression, indices)                   # B x H x W x s^2
    #feature
    feature = tf.expand_dims(feature, axis=4)                                    # B x H x W x Cin x 1
    sr_depth = tf.reduce_sum(feature*weights_gathered, axis=3) + biases_gathered
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale)
    sr = sr_space + data_mid
    return sr#, time_part#, weights_gathered, biases_gathered, data_mid, indices

def hashnet_restore_meta_low(data_mid, FLAGS):
    graph = tf.get_default_graph()
    hashcode = graph.get_tensor_by_name('hash_layer/BiasAdd:0')   # B x H x W x R
    hashcode_hard = tf.cast((tf.sign(hashcode) + 1)/2, dtype=tf.int32)  # B x H x W x R
    basis = [2**i for i in range(FLAGS.regression-1, -1, -1)]
    basis = tf.reshape(tf.constant(basis, dtype=tf.int32), [1, 1, 1, FLAGS.regression])  # 1 x 1 x 1 x R
    indices = tf.reduce_sum(hashcode_hard*basis, axis=3, keep_dims=True)  # B x H x W x1

    #gather filter weights and biases
    weights_regression = graph.get_tensor_by_name('regression_layer/weights:0')  # k x k x Cin x 2^Rxs^2
    biases_regression = graph.get_tensor_by_name('regression_layer/biases:0')    # 2^Rxs^2
    w_shape = tf.shape(weights_regression)
    weights_regression = tf.reshape(weights_regression, [w_shape[0], w_shape[1], w_shape[2], 2**FLAGS.regression, FLAGS.upscale**2])  # k x k x Cin x 2^R x s^2
    weights_regression = tf.transpose(weights_regression, [3, 0, 1, 2, 4])       # 2^R x k x k x Cin x s^2
    # from IPython import embed; embed(); exit()
    weights_gathered = tf.gather_nd(weights_regression, indices)                 # B x H x W x k x k x Cin x s^2
    weights_gathered = tf.squeeze(weights_gathered, [3, 4])                      # B x H x W x Cin x s^2
    biases_regression = tf.reshape(biases_regression, [2**FLAGS.regression, FLAGS.upscale**2])  # 2^R x s^2
    biases_gathered = tf.gather_nd(biases_regression, indices)                   # B x H x W x s^2

    #feature
    feature = tf.expand_dims(graph.get_tensor_by_name('feature_layer2/BiasAdd:0'), axis=4) # B x H x W x Cin x 1
    sr_depth = tf.reduce_sum(feature*weights_gathered, axis=3) + biases_gathered
    sr_space = tf.depth_to_space(sr_depth, FLAGS.upscale)
    sr = sr_space + data_mid
    return sr, sr_space, weights_gathered, biases_gathered, data_mid, indices
