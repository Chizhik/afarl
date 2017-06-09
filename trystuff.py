from network.layer import *


def build_conv(weights_initializer=initializers.xavier_initializer(),
               biases_initializer=tf.constant_initializer(0.1),
               hidden_activation_fn=tf.nn.relu):
    # Dimensions of input: [N, 28, 28, 2] or [N, 56, 56, 2] or ...
    conv_inputs = tf.placeholder('float32', [None, 56, 56, 2], name='conv_inputs')
    print('Shape of conv_inputs: ', conv_inputs.get_shape())
    var = {}
    with tf.variable_scope('test'):
        conv_l1, var['conv_l1_w'], var['conv_l1_b'] = conv2d(conv_inputs,  # input of cnn
                                                                            32,  # number of out channels
                                                                            [8, 8],  # size of kernel (filter)
                                                                            [4, 4],  # strides
                                                                            weights_initializer,
                                                                            biases_initializer,
                                                                            hidden_activation_fn,
                                                                            'NHWC', name='conv_l1')
        print('Shape of conv_l1: ', conv_l1.get_shape())
        # it should be [None, 13, 13, 32]
        conv_l2, var['conv_l2_w'], var['conv_l2_b'] = conv2d(conv_l1,  # input of cnn
                                                                            64,  # number of out channels
                                                                            [4, 4],  # size of kernel (filter)
                                                                            [2, 2],  # strides
                                                                            weights_initializer,
                                                                            biases_initializer,
                                                                            hidden_activation_fn,
                                                                            'NHWC', name='conv_l2')
        print('Shape of conv_l2: ', conv_l2.get_shape())
        # it should be [None, 5, 5, 64]
        conv_l3, var['conv_l3_w'], var['conv_l3_b'] = conv2d(conv_l2,  # input of cnn
                                                                            64,  # number of out channels
                                                                            [3, 3],  # size of kernel (filter)
                                                                            [1, 1],  # strides
                                                                            weights_initializer,
                                                                            biases_initializer,
                                                                            hidden_activation_fn,
                                                                            'NHWC', name='conv_l3')
        print('Shape of conv_l3: ', conv_l3.get_shape())
        # it should be [None, 3, 3, 64]
    conv_out = tf.contrib.layers.flatten(conv_l3)
    conv_out_dim = conv_out.get_shape()[1]
    print('Out dimension of conv: ', conv_out_dim)

with tf.Session() as sess:
    build_conv()

