import tensorflow as tf


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv_net(x, weights, biases):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 56, 56, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.get_shape())
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    print(fc1.get_shape())
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print(fc1.get_shape())
    fc1 = tf.nn.relu(fc1)
    print(fc1.get_shape())

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
