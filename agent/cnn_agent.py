from network.layer import *
from helper import mnist_mask_batch
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('.'))))

# Some thoughts: Since both agent and classifier will train the cnn layers, we need to pay attention to their
# learning rates

class CNNAgent(object):
    def __init__(self,
                 sess,
                 conf,
                 name='Agent'):

        self.sess = sess
        self.var = {}
        # problem
        self.input_dim = conf.input_dim  # for example, [28, 28, 1]
        self.expand_size = conf.expand_size
        # TODO: change conf.n_features to conf.feature_dim
        self.feature_dim = conf.n_features
        self.n_classes = conf.n_classes
        self.train_size = conf.train_size
        self.test_size = conf.test_size
        self.data_type = conf.data_type
        # train Q
        self.update_freq = conf.target_update_freq
        self.discount = conf.discount
        self.policy = conf.policy
        self.max_lr = conf.max_lr # <----------------------------- to be changed
        self.min_lr = conf.min_lr
        self.clf_max_lr = conf.clf_max_lr
        self.clf_min_lr = conf.clf_min_lr
        self.batch_size = conf.batch_size
        self.n_epoch = conf.n_epoch
        self.pre_train_steps = conf.pre_train_steps
        self.startE = conf.eps_start
        self.endE = conf.eps_end
        self.anneling_steps = conf.anneling_steps
        self.eps_decay = (self.startE - self.endE) / self.anneling_steps
        self.eps = self.startE
        # reward
        self.r_cost = conf.r_cost
        self.r_wrong = conf.r_wrong
        self.r_repeat = conf.r_repeat
        self.r_correct = conf.r_correct
        # classfier
        self.clf_hidden_sizes = conf.clf_hidden_sizes
        # for model saving and restoring
        self.name = name
        self.save_dir = os.path.relpath(os.path.dirname(__file__))
        self.save_dir = os.path.join(self.save_dir, conf.save_dir)
        self.save_path = os.path.join(self.save_dir, name + '.ckpt')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def build_conv(self,
                   weights_initializer=initializers.xavier_initializer(),
                   biases_initializer=tf.constant_initializer(0.1),
                   hidden_activation_fn=tf.nn.relu):
        # Dimensions of input: [N, 28, 28, 2] or [N, 56, 56, 2] or ...
        self.conv_inputs = tf.placeholder('float32', [None,
                                                     self.input_dim[0],
                                                     self.input_dim[1],
                                                     2],
                                         name='conv_inputs')
        print('Shape of conv_inputs: ', self.conv_inputs.get_shape())
        with tf.variable_scope(self.name):
            self.conv_l1, self.var['conv_l1_w'], self.var['conv_l1_b'] = conv2d(self.conv_inputs,  # input of cnn
                                                                                32,  # number of out channels
                                                                                [8, 8],  # size of kernel (filter)
                                                                                [4, 4],  # strides
                                                                                weights_initializer,
                                                                                biases_initializer,
                                                                                hidden_activation_fn,
                                                                                'NHWC', name='conv_l1')
            print('Shape of conv_l1: ', self.conv_l1.get_shape())
            # it should be [None, 13, 13, 32]
            self.conv_l2, self.var['conv_l2_w'], self.var['conv_l2_b'] = conv2d(self.conv_l1,  # input of cnn
                                                                                64,  # number of out channels
                                                                                [4, 4],  # size of kernel (filter)
                                                                                [2, 2],  # strides
                                                                                weights_initializer,
                                                                                biases_initializer,
                                                                                hidden_activation_fn,
                                                                                'NHWC', name='conv_l2')
            print('Shape of conv_l2: ', self.conv_l2.get_shape())
            # it should be [None, 5, 5, 64]
            self.conv_l3, self.var['conv_l3_w'], self.var['conv_l3_b'] = conv2d(self.conv_l2,  # input of cnn
                                                                                64,  # number of out channels
                                                                                [3, 3],  # size of kernel (filter)
                                                                                [1, 1],  # strides
                                                                                weights_initializer,
                                                                                biases_initializer,
                                                                                hidden_activation_fn,
                                                                                'NHWC', name='conv_l3')
            print('Shape of conv_l3: ', self.conv_l3.get_shape())
            # it should be [None, 3, 3, 64]
        self.conv_out = tf.contrib.layers.flatten(self.conv_l3)
        self.conv_out_dim = self.conv_out.get_shape()[1]
        print('Out dimension of conv: ', self.conv_out_dim)

    # Make sure to call this method after calling build_conv
    def build_classifier(self):
        layer = self.conv_out
        self.true_class = tf.placeholder('float32', [None, self.n_classes],
            name='true_class')
        for idx, hidden_size in enumerate(self.clf_hidden_sizes):
          w_name, b_name = 'w_c_%d' % idx, 'b_c_%d' % idx
          layer, self.var[w_name], self.var[b_name] = \
                    linear(layer, hidden_size, activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        name='lin_c_%d' % idx)
        self.clf_logits, self.var['w_c_out'], self.var['b_c_out'] = \
                    linear(layer, self.n_classes, activation_fn=None,
                            biases_initializer=tf.zeros_initializer(), name='lin_c_out')
        self.clf_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.clf_logits,
                        labels=self.true_class))
        self.clf_softmax = tf.nn.softmax(self.clf_logits)
        self.clf_pred = tf.argmax(self.clf_softmax, 1)
        self.clf_correct = tf.equal(tf.argmax(self.true_class, 1), self.clf_pred)
        self.clf_accuracy = tf.reduce_mean(tf.cast(self.clf_correct, 'float'))
        self.clf_lr = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdamOptimizer(learning_rate=self.clf_lr)
        self.clf_optim = optimizer.minimize(self.clf_loss)

    # inputs and acquired should be flattened
    def clf_predict(self, inputs, acquired, labels):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        if len(acquired.shape) == 1:
            acquired = acquired.reshape(1, -1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)
        mnist_mask = mnist_mask_batch(acquired, self.expand_size).reshape([-1, self.input_dim[0], self.input_dim[1]])
        inputs = inputs.reshape([-1, self.input_dim[0], self.input_dim[1]])
        # mnist_mask and inputs now are of shape (example) [64, 56, 56] we need to stack them into shape [64, 56, 56, 2]
        conv_in = np.stack([inputs, mnist_mask], axis=3)

        return self.sess.run([self.clf_softmax, self.clf_pred, self.clf_correct],
                             feed_dict={
                                 self.conv_inputs: conv_in,
                                 self.true_class: labels})


    def stat(self, array):
        return np.mean(array), np.amin(array), np.amax(array), np.median(array)



