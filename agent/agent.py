from network.layer import linear
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('.'))))


class Agent(object):
    def __init__(self,
                 sess,
                 conf,
                 name='Agent'):

        self.sess = sess
        self.var = {}
        # problem
        self.input_dim = conf.input_dim
        self.n_features = conf.n_features
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


    def build_classifier(self):
        layer = self.clf_inputs = tf.placeholder('float32', [None, self.input_dim],
            name='clf_inputs')
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

    def clf_predict(self, inputs, acquired, labels):
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        if len(acquired.shape) == 1:
            acquired = acquired.reshape(1, -1)
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)

        clf_inputs = np.concatenate((inputs, acquired), axis=1)
        return self.sess.run([self.clf_softmax, self.clf_pred, self.clf_correct],
                             feed_dict={
                                 self.clf_inputs: clf_inputs,
                                 self.true_class: labels})

    def stat(self, array):
        return np.mean(array), np.amin(array), np.amax(array), np.median(array)



