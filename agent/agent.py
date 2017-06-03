import time
import random
import numpy as np
import tensorflow as tf
#from logging import getLogger
#import matplotlib.pyplot as plt
from .experience import Experience
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('.'))))
from network.layer import linear
from helper import mnist_mask_batch, timeit
# from environment.environment import Environment

# logger = getLogger(__name__)


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

        # experience replay memory init
        observation_dim = self.input_dim - self.n_features
        self.experience = Experience(conf.batch_size, conf.memory_size,
                                     self.n_features, self.n_classes, [observation_dim])
        # # # [self.n_features] should be changed! to real observation dim


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

    def train(self, tr_data, tr_labels, verbose=True):
        self.update_target_q_network()
        total_steps = 0
        n_acquired_history = []
        accuracy_history = []
        history = []
        reward_history = []
        lr = self.max_lr
        clf_lr = self.clf_max_lr
        for epoch in range(self.n_epoch):
            for x, label in zip(tr_data, tr_labels):
                acquired = np.zeros(self.n_features)
                # add initial state to replay memory
                if random.random() > 0.5:
                    self.experience.add(np.zeros(x.shape), acquired, 0, -1, False, label)
                n_acquired = 0
                epi_reward = 0
                terminal = False
                while not terminal:
                    # choose action
                    if total_steps < self.pre_train_steps:
                        action = self.random_action(acquired)
                    else:
                        action = self.choose_action(x, acquired, self.eps, policy=self.policy)
                    if not self.is_terminal(action):
                        # feature acquisition
                        terminal = False
                        self.update_acquired(acquired, action)
                        reward = self.r_cost
                        n_acquired += 1
                        observed = self.get_observed(x, acquired)
                    else:
                        # make a decision (terminal state)
                        terminal = True
                        observed = self.get_observed(x, acquired)
                        prob, pred, correct = self.clf_predict(observed, acquired, label)
                        prob = prob.reshape(-1)
                        pred = pred[0]
                        correct = correct[0]
                        accuracy_history.append(int(correct))
                        assert len(acquired.shape) == 1
                        if correct:
                            sorted_prob = np.sort(prob)
                            reward = sorted_prob[-1] - sorted_prob[-2]
                        else:
                            reward = self.r_wrong
                    epi_reward += reward
                    # save experience
                    self.experience.add(observed, acquired, reward, action, terminal, label)
                    if terminal:
                        if n_acquired == 1 and verbose:
                            print("label", np.argmax(label))
                            print("prediction", pred)
                        break
                reward_history.append(epi_reward)
                assert n_acquired == np.sum(acquired)
                n_acquired_history.append(n_acquired)
                # sample batch
                prestates, unmissing_pre, actions_t, rewards, poststates, unmissing, terminals, labels \
                    = self.experience.sample()
                s_t = np.concatenate((prestates, unmissing_pre), axis=1)
                s_t_plus_1 = np.concatenate((poststates, unmissing), axis=1)
                targets = self.calc_targets(unmissing, s_t_plus_1, poststates, terminals, rewards)
                # train
                clf_inputs = np.concatenate((s_t, s_t_plus_1), axis=0)
                clf_true_class = np.concatenate((labels, labels), axis=0)

                _, _, loss, q_t, clf_accuracy, clf_softmax = self.train_sess_run(targets, actions_t, s_t, lr,
                                                                                 clf_inputs, clf_true_class,
                                                                                 clf_lr)

                total_steps += 1
                if total_steps > self.pre_train_steps and self.eps > self.endE:
                    self.eps -= self.eps_decay

                # update target network parameter
                if total_steps % self.update_freq == self.update_freq - 1:
                    # print("Target Q update")
                    self.update_target_q_network()
                if (total_steps + 1) % 100 == 0:
                    print("------------------(", total_steps + 1, "/",
                          self.n_epoch * self.train_size, ")------------------")
                    print("> current eps    :", self.eps)
                    print("> mean n_acquired:", np.mean(n_acquired_history))
                    print("> accuracy       :", np.mean(accuracy_history))
                    print("> reward         :", np.mean(reward_history))
                    history.append(np.mean(n_acquired_history))
                    n_acquired_history = []  # reset
                    accuracy_history = []
                    reward_history = []
        print("------------------ train done ------------------")
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_path)

        return history

